"""Katman 1 — Ingest.

Mempool'daki (pending) işlemleri WebSocket üzerinden gerçek zamanlı dinler,
normalize edilmiş PendingTx nesneleri üretir ve bir asyncio.Queue'ya basar.

İki mod:
  * Gerçek mod: WSS_URL ayarlıysa node'a `eth_subscribe newPendingTransactions`,
    ardından `eth_getTransactionByHash` ile gövde çekilir.
  * Simülasyon: WSS_URL yoksa sentetik balina/MEV/retail işlemleri üretir,
    böylece node olmadan tüm boru hattı uçtan uca test edilebilir.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Optional

from config import CONFIG
from src.models import PendingTx
from src.decode.tx_decoder import UNISWAP_V2_ROUTER, UNISWAP_V3_ROUTER, WETH

log = logging.getLogger("ingest")

# Simülasyonda kullanılan örnek token (USDC) — WETH karşısı
_SAMPLE_TOKEN = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"


def _v3_single_calldata(weth_in: bool, amount_wei: int) -> str:
    """Gerçekçi V3 exactInputSingle calldata üret (eth_abi ile kodlanmış).

    weth_in=True  → WETH girişi (BUY), weth_in=False → WETH çıkışı (SELL).
    eth_abi yoksa zero-calldata'ya düşer (decode None/UNKNOWN verir)."""
    try:
        from eth_abi import encode  # type: ignore
    except Exception:
        return "0x414bf389" + "0" * 64
    token_in = WETH if weth_in else _SAMPLE_TOKEN
    token_out = _SAMPLE_TOKEN if weth_in else WETH
    # (tokenIn,tokenOut,fee,recipient,deadline,amountIn,amountOutMin,sqrtLimit)
    params = (token_in, token_out, 3000, "0x" + "11" * 20, 0, amount_wei,
              int(amount_wei * 0.99), 0)
    body = encode(
        ["(address,address,uint24,address,uint256,uint256,uint256,uint160)"],
        [params],
    )
    return "0x414bf389" + body.hex()

# Kütüphane opsiyonel: simülasyon modunda websockets gerekmez.
try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover
    websockets = None


class MempoolListener:
    def __init__(self, out_queue: "asyncio.Queue[PendingTx]"):
        self.q = out_queue
        self._stop = asyncio.Event()
        self._w3 = None  # lazy AsyncWeb3, tx gövdesi çekmek için

    async def run(self) -> None:
        if CONFIG.simulation_mode:
            log.warning("WSS_URL yok → SIMÜLASYON modu. Sentetik işlemler üretiliyor.")
            await self._run_simulation()
        else:
            await self._run_websocket()

    def stop(self) -> None:
        self._stop.set()

    # ---------- Gerçek mod ----------
    async def _run_websocket(self) -> None:
        if websockets is None:
            raise RuntimeError("websockets kurulu değil: pip install websockets")
        backoff = 1
        while not self._stop.is_set():
            try:
                async with websockets.connect(CONFIG.wss_url, ping_interval=15) as ws:
                    await ws.send(json.dumps({
                        "id": 1, "method": "eth_subscribe",
                        "params": ["newPendingTransactions"],
                    }))
                    sub = await ws.recv()
                    log.info("Mempool aboneliği aktif: %s", sub)
                    backoff = 1
                    while not self._stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        await self._handle_raw(raw)
            except asyncio.TimeoutError:
                continue  # sessiz dönem, bağlantı canlı
            except Exception as e:  # reconnect
                log.error("WS hatası: %s — %ss sonra yeniden bağlanılıyor", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _handle_raw(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            tx_hash = msg["params"]["result"]
        except (KeyError, TypeError, json.JSONDecodeError):
            return
        # newPendingTransactions sadece hash döndürür; detayı çekmek gerekir.
        tx = await self._fetch_tx(tx_hash)
        if tx is not None:
            await self.q.put(tx)

    async def _ensure_w3(self):
        """AsyncWeb3'ü WebSocket sağlayıcısıyla tembel başlat."""
        if self._w3 is not None:
            return self._w3
        from web3 import AsyncWeb3  # type: ignore
        from web3.providers.persistent import WebSocketProvider  # type: ignore
        self._w3 = AsyncWeb3(WebSocketProvider(CONFIG.wss_url))
        await self._w3.provider.connect()
        return self._w3

    async def _fetch_tx(self, tx_hash: str) -> Optional[PendingTx]:
        """eth_getTransactionByHash ile pending tx gövdesini çek ve normalize et.

        newPendingTransactions yalnızca hash döndürür; gövde için ekstra RPC
        gerekir. Yüksek throughput'ta bunu batch'lemek ya da
        `newPendingTransactionsWithBody` (Alchemy) /
        `alchemy_pendingTransactions` kullanmak tercih edilir."""
        try:
            w3 = await self._ensure_w3()
            tx = await w3.eth.get_transaction(tx_hash)
        except Exception as e:
            # tx mempool'dan düşmüş olabilir (already mined / dropped) — sessiz geç
            log.debug("tx çekilemedi %s: %s", tx_hash[:10], e)
            return None
        # EIP-1559 ise maxFeePerGas, değilse gasPrice
        gas_price = tx.get("maxFeePerGas") or tx.get("gasPrice") or 0
        inp = tx.get("input", "0x")
        return PendingTx(
            tx_hash=tx_hash,
            from_addr=(tx.get("from") or "").lower(),
            to_addr=(tx.get("to") or None) and tx["to"].lower(),
            value_wei=int(tx.get("value", 0)),
            gas_price_wei=int(gas_price),
            input_data=inp.hex() if hasattr(inp, "hex") else str(inp),
        )

    # ---------- Simülasyon ----------
    async def _run_simulation(self) -> None:
        archetypes = ["whale", "mev", "retail", "retail", "mev"]
        while not self._stop.is_set():
            await asyncio.sleep(random.uniform(0.2, 1.0))
            await self.q.put(self._synthetic_tx(random.choice(archetypes)))

    def _synthetic_tx(self, kind: str) -> PendingTx:
        base_fee = 20 * 10**9  # ~20 gwei varsayılan blok baz ücreti
        addr = "0x" + "".join(random.choices("0123456789abcdef", k=40))

        def _hash() -> str:
            return "0x" + "".join(random.choices("0123456789abcdef", k=64))

        # Gerçekçi V2 calldata: swapExactETHForTokens (0x7ff36ab5) imzası ile başlat
        v2_calldata = "0x7ff36ab5" + "0" * 64
        if kind == "whale":
            return PendingTx(
                tx_hash=_hash(), from_addr=addr, to_addr=UNISWAP_V2_ROUTER,
                value_wei=int(random.uniform(80, 400) * 10**18),  # 80-400 ETH
                gas_price_wei=int(base_fee * random.uniform(1.0, 1.3)),
                input_data=v2_calldata,
            )
        if kind == "mev":
            weth_in = random.random() < 0.5  # bazı botlar alır, bazıları satar
            amt = int(random.uniform(0.5, 5) * 10**18)
            return PendingTx(
                tx_hash=_hash(), from_addr=addr, to_addr=UNISWAP_V3_ROUTER,
                value_wei=amt if weth_in else 0,
                gas_price_wei=int(base_fee * random.uniform(3.0, 8.0)),  # aşırı gas
                input_data=_v3_single_calldata(weth_in, amt),  # gerçek V3 calldata
            )
        return PendingTx(  # retail
            tx_hash=_hash(), from_addr=addr, to_addr=UNISWAP_V2_ROUTER,
            value_wei=int(random.uniform(0.05, 2) * 10**18),
            gas_price_wei=int(base_fee * random.uniform(1.0, 1.5)),
            input_data=v2_calldata,
        )
