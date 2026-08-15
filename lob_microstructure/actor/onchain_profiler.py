"""Katman 3 — gerçek on-chain cüzdan profili.

wallet_profiler.py'deki sahte (deterministik) profili gerçek RPC verisiyle
değiştirir:
  * balance_eth  ← eth_getBalance
  * tx_count     ← eth_getTransactionCount (nonce; gönderilen tx sayısı)
  * age_days     ← ilk tx zamanı. Saf RPC'de ilk tx pahalı (blok taraması)
                   olduğundan opsiyonel Etherscan API ile çekilir; yoksa
                   nonce'tan kaba tahmin edilir.
  * lifetime_volume_usd ← yaklaşık (Etherscan/Dune ile zenginleştirilebilir)

Test edilebilirlik için web3 ve http çağrıları enjekte edilebilir
(w3 ve _http parametreleri). Ağ olmadan birim test için sahte enjekte edilir.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Optional

from lob_microstructure.models import WalletProfile

log = logging.getLogger("onchain")

ETH_USD = 3000.0


class OnChainProfiler:
    def __init__(self, w3=None, etherscan_key: Optional[str] = None,
                 http_get=None):
        self._w3 = w3
        self._key = etherscan_key
        self._http = http_get or _default_http_get
        self._cache: dict[str, WalletProfile] = {}

    async def get_profile(self, address: str) -> WalletProfile:
        a = address.lower()
        if a in self._cache:
            return self._cache[a]
        balance_eth = await self._balance(a)
        tx_count = await self._nonce(a)
        age_days = await self._age_days(a, tx_count)
        # Hacim tahmini: bakiye + aktiviteden kaba; gerçek için Etherscan/Dune
        lifetime_volume_usd = round(balance_eth * ETH_USD * (1 + tx_count / 500.0), 2)
        prof = WalletProfile(
            address=a, age_days=float(age_days), tx_count=int(tx_count),
            balance_eth=round(balance_eth, 4), lifetime_volume_usd=lifetime_volume_usd,
        )
        self._cache[a] = prof
        return prof

    async def _balance(self, a: str) -> float:
        if self._w3 is None:
            return 0.0
        try:
            wei = await self._w3.eth.get_balance(a)
            return wei / 10**18
        except Exception as e:
            log.debug("balance hatası %s: %s", a[:10], e)
            return 0.0

    async def _nonce(self, a: str) -> int:
        if self._w3 is None:
            return 0
        try:
            return int(await self._w3.eth.get_transaction_count(a))
        except Exception as e:
            log.debug("nonce hatası %s: %s", a[:10], e)
            return 0

    async def _age_days(self, a: str, tx_count: int) -> float:
        # Etherscan varsa gerçek ilk tx zamanı
        if self._key:
            ts = self._first_tx_ts(a)
            if ts:
                return max(0.0, (time.time() - ts) / 86400.0)
        # Fallback: nonce'tan kaba tahmin (aktif cüzdan ~ daha yaşlı)
        return float(min(tx_count, 2000)) / 2.0

    def _first_tx_ts(self, a: str) -> Optional[int]:
        url = ("https://api.etherscan.io/api?module=account&action=txlist"
               f"&address={a}&startblock=0&endblock=99999999&page=1&offset=1"
               f"&sort=asc&apikey={self._key}")
        try:
            data = self._http(url)
            res = data.get("result") or []
            if res:
                return int(res[0]["timeStamp"])
        except Exception as e:
            log.debug("etherscan ilk tx hatası %s: %s", a[:10], e)
        return None


def _default_http_get(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url, headers={"User-Agent": "microstructure-analyzer"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())
