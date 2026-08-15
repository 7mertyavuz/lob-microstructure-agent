"""Katman 1 köprüsü — `FlowFeed` okuma arayüzü.

Mevcut ingest→decode→actor→features→predict hesaplamalarını **yeniden
yazmaz**; onları sarıp tek bir dışa-dönük `FlowState` struct'ında paketler.
Sözleşme: `docs/00-ORTAK-SOZLESME.md`.

Sim modu birinci sınıftır: `CONFIG.simulation_mode` (yani `WSS_URL` boş) ise
harici bağlantı olmadan, deterministik (seed'e bağlı) sentetik bir işlem
akışıyla geçerli `FlowState` üretir. Canlı modda gerçek `RollingFlow`
beslemesi (main.py'nin worker'ları) ve isteğe bağlı CEX/DEX fiyat akışı
kullanılır.

ÖNEMLİ: `_bucket()` (features/window.py) token kovasını `ActorSignal.dex`
alanına göre seçer (ör. "UniswapV2", "UniswapV3"). Bu yüzden `latest(token)`
çağrısındaki `token` parametresi bu dex bucket adlarından biriyle eşleşmelidir
— gerçek token sembolü değil (mevcut repo tasarımının bir kısıtı, burada
değiştirilmedi).
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

from typing import Callable

from lob_microstructure.config import CONFIG
from lob_microstructure.models import FlowState
from lob_microstructure.book.state import BookState
from lob_microstructure.features.window import RollingFlow
from lob_microstructure.features.lead_lag import LeadLagSpread
from lob_microstructure.predict.direction import predict, predict_toxic
from lob_microstructure.predict.regime import RegimeRouter, THIN, TOXIC
from lob_microstructure.decode.tx_decoder import decode_tx, UNISWAP_V2_ROUTER, UNISWAP_V3_ROUTER, WETH
from lob_microstructure.actor.classifier import classify
from lob_microstructure.models import PendingTx

# ---------------- Rejim eşlemesi (1b) ----------------
REGIME_NORMAL = "normal"
REGIME_TOXIC = "toxic"
REGIME_HIGHVOL = "highvol"
_ALLOWED_REGIMES = (REGIME_NORMAL, REGIME_TOXIC, REGIME_HIGHVOL)


def map_regime(vpin: float, vpin_threshold: float = 0.4,
                highvol_threshold: float = 0.7,
                book_state: BookState | None = None) -> str:
    """Rejimi sözleşme etiketine çevirir (`normal|toxic|highvol`).

    `book_state` verildiğinde karar `predict/regime.py::RegimeRouter`'a
    devredilir; böylece defterin likidite rejimi (spread / eğim / Kyle λ)
    de hesaba katılır. Router'ın `THIN` çıktısı sözleşmede `highvol`e
    eşlenir: ince defter, aynı emrin çok daha büyük fiyat hareketi
    yaratması demektir.

    Defter yoksa eski saf VPIN eşiği korunur — geriye uyumlu.

    NOT: `RegimeRouter` uzun süre bu köprüden hiç çağrılmıyordu; `GaussianHMM2`
    ile birlikte fiilen ölü koddu. Artık defter varsa gerçekten kullanılıyor.
    """
    vpin = vpin or 0.0
    if book_state is not None:
        router = _contract_router(vpin_threshold)
        r = router.regime_of(vpin, book_state)
        if r == THIN:
            return REGIME_HIGHVOL
        if r == TOXIC:
            return REGIME_HIGHVOL if vpin >= highvol_threshold else REGIME_TOXIC
        return REGIME_NORMAL
    if vpin >= highvol_threshold:
        return REGIME_HIGHVOL
    if vpin >= vpin_threshold:
        return REGIME_TOXIC
    return REGIME_NORMAL


def _contract_router(vpin_threshold: float = 0.4) -> RegimeRouter:
    """Rejim ETİKETİ için router. Tahmin fonksiyonları burada kullanılmaz —
    `regime_of()` saf bir sınıflandırıcıdır, yan etkisi yoktur."""
    return RegimeRouter(normal_fn=predict, toxic_fn=predict_toxic,
                        vpin_threshold=vpin_threshold)


# ---------------- Sim modu: deterministik sentetik akış ----------------
_ARCHETYPES = ["whale", "mev", "retail", "retail", "mev"]


def _synthetic_tx(kind: str, rng: random.Random) -> PendingTx:
    """`lob_microstructure/ingest/mempool_listener.py::_synthetic_tx` ile aynı desenler,
    ama enjekte edilebilir `random.Random` ile (modül-seviyeli `random`
    yerine) — böylece `FlowFeed(seed=...)` tam determinizm sağlar."""
    base_fee = 20 * 10**9

    def _hex(n: int) -> str:
        return "".join(rng.choices("0123456789abcdef", k=n))

    addr = "0x" + _hex(40)
    tx_hash = "0x" + _hex(64)
    v2_calldata = "0x7ff36ab5" + "0" * 64

    if kind == "whale":
        return PendingTx(
            tx_hash=tx_hash, from_addr=addr, to_addr=UNISWAP_V2_ROUTER,
            value_wei=int(rng.uniform(80, 400) * 10**18),
            gas_price_wei=int(base_fee * rng.uniform(1.0, 1.3)),
            input_data=v2_calldata,
        )
    if kind == "mev":
        weth_in = rng.random() < 0.5
        amt = int(rng.uniform(0.5, 5) * 10**18)
        return PendingTx(
            tx_hash=tx_hash, from_addr=addr, to_addr=UNISWAP_V3_ROUTER,
            value_wei=amt if weth_in else 0,
            gas_price_wei=int(base_fee * rng.uniform(3.0, 8.0)),
            input_data=_v3_single_calldata(weth_in, amt, rng),
        )
    return PendingTx(  # retail
        tx_hash=tx_hash, from_addr=addr, to_addr=UNISWAP_V2_ROUTER,
        value_wei=int(rng.uniform(0.05, 2) * 10**18),
        gas_price_wei=int(base_fee * rng.uniform(1.0, 1.5)),
        input_data=v2_calldata,
    )


_SAMPLE_TOKEN = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"


def _v3_single_calldata(weth_in: bool, amount_wei: int, rng: random.Random) -> str:
    try:
        from eth_abi import encode  # type: ignore
    except Exception:
        return "0x414bf389" + "0" * 64
    token_in = WETH if weth_in else _SAMPLE_TOKEN
    token_out = _SAMPLE_TOKEN if weth_in else WETH
    params = (token_in, token_out, 3000, "0x" + "11" * 20, 0, amount_wei,
              int(amount_wei * 0.99), 0)
    body = encode(
        ["(address,address,uint24,address,uint256,uint256,uint256,uint160)"],
        [params],
    )
    return "0x414bf389" + body.hex()


class FlowFeed:
    """Katman 1 köprüsü — `latest(token) -> FlowState`.

    mode: "simulation" | "live"; `CONFIG.simulation_mode` (WSS_URL boş) ise
    otomatik "simulation"'a düşer. `print` yapmaz, her zaman `FlowState`
    struct'ı döndürür.
    """

    def __init__(self, mode: str = "simulation", seed: int | None = None,
                 window_sec: float = 60.0, batch_range: tuple[int, int] = (3, 8),
                 book_feed=None, clock: Callable[[], float] | None = None):
        if mode not in ("simulation", "live"):
            raise ValueError("mode 'simulation' ya da 'live' olmalı")
        self.mode = "simulation" if CONFIG.simulation_mode else mode
        self.seed = seed
        self._rng = random.Random(seed)
        self.window_sec = window_sec
        self._batch_range = batch_range
        # Enjekte edilebilir saat sözleşmenin belirlilik kuralı gereği
        # RollingFlow'a geçirilir (yoksa duvar saati).
        self.flow = RollingFlow(window_sec=window_sec, clock=clock)
        # Opsiyonel defter beslemesi. Verilirse `BookState` gerçekten
        # `predict()`'e ulaşır ve D5'in BOOK_WEIGHT=0.35 düzeltmesi ateşlenir;
        # bu yol uzun süre ölüydü (predict book_state olmadan çağrılıyordu).
        self.book_feed = book_feed
        self._live_lead_lag: dict[str, LeadLagSpread] = {}

    # ---------- canlı mod yardımcıları ----------
    def feed_live_signal(self, signal) -> None:
        """Canlı modda main.py worker'larının ürettiği `ActorSignal`'i besler."""
        self.flow.add(signal)

    def update_live_prices(self, token: str, cex: float | None = None,
                            dex: float | None = None) -> None:
        """Canlı modda CEX/DEX fiyat güncellemesi (lead-lag spread için)."""
        ll = self._live_lead_lag.setdefault(token, LeadLagSpread())
        if cex is not None:
            ll.update_cex(cex)
        if dex is not None:
            ll.update_dex(dex)

    # ---------- sim modu ----------
    def _feed_synthetic(self) -> None:
        n = self._rng.randint(*self._batch_range)
        for _ in range(n):
            kind = self._rng.choice(_ARCHETYPES)
            tx = _synthetic_tx(kind, self._rng)
            swap = decode_tx(tx)
            if swap is None:
                continue
            signal = classify(swap)
            self.flow.add(signal)

    def _book_state(self, token: str) -> BookState | None:
        """Bağlı defter beslemesinden `BookState` oku; yoksa None.

        Defter okuması yön oyunun kendisi değildir — `predict()` içinde düşük
        ağırlıklı bir düzeltme ve bir güven çarpanıdır. Sözleşmenin analist
        kuralı korunur: ham metrik verilir, ağırlık kararı verilmez.
        """
        if self.book_feed is None:
            return None
        try:
            return self.book_feed.latest(token)
        except Exception:
            # Defter okunamıyorsa akış tarafı çalışmaya devam eder; sessizce
            # None dönmek, tüm FlowState'i düşürmekten iyidir.
            return None

    def _lead_lag_spread(self, token: str) -> float:
        if self.mode == "simulation":
            return round(self._rng.uniform(-0.01, 0.01), 6)
        ll = self._live_lead_lag.get(token)
        return ll.spread() if ll else 0.0

    # ---------- ortak çıkış ----------
    def _read_state(self, token: str) -> FlowState:
        """Autonomous sentetik beslemeyi TETİKLEMEDEN mevcut RollingFlow
        durumunu okur. `SimEnvironment` (Faz 3) bunu kullanır: enjekte
        edilen emirlerle beslenen akışı, ek sentetik gürültü katmadan
        okumak için."""
        feat = self.flow.features(token)
        book = self._book_state(token)
        # `book` geçirilmezse D5'in defter düzeltmesi (BOOK_WEIGHT=0.35, spoof
        # cezası, likidite güven çarpanı) hiç ateşlenmez — eski davranış buydu.
        pred = predict(feat, book)
        mix = self.flow.actor_mix(token)
        regime = map_regime(feat.vpin, book_state=book)
        lead_lag = self._lead_lag_spread(token)

        return FlowState(
            token=token,
            flow_imbalance=pred.flow_imbalance,
            vpin_toxicity=feat.vpin,
            whale_net_usd=feat.whale_net_usd,
            actor_mix=mix,
            direction_prob_up=pred.prob_up,
            lead_lag_spread=lead_lag,
            regime=regime,
            ts=datetime.now(timezone.utc),
        )

    def latest(self, token: str) -> FlowState:
        if self.mode == "simulation":
            self._feed_synthetic()
        return self._read_state(token)
