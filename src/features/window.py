"""Katman 4 — Feature extraction.

Etiketlenmiş ActorSignal akışını token bazında zaman pencerelerinde toplar ve
aktör-ağırlıklı **order flow imbalance** üretir.

Sezgi: balina alımı güçlü yukarı baskı, balina satışı güçlü aşağı baskı;
retail orta; MEV botları çoğunlukla gürültü/arbitraj olduğundan ağırlığı düşük
tutulur (yön sinyali olarak güvenilmez). İmbalance = (ağırlıklı alış − ağırlıklı
satış) / toplam, [-1, +1] aralığında.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

from src.models import ActorSignal, ActorLabel, Side
from src.features.vpin import VPIN

# Aktör tipine göre yön sinyali ağırlığı
ACTOR_WEIGHT = {
    ActorLabel.WHALE: 1.0,
    ActorLabel.RETAIL: 0.4,
    ActorLabel.MEV_BOT: 0.15,   # botlar yön için zayıf sinyal
    ActorLabel.UNKNOWN: 0.2,
}


@dataclass
class FlowFeatures:
    token: str
    flow_imbalance: float       # -1..+1 (yavaş/tam pencere)
    buy_usd: float
    sell_usd: float
    whale_net_usd: float
    sample_count: int
    window_sec: float
    # --- çok-ufuklu ve toksisite eklentileri ---
    vpin: float = 0.0               # 0..1 akış toksisitesi (VPIN)
    imbalance_fast: float = 0.0     # kısa ufuk imbalance
    window_fast_sec: float = 0.0


class RollingFlow:
    """Token bazında kayan zaman penceresi + VPIN + hızlı/yavaş ufuk.

    fast_frac: hızlı ufuk = window_sec * fast_frac (örn. 60s → 15s).
    """

    def __init__(self, window_sec: float = 60.0, fast_frac: float = 0.25,
                 vpin_bucket_usd: float = 250_000.0):
        self.window_sec = window_sec
        self.window_fast_sec = window_sec * fast_frac
        self.vpin_bucket_usd = vpin_bucket_usd
        # token -> deque[(ts, signed_weighted_usd, side, label, usd)]
        self._buf: dict[str, deque] = defaultdict(deque)
        self._vpin: dict[str, VPIN] = {}

    def add(self, sig: ActorSignal) -> None:
        token = _bucket(sig)
        w = ACTOR_WEIGHT.get(sig.label, 0.2)
        signed = sig.est_value_usd * w * _dir(sig.side)
        self._buf[token].append(
            (time.time(), signed, sig.side, sig.label, sig.est_value_usd)
        )
        # VPIN'i ham (ağırlıksız) hacim ve yönle besle
        if sig.side in (Side.BUY, Side.SELL):
            self._vpin.setdefault(token, VPIN(self.vpin_bucket_usd)).add(
                sig.est_value_usd, sig.side == Side.BUY)

    def _evict(self, token: str, now: float) -> None:
        buf = self._buf[token]
        cutoff = now - self.window_sec
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    @staticmethod
    def _imbalance(buf, since_ts: float) -> float:
        buy = sum(s for ts, s, *_ in buf if s > 0 and ts >= since_ts)
        sell = -sum(s for ts, s, *_ in buf if s < 0 and ts >= since_ts)
        total = buy + sell
        return (buy - sell) / total if total > 0 else 0.0

    def features(self, token: str) -> FlowFeatures:
        now = time.time()
        self._evict(token, now)
        buf = self._buf[token]
        buy = sum(s for _, s, side, _, _ in buf if s > 0)
        sell = -sum(s for _, s, side, _, _ in buf if s < 0)
        whale_net = sum(
            usd * _dir(side)
            for _, _, side, label, usd in buf if label == ActorLabel.WHALE
        )
        total = buy + sell
        imb = (buy - sell) / total if total > 0 else 0.0
        imb_fast = self._imbalance(buf, now - self.window_fast_sec)
        vpin_val = self._vpin[token].value() if token in self._vpin else 0.0
        return FlowFeatures(
            token=token, flow_imbalance=imb, buy_usd=buy, sell_usd=sell,
            whale_net_usd=whale_net, sample_count=len(buf),
            window_sec=self.window_sec,
            vpin=round(vpin_val, 4), imbalance_fast=round(imb_fast, 4),
            window_fast_sec=self.window_fast_sec,
        )

    def tokens(self) -> list[str]:
        return [t for t, b in self._buf.items() if b]

    def actor_mix(self, token: str) -> dict:
        """Pencere icindeki aktor paylari (WHALE/MEV_BOT/RETAIL), hacim-agirlikli.

        CAS entegrasyonu FlowState.actor_mix alani icin -- additif, mevcut
        features() imzasini degistirmez. Bos pencerede esit/notr dagilim
        dondurur (uc aktor icin 1/3'er)."""
        now = time.time()
        self._evict(token, now)
        buf = self._buf[token]
        totals = {
            ActorLabel.WHALE.value: 0.0,
            ActorLabel.MEV_BOT.value: 0.0,
            ActorLabel.RETAIL.value: 0.0,
        }
        grand_total = 0.0
        for _, _, _, label, usd in buf:
            key = label.value if label in (
                ActorLabel.WHALE, ActorLabel.MEV_BOT, ActorLabel.RETAIL
            ) else None
            if key is None:
                continue  # UNKNOWN aktor mix'e dahil edilmez
            totals[key] += abs(usd)
            grand_total += abs(usd)
        if grand_total <= 0:
            n = len(totals)
            return {k: round(1.0 / n, 6) for k in totals}
        return {k: round(v / grand_total, 6) for k, v in totals.items()}


def _dir(side: Side) -> float:
    if side == Side.BUY:
        return 1.0
    if side == Side.SELL:
        return -1.0
    return 0.0  # UNKNOWN yöne katkı vermez ama hacme sayılır


def _bucket(sig: ActorSignal) -> str:
    """Token kovası anahtarı. Gerçekte decode token_out; PoC'de dex bazlı."""
    return sig.dex
