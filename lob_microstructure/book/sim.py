"""SimBookFeed — deterministik, rejim-anahtarlamalı sentetik L2 defteri.

Mevcut `SimOrderbookFeed` (signalcore) rastgele-yürüyüş stub'ının ciddi hâli:
gerçek bir derinlik merdiveni, tape ve emir-yaşam-döngüsü olayları üretir.
Aynı seed → aynı defter dizisi (determinizm kuralı, docs/00-ORTAK-SOZLESME.md).

Üç rejim:
- "calm"  : dar spread, dengeli derinlik, düşük volatilite.
- "toxic" : tek-yönlü baskı, ince karşı taraf, sweep + spoof olayları.
- "wide"  : geniş spread, derinlik boşlukları, yüksek volatilite.

Sim birinci sınıftır: harici bağlantı/anahtar olmadan geçerli defter üretir.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone, timedelta

from lob_microstructure.book.state import BookLevel, Trade, OrderEvent, RawBook

CALM, TOXIC, WIDE = "calm", "toxic", "wide"
_REGIMES = (CALM, TOXIC, WIDE)


class SimBookFeed:
    """Sembol başına durum tutan, deterministik sentetik defter üreteci.

    `next(symbol)` bir adım ilerletip yeni `RawBook` döndürür. Saat enjekte
    edilebilir (varsayılan: sabit epoch'tan sabit adımlı) — böylece tape/olay
    zaman damgaları da deterministiktir.
    """

    def __init__(self, seed: int | None = None, base_price: float = 30_000.0,
                 levels: int = 20, step_sec: float = 1.0,
                 regime_persistence: float = 0.9):
        self.seed = seed
        self._rng = random.Random(seed)
        self.base_price = base_price
        self.levels = levels
        self.step_sec = step_sec
        self.regime_persistence = regime_persistence
        self._mid: dict[str, float] = {}
        self._regime: dict[str, str] = {}
        self._t: float = 0.0
        self._epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # ---------- rejim ----------
    def _next_regime(self, symbol: str) -> str:
        cur = self._regime.get(symbol)
        if cur is None or self._rng.random() > self.regime_persistence:
            cur = self._rng.choice(_REGIMES)
        self._regime[symbol] = cur
        return cur

    def _regime_params(self, regime: str) -> dict:
        if regime == CALM:
            return dict(vol=0.0004, spread_ticks=1, base_qty=5.0,
                        imbalance=0.0, gap_prob=0.02, n_trades=(2, 5),
                        spoof=False, sweep=False)
        if regime == TOXIC:
            return dict(vol=0.0012, spread_ticks=2, base_qty=4.0,
                        imbalance=0.6, gap_prob=0.10, n_trades=(4, 9),
                        spoof=True, sweep=True)
        return dict(vol=0.0018, spread_ticks=6, base_qty=3.0,  # WIDE
                    imbalance=0.1, gap_prob=0.35, n_trades=(1, 4),
                    spoof=False, sweep=False)

    # ---------- üretim ----------
    def next(self, symbol: str) -> RawBook:
        rng = self._rng
        mid = self._mid.get(symbol, self.base_price)
        regime = self._next_regime(symbol)
        p = self._regime_params(regime)

        # 1) mid yürüyüşü
        mid *= (1.0 + rng.gauss(0.0, p["vol"]))
        self._mid[symbol] = mid
        tick = max(mid * 5e-5, 1e-9)

        # 2) merdiven — imbalance: + ise bid tarafı ağır
        imb = p["imbalance"] * (1 if rng.random() < 0.5 else -1)
        half_spread = tick * p["spread_ticks"] / 2.0
        best_bid = mid - half_spread
        best_ask = mid + half_spread

        bids = self._ladder(best_bid, -1, tick, p, rng, side_boost=1.0 + imb)
        asks = self._ladder(best_ask, +1, tick, p, rng, side_boost=1.0 - imb)

        # 3) tape
        trades = self._make_trades(mid, tick, best_bid, best_ask, imb, p, rng)

        # 4) olaylar (add/cancel/trade) — spoof + iceberg dahil
        events = self._make_events(mid, tick, best_bid, best_ask, trades, p, rng)

        # 5) likidasyon kümeleri (rejime göre üst/alt yoğunluk)
        liqs = self._make_liquidations(mid, imb, regime, rng)

        ts = self._epoch + timedelta(seconds=self._t)
        self._t += self.step_sec
        rb = RawBook(symbol=symbol, ts=ts, bids=bids, asks=asks,
                     trades=trades, events=events, liquidations=liqs)
        rb.regime = regime  # type: ignore[attr-defined]  (teşhis/test kolaylığı)
        return rb

    # ---------- yardımcılar ----------
    def _ladder(self, start: float, direction: int, tick: float, p: dict,
                rng: random.Random, side_boost: float) -> list[BookLevel]:
        levels = []
        price = start
        for i in range(self.levels):
            # boşluk: bazen fazladan tick atla (derinlik deliği)
            step = 1
            if i > 0 and rng.random() < p["gap_prob"]:
                step += rng.randint(1, 3)
            price += direction * tick * step
            qty = p["base_qty"] * side_boost * (1.0 + rng.uniform(-0.3, 0.6))
            qty = max(qty, 0.01)
            levels.append(BookLevel(round(price, 8), round(qty, 6)))
        return levels

    def _make_trades(self, mid, tick, best_bid, best_ask, imb, p, rng):
        n = rng.randint(*p["n_trades"])
        trades = []
        # toxic sweep: baskın yönde ardışık, seviye süpüren işlemler
        sweep_side = "BUY" if imb > 0 else "SELL"
        for k in range(n):
            if p["sweep"] and k < max(2, n // 2):
                side = sweep_side
                off = (k + 1) * tick
                price = best_ask + off if side == "BUY" else best_bid - off
            else:
                side = "BUY" if rng.random() < 0.5 + imb / 2 else "SELL"
                price = best_ask if side == "BUY" else best_bid
            qty = p["base_qty"] * rng.uniform(0.2, 1.2)
            ts = self._epoch + timedelta(seconds=self._t + k * 0.01)
            trades.append(Trade(round(price, 8), round(qty, 6), side, ts.timestamp()))
        return trades

    def _make_events(self, mid, tick, best_bid, best_ask, trades, p, rng):
        events = []
        base_ts = self._epoch.timestamp() + self._t

        def dist_bps(price):
            return abs(price - mid) / mid * 1e4 if mid > 0 else 0.0

        # iceberg: baskın seviyede işlem → hemen yenilenme (trade + add aynı fiyat)
        if trades:
            t0 = trades[0]
            events.append(OrderEvent("trade", t0.side, t0.price, t0.qty,
                                     base_ts, dist_bps(t0.price)))
            if rng.random() < (0.6 if p["sweep"] else 0.2):  # iceberg şüphesi
                refill = t0.qty * rng.uniform(0.6, 1.0)
                events.append(OrderEvent("add", t0.side, t0.price, round(refill, 6),
                                         base_ts + 0.02, dist_bps(t0.price)))

        # spoof: mid'den uzakta büyük blok belir → işlem görmeden iptal
        if p["spoof"]:
            for _ in range(rng.randint(1, 3)):
                far = mid + rng.choice([-1, 1]) * tick * rng.randint(15, 40)
                side = "BUY" if far < mid else "SELL"
                qty = p["base_qty"] * rng.uniform(4, 10)  # büyük
                d = dist_bps(far)
                events.append(OrderEvent("add", side, round(far, 8), round(qty, 6),
                                         base_ts, d))
                events.append(OrderEvent("cancel", side, round(far, 8), round(qty, 6),
                                         base_ts + 0.05, d))
        return events

    def _make_liquidations(self, mid, imb, regime, rng):
        if regime == CALM:
            return []
        liqs = []
        # toxic/wide: baskın yönün tersinde likidasyon kümesi (mıknatıs)
        n = rng.randint(1, 4)
        for _ in range(n):
            up = rng.random() < 0.5 + imb / 2
            price = mid * (1 + (0.005 if up else -0.005) * rng.uniform(0.5, 3))
            notional = rng.uniform(50_000, 500_000)
            liqs.append((round(price, 2), round(notional, 2)))
        return liqs
