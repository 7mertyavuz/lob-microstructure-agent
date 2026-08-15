"""BookKeeper — yerel L2 defter kopyası (snapshot + diff senkronizasyonu).

Saf Python; seviye→miktar sözlüğüyle O(1) güncelleme, en iyi N seviye görünümü.
Canlı (Binance depth@100ms diff + REST snapshot) ya da simülatör diff'leriyle
beslenir. Sıra numarası (sequence) boşluğu tespiti hook'u bırakılmıştır —
canlı senkron kaybında yeniden snapshot çağrılır (D1'de bağlanır).

Bu sınıf D0'da tanımlıdır ve `SimBookFeed` gibi RawBook'a çevrilebilir;
D1'de gerçek WS akışına bağlanacaktır.
"""
from __future__ import annotations

from datetime import datetime, timezone

from lob_microstructure.book.state import BookLevel, RawBook


class BookKeeper:
    """Tek sembol için yerel defter durumu."""

    def __init__(self, symbol: str, max_levels: int = 50):
        self.symbol = symbol
        self.max_levels = max_levels
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self.last_update_id: int | None = None

    # ---------- snapshot + diff ----------
    def apply_snapshot(self, bids, asks, update_id: int | None = None) -> None:
        """REST snapshot'ı yükler (mevcut durumu tamamen değiştirir)."""
        self._bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self._asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.last_update_id = update_id

    def apply_diff(self, bids=(), asks=(), update_id: int | None = None) -> None:
        """Diff uygular: qty=0 seviyeyi siler, aksi halde üzerine yazar."""
        for p, q in bids:
            p, q = float(p), float(q)
            if q <= 0:
                self._bids.pop(p, None)
            else:
                self._bids[p] = q
        for p, q in asks:
            p, q = float(p), float(q)
            if q <= 0:
                self._asks.pop(p, None)
            else:
                self._asks[p] = q
        if update_id is not None:
            self.last_update_id = update_id

    # ---------- görünüm ----------
    def top(self, n: int | None = None) -> tuple[list[BookLevel], list[BookLevel]]:
        """En iyi n seviye: (bids fiyat azalan, asks fiyat artan)."""
        n = n or self.max_levels
        bids = sorted(self._bids.items(), key=lambda kv: kv[0], reverse=True)[:n]
        asks = sorted(self._asks.items(), key=lambda kv: kv[0])[:n]
        return ([BookLevel(p, q) for p, q in bids],
                [BookLevel(p, q) for p, q in asks])

    def to_raw(self, trades=None, events=None, liquidations=None,
               ts: datetime | None = None) -> RawBook:
        bids, asks = self.top()
        return RawBook(
            symbol=self.symbol,
            ts=ts or datetime.now(timezone.utc),
            bids=bids,
            asks=asks,
            trades=list(trades or []),
            events=list(events or []),
            liquidations=list(liquidations or []),
        )
