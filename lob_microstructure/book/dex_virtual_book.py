"""DEX sanal defteri — Uniswap V3-stil concentrated liquidity.

D4 hedefi: AMM'yi defter gibi okumak. Havuzun tick basina likidite
dagilimi birebir bir L2 defteridir. `DexVirtualBook`:

- Sembol basina sanal V3 havuzu kurar (sim modda deterministik).
- Her tick bir fiyat seviyesidir; likidite o aralikta ne kadar swap
  tasyabilecegini soyler.
- Pending swap'lerin beklenen fiyat etkisini hesaplar.
- `to_raw_book()` ile mevcut `BookFeed`/`features.py` zincirine takilir.

Formul (saf Python, numpy yok — V3 Whitepaper):
  price(tick) = 1.0001^tick
  sqrt_price  = sqrt(price)
  Δtoken1 = L * (sqrtPb - sqrtPa)
  Δtoken0 = L * (1/sqrtPa - 1/sqrtPb)

Sim modu birinci siniftir; canli modda RPC ile gercek slot0/tick-bitmap
okunabilir (iskelet hazir, D4 kapsaminda simulasyon uzerinden kanitlanir).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from lob_microstructure.book.state import BookLevel, RawBook, Trade, OrderEvent


# V3 pool parametreleri (tick spacing 60 ≈ %0.6 fee tier, örn. 0.3% veya 1%)
DEFAULT_TICK_SPACING = 60
TICK_BASE = 1.0001


def tick_to_price(tick: int) -> float:
    return TICK_BASE ** tick


def price_to_tick(price: float) -> int:
    return int(round(math.log(price, TICK_BASE)))


def sqrt_price(tick: int) -> float:
    return math.sqrt(tick_to_price(tick))


@dataclass
class _Position:
    """Bir [tick_lower, tick_upper] araligindaki likidite."""
    tick_lower: int
    tick_upper: int
    liquidity: float  # L


@dataclass
class DexVirtualBook:
    """Sanal Uniswap V3 defteri.

    Args:
        symbol: sembol adi (ör. "ETHUSDT").
        base_price: baslangic fiyati.
        seed: determinizm icin; None ise rastgele.
        tick_spacing: havuzun tick araligi.
        depth_levels: defter her iki tarafa kac seviye gosterir.
    """
    symbol: str
    base_price: float = 3_000.0
    seed: int | None = None
    tick_spacing: int = DEFAULT_TICK_SPACING
    depth_levels: int = 20
    _positions: list[_Position] = field(default_factory=list, repr=False)
    _current_tick: int = field(default=0, repr=False)
    _mid: float = field(default=0.0, repr=False)
    _rng: random.Random = field(default_factory=lambda: random.Random(None), repr=False)
    _epoch: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc), repr=False)
    _t: float = field(default=0.0, repr=False)
    _pending_swaps: list[tuple[float, str, str]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._current_tick = price_to_tick(self.base_price)
        self._mid = self.base_price
        self._build_liquidity()

    # ---------- likidite insasi ----------
    def _build_liquidity(self) -> None:
        """Mevcut fiyat etrafinda Gaussian dagilimli LP pozisyonlari olustur."""
        center = self._current_tick
        width_ticks = self.tick_spacing * 50  # ~±%30'lik bir aralik
        n_positions = 40
        self._positions = []
        for _ in range(n_positions):
            # LP'ler fiyatin etrafinda kümelenmis
            mu = center + int(self._rng.gauss(0, width_ticks / 4))
            half = int(self._rng.uniform(self.tick_spacing, width_ticks))
            lower = (mu - half) // self.tick_spacing * self.tick_spacing
            upper = (mu + half) // self.tick_spacing * self.tick_spacing
            if upper <= lower:
                upper = lower + self.tick_spacing
            # Likidite miktari: dar araliklar daha yogun
            l = self._rng.uniform(1_000_000.0, 10_000_000.0) / (half / self.tick_spacing)
            self._positions.append(_Position(lower, upper, l))

    # ---------- fiyat etkisi ----------
    def _liquidity_at_tick(self, tick: int) -> float:
        """Tick'te aktif toplam likidite (L)."""
        return sum(
            p.liquidity for p in self._positions
            if p.tick_lower <= tick < p.tick_upper
        )

    def simulate_swap(self, amount_in: float, zero_for_one: bool,
                      start_tick: int | None = None) -> tuple[float, float, int]:
        """Bir swap'in fiyat etkisini hesapla.

        Args:
            amount_in: satilan token miktari (token0 ise zero_for_one=True).
            zero_for_one: True → token0 sat, token1 al; fiyat duser.
                          False → token1 sat, token0 al; fiyat yukselir.
        Returns:
            (amount_out, price_impact_pct, end_tick)
        """
        tick = start_tick if start_tick is not None else self._current_tick
        remaining = amount_in
        amount_out = 0.0
        direction = -1 if zero_for_one else 1  # fiyat duser/yukselir

        while remaining > 1e-12:
            # Bir sonraki gecerli tick
            next_tick = tick + direction * self.tick_spacing
            # Tick araligindaki likidite
            l = self._liquidity_at_tick(tick)
            if l <= 0:
                tick = next_tick
                if abs(tick - self._current_tick) > self.tick_spacing * self.depth_levels * 2:
                    break
                continue

            sqrt_a = sqrt_price(min(tick, next_tick))
            sqrt_b = sqrt_price(max(tick, next_tick))

            if zero_for_one:
                # token0 satiyoruz, Δtoken0 = L*(1/sqrtA - 1/sqrtB)
                max_in_tick = l * (1.0 / sqrt_a - 1.0 / sqrt_b)
                if max_in_tick <= 0:
                    break
                take = min(remaining, max_in_tick)
                ratio = take / max_in_tick
                # Δtoken1 = L*(sqrtB - sqrtA)
                amount_out += ratio * l * (sqrt_b - sqrt_a)
            else:
                # token1 satiyoruz, Δtoken1 = L*(sqrtB - sqrtA)
                max_in_tick = l * (sqrt_b - sqrt_a)
                if max_in_tick <= 0:
                    break
                take = min(remaining, max_in_tick)
                ratio = take / max_in_tick
                # Δtoken0 = L*(1/sqrtA - 1/sqrtB)
                amount_out += ratio * l * (1.0 / sqrt_a - 1.0 / sqrt_b)

            remaining -= take
            if take >= max_in_tick * (1 - 1e-12):
                tick = next_tick
            else:
                break

        start_price = tick_to_price(self._current_tick)
        end_price = tick_to_price(tick)
        impact = (end_price - start_price) / start_price if start_price else 0.0
        return amount_out, impact, tick

    def inject_pending_swap(self, amount_usd: float, side: str,
                            actor_label: str = "UNKNOWN") -> None:
        """Mempool'dan gelen swap'i kuyruga al (fiyat etkisi hesaplanacak)."""
        self._pending_swaps.append((amount_usd, side, actor_label))

    def clear_pending_swaps(self) -> list[tuple[float, str, str]]:
        """Kuyrugu bosalt ve onceki listeyi dondur."""
        old = self._pending_swaps
        self._pending_swaps = []
        return old

    # ---------- defter gorunumu ----------
    def _side_depth(self, direction: int) -> list[BookLevel]:
        """direction=-1 bid (alt), +1 ask (ust) derinlik merdiveni."""
        levels = []
        for i in range(1, self.depth_levels + 1):
            tick = self._current_tick + direction * i * self.tick_spacing
            price = tick_to_price(tick)
            # O tick araligindaki toplam likiditeyi USD-benzeri notional olarak ifade et
            l = self._liquidity_at_tick(tick)
            if l <= 0:
                qty = 0.0
            else:
                sqrt_a = sqrt_price(min(tick, tick + self.tick_spacing))
                sqrt_b = sqrt_price(max(tick, tick + self.tick_spacing))
                # Notional = Δtoken1 (fiyat * token0) yaklasimi
                qty = l * (sqrt_b - sqrt_a) * price
            levels.append(BookLevel(round(price, 8), round(qty, 4)))
        return levels

    def to_raw_book(self, include_pending: bool = True) -> RawBook:
        """Sanal V3 havuzunu `RawBook` olarak dondur; boylece `BookFeed`
        ayni `features.py` zincirini kullanir."""
        bids = self._side_depth(-1)
        asks = self._side_depth(+1)

        trades: list[Trade] = []
        events: list[OrderEvent] = []

        if include_pending:
            for amount_usd, side, actor in self._pending_swaps:
                zero_for_one = side == "SELL"
                amount_out, impact, end_tick = self.simulate_swap(
                    amount_usd, zero_for_one=zero_for_one
                )
                # Trade olarak isle: fiyat etkisi gerceklesmis gibi
                price = tick_to_price(end_tick)
                qty = amount_usd / price if price > 0 else 0.0
                trades.append(Trade(
                    price=round(price, 8),
                    qty=round(qty, 6),
                    side="SELL" if zero_for_one else "BUY",
                    ts=self._t,
                ))
                # L3-türevi olay: agresif emir
                events.append(OrderEvent(
                    kind="trade",
                    side="SELL" if zero_for_one else "BUY",
                    price=round(price, 8),
                    qty=round(qty, 6),
                    ts=self._t,
                    dist_bps=abs(impact) * 1e4,
                ))
                # Bir kisim durumda "absorption" veya "sweep" gibi
                # ek olaylar eklenebilir; simdi temel kuyruk bosaltiliyor.
            self._pending_swaps = []

        # Zaman adimi
        self._t += 1.0
        ts = self._epoch if self._t == 1.0 else self._epoch.replace(
            microsecond=int(self._t * 1_000)
        )

        return RawBook(
            symbol=self.symbol,
            ts=ts,
            bids=bids,
            asks=asks,
            trades=trades,
            events=events,
        )

    def step(self, price_change_pct: float | None = None) -> RawBook:
        """Bir zaman adimi ilerlet: fiyat hafifce yurur, sonra RawBook döner."""
        if price_change_pct is None:
            price_change_pct = self._rng.gauss(0, 0.0003)
        new_price = self._mid * (1.0 + price_change_pct)
        self._mid = new_price
        self._current_tick = price_to_tick(new_price)
        return self.to_raw_book()
