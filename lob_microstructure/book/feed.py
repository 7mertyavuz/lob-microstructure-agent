"""BookFeed — Katman 1 defter köprüsü: `latest(symbol) -> BookState`.

`FlowFeed`'e paralel okuma arayüzü. Ham defteri (sim: `SimBookFeed`;
canlı: `BookKeeper`) alır, `features.py`'deki saf fonksiyonlarla `BookState`
üretir. Analist rolü: ağırlık kararı vermez, ham/temiz metrik verir.

Sim modu birinci sınıftır ve `CONFIG.simulation_mode` (WSS_URL boş) ise
otomatik seçilir; aynı seed → aynı `BookState` dizisi.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from lob_microstructure.config import CONFIG
from lob_microstructure.book.state import BookState, RawBook, OrderEvent
from lob_microstructure.book.sim import SimBookFeed
from lob_microstructure.book.keeper import BookKeeper
from lob_microstructure.book.dex_virtual_book import DexVirtualBook
from lob_microstructure.book import features as F
from lob_microstructure.features.fourier import book_noise
from lob_microstructure.features.jumps import jumps_from_prices


def refills_from_events(events) -> list[tuple[float, float]]:
    """Olay dizisinden iceberg refill çiftleri çıkarır: bir "trade"i, aynı
    fiyat/yöndeki sonraki "add" (yenilenme) izliyorsa (işlem_qty, yenilenen_qty)."""
    out = []
    for i, e in enumerate(events):
        if e.kind != "trade":
            continue
        for f in events[i + 1:]:
            if f.kind == "add" and f.side == e.side and abs(f.price - e.price) < 1e-9:
                out.append((e.qty, f.qty))
                break
    return out


class BookFeed:
    """`latest(symbol) -> BookState`.

    mode: "simulation" | "live"
    venue: "cex" | "dex"  -- "dex" V3 sanal defter (D4), "cex" klasik L2.
    """

    def __init__(self, mode: str = "simulation", seed: int | None = None,
                 base_price: float = 30_000.0, spread_hist: int = 64,
                 venue: str = "cex", mid_hist: int = 256):
        if mode not in ("simulation", "live"):
            raise ValueError("mode 'simulation' ya da 'live' olmalı")
        if venue not in ("cex", "dex"):
            raise ValueError("venue 'cex' ya da 'dex' olmalı")
        self.mode = "simulation" if CONFIG.simulation_mode else mode
        self.venue = venue
        self.seed = seed
        self.base_price = base_price
        self._sim = SimBookFeed(seed=seed, base_price=base_price)
        self._keepers: dict[str, BookKeeper] = {}
        self._dex_books: dict[str, DexVirtualBook] = {}
        # sembol-başına durum
        self._prev_best: dict[str, tuple[float, float, float, float]] = {}
        self._prev_mid: dict[str, float] = {}
        self._spread_hist: dict[str, deque] = {}
        self._spread_hist_len = spread_hist
        # Fourier gürültü filtresi ve Lee-Mykland sıçrama testi için mid geçmişi.
        self._mid_hist: dict[str, deque] = {}
        self._mid_hist_len = mid_hist

    # ---------- canlı mod: dış WS/diff beslemesi ----------
    def keeper(self, symbol: str) -> BookKeeper:
        """Canlı modda dışarıdan (D1 WS worker'ı) beslenecek defter tutucu."""
        return self._keepers.setdefault(symbol, BookKeeper(symbol))

    def dex_book(self, symbol: str) -> DexVirtualBook:
        """D4: sembol basina sanal V3 defteri (yoksa olustur)."""
        if symbol not in self._dex_books:
            self._dex_books[symbol] = DexVirtualBook(
                symbol=symbol,
                base_price=self.base_price,
                seed=self.seed,
            )
        return self._dex_books[symbol]

    def inject_dex_swap(self, symbol: str, amount_usd: float, side: str,
                        actor_label: str = "UNKNOWN") -> None:
        """D4: V3 sanal defterine pending swap enjekte et."""
        self.dex_book(symbol).inject_pending_swap(amount_usd, side, actor_label)

    # ---------- ham defter ----------
    def _raw(self, symbol: str) -> RawBook:
        if self.venue == "dex":
            return self.dex_book(symbol).step()
        if self.mode == "simulation":
            return self._sim.next(symbol)
        kp = self.keeper(symbol)
        return kp.to_raw(ts=datetime.now(timezone.utc))

    # ---------- okuma ----------
    def latest(self, symbol: str) -> BookState:
        rb = self._raw(symbol)
        mid = rb.mid
        bb = rb.best_bid
        ba = rb.best_ask

        # --- D2 çekirdek ---
        depth_imb = F.depth_imbalance(rb.bids, rb.asks, mid)
        if bb and ba:
            micro = F.microprice(bb.price, bb.qty, ba.price, ba.qty)
            q_imb = F.queue_imbalance(bb.qty, ba.qty)
            spread_bps = (rb.spread / mid * 1e4) if mid > 0 else 0.0
        else:
            micro = mid
            q_imb = 0.0
            spread_bps = 0.0

        # event-bazlı OFI (önceki en iyi kotasyona göre)
        prev = self._prev_best.get(symbol)
        if prev and bb and ba:
            ofi = F.ofi_event(prev[0], prev[1], prev[2], prev[3],
                              bb.price, bb.qty, ba.price, ba.qty)
        else:
            ofi = 0.0
        if bb and ba:
            self._prev_best[symbol] = (bb.price, bb.qty, ba.price, ba.qty)

        slope = F.book_slope(rb)
        klambda = F.kyle_lambda(rb)

        # spread z — geçmiş uzun süre doldurulup HİÇ okunmuyordu; artık
        # BookState'e çıkıyor. Ani genişleme bilgili akış şüphesidir.
        hist = self._spread_hist.setdefault(symbol, deque(maxlen=self._spread_hist_len))
        sp_z = F.spread_z(rb.spread, hist)

        # --- D3 L3-türevi ---
        refills = refills_from_events(rb.events)
        ice = F.iceberg_score(refills)
        spoof = F.spoof_score(rb.events)
        prev_mid = self._prev_mid.get(symbol, mid)
        price_change = mid - prev_mid
        self._prev_mid[symbol] = mid
        ref_move = max(mid * 5e-4, 1e-9)
        absorp = F.absorption(rb.trades, price_change, ref_move)
        liq_skew = F.liq_map_skew(rb.liquidations, mid)

        hist.append(rb.spread)

        # Zaten hesaplanabilen ama yüzeye çıkmayan iki metrik.
        sweep_val, sweep_dir = F.sweep_score(rb.trades)
        gaps = F.liquidity_gaps(rb.bids + rb.asks, mid)

        # Fourier gürültü filtresi + Lee-Mykland sıçrama testi, mid geçmişi
        # üzerinde. Geçmiş kısa olduğunda ikisi de sessizce 0 döndürür.
        mh = self._mid_hist.setdefault(symbol, deque(maxlen=self._mid_hist_len))
        mh.append(mid)
        noise = book_noise(list(mh))["noise_ratio"] if len(mh) >= 8 else 0.0
        jp = jumps_from_prices(list(mh)) if len(mh) >= 24 else None

        return BookState(
            symbol=symbol,
            spread_bps=max(0.0, spread_bps),
            microprice=micro,
            depth_imbalance=depth_imb,
            ofi=ofi,
            queue_imbalance=q_imb,
            book_slope=slope,
            kyle_lambda=klambda,
            iceberg_score=ice,
            spoof_score=spoof,
            absorption=absorp,
            liq_map_skew=liq_skew,
            sweep=round(sweep_val * sweep_dir, 6),
            liquidity_gaps=round(gaps, 6),
            spread_z=round(sp_z, 6),
            noise_ratio=round(noise, 6),
            lambda_jump=round(jp["lambda_jump"], 8) if jp else 0.0,
            jump_mean=round(jp["jump_mean"], 8) if jp else 0.0,
            jump_std=round(jp["jump_std"], 8) if jp else 0.0,
            ts=rb.ts if isinstance(rb.ts, datetime) else datetime.now(timezone.utc),
        )
