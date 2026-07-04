"""Defter okuma özellikleri — saf, yan-etkisiz, tek tek testli fonksiyonlar.

İki grup:
- D2 çekirdek L2 okumaları: depth imbalance, microprice, event-bazlı OFI,
  book slope / Kyle's λ, queue imbalance / spread z, likidite boşlukları.
- D3 L3-türevi okumalar: iceberg, spoofing, absorption, sweep, likidasyon skew.

Hepsi kural-tabanlı ve parametriktir ("sezgi kodlama yok"): eşikler argüman,
çıktılar (skorlar) 0..1 ya da [-1,1] normalize. Aynı girdi → aynı çıktı.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.book.state import BookLevel, Trade, OrderEvent, RawBook


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _bps(price: float, mid: float) -> float:
    """`price`'ın mid'e mutlak uzaklığı (bps). mid<=0 ise 0."""
    if mid <= 0:
        return 0.0
    return abs(price - mid) / mid * 1e4


# ============================================================
# D2 — Çekirdek L2 okuma özellikleri
# ============================================================

def depth_imbalance(bids: Sequence[BookLevel], asks: Sequence[BookLevel],
                    mid: float | None = None, decay_bps: float = 5.0) -> float:
    """Çok-seviyeli, mid'e uzaklıkla üstel sönümlü derinlik dengesizliği.

    w = exp(-dist_bps / decay_bps); imb = (Σw·bid − Σw·ask)/(Σw·bid + Σw·ask).
    Tek-seviyeli naif imbalance'tan daha dayanıklı: uzak seviyeler az sayılır.
    Sonuç [-1, 1]; + = alış tarafı ağır. Boş defterde 0.
    """
    if not bids and not asks:
        return 0.0
    if mid is None:
        bb = bids[0].price if bids else 0.0
        ba = asks[0].price if asks else 0.0
        mid = (bb + ba) / 2.0 if bids and asks else (bb or ba)
    wb = sum(q * math.exp(-_bps(p, mid) / decay_bps) for p, q in bids)
    wa = sum(q * math.exp(-_bps(p, mid) / decay_bps) for p, q in asks)
    denom = wb + wa
    if denom <= 0:
        return 0.0
    return _clamp((wb - wa) / denom, -1.0, 1.0)


def microprice(best_bid: float, bid_qty: float,
               best_ask: float, ask_qty: float) -> float:
    """Stoikov microprice: (ask_qty·bid + bid_qty·ask)/(bid_qty+ask_qty).

    Bid tarafında derinlik ağırsa fiyat ask'e doğru çekilir (yukarı baskı) —
    mid'den daha adil bir fiyat. Kuyruklar boşsa mid'e düşer.
    """
    denom = bid_qty + ask_qty
    if denom <= 0:
        return (best_bid + best_ask) / 2.0
    return (ask_qty * best_bid + bid_qty * best_ask) / denom


def ofi_event(prev_bid: float, prev_bid_qty: float,
              prev_ask: float, prev_ask_qty: float,
              bid: float, bid_qty: float,
              ask: float, ask_qty: float) -> float:
    """Cont-Kukanov-Stoikov event-bazlı Order Flow Imbalance artışı.

    e = I(bid≥bid₋)·q_bid − I(bid≤bid₋)·q_bid₋
        − I(ask≤ask₋)·q_ask + I(ask≥ask₋)·q_ask₋

    + = alış baskısı (bid yükseldi/derinleşti ya da ask geri çekildi).
    İşlem-bazlı OFI'yi tamamlar; ikisi ayrı faktördür (korelasyonu
    factor_tracker ölçer).
    """
    e = 0.0
    if bid >= prev_bid:
        e += bid_qty
    if bid <= prev_bid:
        e -= prev_bid_qty
    if ask <= prev_ask:
        e -= ask_qty
    if ask >= prev_ask:
        e += prev_ask_qty
    return e


def _cumulative(levels: Sequence[BookLevel], mid: float):
    """(dist_bps, kümülatif_qty) dizileri döndürür."""
    dists, cums = [], []
    c = 0.0
    for p, q in levels:
        c += q
        dists.append(_bps(p, mid))
        cums.append(c)
    return dists, cums


def book_slope(book: RawBook, decay_levels: int = 10) -> float:
    """Defter eğimi / esneklik: kümülatif derinliğin mesafeyle artış hızı.

    Her iki tarafın (dist_bps, kümülatif_qty) noktalarına doğru uydurulur;
    eğim (qty/bps) döner. Yüksek eğim = mid civarı yoğun, dirençli defter.
    Boş/yetersiz veri → 0.
    """
    mid = book.mid
    if mid <= 0:
        return 0.0
    db, cb = _cumulative(book.bids[:decay_levels], mid)
    da, ca = _cumulative(book.asks[:decay_levels], mid)
    xs = np.asarray(db + da, dtype=float)
    ys = np.asarray(cb + ca, dtype=float)
    if len(xs) < 2 or np.ptp(xs) == 0:
        return 0.0
    slope = float(np.polyfit(xs, ys, 1)[0])
    return max(0.0, slope)


def kyle_lambda(book: RawBook, max_levels: int = 10) -> float:
    """Kyle's λ — hacim başına fiyat etkisi (mutlak fiyat / qty).

    Defteri yürüterek her seviye için (|fiyat−mid|)/kümülatif_qty ortalanır;
    iki taraf birleştirilir. Büyük λ = ince defter, yüksek slipaj.
    `PaperExecutor` slipaj modelinin gerçek girdisi olur. ≥0.
    """
    mid = book.mid
    if mid <= 0:
        return 0.0
    impacts = []
    for levels in (book.bids[:max_levels], book.asks[:max_levels]):
        c = 0.0
        for p, q in levels:
            c += q
            if c > 0:
                impacts.append(abs(p - mid) / c)
    if not impacts:
        return 0.0
    return float(np.mean(impacts))


def queue_imbalance(bid_qty: float, ask_qty: float) -> float:
    """En iyi seviye kuyruk dengesizliği: (bid−ask)/(bid+ask), [-1,1]."""
    denom = bid_qty + ask_qty
    if denom <= 0:
        return 0.0
    return _clamp((bid_qty - ask_qty) / denom, -1.0, 1.0)


def spread_z(spread: float, history: Sequence[float]) -> float:
    """Spread'in kendi geçmişine göre z-skoru. Ani genişleme (yüksek +z) =
    bilgili akış / haber öncüsü şüphesi. Yetersiz/sabit geçmiş → 0."""
    h = np.asarray(history, dtype=float)
    if len(h) < 2:
        return 0.0
    sd = float(np.std(h))
    if sd == 0:
        return 0.0
    return (spread - float(np.mean(h))) / sd


def liquidity_gaps(levels: Sequence[BookLevel], mid: float,
                   gap_factor: float = 3.0) -> float:
    """Derinlik profilindeki boşluk skoru: ardışık seviyeler arası fiyat
    sıçraması tipik sıçramanın `gap_factor` katından büyükse boşluk sayılır.

    İnce bölgeye girince fiyatın hızlanma beklentisini niceler. Skor [0,1] =
    boşluklu geçiş oranı. <3 seviye → 0.
    """
    if len(levels) < 3:
        return 0.0
    steps = [abs(levels[i + 1].price - levels[i].price) for i in range(len(levels) - 1)]
    med = float(np.median(steps))
    if med <= 0:
        return 0.0
    gaps = sum(1 for s in steps if s > gap_factor * med)
    return _clamp(gaps / len(steps), 0.0, 1.0)


# ============================================================
# D3 — L3-türevi okumalar (defterin "niyeti")
# ============================================================
# Skorlar kanıt değil ŞÜPHEDİR: tek başına yön oyu vermez, güven çarpanı olur.

def iceberg_score(refills: Sequence[tuple[float, float]],
                  min_ratio: float = 0.5, target_count: int = 3) -> float:
    """Iceberg (gizli likidite) şüphesi [0,1].

    `refills`: aynı fiyat seviyesinde işlemden HEMEN sonra ölçülen
    (işlem_gören_qty, yenilenen_qty) çiftleri. Güçlü yenilenme (yüksek
    yenilenen/işlem oranı) + tekrar sayısı → gizli alıcı/satıcı = güçlü seviye.

    Skor = tekrar-sayısı faktörü × ortalama yenilenme oranı faktörü.
    """
    if not refills:
        return 0.0
    ratios = []
    hits = 0
    for traded, refilled in refills:
        if traded <= 0:
            continue
        r = refilled / traded
        ratios.append(min(r, 1.0))
        if r >= min_ratio:
            hits += 1
    if not ratios:
        return 0.0
    count_factor = min(1.0, hits / target_count)
    ratio_factor = float(np.mean(ratios))
    return _clamp(count_factor * ratio_factor, 0.0, 1.0)


def spoof_score(events: Sequence[OrderEvent],
                min_dist_bps: float = 10.0,
                large_qty: float = 0.0) -> float:
    """Spoofing / katmanlama şüphesi [0,1].

    Mid'den uzakta (≥`min_dist_bps`) beliren büyük (≥`large_qty`) pasif
    blokların İŞLEM GÖRMEDEN iptali. Skor = iptal edilen notional /
    (iptal + işlem gören) notional — uzak-büyük emirler arasında.

    `large_qty=0` ise, uzak add'lerin medyan qty'si eşik olur (göreli büyüklük).
    Yüksek spoof_score iken depth_imbalance'ın güveni kısılmalıdır.
    """
    far_adds = [e for e in events if e.kind == "add" and e.dist_bps >= min_dist_bps]
    if not far_adds:
        return 0.0
    thr = large_qty if large_qty > 0 else float(np.median([e.qty for e in far_adds]))
    cancel_notional = 0.0
    trade_notional = 0.0
    for e in events:
        if e.dist_bps < min_dist_bps or e.qty < thr:
            continue
        if e.kind == "cancel":
            cancel_notional += e.qty
        elif e.kind == "trade":
            trade_notional += e.qty
    denom = cancel_notional + trade_notional
    if denom <= 0:
        return 0.0
    return _clamp(cancel_notional / denom, 0.0, 1.0)


def absorption(trades: Sequence[Trade], price_change: float,
               ref_move: float) -> float:
    """Emilim [-1,1]. + = satış baskısı emiliyor (bid güçlü, dip adayı);
    − = alış baskısı emiliyor (dağıtım).

    Mantık: baskın agresör yönü fiyatı kendi yönünde İTEMİYORSA emilim var.
    `pressure` = (sell_vol − buy_vol)/toplam (+ = satış agresörü baskın).
    Satış baskısı fiyatı düşürmeliydi; düşmediyse (price_change ≥ 0) bid emmiş.

    `ref_move`: "normal" beklenen fiyat hareketi ölçeği (aynı birimde);
    gerçek hareket bu ölçeğe göre normalize edilir.
    """
    if not trades or ref_move <= 0:
        return 0.0
    buy = sum(t.qty for t in trades if t.side == "BUY")
    sell = sum(t.qty for t in trades if t.side == "SELL")
    total = buy + sell
    if total <= 0:
        return 0.0
    pressure = (sell - buy) / total          # + satış baskın, − alış baskın
    # Baskının beklenen yönü: satış → aşağı (negatif), alış → yukarı (pozitif).
    expected_dir = -1.0 if pressure > 0 else 1.0
    # Fiyat beklenen yönde ne kadar gittiyse emilim o kadar AZ.
    realized = (price_change / ref_move) * expected_dir   # + = baskı yönünde gitti
    resistance = _clamp(1.0 - realized, 0.0, 2.0) / 2.0   # 0..1 (1 = tam emilim)
    return _clamp(pressure * 2.0 * resistance, -1.0, 1.0)


def sweep_score(trades: Sequence[Trade], target_levels: int = 4) -> tuple[float, int]:
    """Sweep (tek yönde çok seviyeyi süpüren agresif market emirleri) skoru.

    Aynı yönde ardışık işlemlerin taradığı FARKLI fiyat seviyesi sayısına
    bakar (buy → artan fiyat, sell → azalan fiyat). Momentum ateşleyici;
    whale sinyaliyle çapraz teyit edilir.

    Döner: (skor [0,1], yön) — yön +1 buy sweep, −1 sell sweep, 0 yok.
    """
    if not trades:
        return 0.0, 0
    best_len, best_dir = 0, 0
    best_prices: set = set()
    i = 0
    n = len(trades)
    while i < n:
        side = trades[i].side
        prices = {round(trades[i].price, 10)}
        j = i + 1
        last = trades[i].price
        while j < n and trades[j].side == side:
            p = trades[j].price
            monotone = (p >= last) if side == "BUY" else (p <= last)
            if not monotone:
                break
            prices.add(round(p, 10))
            last = p
            j += 1
        if len(prices) > best_len:
            best_len = len(prices)
            best_dir = 1 if side == "BUY" else -1
            best_prices = prices
        i = max(j, i + 1)
    score = _clamp(best_len / target_levels, 0.0, 1.0)
    if best_len < 2:
        return 0.0, 0
    return score, best_dir


def liq_map_skew(liquidations: Sequence[tuple[float, float]], mid: float) -> float:
    """Likidasyon haritası skew'i [-1,1]. + = yoğunluk mid ÜSTÜNDE (yukarı
    mıknatıs; short likidasyonları), − = ALTINDA (aşağı mıknatıs; long).

    `liquidations`: (fiyat, notional_usd) — gerçek forceOrder olayları ya da
    OI/funding'den tahmini kaldıraç kümeleri.
    """
    if not liquidations or mid <= 0:
        return 0.0
    above = sum(n for p, n in liquidations if p > mid)
    below = sum(n for p, n in liquidations if p < mid)
    denom = above + below
    if denom <= 0:
        return 0.0
    return _clamp((above - below) / denom, -1.0, 1.0)
