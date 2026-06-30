"""CEX-DEX Lider-Takipçi (Lead-Lag) spread özelliği.

Fiyat keşfi çoğu zaman hacimli CEX'te (Binance/Bybit/OKX) olur; DEX havuz
fiyatı mikro-gecikmeyle takip eder (varlığa göre değişir: BTC'de CEX lider,
ETH/SOL'de DEX lider olabilir — korelasyon ~0.98-0.99). Gecikme-düzeltmeli
spread, kısa vadeli yön için güçlü öngörü verir:

    spread = (dex_price - cex_price_lagged) / cex_price_lagged

DEX, CEX'in altındaysa (negatif spread) DEX'in yukarı yakınsaması beklenir →
LONG bias; üstündeyse SHORT bias.

CEX fiyat kaynağı enjekte edilebilir (Binance WS client'ı stub olarak verilir);
böylece spread mantığı canlı feed olmadan test edilir.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def estimate_lag(cex: list[float], dex: list[float], max_lag: int = 20) -> int:
    """DEX'in CEX'i kaç adım gecikmeyle takip ettiğini çapraz-korelasyonla bul.
    Pozitif lag → DEX, CEX'in `lag` adım gerisinden gelir (CEX lider)."""
    c = np.asarray(cex, dtype=float)
    d = np.asarray(dex, dtype=float)
    n = min(len(c), len(d))
    if n < max_lag + 5:
        return 0
    c, d = c[-n:], d[-n:]
    dc = np.diff(c)
    dd = np.diff(d)
    best_lag, best_corr = 0, -2.0
    for lag in range(0, max_lag + 1):
        if lag >= len(dd):
            break
        a = dc[: len(dc) - lag]
        b = dd[lag:]
        if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag


class LeadLagSpread:
    """Gecikme-düzeltmeli CEX-DEX spread'i ve yön sinyali."""

    def __init__(self, lag: int = 1, maxlen: int = 256):
        self.lag = max(0, int(lag))
        self.cex: deque[float] = deque(maxlen=maxlen)
        self.dex: deque[float] = deque(maxlen=maxlen)

    def update_cex(self, price: float) -> None:
        self.cex.append(float(price))

    def update_dex(self, price: float) -> None:
        self.dex.append(float(price))

    def spread(self) -> float:
        """(dex_son - cex_{son-lag}) / cex_{son-lag}. Veri yetersizse 0."""
        if not self.dex or len(self.cex) <= self.lag:
            return 0.0
        dex_now = self.dex[-1]
        cex_lagged = self.cex[-1 - self.lag]
        if cex_lagged == 0:
            return 0.0
        return (dex_now - cex_lagged) / cex_lagged

    def directional_bias(self, deadband: float = 0.0005) -> int:
        """+1 LONG (DEX ucuz, yukarı yakınsar), -1 SHORT, 0 nötr."""
        s = self.spread()
        if s < -deadband:
            return 1
        if s > deadband:
            return -1
        return 0

    def refresh_lag(self, max_lag: int = 20) -> int:
        self.lag = estimate_lag(list(self.cex), list(self.dex), max_lag)
        return self.lag
