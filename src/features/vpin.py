"""VPIN — Volume-Synchronized Probability of Informed Trading.

Easley & López de Prado'nun akış-toksisitesi metriği. Hacmi eşit "bucket"lara
böler; her bucket'ta |alış_hacmi − satış_hacmi| / bucket_hacmi hesaplanır ve
son N bucket'ın ortalaması VPIN'dir (0..1). Yüksek VPIN = bilgili/tek-yönlü
(toksik) akış → fiyat sıçramaları ve volatilite öngörüsü.

Burada hacim USD cinsindendir; alış/satış işaretleri sinyal yönünden gelir.
"""
from __future__ import annotations

from collections import deque


class VPIN:
    def __init__(self, bucket_volume_usd: float = 250_000.0, n_buckets: int = 50):
        self.bucket_volume = float(bucket_volume_usd)
        self.buckets: deque[float] = deque(maxlen=n_buckets)
        self._buy = 0.0
        self._sell = 0.0
        self._vol = 0.0

    def add(self, usd: float, is_buy: bool) -> None:
        usd = abs(float(usd))
        if usd <= 0:
            return
        if is_buy:
            self._buy += usd
        else:
            self._sell += usd
        self._vol += usd
        # Bucket(lar) doldukça finalize et (büyük işlem birden çok bucket doldurabilir)
        while self._vol >= self.bucket_volume:
            frac = self.bucket_volume / self._vol if self._vol else 1.0
            b = self._buy * frac
            s = self._sell * frac
            self.buckets.append(abs(b - s) / self.bucket_volume)
            # kalanı bir sonraki bucket'a taşı
            self._buy -= b
            self._sell -= s
            self._vol -= self.bucket_volume

    def value(self) -> float:
        if self.buckets:
            return sum(self.buckets) / len(self.buckets)
        # Henüz tam bucket yoksa kısmi tahmin
        tot = self._buy + self._sell
        return abs(self._buy - self._sell) / tot if tot > 0 else 0.0
