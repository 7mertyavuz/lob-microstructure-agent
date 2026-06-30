"""Online learning + feature decay — sürekli öğrenme ve körleşme izleme.

Kripto mikro-yapısı hızlı değişir (concept drift); bugünkü alfa 3 ay sonra
rakip botlar yüzünden kaybolabilir. İki bileşen:

  * OnlineLogReg: her yeni blok/örnekte tahmin hatasına göre ağırlıkları
    küçük SGD adımıyla günceller. Üstel unutma (forgetting) ile eski veriyi
    yavaşça unutur → ani rejim değişimine hızlı tepki.
  * FeatureMonitor: her özelliğin tahmin gücünü (EWMA |korelasyon|) izler;
    zamanla düşenleri "decayed" olarak işaretler → bot körleşmeden uyarır.

Pür numpy; batch modelle (train.py) birlikte hibrit çalışır.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class OnlineLogReg:
    def __init__(self, n_features: int, lr: float = 0.05, l2: float = 1e-4,
                 forget: float = 1.0):
        """forget<1.0 → her adımda ağırlıkları hafifçe küçült (üstel unutma)."""
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.forget = forget

    def predict_proba(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(_sigmoid(x @ self.w + self.b))

    def partial_fit(self, x, y: float) -> float:
        """Tek örnekle bir SGD adımı. Güncellemeden ÖNCEki kaybı döndürür
        (online değerlendirme / drift izleme için)."""
        x = np.asarray(x, dtype=float)
        p = self.predict_proba(x)
        eps = 1e-9
        loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        err = p - y
        if self.forget < 1.0:
            self.w *= self.forget
        self.w -= self.lr * (err * x + self.l2 * self.w)
        self.b -= self.lr * err
        return float(loss)

    def update_block(self, X, y) -> float:
        """Bir bloğun örnekleriyle güncelle; ortalama (prequential) kaybı döndür."""
        losses = [self.partial_fit(xi, yi) for xi, yi in zip(np.asarray(X), np.asarray(y))]
        return float(np.mean(losses)) if losses else 0.0


class FeatureMonitor:
    """Her özellik için tahmin gücünün EWMA'sını izler (feature decay)."""

    def __init__(self, n_features: int, names: list[str] | None = None,
                 halflife: float = 200.0):
        self.names = names or [f"f{i}" for i in range(n_features)]
        self.alpha = 1 - 0.5 ** (1.0 / halflife)   # EWMA katsayısı
        self._ex = np.zeros(n_features)            # E[x]
        self._ey = 0.0                              # E[y]
        self._exy = np.zeros(n_features)           # E[x*y]
        self._ex2 = np.zeros(n_features)           # E[x^2]
        self._ey2 = 0.0
        self._init = False

    def update(self, x, y: float) -> None:
        x = np.asarray(x, dtype=float)
        a = self.alpha
        if not self._init:
            self._ex, self._ey = x.copy(), float(y)
            self._exy, self._ex2 = x * y, x * x
            self._ey2 = y * y
            self._init = True
            return
        self._ex = (1 - a) * self._ex + a * x
        self._ey = (1 - a) * self._ey + a * y
        self._exy = (1 - a) * self._exy + a * (x * y)
        self._ex2 = (1 - a) * self._ex2 + a * (x * x)
        self._ey2 = (1 - a) * self._ey2 + a * (y * y)

    def importances(self) -> dict[str, float]:
        """Özellik→|EWMA korelasyon| (0..1)."""
        cov = self._exy - self._ex * self._ey
        vx = np.maximum(self._ex2 - self._ex ** 2, 1e-12)
        vy = max(self._ey2 - self._ey ** 2, 1e-12)
        corr = np.abs(cov / np.sqrt(vx * vy))
        return {n: float(min(c, 1.0)) for n, c in zip(self.names, corr)}

    def decayed(self, threshold: float = 0.05) -> list[str]:
        """Tahmin gücü eşik altına düşmüş özellikler."""
        return [n for n, v in self.importances().items() if v < threshold]
