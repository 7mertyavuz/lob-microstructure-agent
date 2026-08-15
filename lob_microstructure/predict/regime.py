"""Rejim değişim mekanizması — model yönlendirici (router).

Fikir: "Normal rejim"de basit/şeffaf lineer (veya Hawkes) model domine eder;
"Toksik/yüksek-volatilite rejimi"nde nonlineer (LSTM/CNN, burada pluggable
stand-in) devreye girer. Rejim iki yoldan saptanır:

  1) VPIN eşiği (basit, kararlı): vpin >= threshold → TOKSİK.
  2) 2-durumlu Gaussian HMM (|getiri| ya da vpin serisi üzerinde): yüksek
     varyanslı durum → TOKSİK. Veriye uyum sağlar, eşik elle ayarlanmaz.

Router, regime'e göre verilen iki tahmin fonksiyonundan birini çağırır.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lob_microstructure.book.state import BookState

NORMAL = "NORMAL"
TOXIC = "TOXIC"
THIN = "THIN"


# ----------------- 2-durum Gaussian HMM -----------------
class GaussianHMM2:
    """Kompakt 2-durumlu Gaussian HMM (EM/Baum-Welch). 1-B seri için.

    Genelde |getiri| veya VPIN serisine fit edilir; yüksek ortalama/varyanslı
    durum 'toksik' kabul edilir."""

    def __init__(self, n_iter: int = 50, seed: int = 0):
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)
        self.mu = None
        self.var = None
        self.trans = None
        self.pi = None
        self.toxic_state = 1

    def _gauss(self, x, mu, var):
        var = max(var, 1e-9)
        return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)

    def fit(self, x) -> "GaussianHMM2":
        x = np.asarray(x, dtype=float)
        n = len(x)
        # init: düşük/yüksek medyan ikiye bölerek
        lo, hi = np.quantile(x, 0.25), np.quantile(x, 0.75)
        self.mu = np.array([lo, hi], dtype=float)
        self.var = np.array([np.var(x) + 1e-6, np.var(x) + 1e-6])
        self.trans = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.pi = np.array([0.5, 0.5])

        for _ in range(self.n_iter):
            B = np.stack([self._gauss(x, self.mu[s], self.var[s]) for s in range(2)], 1)
            B = np.clip(B, 1e-300, None)
            # forward-backward (ölçekli)
            alpha = np.zeros((n, 2)); c = np.zeros(n)
            alpha[0] = self.pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
            for t in range(1, n):
                alpha[t] = (alpha[t - 1] @ self.trans) * B[t]
                c[t] = alpha[t].sum(); alpha[t] /= c[t]
            beta = np.zeros((n, 2)); beta[-1] = 1.0
            for t in range(n - 2, -1, -1):
                beta[t] = (self.trans @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            # xi toplamı
            xi = np.zeros((2, 2))
            for t in range(n - 1):
                denom = (alpha[t][:, None] * self.trans * (B[t + 1] * beta[t + 1])[None, :])
                xi += denom / denom.sum()
            # M-step
            self.pi = gamma[0]
            self.trans = xi / xi.sum(1, keepdims=True)
            w = gamma.sum(0)
            self.mu = (gamma * x[:, None]).sum(0) / w
            self.var = (gamma * (x[:, None] - self.mu) ** 2).sum(0) / w
        self.toxic_state = int(np.argmax(self.mu))  # yüksek ortalama = toksik
        return self

    def predict_states(self, x) -> np.ndarray:
        """Her gözlem için posterior argmax (0/1). 1 = toksik durum."""
        x = np.asarray(x, dtype=float)
        B = np.stack([self._gauss(x, self.mu[s], self.var[s]) for s in range(2)], 1)
        B = np.clip(B, 1e-300, None)
        n = len(x); alpha = np.zeros((n, 2))
        alpha[0] = self.pi * B[0]; alpha[0] /= alpha[0].sum()
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ self.trans) * B[t]; alpha[t] /= alpha[t].sum()
        states = alpha.argmax(1)
        return (states == self.toxic_state).astype(int)


# ----------------- Router -----------------
@dataclass
class RoutedPrediction:
    regime: str
    prob_up: float
    used_model: str


class RegimeRouter:
    """VPIN + defter okuma ile rejim yonlendirmesi.

    normal_fn / toxic_fn / thin_fn: FlowFeatures (ve opsiyonel BookState)
    alip PricePrediction donduren cagrılabilirler.
    """

    def __init__(self, normal_fn: Callable, toxic_fn: Callable,
                 thin_fn: Callable | None = None,
                 vpin_threshold: float = 0.4,
                 spread_bps_threshold: float = 50.0,
                 slope_threshold: float = 5.0,
                 lambda_threshold: float = 1e-5):
        self.normal_fn = normal_fn
        self.toxic_fn = toxic_fn
        self.thin_fn = thin_fn or toxic_fn
        self.vpin_threshold = vpin_threshold
        self.spread_bps_threshold = spread_bps_threshold
        self.slope_threshold = slope_threshold
        self.lambda_threshold = lambda_threshold

    def regime_of(self, vpin: float, book_state: BookState | None = None) -> str:
        if vpin >= self.vpin_threshold:
            return TOXIC
        if book_state is None:
            return NORMAL
        # D5: likidite rejimi -- ince defter de toksik gibi nonlineer modele gider
        if book_state.spread_bps >= self.spread_bps_threshold:
            return THIN
        if book_state.book_slope >= self.slope_threshold:
            return THIN
        if book_state.kyle_lambda >= self.lambda_threshold:
            return THIN
        return NORMAL

    def predict(self, feat, book_state: BookState | None = None):
        vpin = getattr(feat, "vpin", 0.0) or 0.0
        regime = self.regime_of(vpin, book_state)
        if regime == TOXIC:
            fn = self.toxic_fn
        elif regime == THIN:
            fn = self.thin_fn
        else:
            fn = self.normal_fn
        pred = fn(feat, book_state)
        return regime, pred
