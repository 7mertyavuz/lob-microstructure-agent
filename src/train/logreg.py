"""Pür-numpy logistic regression — sklearn bağımlılığı yok.

Gradient descent ile logloss minimize eder. Çıktı katsayıları doğrudan
Katman 5'in (predict/direction.py) beklediği biçimle uyumludur:
    z = b0 + b1*imbalance + b2*tanh(whale_net/scale)
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogReg:
    def __init__(self, lr: float = 0.1, epochs: int = 2000, l2: float = 1e-4):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: np.ndarray | None = None   # [b1, b2, ...]
        self.b: float = 0.0                 # b0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogReg":
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = _sigmoid(X @ self.w + self.b)
            err = p - y
            grad_w = X.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.w + self.b)

    def coeffs(self) -> dict:
        return {"b0": float(self.b),
                "b1": float(self.w[0]),
                "b2": float(self.w[1])}
