"""NumpyMLP — pür-numpy çok katmanlı algılayıcı (1 gizli katman).

Regime router'ın "toksik/yüksek-volatilite" bacağındaki nonlineer derin model.
predict_toxic placeholder'ının yerini alır: lineer modelin yakalayamadığı
nonlineer etkileşimleri (ör. imbalance × toksisite, eşik etkileri) öğrenir.

Gerçek sistemde fracdiff özellikleriyle (durağan + hafızalı) beslenir.
Tam forward + backprop; sklearn/torch yok.
"""
from __future__ import annotations

import numpy as np

from src.features.window import FlowFeatures
from src.models import PricePrediction


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class NumpyMLP:
    def __init__(self, n_in: int, n_hidden: int = 16, lr: float = 0.05,
                 epochs: int = 3000, l2: float = 1e-4, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Xavier/He init
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, 1))
        self.b2 = 0.0
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = (a1 @ self.W2).ravel() + self.b2
        p = _sigmoid(z2)
        return z1, a1, p

    def fit(self, X, y) -> "NumpyMLP":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        for _ in range(self.epochs):
            z1, a1, p = self._forward(X)
            dz2 = (p - y) / n                      # (n,)
            dW2 = a1.T @ dz2[:, None] + self.l2 * self.W2
            db2 = dz2.sum()
            da1 = dz2[:, None] * self.W2.ravel()[None, :]
            dz1 = da1 * (1 - a1 ** 2)              # tanh'
            dW1 = X.T @ dz1 + self.l2 * self.W1
            db1 = dz1.sum(0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._forward(X)[2]

    def save(self, path: str) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=np.array([self.b2]))

    @classmethod
    def load(cls, path: str) -> "NumpyMLP":
        d = np.load(path)
        m = cls(d["W1"].shape[0], d["W1"].shape[1])
        m.W1, m.b1, m.W2, m.b2 = d["W1"], d["b1"], d["W2"], float(d["b2"][0])
        return m


def features_to_vector(feat: FlowFeatures) -> list[float]:
    """FlowFeatures → MLP girdi vektörü. Gerçekte fracdiff serileri eklenir."""
    fast = getattr(feat, "imbalance_fast", 0.0) or 0.0
    vpin = getattr(feat, "vpin", 0.0) or 0.0
    whale = np.tanh(feat.whale_net_usd / 250_000.0)
    return [feat.flow_imbalance, fast, vpin, float(whale)]


def make_toxic_predictor(model: NumpyMLP):
    """Eğitilmiş MLP'yi RegimeRouter'ın toxic_fn'i olarak saran fabrika.

    Dönen fonksiyon FlowFeatures alıp PricePrediction döndürür — predict_toxic
    ile aynı arayüz, böylece doğrudan RegimeRouter(normal_fn, toxic_fn)'e geçer.
    """
    def _predict(feat: FlowFeatures) -> PricePrediction:
        x = features_to_vector(feat)
        p = float(model.predict_proba(x)[0])
        vpin = getattr(feat, "vpin", 0.0) or 0.0
        if feat.sample_count < 3:
            p = 0.5 + (p - 0.5) * (feat.sample_count / 3)
        return PricePrediction(
            token=feat.token, prob_up=round(p, 3),
            flow_imbalance=round(getattr(feat, "imbalance_fast", 0.0) or 0.0, 3),
            window_sec=feat.window_sec, sample_count=feat.sample_count,
            whale_net_usd=round(feat.whale_net_usd, 2), toxicity=round(vpin, 3),
        )
    return _predict
