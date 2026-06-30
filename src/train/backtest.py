"""Backtest & değerlendirme — train/test split, metrikler, katsayı kalıcılaştırma."""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np

from src.train.logreg import LogReg, _sigmoid
from src.train.dataset import WHALE_SCALE

COEFF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "direction_coeffs.json",
)


def split(X, y, test_frac=0.25, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    eps = 1e-9
    acc = float(((p >= 0.5).astype(float) == y).mean())
    logloss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    return {"accuracy": round(acc, 4), "logloss": round(logloss, 4), "auc": round(_auc(y, p), 4)}


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U ile ROC-AUC (numpy)."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # ortalama rank (tie düzeltmesi basit)
    r_pos = ranks[y == 1].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def save_coeffs(coeffs: dict, path: str = COEFF_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {**coeffs, "whale_scale": WHALE_SCALE}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def baseline_metrics(X, y, b=(0.0, 2.2, 1.3)) -> dict:
    """Elle girili (kalibre edilmemiş) katsayıların performansı."""
    b0, b1, b2 = b
    p = _sigmoid(b0 + b1 * X[:, 0] + b2 * X[:, 1])
    return metrics(y, p)
