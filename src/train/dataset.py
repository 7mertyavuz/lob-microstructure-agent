"""Eğitim verisi katmanı.

İki kaynak:
  * load_csv: gerçek geçmiş veri. Beklenen kolonlar:
        flow_imbalance, whale_net_usd, label
    label = sonraki ufuk içinde fiyat yükseldi(1) / yükselmedi(0).
    Bu CSV, canlı pipeline'ın sinyalleri ile bir sonraki blok/dakikadaki
    fiyat hareketinin offline join'inden üretilir (etiketleme aşaması).
  * make_synthetic: bilinen bir ilişkiden sentetik veri (testler ve
    katsayı geri-kazanımını doğrulamak için).

Özellik vektörü modelle aynıdır: x = [imbalance, tanh(whale_net/scale)].
"""
from __future__ import annotations

import csv
import math
from typing import Tuple

import numpy as np

WHALE_SCALE = 250_000.0


def featurize(flow_imbalance: float, whale_net_usd: float) -> list[float]:
    return [flow_imbalance, math.tanh(whale_net_usd / WHALE_SCALE)]


def make_synthetic(
    n: int = 5000,
    true_b: Tuple[float, float, float] = (0.0, 2.5, 1.5),
    noise: float = 0.0,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinen katsayılardan etiketli veri üret. Trainer bu katsayıları
    yaklaşık geri kazanabilmeli."""
    rng = np.random.default_rng(seed)
    imb = rng.uniform(-1, 1, n)
    whale_net = rng.normal(0, 1.2 * WHALE_SCALE, n)
    x2 = np.tanh(whale_net / WHALE_SCALE)
    X = np.column_stack([imb, x2])
    b0, b1, b2 = true_b
    z = b0 + b1 * imb + b2 * x2 + rng.normal(0, noise, n)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(0, 1, n) < p).astype(float)
    return X, y


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    rows_x, rows_y = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows_x.append(featurize(float(r["flow_imbalance"]),
                                    float(r["whale_net_usd"])))
            rows_y.append(float(r["label"]))
    return np.array(rows_x), np.array(rows_y)
