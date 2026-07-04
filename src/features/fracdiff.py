"""Kesirli fark alma — Fixed-Width Window Fractional Differencing (FFD).

López de Prado, *Advances in Financial Machine Learning*, Bölüm 5.

Amaç: LSTM/CNN'e beslenecek fiyat/özellik serilerinde **durağanlık**
sağlarken **hafızayı** (uzun-vade bağımlılık) korumak. Tamsayı fark alma
(getiri) durağanlık verir ama hafızayı yok eder; d∈(0,1) kesirli fark ikisini
dengeler.

Ağırlıklar:
    w_0 = 1,  w_k = -w_{k-1} * (d - k + 1) / k
|w_k| < thres olunca pencere kesilir (sabit genişlik). En yeni gözlemin
ağırlığı 1'dir.
"""
from __future__ import annotations

import numpy as np


def ffd_weights(d: float, thres: float = 1e-5) -> np.ndarray:
    """En eskiden en yeniye sıralı ağırlık vektörü (son eleman = 1.0)."""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])  # ters: en yeni gözlem ağırlığı 1.0


def frac_diff_ffd(series, d: float, thres: float = 1e-5) -> np.ndarray:
    """Seriyi kesirli farkla dönüştür. İlk `width` eleman NaN (yeterli geçmiş yok).

    d=0 → seri (değişmez), d=1 → standart birinci fark."""
    s = np.asarray(series, dtype=float)
    w = ffd_weights(d, thres)
    width = len(w) - 1
    out = np.full(len(s), np.nan)
    if width >= len(s):
        return out
    for i in range(width, len(s)):
        out[i] = float(np.dot(w, s[i - width:i + 1]))
    return out


def min_ffd_d(series, max_d: float = 1.0, step: float = 0.05,
              thres: float = 1e-5) -> float:
    """Durağanlığı sağlayan en küçük d'yi kaba ızgara ile bul (varyans
    stabilizasyonu vekili). Gerçekte ADF testi kullanılır; burada hafif bir
    vekil: kesirli fark serisinin varyansı sınırlı/sonlu hâle geldiğinde döner."""
    s = np.asarray(series, dtype=float)
    base = np.nanvar(np.diff(s)) or 1.0
    d = 0.0
    while d <= max_d:
        fd = frac_diff_ffd(s, d, thres)
        v = np.nanvar(fd)
        if np.isfinite(v) and v <= base * 3:   # makul ölçekte → durağan vekili
            return round(d, 4)
        d += step
    return round(max_d, 4)
