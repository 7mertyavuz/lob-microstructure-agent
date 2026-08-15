"""Fourier HFT gürültü filtresi (gemini planı Faz 1, Heimdall F6).

Emir defteri serisi iki şeyin toplamıdır: **gerçek likidite dalgaları** (yavaş,
yönlü, bilgi taşır) ve **HFT churn'ü** (hızlı, salınımlı, çoğu anında iptal
edilen sahte emirler). Spektral olarak bunlar ayrışır — churn enerjisi yüksek
frekanslarda toplanır.

`noise_ratio` = yüksek frekanslardaki enerji payı. Yüksekse defterin gösterdiği
derinlik dengesizliğine daha az güvenilir; `spoof_score`'a ek bir teyit verir.

PENCERELEME NEDEN ZORUNLU
------------------------
Sonlu bir seriye doğrudan FFT uygulamak, serinin periyodik olduğunu varsayar.
Baştaki ve sondaki değerler farklıysa bu varsayım sanal bir süreksizlik yaratır
ve o süreksizliğin enerjisi TÜM spektruma yayılır (spectral leakage). Sonuç:
tamamen düzgün bir trendde bile yapay bir "yüksek frekans gürültüsü" ölçersin.

Bu yüzden önce doğrusal trend çıkarılır, sonra Hann penceresi uygulanır
(Grzelak Ders 8'in Fourier disiplini). İkisi atlanırsa filtre sessizce yanlış
çalışır — hata vermez, sadece yanlış sayı üretir.
"""
from __future__ import annotations

import numpy as np

MIN_SAMPLES = 8
DEFAULT_CUTOFF = 0.25       # üst %75 frekans bandı "gürültü" sayılır


def _detrend(x: np.ndarray) -> np.ndarray:
    """Doğrusal trendi çıkar. Trend, DC'ye yakın devasa bir bileşendir ve
    bırakılırsa gürültü oranını sistematik olarak KÜÇÜK gösterir."""
    n = len(x)
    t = np.arange(n, dtype=float)
    # np.polyfit yerine kapalı form: bağımlılık yok, daha hızlı, aynı sonuç
    tm, xm = t.mean(), x.mean()
    denom = float(((t - tm) ** 2).sum())
    slope = float(((t - tm) * (x - xm)).sum() / denom) if denom > 0 else 0.0
    return x - (xm + slope * (t - tm))


def spectral_noise_ratio(series, *, cutoff_frac: float = DEFAULT_CUTOFF) -> float:
    """Yüksek frekans enerji payı, [0,1].

    0 → tamamen düz/yavaş hareket (temiz likidite)
    1 → tüm enerji hızlı salınımda (HFT churn)
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_SAMPLES:
        return 0.0

    x = _detrend(x)
    if np.allclose(x, 0.0):
        return 0.0

    w = np.hanning(len(x))
    power = np.abs(np.fft.rfft(x * w)) ** 2
    power = power[1:]                       # DC bileşenini at
    total = float(power.sum())
    if total <= 0.0:
        return 0.0

    k = max(1, int(len(power) * cutoff_frac))
    return float(np.clip(power[k:].sum() / total, 0.0, 1.0))


def lowpass(series, *, cutoff_frac: float = DEFAULT_CUTOFF) -> np.ndarray:
    """Alçak-geçiren filtre: yalnızca gerçek likidite dalgalarını bırakır.

    Pencereleme burada UYGULANMAZ — filtrelenmiş sinyalin genliğini bozardı.
    Bu fonksiyon ölçüm değil rekonstrüksiyon içindir.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_SAMPLES:
        return x

    spec = np.fft.rfft(x)
    k = max(1, int(len(spec) * cutoff_frac))
    spec[k:] = 0.0
    return np.fft.irfft(spec, n=len(x))


def book_noise(mid_series, depth_series=None, *, cutoff_frac: float = DEFAULT_CUTOFF) -> dict:
    """Defter serilerinden birleşik gürültü teşhisi.

    Returns {noise_ratio, mid_noise, depth_noise, n}.
    Derinlik serisi verilirse ikisinin ortalaması alınır: fiyat sakinken
    derinliğin çırpınması da churn işaretidir.
    """
    mid_noise = spectral_noise_ratio(mid_series, cutoff_frac=cutoff_frac)
    n = len(np.asarray(mid_series, dtype=float))
    if depth_series is None:
        return {"noise_ratio": round(mid_noise, 6), "mid_noise": round(mid_noise, 6),
                "depth_noise": None, "n": n}
    depth_noise = spectral_noise_ratio(depth_series, cutoff_frac=cutoff_frac)
    combined = 0.5 * (mid_noise + depth_noise)
    return {"noise_ratio": round(combined, 6), "mid_noise": round(mid_noise, 6),
            "depth_noise": round(depth_noise, 6), "n": n}
