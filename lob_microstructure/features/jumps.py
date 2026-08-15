"""Lee-Mykland sıçrama testi — likidasyon gap'lerinin istatistiksel tespiti.

`sweep_score` / `absorption` eşik sezgisidir: "getiri şu kadardan büyükse
sıçrama". Sorun, "şu kadar"ın rejime bağlı olmasıdır — sakin piyasada sıçrama
olan hareket, çalkantılı piyasada normaldir.

Lee & Mykland (2008) bunu yerel volatiliteye normalize ederek çözer ve kritik
değeri Gumbel dağılımından alır, yani "bu büyüklükte bir getiri sıçrama YOKKEN
ne sıklıkla görülürdü?" sorusunu cevaplar.

BIPOWER VARIATION NEDEN
-----------------------
Yerel volatilite ardışık mutlak getirilerin ÇARPIMLARININ ortalamasıyla
tahmin edilir. Sebep: tek bir sıçrama bu toplamda yalnızca iki terime girer ve
komşusuyla çarpılınca seyrelir. Düz bir kayan standart sapma ise sıçramayı
kendi tahminine emer ve sıçramayı GİZLER — aradığın şeyi ölçüne dahil edersen
bulamazsın.

Çıktısı Heimdall'ın Bates yol üretecini besler: `lambda_jump` doğrudan
sıçrama yoğunluğu, `jump_mean`/`jump_std` sıçrama büyüklüğü dağılımı.
İki repo arasındaki ilk gerçek matematiksel bağ.
"""
from __future__ import annotations

import numpy as np

# Gumbel kritik değeri, alpha = %1:  beta* = -log(-log(1 - alpha))
BETA_STAR_1PCT = 4.6001
BETA_STAR_5PCT = 2.9702
_C_LM = np.sqrt(2.0 / np.pi)        # standart normal için E|Z|
_TINY = 1e-12


def lee_mykland(returns, *, window: int | None = None,
                beta_star: float = BETA_STAR_1PCT) -> dict:
    """Log-getiri serisinde sıçrama tespiti.

    Returns {lambda_jump, jump_mean, jump_std, n_jumps, n_obs, window, indices}
      lambda_jump : bar başına sıçrama sayısı (Poisson yoğunluğu)
      jump_mean   : tespit edilen sıçramaların ortalaması (log uzayında)
      jump_std    : sıçrama büyüklüğü standart sapması
      indices     : sıçrama tespit edilen orijinal indeksler
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    empty = {"lambda_jump": 0.0, "jump_mean": 0.0, "jump_std": 0.0,
             "n_jumps": 0, "n_obs": n, "window": window or 0, "indices": []}

    if window is None:
        # Lee-Mykland penceresinin O(sqrt(n)) olması gerekir; katsayı pratik seçim.
        window = int(np.clip(int(np.sqrt(n)) * 2, 16, max(16, n // 4)))
    if n < window + 4:
        return {**empty, "window": window}

    absr = np.abs(r)
    bp = absr[1:] * absr[:-1]                   # |r_j| * |r_{j-1}|
    csum = np.concatenate([[0.0], np.cumsum(bp)])
    k = window - 2
    if k < 1:
        return {**empty, "window": window}

    # sigma_hat[i] yalnızca i'den ÖNCEKİ k bipower terimini kullanır — ileriye
    # bakma yok, bu bir tahmin değil teşhis olduğu için de kritik.
    sigma2 = (csum[k:] - csum[:-k]) / k
    sigma = np.sqrt(np.clip(sigma2, _TINY, None)) / _C_LM

    start = window - 1
    tested = r[start:start + len(sigma)]
    sigma = sigma[:len(tested)]
    if len(tested) == 0:
        return {**empty, "window": window}

    stat = np.abs(tested) / np.clip(sigma, _TINY, None)

    m = len(stat)
    log_m = np.log(max(m, 3))
    sqrt2log = np.sqrt(2.0 * log_m)
    c_n = sqrt2log / _C_LM - (np.log(np.pi) + np.log(log_m)) / (2.0 * _C_LM * sqrt2log)
    s_n = 1.0 / (_C_LM * sqrt2log)

    is_jump = (stat - c_n) / s_n > beta_star
    jumps = tested[is_jump]
    n_jumps = int(is_jump.sum())
    if n_jumps == 0:
        return {**empty, "n_obs": m, "window": window}

    return {
        "lambda_jump": float(n_jumps / m),
        "jump_mean": float(np.mean(jumps)),
        "jump_std": (float(np.std(jumps)) if n_jumps > 1
                     else float(abs(np.mean(jumps)) * 0.5)),
        "n_jumps": n_jumps,
        "n_obs": m,
        "window": window,
        "indices": (np.flatnonzero(is_jump) + start).tolist(),
    }


def jumps_from_prices(prices, **kw) -> dict:
    """Fiyat serisinden doğrudan sıçrama testi (log getiriye çevirip çağırır)."""
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < 3:
        return lee_mykland(np.array([]))
    return lee_mykland(np.diff(np.log(p)), **kw)
