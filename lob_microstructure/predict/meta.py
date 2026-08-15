"""Meta-Etiketleme (Meta-Labeling) — López de Prado, AFML Bölüm 3.

İki katmanlı karar:
  * BİRİNCİL model yön söyler (al/sat). Yüksek hatırlama (recall), düşük
    kesinlik (precision) olabilir.
  * META model, birincil modelin O AN haklı olup olmayacağını tahmin eder.
    Çıktısı **pozisyon büyüklüğü** (sizing) olarak kullanılır: 0 = işleme
    girme, 1 = tam boyut. Böylece yön ile boyutlandırma ayrışır.

Meta-etiket: y_meta = 1 eğer (birincil yön == gerçek yön), aksi 0.
Meta özellikler: ham özellikler + birincil modelin güveni (|p-0.5|).
Meta model: pür-numpy LogReg (sklearn yok).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lob_microstructure.train.logreg import LogReg
from lob_microstructure.book.state import BookState


def primary_direction(prob_up: np.ndarray) -> np.ndarray:
    """prob_up → yön (1 = yukarı/long, 0 = aşağı/short)."""
    return (np.asarray(prob_up) >= 0.5).astype(int)


def make_meta_dataset(X: np.ndarray, y: np.ndarray, primary_prob: np.ndarray):
    """Meta eğitim seti üret.

    X: birincil özellikler (n,d). y: gerçek yön (0/1). primary_prob: birincil
    modelin prob_up'ı. Döner: (X_meta, y_meta).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    p = np.asarray(primary_prob, dtype=float)
    direction = primary_direction(p)
    confidence = np.abs(p - 0.5)[:, None]            # birincil güven
    X_meta = np.hstack([X, confidence])
    y_meta = (direction == y).astype(float)          # birincil haklı mı?
    return X_meta, y_meta


@dataclass
class MetaDecision:
    direction: int        # 1 long, 0 short
    size: float           # 0..1 pozisyon büyüklüğü (meta olasılık)
    take: bool            # size >= eşik mi


class MetaLabeler:
    def __init__(self, size_threshold: float = 0.5):
        self.meta = LogReg(lr=0.2, epochs=3000)
        self.size_threshold = size_threshold

    def fit(self, X, y, primary_prob) -> "MetaLabeler":
        Xm, ym = make_meta_dataset(X, y, primary_prob)
        self.meta.fit(Xm, ym)
        return self

    def decide(self, x_row, primary_prob_row,
               book_state: BookState | None = None) -> MetaDecision:
        x = np.asarray(x_row, dtype=float).reshape(1, -1)
        conf = np.array([[abs(primary_prob_row - 0.5)]])
        xm = np.hstack([x, conf])
        size = float(self.meta.predict_proba(xm)[0])

        # D5: likidite riskine gore boyut azaltma
        if book_state is not None:
            liq_mult = 1.0
            if book_state.kyle_lambda > 0:
                liq_mult *= max(0.5, 1.0 - book_state.kyle_lambda / 1e-5)
            if book_state.book_slope > 0:
                liq_mult *= max(0.5, 1.0 - book_state.book_slope / 10.0)
            spoof = getattr(book_state, "spoof_score", 0.0) or 0.0
            liq_mult *= max(0.5, 1.0 - spoof)
            size *= liq_mult

        direction = int(primary_prob_row >= 0.5)
        return MetaDecision(direction=direction, size=round(size, 3),
                            take=size >= self.size_threshold)

    def sizes(self, X, primary_prob) -> np.ndarray:
        """Toplu meta boyut (0..1)."""
        Xm, _ = make_meta_dataset(X, np.zeros(len(X)), primary_prob)
        return self.meta.predict_proba(Xm)
