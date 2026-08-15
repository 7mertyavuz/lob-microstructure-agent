"""Katman 5 — Yön tahmini.

FlowFeatures'tan kısa vadeli fiyat yönü olasılığı (prob_up) üretir.

Logistic birleştirici:
    z = b0 + b1*imbalance + b2*tanh(whale_net/scale)
    prob_up = sigmoid(z)

Katsayılar models/direction_coeffs.json'dan yüklenir (varsa); yoksa makul
varsayılanlara düşer. Katsayılar `train.py` ile geçmiş veriden öğrenilir.

DÜRÜST NOT: kısa vadeli yön tahmini gürültülüdür; iyi modeller bile marjinal
isabet verir. Çıktı her zaman OLASILIK + örnek sayısıdır, kesinlik değil.
"""
from __future__ import annotations

import json
import logging
import math
import os

from lob_microstructure.features.window import FlowFeatures
from lob_microstructure.models import PricePrediction
from lob_microstructure.book.state import BookState

log = logging.getLogger("predict")

# Varsayılan (kalibre edilmemiş) katsayılar
B0, B1, B2 = 0.0, 2.2, 1.3
WHALE_SCALE = 250_000.0
MIN_SAMPLES = 3

_COEFF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "direction_coeffs.json",
)


# Katsayıların NEREDEN geldiği. Tüketicinin bunu bilmesi gerekir: sentetik
# veriyle eğitilmiş katsayı, gerçek piyasa kalibrasyonu DEĞİLDİR. Eskiden
# dosya yoksa sessizce varsayılana düşülüyordu ve dışarıdan ayırt edilemiyordu.
COEFF_SOURCE = "defaults_uncalibrated"
COEFF_PROVENANCE = "elle secilmis varsayilanlar; hicbir veriye fit edilmedi"


def _load_coeffs() -> None:
    """Eğitilmiş katsayılar varsa yükle (import sırasında bir kez)."""
    global B0, B1, B2, WHALE_SCALE, COEFF_SOURCE, COEFF_PROVENANCE
    try:
        # utf-8-sig: dosya Notepad / PowerShell `Set-Content -Encoding utf8`
        # ile yazıldıysa başında BOM olur ve düz json.load patlar.
        with open(_COEFF_PATH, encoding="utf-8-sig") as f:
            c = json.load(f)
        B0, B1, B2 = c["b0"], c["b1"], c["b2"]
        WHALE_SCALE = c.get("whale_scale", WHALE_SCALE)
        COEFF_SOURCE = c.get("source", "trained")
        COEFF_PROVENANCE = c.get("provenance", "bilinmiyor")
        log.info("Katsayılar yüklendi (%s): b0=%.3f b1=%.3f b2=%.3f",
                 COEFF_SOURCE, B0, B1, B2)
        if COEFF_SOURCE.startswith("synthetic"):
            log.warning("DİKKAT: katsayılar SENTETİK veriden; gerçek piyasa "
                        "kalibrasyonu değil. %s", COEFF_PROVENANCE)
    except FileNotFoundError:
        log.info("Eğitilmiş katsayı yok; varsayılanlar kullanılıyor (train.py ile eğit).")
    except Exception as e:
        log.warning("Katsayı yükleme hatası, varsayılana dönülüyor: %s", e)


def coeff_info() -> dict:
    """Yüklü katsayıların kaynağı ve değerleri — teşhis/rapor için."""
    return {"source": COEFF_SOURCE, "provenance": COEFF_PROVENANCE,
            "b0": B0, "b1": B1, "b2": B2, "whale_scale": WHALE_SCALE}


_load_coeffs()


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


# Hızlı ufuk ağırlığı: kısa vadeli imbalance daha taze sinyal taşır
FAST_WEIGHT = 0.6


# Defter okuma katkisi (D5): dusuk agirlik, guven carpani olarak kullanilir.
BOOK_WEIGHT = 0.35
BOOK_SPOOF_PENALTY = 0.4
BOOK_LIQUIDITY_PENALTY = 0.25


def _book_adjustment(book: BookState | None) -> tuple[float, float]:
    """Defter okumalarindan ek skor ve guven carpani dondur.

    Returns:
        (extra_z, confidence_multiplier)
    """
    if book is None:
        return 0.0, 1.0
    # Yon sinyali: derinlik dengesizligi + emilim teyidi; spoof cezasi
    spoof = getattr(book, "spoof_score", 0.0) or 0.0
    signal = book.depth_imbalance * (1 - BOOK_SPOOF_PENALTY * spoof)
    signal += 0.3 * getattr(book, "absorption", 0.0)
    extra_z = BOOK_WEIGHT * signal

    # Dar/incede veya yuksek fiyat etkisinde guveni kis
    slope_penalty = 1.0 - BOOK_LIQUIDITY_PENALTY * min(1.0, book.book_slope / 10.0)
    lambda_penalty = 1.0 - BOOK_LIQUIDITY_PENALTY * min(1.0, book.kyle_lambda / 1e-5)
    mult = max(0.5, slope_penalty * lambda_penalty)
    return extra_z, mult


def predict(feat: FlowFeatures, book_state: BookState | None = None) -> PricePrediction:
    # Çok-ufuklu imbalance harmanı (hızlı + yavaş)
    fast = getattr(feat, "imbalance_fast", 0.0) or 0.0
    imbalance = FAST_WEIGHT * fast + (1 - FAST_WEIGHT) * feat.flow_imbalance

    z = B0 + B1 * imbalance + B2 * math.tanh(feat.whale_net_usd / WHALE_SCALE)
    extra_z, book_mult = _book_adjustment(book_state)
    z += extra_z
    prob_up = _sigmoid(z)

    # Defter okuma guven carpani
    prob_up = 0.5 + (prob_up - 0.5) * book_mult

    # Az veri varsa güveni 0.5'e doğru çek (belirsizliği yansıt)
    if feat.sample_count < MIN_SAMPLES:
        prob_up = 0.5 + (prob_up - 0.5) * (feat.sample_count / MIN_SAMPLES)

    # Yüksek VPIN (toksik akış) → tahmini 0.5'e doğru hafifçe yumuşat (belirsizlik)
    vpin = getattr(feat, "vpin", 0.0) or 0.0
    prob_up = 0.5 + (prob_up - 0.5) * (1 - 0.3 * vpin)

    return PricePrediction(
        token=feat.token,
        prob_up=round(prob_up, 3),
        flow_imbalance=round(imbalance, 3),
        window_sec=feat.window_sec,
        sample_count=feat.sample_count,
        whale_net_usd=round(feat.whale_net_usd, 2),
        toxicity=round(vpin, 3),
    )


def predict_toxic(feat: FlowFeatures, book_state: BookState | None = None) -> PricePrediction:
    """Toksik/yüksek-volatilite rejimi tahmincisi (nonlineer stand-in).

    NOT: Burası gerçek sistemde OFFLINE eğitilmiş bir LSTM/CNN ile değiştirilir
    (fracdiff özellikleriyle beslenir). Stand-in olarak: en taze sinyale (hızlı
    imbalance) ağırlık verir, ama yüksek belirsizlik nedeniyle büyüklüğü daha
    güçlü bastırır — yüksek-vol rejiminde 'temkinli ama hızlı' davranış."""
    fast = getattr(feat, "imbalance_fast", 0.0) or 0.0
    vpin = getattr(feat, "vpin", 0.0) or 0.0
    # Daha çok hızlı ufka yaslan, tanh ile sıkıştır
    signal = math.tanh(1.5 * fast + 0.8 * math.tanh(feat.whale_net_usd / WHALE_SCALE))

    # D5: defter okuma katkisi (toksik rejimde daha agresif guven kisilir)
    extra_z, book_mult = _book_adjustment(book_state)
    signal += extra_z
    prob_up = _sigmoid(2.0 * signal)
    prob_up = 0.5 + (prob_up - 0.5) * book_mult

    # Toksik rejim → büyüklüğü güçlü bastır (0.5'e çek)
    prob_up = 0.5 + (prob_up - 0.5) * (1 - 0.5 * vpin)
    if feat.sample_count < MIN_SAMPLES:
        prob_up = 0.5 + (prob_up - 0.5) * (feat.sample_count / MIN_SAMPLES)
    return PricePrediction(
        token=feat.token, prob_up=round(prob_up, 3),
        flow_imbalance=round(fast, 3), window_sec=feat.window_sec,
        sample_count=feat.sample_count, whale_net_usd=round(feat.whale_net_usd, 2),
        toxicity=round(vpin, 3),
    )
