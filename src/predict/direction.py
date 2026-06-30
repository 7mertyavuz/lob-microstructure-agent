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

from src.features.window import FlowFeatures
from src.models import PricePrediction

log = logging.getLogger("predict")

# Varsayılan (kalibre edilmemiş) katsayılar
B0, B1, B2 = 0.0, 2.2, 1.3
WHALE_SCALE = 250_000.0
MIN_SAMPLES = 3

_COEFF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "direction_coeffs.json",
)


def _load_coeffs() -> None:
    """Eğitilmiş katsayılar varsa yükle (import sırasında bir kez)."""
    global B0, B1, B2, WHALE_SCALE
    try:
        with open(_COEFF_PATH) as f:
            c = json.load(f)
        B0, B1, B2 = c["b0"], c["b1"], c["b2"]
        WHALE_SCALE = c.get("whale_scale", WHALE_SCALE)
        log.info("Eğitilmiş katsayılar yüklendi: b0=%.3f b1=%.3f b2=%.3f", B0, B1, B2)
    except FileNotFoundError:
        log.info("Eğitilmiş katsayı yok; varsayılanlar kullanılıyor (train.py ile eğit).")
    except Exception as e:
        log.warning("Katsayı yükleme hatası, varsayılana dönülüyor: %s", e)


_load_coeffs()


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


# Hızlı ufuk ağırlığı: kısa vadeli imbalance daha taze sinyal taşır
FAST_WEIGHT = 0.6


def predict(feat: FlowFeatures) -> PricePrediction:
    # Çok-ufuklu imbalance harmanı (hızlı + yavaş)
    fast = getattr(feat, "imbalance_fast", 0.0) or 0.0
    imbalance = FAST_WEIGHT * fast + (1 - FAST_WEIGHT) * feat.flow_imbalance

    z = B0 + B1 * imbalance + B2 * math.tanh(feat.whale_net_usd / WHALE_SCALE)
    prob_up = _sigmoid(z)

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


def predict_toxic(feat: FlowFeatures) -> PricePrediction:
    """Toksik/yüksek-volatilite rejimi tahmincisi (nonlineer stand-in).

    NOT: Burası gerçek sistemde OFFLINE eğitilmiş bir LSTM/CNN ile değiştirilir
    (fracdiff özellikleriyle beslenir). Stand-in olarak: en taze sinyale (hızlı
    imbalance) ağırlık verir, ama yüksek belirsizlik nedeniyle büyüklüğü daha
    güçlü bastırır — yüksek-vol rejiminde 'temkinli ama hızlı' davranış."""
    fast = getattr(feat, "imbalance_fast", 0.0) or 0.0
    vpin = getattr(feat, "vpin", 0.0) or 0.0
    # Daha çok hızlı ufka yaslan, tanh ile sıkıştır
    signal = math.tanh(1.5 * fast + 0.8 * math.tanh(feat.whale_net_usd / WHALE_SCALE))
    prob_up = _sigmoid(2.0 * signal)
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
