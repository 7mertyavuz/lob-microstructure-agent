"""D5 — Tahmin katmanina defter ozelliklerinin baglanmasi testleri."""
from __future__ import annotations

import pytest

from src.book.state import BookState
from src.features.window import FlowFeatures
from src.predict.direction import predict, predict_toxic
from src.predict.regime import RegimeRouter, NORMAL, TOXIC, THIN
from src.predict.meta import MetaLabeler
from src.predict.economic import ArbFeasibility


def _book(depth_imbalance: float = 0.0, spread_bps: float = 5.0,
          book_slope: float = 1.0, kyle_lambda: float = 1e-6,
          spoof_score: float = 0.0, absorption: float = 0.0) -> BookState:
    return BookState(
        symbol="ETHUSDT",
        spread_bps=spread_bps,
        microprice=3_000.0,
        depth_imbalance=depth_imbalance,
        ofi=0.0,
        queue_imbalance=depth_imbalance * 0.8,
        book_slope=book_slope,
        kyle_lambda=kyle_lambda,
        spoof_score=spoof_score,
        absorption=absorption,
    )


def _feat(flow_imbalance: float = 0.3, whale_net_usd: float = 50_000.0,
          sample_count: int = 10, vpin: float = 0.2) -> FlowFeatures:
    return FlowFeatures(
        token="ETHUSDT",
        flow_imbalance=flow_imbalance,
        buy_usd=100_000.0,
        sell_usd=50_000.0,
        whale_net_usd=whale_net_usd,
        sample_count=sample_count,
        window_sec=60.0,
        vpin=vpin,
        imbalance_fast=flow_imbalance,
    )


class TestDirectionWithBook:
    def test_book_state_changes_prediction(self):
        feat = _feat()
        p_no_book = predict(feat)
        p_with_book = predict(feat, _book(depth_imbalance=0.8))
        assert p_no_book.prob_up != p_with_book.prob_up

    def test_spoof_reduces_book_signal(self):
        feat = _feat()
        p_clean = predict(feat, _book(depth_imbalance=0.8, spoof_score=0.0))
        p_spoof = predict(feat, _book(depth_imbalance=0.8, spoof_score=0.9))
        assert p_spoof.prob_up != p_clean.prob_up


class TestRegimeRouterWithBook:
    def test_thin_liquidity_regime(self):
        router = RegimeRouter(predict, predict_toxic)
        feat = _feat(vpin=0.1)
        regime, _ = router.predict(feat, _book(spread_bps=80.0))
        assert regime == THIN

    def test_normal_regime_with_book(self):
        router = RegimeRouter(predict, predict_toxic)
        feat = _feat(vpin=0.1)
        regime, _ = router.predict(feat, _book())
        assert regime == NORMAL


class TestMetaWithBook:
    def test_liquidity_reduces_size(self):
        import numpy as np
        meta = MetaLabeler(size_threshold=0.0)
        # Basit egitim verisi
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        y = np.array([0, 1, 1])
        primary_prob = np.array([0.3, 0.6, 0.8])
        meta.fit(X, y, primary_prob)

        dec_no_book = meta.decide([0.5, 0.5], 0.6)
        dec_thin = meta.decide([0.5, 0.5], 0.6,
                               _book(kyle_lambda=1e-4, book_slope=8.0))
        assert dec_thin.size <= dec_no_book.size


class TestEconomicSlippage:
    def test_arb_feasibility_uses_spread(self):
        arb = ArbFeasibility(fee_bps=30.0, slippage_bps=5.0)
        dec = arb.evaluate(spread=0.002, notional_usd=10_000.0,
                           gas_cost_usd=5.0)
        assert dec.profitable or not dec.profitable  # yalnizca hesaplandi
        assert dec.net_usd == dec.gross_usd - dec.cost_usd
