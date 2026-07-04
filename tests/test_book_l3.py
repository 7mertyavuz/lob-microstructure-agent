"""Faz D3 -- L3-türevi okumalar, bilinen-cevaplı senaryolar.

Skorlar 'kanıt değil şüphe': burada yalnızca doğru yönde/aralıkta tepki
verdikleri kontrol edilir (kontrollü deney mantığı, D7'nin ön provası)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.book.state import Trade, OrderEvent
from src.book import features as F


# ---------- iceberg ----------
def test_iceberg_high_when_repeated_full_refill():
    refills = [(10, 10), (10, 9), (10, 10)]  # her işlemden sonra tam yenilenme
    assert F.iceberg_score(refills) > 0.7


def test_iceberg_zero_when_no_refill():
    assert F.iceberg_score([]) == 0.0
    assert F.iceberg_score([(10, 0), (10, 0)]) == 0.0


def test_iceberg_low_for_single_weak_refill():
    assert F.iceberg_score([(10, 1)]) < 0.3


# ---------- spoof ----------
def test_spoof_high_when_far_large_all_cancelled():
    ev = [
        OrderEvent("add", "SELL", 110, 100, 0.0, 30),
        OrderEvent("cancel", "SELL", 110, 100, 0.1, 30),
        OrderEvent("add", "SELL", 111, 100, 0.2, 35),
        OrderEvent("cancel", "SELL", 111, 100, 0.3, 35),
    ]
    assert F.spoof_score(ev) > 0.9


def test_spoof_zero_when_far_orders_trade():
    ev = [
        OrderEvent("add", "SELL", 110, 100, 0.0, 30),
        OrderEvent("trade", "SELL", 110, 100, 0.1, 30),
    ]
    assert F.spoof_score(ev) == 0.0


def test_spoof_ignores_near_orders():
    ev = [
        OrderEvent("add", "BUY", 100, 100, 0.0, 1),
        OrderEvent("cancel", "BUY", 100, 100, 0.1, 1),
    ]
    assert F.spoof_score(ev, min_dist_bps=10) == 0.0


# ---------- absorption ----------
def _sells(n, qty=5.0):
    return [Trade(100.0, qty, "SELL", i * 0.01) for i in range(n)]


def _buys(n, qty=5.0):
    return [Trade(100.0, qty, "BUY", i * 0.01) for i in range(n)]


def test_absorption_positive_when_sells_fail_to_move_price():
    # yoğun satış, fiyat düşmedi → bid emiyor → +
    a = F.absorption(_sells(6), price_change=0.0, ref_move=1.0)
    assert a > 0.5


def test_absorption_negative_when_buys_fail_to_move_price():
    a = F.absorption(_buys(6), price_change=0.0, ref_move=1.0)
    assert a < -0.5


def test_absorption_zero_when_price_moves_as_expected():
    # satış baskısı fiyatı tam beklendiği kadar düşürdü → emilim yok
    a = F.absorption(_sells(6), price_change=-1.0, ref_move=1.0)
    assert abs(a) < 1e-6


def test_absorption_empty_zero():
    assert F.absorption([], 0.0, 1.0) == 0.0


# ---------- sweep ----------
def test_sweep_detects_multi_level_buy():
    trades = [Trade(100 + i, 5, "BUY", i * 0.01) for i in range(4)]
    score, direction = F.sweep_score(trades, target_levels=4)
    assert score == 1.0 and direction == 1


def test_sweep_detects_sell_direction():
    trades = [Trade(100 - i, 5, "SELL", i * 0.01) for i in range(3)]
    score, direction = F.sweep_score(trades, target_levels=4)
    assert direction == -1 and score > 0


def test_sweep_zero_for_single_trade():
    score, direction = F.sweep_score([Trade(100, 5, "BUY", 0.0)])
    assert score == 0.0 and direction == 0


# ---------- liquidation map skew ----------
def test_liq_skew_positive_above_mid():
    liqs = [(105, 200_000), (106, 300_000)]
    assert F.liq_map_skew(liqs, mid=100) == 1.0


def test_liq_skew_negative_below_mid():
    liqs = [(95, 200_000), (94, 300_000)]
    assert F.liq_map_skew(liqs, mid=100) == -1.0


def test_liq_skew_balanced_zero():
    liqs = [(105, 100_000), (95, 100_000)]
    assert abs(F.liq_map_skew(liqs, mid=100)) < 1e-9


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
