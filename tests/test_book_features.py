"""Faz D2 -- çekirdek L2 okuma özellikleri, bilinen-cevaplı senaryolar."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.book.state import BookLevel, RawBook
from src.book import features as F
from datetime import datetime, timezone


def _rb(bids, asks, **kw):
    return RawBook(symbol="X", ts=datetime.now(timezone.utc),
                   bids=[BookLevel(*b) for b in bids],
                   asks=[BookLevel(*a) for a in asks], **kw)


# ---------- microprice ----------
def test_microprice_symmetric_is_mid():
    assert F.microprice(100.0, 5.0, 102.0, 5.0) == 101.0


def test_microprice_heavy_bid_pulls_toward_ask():
    # bid tarafı ağır → fiyat ask'e (yukarı) çekilir
    m = F.microprice(100.0, 10.0, 101.0, 0.0)
    assert abs(m - 101.0) < 1e-9


def test_microprice_empty_queues_falls_back_to_mid():
    assert F.microprice(100.0, 0.0, 102.0, 0.0) == 101.0


# ---------- depth imbalance ----------
def test_depth_imbalance_all_bid_positive_one():
    v = F.depth_imbalance([BookLevel(100, 10)], [], mid=100)
    assert abs(v - 1.0) < 1e-9


def test_depth_imbalance_all_ask_negative_one():
    v = F.depth_imbalance([], [BookLevel(101, 10)], mid=100.5)
    assert abs(v + 1.0) < 1e-9


def test_depth_imbalance_symmetric_zero():
    v = F.depth_imbalance([BookLevel(99, 10)], [BookLevel(101, 10)], mid=100)
    assert abs(v) < 1e-9


def test_depth_imbalance_far_levels_downweighted():
    # uzak bid, yakın ask: ağırlıklı imbalance negatife yakın olmalı
    near = F.depth_imbalance([BookLevel(90, 10)], [BookLevel(100.1, 10)], mid=100)
    assert near < 0


# ---------- OFI ----------
def test_ofi_bid_up_is_positive():
    # bid fiyatı yükseldi → alış baskısı
    e = F.ofi_event(100, 5, 101, 5, 100.5, 4, 101, 5)
    assert e > 0


def test_ofi_ask_up_adds_prev_ask_qty():
    # ask yükseldi (geri çekildi) → alış baskısı bileşeni +
    e = F.ofi_event(100, 5, 101, 5, 100, 5, 101.5, 3)
    assert e > 0


def test_ofi_no_change_zero():
    e = F.ofi_event(100, 5, 101, 5, 100, 5, 101, 5)
    # bid==prev: +q_bid -q_prevbid = 0 ; ask==prev: -q_ask + q_prevask = 0
    assert abs(e) < 1e-9


# ---------- queue imbalance ----------
def test_queue_imbalance_range():
    assert F.queue_imbalance(10, 0) == 1.0
    assert F.queue_imbalance(0, 10) == -1.0
    assert F.queue_imbalance(5, 5) == 0.0
    assert F.queue_imbalance(0, 0) == 0.0


# ---------- book slope & kyle lambda ----------
def test_book_slope_positive_for_normal_book():
    bids = [(100 - i, 5) for i in range(1, 8)]
    asks = [(100 + i, 5) for i in range(1, 8)]
    s = F.book_slope(_rb(bids, asks))
    assert s > 0


def test_kyle_lambda_thin_book_higher_than_thick():
    bids_thin = [(100 - i, 1) for i in range(1, 6)]
    asks_thin = [(100 + i, 1) for i in range(1, 6)]
    bids_thick = [(100 - i, 50) for i in range(1, 6)]
    asks_thick = [(100 + i, 50) for i in range(1, 6)]
    lam_thin = F.kyle_lambda(_rb(bids_thin, asks_thin))
    lam_thick = F.kyle_lambda(_rb(bids_thick, asks_thick))
    assert lam_thin > lam_thick > 0


# ---------- spread z ----------
def test_spread_z_spike_positive():
    hist = [1.0] * 10 + [1.1, 0.9]
    z = F.spread_z(5.0, hist)
    assert z > 2


def test_spread_z_insufficient_history_zero():
    assert F.spread_z(5.0, [1.0]) == 0.0


# ---------- liquidity gaps ----------
def test_liquidity_gaps_uniform_zero():
    levels = [BookLevel(100 + i, 5) for i in range(6)]
    assert F.liquidity_gaps(levels, mid=100) == 0.0


def test_liquidity_gaps_detects_hole():
    # düzenli seviyeler + bir büyük atlama
    levels = [BookLevel(100, 5), BookLevel(101, 5), BookLevel(102, 5),
              BookLevel(120, 5), BookLevel(121, 5)]
    assert F.liquidity_gaps(levels, mid=100) > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
