"""Faz D0 -- BookFeed / SimBookFeed uçtan uca sözleşme testleri."""
import sys
import os
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_microstructure.book import BookFeed, SimBookFeed, BookState
from lob_microstructure.book.keeper import BookKeeper

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

_FIELDS = ("symbol", "spread_bps", "microprice", "depth_imbalance", "ofi",
           "queue_imbalance", "book_slope", "kyle_lambda", "iceberg_score",
           "spoof_score", "absorption", "liq_map_skew", "ts")


def test_latest_returns_full_bookstate():
    f = BookFeed(mode="simulation", seed=1)
    bs = f.latest("BTCUSDT")
    assert isinstance(bs, BookState)
    d = asdict(bs)
    for k in _FIELDS:
        assert k in d and d[k] is not None, f"eksik alan: {k}"


def test_invalid_mode_raises():
    try:
        BookFeed(mode="bogus")
        assert False
    except ValueError:
        pass


def test_determinism_same_seed_same_sequence():
    f1 = BookFeed(mode="simulation", seed=42)
    f2 = BookFeed(mode="simulation", seed=42)
    for _ in range(6):
        a = asdict(f1.latest("BTCUSDT")); a.pop("ts")
        b = asdict(f2.latest("BTCUSDT")); b.pop("ts")
        assert a == b


def test_value_ranges():
    f = BookFeed(mode="simulation", seed=7)
    for sym in SYMBOLS:
        for _ in range(20):
            bs = f.latest(sym)
            assert bs.spread_bps >= 0.0
            assert bs.microprice > 0.0
            assert -1.0 - 1e-9 <= bs.depth_imbalance <= 1.0 + 1e-9
            assert -1.0 - 1e-9 <= bs.queue_imbalance <= 1.0 + 1e-9
            assert 0.0 <= bs.iceberg_score <= 1.0
            assert 0.0 <= bs.spoof_score <= 1.0
            assert -1.0 - 1e-9 <= bs.absorption <= 1.0 + 1e-9
            assert -1.0 - 1e-9 <= bs.liq_map_skew <= 1.0 + 1e-9
            assert bs.book_slope >= 0.0
            assert bs.kyle_lambda >= 0.0


def test_microprice_between_best_bid_ask():
    sim = SimBookFeed(seed=3)
    for _ in range(10):
        rb = sim.next("BTCUSDT")
        assert rb.best_bid.price < rb.best_ask.price  # geçerli defter
        assert rb.best_bid.price <= rb.mid <= rb.best_ask.price


def test_ts_tz_aware():
    f = BookFeed(mode="simulation", seed=8)
    bs = f.latest("BTCUSDT")
    assert isinstance(bs.ts, datetime)
    assert bs.ts.tzinfo is not None


def test_sim_regimes_all_appear():
    sim = SimBookFeed(seed=11)
    seen = set()
    for _ in range(200):
        rb = sim.next("BTCUSDT")
        seen.add(getattr(rb, "regime", None))
    assert {"calm", "toxic", "wide"}.issubset(seen)


def test_live_mode_empty_keeper_neutral_state():
    # WSS_URL yoksa CONFIG.simulation_mode True -> otomatik simulation'a düşer,
    # yine de geçerli BookState üretmeli.
    f = BookFeed(mode="live", seed=1)
    bs = f.latest("BTCUSDT")
    assert isinstance(bs, BookState)


def test_keeper_snapshot_diff():
    kp = BookKeeper("BTCUSDT")
    kp.apply_snapshot(bids=[(100, 5), (99, 3)], asks=[(101, 4), (102, 2)])
    bids, asks = kp.top()
    assert bids[0].price == 100 and asks[0].price == 101
    kp.apply_diff(bids=[(100, 0)], asks=[(101, 8)])  # 100'ü sil, 101'i güncelle
    bids, asks = kp.top()
    assert bids[0].price == 99
    assert asks[0].price == 101 and asks[0].qty == 8


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
