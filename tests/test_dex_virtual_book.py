"""D4 — DEX sanal defteri testleri."""
from __future__ import annotations

import pytest

from lob_microstructure.book.dex_virtual_book import DexVirtualBook, tick_to_price, price_to_tick
from lob_microstructure.book import BookFeed, BookState


class TestTickMath:
    def test_tick_to_price_roundtrip(self):
        for tick in (0, 60, -60, 200000, -200000):
            assert price_to_tick(tick_to_price(tick)) == pytest.approx(tick, abs=1)


class TestDexVirtualBook:
    def test_construct_and_raw_book(self):
        book = DexVirtualBook(symbol="ETHUSDT", base_price=3_000.0, seed=42)
        rb = book.to_raw_book()
        assert rb.symbol == "ETHUSDT"
        assert len(rb.bids) == book.depth_levels
        assert len(rb.asks) == book.depth_levels
        assert rb.bids[0].price < rb.asks[0].price

    def test_determinism(self):
        b1 = DexVirtualBook(symbol="ETHUSDT", base_price=3_000.0, seed=7)
        b2 = DexVirtualBook(symbol="ETHUSDT", base_price=3_000.0, seed=7)
        for _ in range(5):
            r1, r2 = b1.step(), b2.step()
            assert len(r1.bids) == len(r2.bids)
            assert [p for p, _ in r1.bids] == [p for p, _ in r2.bids]
            assert [q for _, q in r1.bids] == [q for _, q in r2.bids]

    def test_swap_changes_price(self):
        book = DexVirtualBook(symbol="ETHUSDT", base_price=3_000.0, seed=1)
        start = book._current_tick
        amount_out, impact, end_tick = book.simulate_swap(
            amount_in=50_000.0, zero_for_one=True
        )
        assert end_tick != start or abs(impact) < 1e-9
        assert impact <= 0.0  # satis fiyati dusurur
        assert amount_out > 0.0

    def test_pending_swap_becomes_trade(self):
        book = DexVirtualBook(symbol="ETHUSDT", base_price=3_000.0, seed=1)
        book.inject_pending_swap(20_000.0, "BUY", "WHALE")
        rb = book.to_raw_book()
        assert len(rb.trades) == 1
        assert rb.trades[0].side == "BUY"
        assert len(rb.events) == 1
        assert rb.events[0].kind == "trade"


class TestBookFeedDexVenue:
    def test_book_feed_dex_returns_state(self):
        feed = BookFeed(mode="simulation", venue="dex", seed=3, base_price=3_000.0)
        state = feed.latest("ETHUSDT")
        assert isinstance(state, BookState)
        assert state.symbol == "ETHUSDT"
        assert state.spread_bps >= 0.0
        assert -1.0 <= state.depth_imbalance <= 1.0

    def test_book_feed_dex_inject_swap(self):
        feed = BookFeed(mode="simulation", venue="dex", seed=3, base_price=3_000.0)
        feed.inject_dex_swap("ETHUSDT", 50_000.0, "SELL", "WHALE")
        state = feed.latest("ETHUSDT")
        assert isinstance(state, BookState)
