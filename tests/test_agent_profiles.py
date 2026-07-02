"""Faz 2 -- ajan profilleri + saf MEV karar fonksiyonlari testleri."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.actor.agent_profiles import PROFILES, AgentProfile, WHALE, MEV_BOT, RETAIL
from src.mev.decision import (
    decide_sandwich, decide_jit, decide_arbitrage, decide_builder_tip,
)


def test_three_profiles_present():
    assert set(PROFILES.keys()) == {"WHALE", "MEV_BOT", "RETAIL"}
    for name, prof in PROFILES.items():
        assert isinstance(prof, AgentProfile)
        lo, hi = prof.size_usd
        assert lo < hi
        assert prof.freq in ("low", "mid", "high")
        assert 0.0 <= prof.aggression <= 1.0
        assert isinstance(prof.trigger, str) and prof.trigger


def test_profile_field_contract_matches_plan():
    assert WHALE.size_usd == (50_000, 5_000_000)
    assert WHALE.freq == "low"
    assert MEV_BOT.freq == "high"
    assert MEV_BOT.aggression == 1.0
    assert RETAIL.size_usd == (50, 5_000)


def test_whale_size_consistent_with_classifier_threshold():
    # src/actor/classifier.py CONFIG.whale_usd_threshold varsayilan 100_000;
    # WHALE profilinin ust siniri bu esigin cok uzerinde olmali (celiskisiz).
    from config import CONFIG
    assert WHALE.size_usd[1] >= CONFIG.whale_usd_threshold


def _sandwich_swaps():
    return [
        {"address": "0xATT", "token": "0xTKN", "side": "BUY", "tx_index": 1, "est_value_usd": 50000},
        {"address": "0xVIC", "token": "0xTKN", "side": "BUY", "tx_index": 2, "est_value_usd": 8000},
        {"address": "0xATT", "token": "0xTKN", "side": "SELL", "tx_index": 3, "est_value_usd": 51000},
    ]


def test_decide_sandwich_pure_and_deterministic():
    swaps = _sandwich_swaps()
    r1 = decide_sandwich(swaps)
    r2 = decide_sandwich(swaps)
    assert len(r1) == len(r2) == 1
    assert r1[0] == r2[0]
    # girdi degistirilmemis olmali
    assert swaps == _sandwich_swaps()


def _jit_events():
    return [
        {"address": "0xJIT", "pool": "0xPOOL", "kind": "MINT", "tx_index": 1, "liquidity_usd": 500000},
        {"address": "0xVIC", "pool": "0xPOOL", "kind": "SWAP", "tx_index": 2},
        {"address": "0xJIT", "pool": "0xPOOL", "kind": "BURN", "tx_index": 3},
    ]


def test_decide_jit_pure_and_deterministic():
    events = _jit_events()
    r1 = decide_jit(events)
    r2 = decide_jit(events)
    assert len(r1) == len(r2) == 1
    assert r1[0].provider == "0xjit" and r1[0].victim == "0xvic"


def _arb_transfers(actor="0xARB"):
    weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    other = "0xTOKENB"
    return [
        {"token": weth, "amount": 10**18, "from": "0xpool", "to": actor},
        {"token": other, "amount": 500, "from": actor, "to": "0xpool2"},
        {"token": other, "amount": 500, "from": "0xpool2", "to": actor},
    ]


def test_decide_arbitrage_pure_and_deterministic():
    transfers = _arb_transfers()
    r1 = decide_arbitrage(transfers, "0xARB")
    r2 = decide_arbitrage(transfers, "0xARB")
    assert r1 == r2
    assert r1.is_arb is True


def test_decide_builder_tip_pure_and_deterministic():
    trace = [{"to": "0xfee", "value": 2 * 10**16}]
    r1 = decide_builder_tip(21000, 40 * 10**9, 20 * 10**9, trace=trace, fee_recipient="0xfee")
    r2 = decide_builder_tip(21000, 40 * 10**9, 20 * 10**9, trace=trace, fee_recipient="0xfee")
    assert r1[0] == r2[0]
    assert r1[1] == r2[1]
    assert r1[0].direct_builder_payment is True
    assert 0.0 <= r1[1] <= 1.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
