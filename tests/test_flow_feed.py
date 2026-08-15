"""Faz 1 -- FlowFeed adaptoru testleri (docs/00-ORTAK-SOZLESME.md sozlesmesi)."""
import sys
import os
from dataclasses import asdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_microstructure.api.flow_feed import FlowFeed, map_regime, REGIME_NORMAL, REGIME_TOXIC, REGIME_HIGHVOL
from lob_microstructure.models import FlowState

TOKENS = ["UniswapV2", "UniswapV3"]


def test_latest_returns_full_flowstate_sim_mode():
    f = FlowFeed(mode="simulation", seed=1)
    fs = f.latest("UniswapV2")
    assert isinstance(fs, FlowState)
    d = asdict(fs)
    for key in ("token", "flow_imbalance", "vpin_toxicity", "whale_net_usd",
                "actor_mix", "direction_prob_up", "lead_lag_spread", "regime", "ts"):
        assert key in d, f"eksik alan: {key}"
        assert d[key] is not None


def test_no_print_returns_struct():
    # latest() bir FlowState struct'i dondurmeli, print/None degil
    f = FlowFeed(mode="simulation", seed=2)
    result = f.latest("UniswapV2")
    assert result is not None
    assert type(result).__name__ == "FlowState"


def test_actor_mix_sums_to_one():
    f = FlowFeed(mode="simulation", seed=3)
    for token in TOKENS:
        fs = f.latest(token)
        total = sum(fs.actor_mix.values())
        assert abs(total - 1.0) < 1e-6, f"actor_mix toplami 1.0 degil: {total}"
        assert set(fs.actor_mix.keys()) == {"WHALE", "MEV_BOT", "RETAIL"}


def test_actor_mix_empty_window_neutral():
    f = FlowFeed(mode="simulation", seed=4)
    # hic sentetik akis beslenmemis bir token icin actor_mix dogrudan cagrildiginda
    mix = f.flow.actor_mix("BOS_TOKEN")
    assert abs(sum(mix.values()) - 1.0) < 1e-5
    assert all(abs(v - 1 / 3) < 1e-5 for v in mix.values())


def test_regime_only_allowed_values():
    f = FlowFeed(mode="simulation", seed=5)
    for token in TOKENS:
        for _ in range(5):
            fs = f.latest(token)
            assert fs.regime in ("normal", "toxic", "highvol")


def test_map_regime_thresholds():
    assert map_regime(0.0) == REGIME_NORMAL
    assert map_regime(0.39) == REGIME_NORMAL
    assert map_regime(0.4) == REGIME_TOXIC
    assert map_regime(0.69) == REGIME_TOXIC
    assert map_regime(0.7) == REGIME_HIGHVOL
    assert map_regime(1.0) == REGIME_HIGHVOL


def test_determinism_same_seed_same_output():
    f1 = FlowFeed(mode="simulation", seed=123)
    f2 = FlowFeed(mode="simulation", seed=123)
    a = asdict(f1.latest("UniswapV2"))
    b = asdict(f2.latest("UniswapV2"))
    a.pop("ts"); b.pop("ts")
    assert a == b, "ayni seed ayni cikti vermeli (ts haric)"


def test_determinism_multi_step_sequence():
    f1 = FlowFeed(mode="simulation", seed=555)
    f2 = FlowFeed(mode="simulation", seed=555)
    seq1, seq2 = [], []
    for _ in range(4):
        d1 = asdict(f1.latest("UniswapV3")); d1.pop("ts")
        d2 = asdict(f2.latest("UniswapV3")); d2.pop("ts")
        seq1.append(d1)
        seq2.append(d2)
    assert seq1 == seq2


def test_value_ranges():
    f = FlowFeed(mode="simulation", seed=6)
    for token in TOKENS:
        for _ in range(5):
            fs = f.latest(token)
            assert -1.0 - 1e-9 <= fs.flow_imbalance <= 1.0 + 1e-9
            assert 0.0 - 1e-9 <= fs.vpin_toxicity <= 1.0 + 1e-9
            assert 0.0 - 1e-9 <= fs.direction_prob_up <= 1.0 + 1e-9


def test_ts_is_utc_tz_aware():
    f = FlowFeed(mode="simulation", seed=8)
    fs = f.latest("UniswapV2")
    assert isinstance(fs.ts, datetime)
    assert fs.ts.tzinfo is not None
    assert fs.ts.tzinfo == timezone.utc or fs.ts.utcoffset().total_seconds() == 0


def test_live_mode_valid_flowstate_no_external_deps():
    # live mod da (WSS_URL yoksa CONFIG.simulation_mode True oldugundan
    # otomatik simulation'a duser) gecerli bir FlowState uretmeli.
    f = FlowFeed(mode="live", seed=9)
    fs = f.latest("UniswapV2")
    assert isinstance(fs, FlowState)


def test_invalid_mode_raises():
    try:
        FlowFeed(mode="bogus")
        assert False, "gecersiz mode ValueError firlatmali"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
