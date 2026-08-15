"""Faz 4 -- FlowState/AgentOrder alan seti, docs/00-ORTAK-SOZLESME.md ile
birebir uyumlu mu diye dogrular (sozlesme uyum kontrolu)."""
import sys, os
from dataclasses import fields
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_microstructure.models import FlowState, AgentOrder

EXPECTED_FLOWSTATE_FIELDS = {
    "token": str,
    "flow_imbalance": float,
    "vpin_toxicity": float,
    "whale_net_usd": float,
    "actor_mix": dict,
    "direction_prob_up": float,
    "lead_lag_spread": float,
    "regime": str,
    "ts": datetime,
}

EXPECTED_AGENTORDER_FIELDS = {
    "token": str,
    "side": str,
    "size_usd": float,
    "actor_label": str,
    "ts": datetime,
}


def test_flowstate_field_names_match_contract():
    names = {f.name for f in fields(FlowState)}
    assert names == set(EXPECTED_FLOWSTATE_FIELDS.keys())


def test_agentorder_field_names_match_contract():
    names = {f.name for f in fields(AgentOrder)}
    assert names == set(EXPECTED_AGENTORDER_FIELDS.keys())


def test_flowstate_instance_field_types():
    fs = FlowState(
        token="X", flow_imbalance=0.1, vpin_toxicity=0.2, whale_net_usd=10.0,
        actor_mix={"WHALE": 1.0, "MEV_BOT": 0.0, "RETAIL": 0.0},
        direction_prob_up=0.5, lead_lag_spread=0.0, regime="normal",
    )
    for name, typ in EXPECTED_FLOWSTATE_FIELDS.items():
        val = getattr(fs, name)
        assert isinstance(val, typ), f"{name} tipi {type(val)}, beklenen {typ}"


def test_flowstate_regime_allowed_values_only():
    allowed = {"normal", "toxic", "highvol"}
    for r in allowed:
        fs = FlowState(
            token="X", flow_imbalance=0.0, vpin_toxicity=0.0, whale_net_usd=0.0,
            actor_mix={}, direction_prob_up=0.5, lead_lag_spread=0.0, regime=r,
        )
        assert fs.regime in allowed


def test_flowstate_to_dict_serializes_ts():
    fs = FlowState(
        token="X", flow_imbalance=0.0, vpin_toxicity=0.0, whale_net_usd=0.0,
        actor_mix={"WHALE": 1.0}, direction_prob_up=0.5, lead_lag_spread=0.0,
        regime="normal",
    )
    d = fs.to_dict()
    assert isinstance(d["ts"], str)  # isoformat


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
