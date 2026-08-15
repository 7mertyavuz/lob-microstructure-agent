import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lob_microstructure.mev.builder_tip import (priority_fee_per_gas, coinbase_transfer_wei,
                                 builder_payment, mev_score_from_payment)
from lob_microstructure.mev.jit_liquidity import detect_jit
from lob_microstructure.features.lead_lag import LeadLagSpread, estimate_lag
from lob_microstructure.train.online import OnlineLogReg, FeatureMonitor
from lob_microstructure.actor.classifier import classify
from lob_microstructure.models import PendingTx, DecodedSwap, Side, ActorLabel
from lob_microstructure.decode.tx_decoder import UNISWAP_V2_ROUTER


# ---------- Builder tip / coinbase ----------
def test_coinbase_transfer_sum():
    trace = [{"to": "0xBUILDER", "value": 10**18}, {"to": "0xOTHER", "value": 5},
             {"to": "0xbuilder", "value": 2 * 10**18}]
    assert coinbase_transfer_wei(trace, "0xBUILDER") == 3 * 10**18


def test_priority_fee():
    assert priority_fee_per_gas(30 * 10**9, 20 * 10**9) == 10 * 10**9
    assert priority_fee_per_gas(10 * 10**9, 20 * 10**9) == 0


def test_payment_flags_mev():
    pay = builder_payment(gas_used=200000, gas_price_wei=25*10**9, base_fee_wei=20*10**9,
                          trace=[{"to": "0xfee", "value": 5*10**17}], fee_recipient="0xFEE")
    assert pay.direct_builder_payment is True
    assert mev_score_from_payment(pay, 20*10**9) >= 0.8


def test_classify_coinbase_forces_mev():
    tx = PendingTx("0x"+"ab"*32, "0x"+"11"*20, UNISWAP_V2_ROUTER,
                   10**18, 21*10**9, "0x7ff36ab5")
    swap = DecodedSwap(tx=tx, dex="UniswapV2", method="swapExactETHForTokens", side=Side.BUY)
    sig = classify(swap, base_fee_wei=20*10**9, coinbase_wei=3*10**17)
    assert sig.label == ActorLabel.MEV_BOT
    assert any("coinbase" in r for r in sig.reasons)


# ---------- JIT ----------
def test_detect_jit_pattern():
    events = [
        {"address": "0xBOT", "pool": "0xP", "kind": "MINT", "tx_index": 5, "liquidity_usd": 1e6},
        {"address": "0xUSER", "pool": "0xP", "kind": "SWAP", "tx_index": 6},
        {"address": "0xBOT", "pool": "0xP", "kind": "BURN", "tx_index": 7},
    ]
    res = detect_jit(events)
    assert len(res) == 1 and res[0].provider == "0xbot" and res[0].victim == "0xuser"


def test_no_jit_without_victim_between():
    events = [
        {"address": "0xBOT", "pool": "0xP", "kind": "MINT", "tx_index": 5},
        {"address": "0xBOT", "pool": "0xP", "kind": "BURN", "tx_index": 6},
    ]
    assert detect_jit(events) == []


# ---------- Lead-lag ----------
def test_estimate_lag_recovers_known_lag():
    rng = np.random.default_rng(0)
    cex = np.cumsum(rng.normal(0, 1, 500))
    lag = 3
    dex = np.empty_like(cex)
    dex[:lag] = cex[0]
    dex[lag:] = cex[:-lag]            # DEX, CEX'i 3 adım gecikmeyle takip eder
    est = estimate_lag(list(cex), list(dex), max_lag=10)
    assert abs(est - lag) <= 1


def test_leadlag_directional_bias():
    ll = LeadLagSpread(lag=0)
    for p in [100, 101, 102]:
        ll.update_cex(p)
    ll.update_dex(99)                 # DEX, CEX'in altında → LONG bias
    assert ll.directional_bias() == 1
    ll.update_dex(105)                # DEX üstte → SHORT
    assert ll.directional_bias() == -1


# ---------- Online learning ----------
def test_online_adapts_to_drift():
    rng = np.random.default_rng(1)
    olr = OnlineLogReg(n_features=2, lr=0.1, forget=0.999)
    def gen(b, n=1500):
        X = rng.uniform(-1, 1, (n, 2))
        z = b[0]*X[:, 0] + b[1]*X[:, 1]
        y = (rng.uniform(0, 1, n) < 1/(1+np.exp(-z))).astype(float)
        return X, y
    # rejim 1
    X1, y1 = gen([3.0, 0.0])
    for x, t in zip(X1, y1): olr.partial_fit(x, t)
    # rejim DEĞİŞİR: ikinci özellik baskın olur
    X2, y2 = gen([0.0, 3.0])
    early = np.mean([olr.partial_fit(x, t) for x, t in zip(X2[:300], y2[:300])])
    late = np.mean([olr.partial_fit(x, t) for x, t in zip(X2[-300:], y2[-300:])])
    assert late < early              # drift sonrası kayıp düşüyor (adapte oluyor)


def test_feature_monitor_detects_decay():
    rng = np.random.default_rng(2)
    mon = FeatureMonitor(n_features=2, names=["useful", "noise"], halflife=100)
    for _ in range(2000):
        y = rng.integers(0, 2)
        x0 = y + rng.normal(0, 0.3)    # 'useful' y ile korele
        x1 = rng.normal(0, 1)          # 'noise' alakasız
        mon.update([x0, x1], y)
    imp = mon.importances()
    assert imp["useful"] > imp["noise"]
    assert "noise" in mon.decayed(threshold=0.15)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} OK")
    print("alfa modülleri testleri GEÇTİ")
