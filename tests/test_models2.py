import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.predict.mlp import NumpyMLP, make_toxic_predictor, features_to_vector
from src.predict.regime import RegimeRouter, TOXIC
from src.predict.direction import predict
from src.predict.economic import net_pnl_usd, economic_label, ArbFeasibility, gas_cost_usd
from src.mev.arbitrage import detect_atomic_arb, net_positions, WETH
from src.features.window import FlowFeatures
from src.train.logreg import LogReg


# ---------- NumpyMLP ----------
def test_mlp_learns_xor():
    # XOR: lineer model ÇÖZEMEZ, MLP çözmeli
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0], float)
    mlp = NumpyMLP(2, n_hidden=8, lr=0.5, epochs=4000, seed=1).fit(X, y)
    pred = (mlp.predict_proba(X) >= 0.5).astype(int)
    assert (pred == y).all()
    # lineer baseline başarısız olmalı (kontrol)
    lr = LogReg(epochs=2000).fit(X, y)
    lin = (lr.predict_proba(X) >= 0.5).astype(int)
    assert not (lin == y).all()


def test_mlp_beats_logreg_on_circle():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (1500, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 0.5).astype(float)   # daire (nonlineer)
    mlp = NumpyMLP(2, 16, lr=0.3, epochs=3000, seed=2).fit(X, y)
    lr = LogReg(epochs=2000).fit(X, y)
    acc_mlp = ((mlp.predict_proba(X) >= 0.5) == y).mean()
    acc_lr = ((lr.predict_proba(X) >= 0.5) == y).mean()
    assert acc_mlp > acc_lr + 0.1


def test_mlp_router_integration():
    mlp = NumpyMLP(4, 8, epochs=200, seed=3)   # eğitimsiz bile arayüz çalışmalı
    toxic_fn = make_toxic_predictor(mlp)
    router = RegimeRouter(predict, toxic_fn, vpin_threshold=0.4)
    feat = FlowFeatures("T", 0.5, 1e5, 1e4, 3e5, 6, 60, vpin=0.7, imbalance_fast=0.5)
    regime, pred = router.predict(feat)
    assert regime == TOXIC
    assert 0.0 <= pred.prob_up <= 1.0


# ---------- Economic ----------
def test_net_pnl_and_label():
    # long, +%1 hareket, $100k notional, $50 gas
    pnl = net_pnl_usd(1, 0.01, 100_000, 50, fee_bps=30, slippage_bps=5)
    assert pnl > 0
    assert economic_label(1, 0.01, 100_000, 50) == 1
    # küçük hareket maliyeti karşılamaz
    assert economic_label(1, 0.0001, 100_000, 50) == 0


def test_arb_feasibility():
    f = ArbFeasibility(fee_bps=30, slippage_bps=5)
    # %1.5 spread, $50k, $30 gas → çift-bacak %0.7 maliyeti aşar, kârlı
    assert f.evaluate(0.015, 50_000, 30).profitable
    # %0.05 ve %0.5 spread çift-bacak %0.7 maliyeti karşılamaz (ekonomik gerçeklik)
    assert not f.evaluate(0.0005, 50_000, 30).profitable
    assert not f.evaluate(0.005, 50_000, 30).profitable


def test_gas_cost():
    # 200k gas @ 20 gwei @ $3000 ≈ $12
    c = gas_cost_usd(200_000, 20 * 10**9, 3000)
    assert 10 < c < 14


# ---------- Atomic arbitrage ----------
def test_detect_atomic_arb():
    a = "0xBOT"
    transfers = [
        {"from": a, "to": "0xP1", "token": WETH, "amount": 10 * 10**18},      # WETH çıkar
        {"from": "0xP1", "to": a, "token": "0xUSDC", "amount": 30000 * 10**6},
        {"from": a, "to": "0xP2", "token": "0xUSDC", "amount": 30000 * 10**6}, # USDC kapanır
        {"from": "0xP2", "to": a, "token": WETH, "amount": 11 * 10**18},       # daha çok WETH döner
    ]
    res = detect_atomic_arb(transfers, a)
    assert res.is_arb
    assert res.profit_wei == 1 * 10**18   # net +1 WETH kâr


def test_not_arb_when_open_position():
    a = "0xUSER"
    transfers = [
        {"from": a, "to": "0xP", "token": WETH, "amount": 5 * 10**18},
        {"from": "0xP", "to": a, "token": "0xTKN", "amount": 1000 * 10**18},   # token elde kaldı
    ]
    assert detect_atomic_arb(transfers, a).is_arb is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} OK")
    print("model2 testleri GEÇTİ")
