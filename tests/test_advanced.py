import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.features.fracdiff import ffd_weights, frac_diff_ffd
from src.features.window import FlowFeatures
from src.predict.regime import GaussianHMM2, RegimeRouter, NORMAL, TOXIC
from src.predict.direction import predict, predict_toxic
from src.predict.meta import MetaLabeler, primary_direction
from src.train.dataset import make_synthetic
from src.train.logreg import LogReg
from src.train.backtest import split, metrics


# ---------- Fractional differencing ----------
def test_ffd_weights_recent_is_one():
    w = ffd_weights(0.4)
    assert abs(w[-1] - 1.0) < 1e-12      # en yeni gözlem ağırlığı 1
    assert len(w) > 1


def test_ffd_d0_is_identity():
    s = np.linspace(1, 10, 30)
    fd = frac_diff_ffd(s, 0.0)
    assert np.allclose(fd, s)            # d=0 → değişmez


def test_ffd_d1_is_first_diff():
    rng = np.random.default_rng(0)
    s = np.cumsum(rng.normal(0, 1, 200))
    fd = frac_diff_ffd(s, 1.0)
    expected = np.diff(s)
    assert np.allclose(fd[1:], expected, atol=1e-9)


def test_ffd_reduces_variance_growth():
    # Random walk (durağan değil) vs kesirli fark (d=0.4) → daha küçük varyans
    rng = np.random.default_rng(1)
    s = np.cumsum(rng.normal(0, 1, 2000))
    fd = frac_diff_ffd(s, 0.4)
    assert np.nanvar(fd) < np.var(s)


# ---------- Regime: HMM ----------
def test_hmm_separates_two_variance_regimes():
    rng = np.random.default_rng(2)
    calm = np.abs(rng.normal(0, 0.2, 400))
    storm = np.abs(rng.normal(0, 2.0, 400))
    x = np.concatenate([calm, storm])
    truth = np.concatenate([np.zeros(400), np.ones(400)])
    hmm = GaussianHMM2(n_iter=40).fit(x)
    pred = hmm.predict_states(x)
    acc = max((pred == truth).mean(), (pred != truth).mean())
    assert acc > 0.85                    # iki rejimi büyük ölçüde ayırıyor


# ---------- Regime: router ----------
def test_router_picks_model_by_vpin():
    router = RegimeRouter(predict, predict_toxic, vpin_threshold=0.4)
    calm = FlowFeatures("T", 0.8, 1e5, 1e4, 5e5, 6, 60, vpin=0.1, imbalance_fast=0.8)
    toxic = FlowFeatures("T", 0.8, 1e5, 1e4, 5e5, 6, 60, vpin=0.7, imbalance_fast=0.8)
    r1, _ = router.predict(calm); r2, _ = router.predict(toxic)
    assert r1 == NORMAL and r2 == TOXIC


# ---------- Meta-labeling ----------
def test_meta_improves_precision():
    X, y = make_synthetic(n=8000, true_b=(0.0, 2.5, 1.5), noise=1.2, seed=4)
    Xtr, ytr, Xte, yte = split(X, y)
    primary = LogReg(lr=0.2, epochs=3000).fit(Xtr, ytr)
    p_tr = primary.predict_proba(Xtr)
    p_te = primary.predict_proba(Xte)

    meta = MetaLabeler(size_threshold=0.6).fit(Xtr, ytr, p_tr)
    sizes = meta.sizes(Xte, p_te)
    dir_te = primary_direction(p_te)

    base_acc = (dir_te == yte).mean()                 # tüm işlemler
    take = sizes >= 0.6
    if take.sum() > 30:                               # yeterli örnek varsa
        sel_acc = (dir_te[take] == yte[take]).mean()  # meta'nın seçtikleri
        assert sel_acc >= base_acc                    # kesinlik artmalı (azalmamalı)
    assert sizes.min() >= 0.0 and sizes.max() <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} OK")
    print("ileri stack testleri GEÇTİ")
