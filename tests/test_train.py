import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.train.dataset import make_synthetic
from src.train.logreg import LogReg
from src.train.backtest import split, metrics


def test_recovers_true_coeffs():
    # Gürültüsüz veride trainer gerçek katsayıları (2.5, 1.5) yaklaşık bulmalı
    X, y = make_synthetic(n=8000, true_b=(0.0, 2.5, 1.5), noise=0.0, seed=3)
    m = LogReg(lr=0.3, epochs=5000).fit(X, y)
    c = m.coeffs()
    assert abs(c["b1"] - 2.5) < 0.6, c
    assert abs(c["b2"] - 1.5) < 0.6, c


def test_beats_random():
    import numpy as np
    X, y = make_synthetic(n=8000, true_b=(0.0, 2.5, 1.5), noise=1.0, seed=5)
    Xtr, ytr, Xte, yte = split(X, y)
    m = LogReg(lr=0.2, epochs=4000).fit(Xtr, ytr)
    learned = metrics(yte, m.predict_proba(Xte))
    # Trivial (hep 0.5) tahmincinin logloss'u ~0.693; öğrenilen bundan iyi olmalı.
    trivial_logloss = metrics(yte, np.full_like(yte, 0.5))["logloss"]
    assert learned["accuracy"] > 0.55          # rastgeleden (0.5) iyi
    assert learned["auc"] > 0.65               # ayırt ediciliği var
    assert learned["logloss"] < trivial_logloss   # trivial'dan iyi


def test_auc_bounds():
    X, y = make_synthetic(n=2000, seed=1)
    m = LogReg(epochs=1000).fit(X, y)
    auc = metrics(y, m.predict_proba(X))["auc"]
    assert 0.5 <= auc <= 1.0


if __name__ == "__main__":
    test_recovers_true_coeffs(); test_beats_random(); test_auc_bounds()
    print("train testleri GEÇTİ")
