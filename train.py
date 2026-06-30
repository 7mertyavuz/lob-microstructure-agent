"""Yön modelini eğit ve katsayıları kaydet.

Kullanım:
    python train.py                 # sentetik veriyle (PoC / katsayı geri-kazanım)
    python train.py veri.csv        # gerçek etiketli CSV ile
                                    # kolonlar: flow_imbalance, whale_net_usd, label

Eğitilen katsayılar models/direction_coeffs.json'a yazılır; Katman 5
(predict/direction.py) sonraki çalıştırmada bunları otomatik yükler.
"""
from __future__ import annotations

import sys

from src.train.dataset import make_synthetic, load_csv
from src.train.logreg import LogReg
from src.train.backtest import split, metrics, save_coeffs, baseline_metrics


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        print(f"CSV yükleniyor: {argv[1]}")
        X, y = load_csv(argv[1])
    else:
        print("Sentetik veri (gerçek ilişki b=(0, 2.5, 1.5)) üretiliyor…")
        X, y = make_synthetic(n=8000, true_b=(0.0, 2.5, 1.5), noise=1.0)

    Xtr, ytr, Xte, yte = split(X, y)
    model = LogReg(lr=0.2, epochs=4000).fit(Xtr, ytr)

    base = baseline_metrics(Xte, yte)
    test = metrics(yte, model.predict_proba(Xte))
    coeffs = model.coeffs()

    print(f"\nÖrnek: {len(y)} (train {len(ytr)} / test {len(yte)})")
    print(f"Öğrenilen katsayılar: b0={coeffs['b0']:+.3f} b1={coeffs['b1']:+.3f} b2={coeffs['b2']:+.3f}")
    print(f"Baseline (elle):  {base}")
    print(f"Öğrenilen (test): {test}")
    delta = test["accuracy"] - base["accuracy"]
    print(f"Accuracy farkı:   {delta:+.4f}")

    path = save_coeffs(coeffs)
    print(f"\nKatsayılar kaydedildi → {path}")
    print("Katman 5 bir sonraki çalıştırmada bunları otomatik yükleyecek.")


if __name__ == "__main__":
    main(sys.argv)
