"""Fourier gürültü filtresi, Lee-Mykland sıçrama testi ve ölü yolların bağlanması."""
from __future__ import annotations

import numpy as np
import pytest

from lob_microstructure.api.flow_feed import FlowFeed, map_regime
from lob_microstructure.book.feed import BookFeed
from lob_microstructure.book.state import BookState
from lob_microstructure.features.fourier import (
    book_noise,
    lowpass,
    spectral_noise_ratio,
)
from lob_microstructure.features.jumps import jumps_from_prices, lee_mykland
from lob_microstructure.features.window import ManualClock, RollingFlow
from lob_microstructure.models import ActorLabel, ActorSignal, Side


# ── Fourier ───────────────────────────────────────────────────────
def test_noise_ratio_is_low_for_a_slow_wave():
    t = np.linspace(0, 4 * np.pi, 256)
    assert spectral_noise_ratio(np.sin(t)) < 0.1


def test_noise_ratio_is_high_for_fast_churn():
    t = np.arange(256)
    churn = np.sin(t * 2.0)          # neredeyse Nyquist
    assert spectral_noise_ratio(churn) > 0.7


def test_noise_ratio_separates_signal_from_churn():
    t = np.linspace(0, 4 * np.pi, 256)
    clean = np.sin(t)
    noisy = clean + 0.5 * np.sin(t * 40)
    assert spectral_noise_ratio(noisy) > spectral_noise_ratio(clean)


def test_pure_trend_is_not_mistaken_for_noise():
    """Detrend + Hann olmadan düz bir rampa yapay yüksek frekans üretirdi.

    Bu, pencerelemenin atlanması hâlinde sessizce yanlış çalışacak tam olarak
    o durumdur — hata vermez, sadece yanlış sayı verir.
    """
    assert spectral_noise_ratio(np.arange(128, dtype=float)) < 0.15


def test_noise_ratio_is_bounded():
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = spectral_noise_ratio(rng.normal(size=128))
        assert 0.0 <= v <= 1.0


def test_short_series_returns_zero_not_garbage():
    assert spectral_noise_ratio([1.0, 2.0, 3.0]) == 0.0


def test_constant_series_returns_zero():
    assert spectral_noise_ratio([5.0] * 64) == 0.0


def test_lowpass_removes_high_frequency_component():
    t = np.linspace(0, 4 * np.pi, 256)
    clean, noisy = np.sin(t), np.sin(t) + 0.5 * np.sin(t * 40)
    filt = lowpass(noisy)
    assert np.mean(np.abs(filt - clean)) < np.mean(np.abs(noisy - clean))


def test_book_noise_combines_mid_and_depth():
    t = np.linspace(0, 4 * np.pi, 128)
    out = book_noise(np.sin(t), np.sin(t * 30))
    assert out["depth_noise"] > out["mid_noise"]
    assert out["mid_noise"] <= out["noise_ratio"] <= out["depth_noise"]


# ── Lee-Mykland ───────────────────────────────────────────────────
def test_jump_test_finds_an_injected_jump():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0, 0.001, 800)
    r[400] = 0.05
    out = lee_mykland(r)
    assert out["n_jumps"] >= 1 and out["lambda_jump"] > 0
    assert 400 in out["indices"]


def test_jump_test_is_quiet_on_clean_noise():
    out = lee_mykland(np.random.default_rng(6).normal(0.0, 0.001, 1500))
    assert out["n_jumps"] <= 5


def test_jump_test_is_robust_to_a_volatility_regime_change():
    """Kalıcı vol artışı sıçrama DEĞİLDİR; bipower yerel tahmin bunu ayırt eder.

    Düz bir kayan standart sapma da bunu yapardı; asıl fark, bipower'ın tek bir
    sıçramayı kendi tahminine ememesidir (test_finds_an_injected_jump).
    """
    rng = np.random.default_rng(7)
    calm = rng.normal(0.0, 0.001, 600)
    wild = rng.normal(0.0, 0.006, 600)
    out = lee_mykland(np.concatenate([calm, wild]))
    assert out["n_jumps"] <= 12, f"rejim degisimi sicrama sanildi: {out['n_jumps']}"


def test_jumps_from_prices_matches_returns_path():
    rng = np.random.default_rng(8)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 500)))
    p[250] *= 1.05
    assert jumps_from_prices(p)["n_jumps"] >= 1


def test_jump_test_handles_short_and_degenerate_input():
    assert lee_mykland(np.array([]))["n_jumps"] == 0
    assert jumps_from_prices([1.0, 2.0])["n_jumps"] == 0


# ── determinizm (sözleşmenin belirlilik kuralı) ───────────────────
def _sig(usd: float, side: Side, label: ActorLabel) -> ActorSignal:
    return ActorSignal(tx_hash="0x1", address="0xa", label=label, side=side,
                       dex="UniswapV2", method="swap", est_value_usd=usd,
                       confidence=1.0)


def test_rolling_flow_with_manual_clock_is_deterministic():
    """Duvar saatiyle bu garanti TUTMAZ — sözleşmenin belirlilik kuralı."""
    def run():
        clk = ManualClock(start=1000.0, step=1.0)
        rf = RollingFlow(window_sec=60.0, clock=clk)
        for i in range(10):
            rf.add(_sig(1000.0 * (i + 1), Side.BUY if i % 2 else Side.SELL,
                        ActorLabel.WHALE))
            clk.tick()
        return rf.features("UniswapV2")

    a, b = run(), run()
    assert a.flow_imbalance == b.flow_imbalance
    assert a.whale_net_usd == b.whale_net_usd
    assert a.sample_count == b.sample_count


def test_manual_clock_eviction_is_exact():
    """Pencere dışına düşen kayıt tam olarak zamanla boşalmalı."""
    clk = ManualClock(start=0.0)
    rf = RollingFlow(window_sec=10.0, clock=clk)
    rf.add(_sig(1000.0, Side.BUY, ActorLabel.RETAIL))
    assert rf.features("UniswapV2").sample_count == 1
    clk.set(20.0)
    assert rf.features("UniswapV2").sample_count == 0


def test_flow_feed_passes_the_clock_through():
    clk = ManualClock(start=500.0)
    feed = FlowFeed(mode="simulation", seed=1, clock=clk)
    assert feed.flow._clock is clk


# ── ölü yolların bağlanması ───────────────────────────────────────
def test_map_regime_without_book_keeps_the_old_vpin_rule():
    assert map_regime(0.1) == "normal"
    assert map_regime(0.5) == "toxic"
    assert map_regime(0.8) == "highvol"


def test_map_regime_uses_the_book_when_supplied():
    """Ince/pahalı defter, VPIN düşük olsa bile highvol'e yönlendirir.

    `RegimeRouter` uzun süre bu köprüden hiç çağrılmıyordu (ölü kod).
    """
    thin = BookState(symbol="X", spread_bps=200.0, microprice=100.0,
                     depth_imbalance=0.0, ofi=0.0, queue_imbalance=0.0,
                     book_slope=0.0, kyle_lambda=0.0)
    assert map_regime(0.1) == "normal"
    assert map_regime(0.1, book_state=thin) == "highvol"


def test_map_regime_normal_book_stays_normal():
    calm = BookState(symbol="X", spread_bps=1.0, microprice=100.0,
                     depth_imbalance=0.0, ofi=0.0, queue_imbalance=0.0,
                     book_slope=0.0, kyle_lambda=0.0)
    assert map_regime(0.1, book_state=calm) == "normal"


def test_flow_feed_wires_book_state_into_predict():
    """book_feed verilince D5 defter düzeltmesi gerçekten ateşlenmeli.

    Eskiden `predict(feat)` book_state OLMADAN çağrılıyordu, dolayısıyla
    BOOK_WEIGHT=0.35 yolu hiç çalışmıyordu.
    """
    bf = BookFeed(mode="simulation", seed=3)
    with_book = FlowFeed(mode="simulation", seed=7, book_feed=bf)
    without = FlowFeed(mode="simulation", seed=7)
    a = with_book.latest("UniswapV2")
    b = without.latest("UniswapV2")
    # Aynı seed, aynı sentetik akış; tek fark defterin devrede olması.
    assert a.direction_prob_up != b.direction_prob_up


def test_flow_feed_survives_a_broken_book_feed():
    class Broken:
        def latest(self, _):
            raise RuntimeError("defter yok")

    feed = FlowFeed(mode="simulation", seed=1, book_feed=Broken())
    st = feed.latest("UniswapV2")
    assert 0.0 <= st.direction_prob_up <= 1.0     # akış tarafı çalışmaya devam eder


# ── yeni BookState alanları ───────────────────────────────────────
def test_book_state_exposes_the_previously_hidden_metrics():
    bf = BookFeed(mode="simulation", seed=11)
    st = None
    for _ in range(30):          # geçmiş birikmeli ki noise/jump hesaplansın
        st = bf.latest("BTCUSDT")
    assert hasattr(st, "sweep") and hasattr(st, "liquidity_gaps")
    assert hasattr(st, "spread_z") and hasattr(st, "noise_ratio")
    assert -1.0 <= st.sweep <= 1.0
    assert 0.0 <= st.liquidity_gaps <= 1.0
    assert 0.0 <= st.noise_ratio <= 1.0
    assert st.lambda_jump >= 0.0


def test_book_feed_stays_deterministic_with_the_new_fields():
    def run():
        bf = BookFeed(mode="simulation", seed=21)
        return [bf.latest("BTCUSDT") for _ in range(25)][-1]

    a, b = run(), run()
    assert a.noise_ratio == b.noise_ratio
    assert a.lambda_jump == b.lambda_jump
    assert a.spread_z == b.spread_z


def test_book_state_to_dict_includes_new_fields():
    bf = BookFeed(mode="simulation", seed=5)
    d = bf.latest("BTCUSDT").to_dict()
    for key in ("sweep", "liquidity_gaps", "spread_z", "noise_ratio",
                "lambda_jump", "jump_mean", "jump_std"):
        assert key in d, f"sözleşme alanı eksik: {key}"


@pytest.mark.parametrize("n", [1, 5, 40])
def test_book_feed_never_crashes_on_short_history(n):
    bf = BookFeed(mode="simulation", seed=2)
    for _ in range(n):
        st = bf.latest("ETHUSDT")
    assert st.noise_ratio >= 0.0
