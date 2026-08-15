import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_microstructure.models import ActorSignal, ActorLabel, Side
from lob_microstructure.features.window import RollingFlow
from lob_microstructure.predict.direction import predict


def _sig(label, side, usd):
    return ActorSignal(
        tx_hash="0x" + "ee" * 32, address="0x" + "33" * 20, label=label,
        side=side, dex="UniswapV2", method="swapExactETHForTokens",
        est_value_usd=usd, confidence=0.9,
    )


def test_whale_buys_push_up():
    flow = RollingFlow(window_sec=60)
    for _ in range(5):
        flow.add(_sig(ActorLabel.WHALE, Side.BUY, 300_000))
    pred = predict(flow.features("UniswapV2"))
    assert pred.direction == "YUKARI"
    assert pred.prob_up > 0.58
    assert pred.whale_net_usd > 0


def test_whale_sells_push_down():
    flow = RollingFlow(window_sec=60)
    for _ in range(5):
        flow.add(_sig(ActorLabel.WHALE, Side.SELL, 300_000))
    pred = predict(flow.features("UniswapV2"))
    assert pred.direction == "AŞAĞI"
    assert pred.prob_up < 0.42


def test_balanced_is_neutralish():
    flow = RollingFlow(window_sec=60)
    flow.add(_sig(ActorLabel.WHALE, Side.BUY, 200_000))
    flow.add(_sig(ActorLabel.WHALE, Side.SELL, 200_000))
    pred = predict(flow.features("UniswapV2"))
    assert 0.40 <= pred.prob_up <= 0.60


def test_prob_bounds():
    flow = RollingFlow(window_sec=60)
    for _ in range(20):
        flow.add(_sig(ActorLabel.WHALE, Side.BUY, 5_000_000))
    pred = predict(flow.features("UniswapV2"))
    assert 0.0 <= pred.prob_up <= 1.0


if __name__ == "__main__":
    test_whale_buys_push_up(); test_whale_sells_push_down()
    test_balanced_is_neutralish(); test_prob_bounds()
    print("predict testleri GEÇTİ")
