import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PendingTx, DecodedSwap, Side, ActorLabel
from src.actor.classifier import classify
from src.decode.tx_decoder import UNISWAP_V2_ROUTER, UNISWAP_V3_ROUTER

BASE = 20 * 10**9


def _swap(value_wei, gas_wei, side=Side.BUY, dex="UniswapV2"):
    tx = PendingTx(
        tx_hash="0x" + "cd" * 32, from_addr="0x" + "22" * 20,
        to_addr=UNISWAP_V2_ROUTER, value_wei=value_wei,
        gas_price_wei=gas_wei, input_data="0x7ff36ab5",
    )
    return DecodedSwap(tx=tx, dex=dex, method="swapExactETHForTokens", side=side)


def test_whale_large_value():
    # 200 ETH ~ $600k > eşik → WHALE beklenir (gas normal)
    sig = classify(_swap(200 * 10**18, int(BASE * 1.1)), base_fee_wei=BASE)
    assert sig.label == ActorLabel.WHALE
    assert sig.est_value_usd > 100_000


def test_mev_high_gas():
    # küçük değer ama gas 5x → MEV_BOT beklenir
    sig = classify(_swap(1 * 10**18, int(BASE * 5)), base_fee_wei=BASE)
    assert sig.label == ActorLabel.MEV_BOT


def test_retail_small_normal():
    sig = classify(_swap(int(0.2 * 10**18), int(BASE * 1.1)), base_fee_wei=BASE)
    assert sig.label == ActorLabel.RETAIL


def test_confidence_bounds():
    sig = classify(_swap(200 * 10**18, int(BASE * 1.1)), base_fee_wei=BASE)
    assert 0.0 <= sig.confidence <= 1.0


if __name__ == "__main__":
    test_whale_large_value(); test_mev_high_gas()
    test_retail_small_normal(); test_confidence_bounds()
    print("classifier testleri GEÇTİ")
