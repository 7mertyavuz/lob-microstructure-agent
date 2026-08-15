import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_microstructure.models import PendingTx, Side
from lob_microstructure.decode.tx_decoder import decode_tx, UNISWAP_V2_ROUTER


def _tx(input_data, value=10**18):
    return PendingTx(
        tx_hash="0x" + "ab" * 32, from_addr="0x" + "11" * 20,
        to_addr=UNISWAP_V2_ROUTER, value_wei=value,
        gas_price_wei=20 * 10**9, input_data=input_data,
    )


def test_v2_buy_selector():
    swap = decode_tx(_tx("0x7ff36ab5" + "0" * 64))
    assert swap is not None
    assert swap.dex == "UniswapV2"
    assert swap.method == "swapExactETHForTokens"
    assert swap.side == Side.BUY


def test_v2_sell_selector():
    swap = decode_tx(_tx("0x18cbafe5" + "0" * 64))
    assert swap is not None
    assert swap.side == Side.SELL


def test_non_swap_returns_none():
    assert decode_tx(_tx("0xdeadbeef" + "0" * 64)) is None


def test_garbage_input():
    assert decode_tx(_tx("0x")) is None


if __name__ == "__main__":
    test_v2_buy_selector(); test_v2_sell_selector()
    test_non_swap_returns_none(); test_garbage_input()
    print("decoder testleri GEÇTİ")
