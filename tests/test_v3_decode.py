import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eth_abi import encode
from src.models import PendingTx, Side
from src.decode.tx_decoder import decode_tx, UNISWAP_V3_ROUTER, WETH

TOKEN = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"  # USDC


def _tx(input_data, value=0):
    return PendingTx(
        tx_hash="0x" + "ab" * 32, from_addr="0x" + "11" * 20,
        to_addr=UNISWAP_V3_ROUTER, value_wei=value,
        gas_price_wei=20 * 10**9, input_data=input_data,
    )


def _single(token_in, token_out, amount_in):
    params = (token_in, token_out, 3000, "0x" + "11" * 20, 0,
              amount_in, int(amount_in * 0.99), 0)
    body = encode(
        ["(address,address,uint24,address,uint256,uint256,uint256,uint160)"],
        [params])
    return "0x414bf389" + body.hex()


def test_v3_buy_weth_in():
    swap = decode_tx(_tx(_single(WETH, TOKEN, 3 * 10**18), value=3 * 10**18))
    assert swap is not None and swap.dex == "UniswapV3"
    assert swap.method == "exactInputSingle"
    assert swap.side == Side.BUY
    assert swap.token_in == WETH and swap.token_out == TOKEN
    assert swap.amount_in_wei == 3 * 10**18


def test_v3_sell_weth_out():
    swap = decode_tx(_tx(_single(TOKEN, WETH, 5000 * 10**6)))
    assert swap is not None
    assert swap.side == Side.SELL
    # SELL'de WETH tarafı amountOutMinimum ile temsil edilir (>0)
    assert swap.amount_in_wei is not None and swap.amount_in_wei > 0


def test_v3_path_exact_input():
    # packed path: WETH(20) + fee(3) + TOKEN(20)
    path = bytes.fromhex(WETH[2:]) + (3000).to_bytes(3, "big") + bytes.fromhex(TOKEN[2:])
    body = encode(["(bytes,address,uint256,uint256,uint256)"],
                  [(path, b"\x11" * 20, 0, 2 * 10**18, 10**18)])
    swap = decode_tx(_tx("0xc04b8d59" + body.hex(), value=2 * 10**18))
    assert swap is not None and swap.method == "exactInput"
    assert swap.side == Side.BUY
    assert swap.token_in == WETH and swap.token_out == TOKEN


if __name__ == "__main__":
    test_v3_buy_weth_in(); test_v3_sell_weth_out(); test_v3_path_exact_input()
    print("V3 decode testleri GEÇTİ")
