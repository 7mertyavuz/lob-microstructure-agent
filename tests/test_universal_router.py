import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eth_abi import encode
from src.models import PendingTx, Side
from src.decode.tx_decoder import decode_tx, WETH

TOKEN = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"


def _ur_tx(commands: bytes, inputs: list[bytes], value=0, with_deadline=True):
    if with_deadline:
        sel = "0x3593564c"
        body = encode(["bytes", "bytes[]", "uint256"], [commands, inputs, 0])
    else:
        sel = "0x24856bc3"
        body = encode(["bytes", "bytes[]"], [commands, inputs])
    return PendingTx("0x" + "ab" * 32, "0x" + "11" * 20, "0x" + "22" * 20,
                     value, 20 * 10**9, sel + body.hex())


def test_ur_v2_swap_exact_in_buy():
    # command 0x08 = V2_SWAP_EXACT_IN, path = [WETH, TOKEN] → BUY
    inp = encode(["address", "uint256", "uint256", "address[]", "bool"],
                 ["0x" + "11" * 20, 3 * 10**18, 0, [WETH, TOKEN], True])
    swap = decode_tx(_ur_tx(b"\x08", [inp], value=3 * 10**18))
    assert swap is not None and swap.method == "V2_SWAP_EXACT_IN"
    assert swap.side == Side.BUY
    assert swap.token_in == WETH and swap.token_out == TOKEN


def test_ur_v3_swap_exact_in_sell():
    # command 0x00 = V3_SWAP_EXACT_IN, packed path TOKEN→WETH → SELL
    path = bytes.fromhex(TOKEN[2:]) + (3000).to_bytes(3, "big") + bytes.fromhex(WETH[2:])
    inp = encode(["address", "uint256", "uint256", "bytes", "bool"],
                 ["0x" + "11" * 20, 5000 * 10**6, 0, path, True])
    swap = decode_tx(_ur_tx(b"\x00", [inp]))
    assert swap is not None and swap.method == "V3_SWAP_EXACT_IN"
    assert swap.side == Side.SELL


def test_ur_v4_recognized():
    swap = decode_tx(_ur_tx(b"\x10", [b"\x00"]))
    assert swap is not None and swap.dex == "UniswapV4" and swap.method == "V4_SWAP"


def test_ur_allow_revert_flag_masked():
    # 0x88 = 0x08 | 0x80 (allow_revert) → yine V2_SWAP_EXACT_IN olarak çözülmeli
    inp = encode(["address", "uint256", "uint256", "address[]", "bool"],
                 ["0x" + "11" * 20, 10**18, 0, [WETH, TOKEN], True])
    swap = decode_tx(_ur_tx(b"\x88", [inp], value=10**18))
    assert swap is not None and swap.method == "V2_SWAP_EXACT_IN"


if __name__ == "__main__":
    test_ur_v2_swap_exact_in_buy(); test_ur_v3_swap_exact_in_sell()
    test_ur_v4_recognized(); test_ur_allow_revert_flag_masked()
    print("Universal Router testleri GEÇTİ")
