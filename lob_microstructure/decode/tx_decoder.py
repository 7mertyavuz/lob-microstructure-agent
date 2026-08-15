"""Katman 2 — Reconstruction & Decoding.

Ham hex calldata'nın ilk 4 byte'ı (function selector) ile DEX router
çağrısını tanır, parametreleri eth-abi ile çözer.

Kapsam:
  * Uniswap V2: swapExact{ETHForTokens, TokensForETH, TokensForTokens}
  * Uniswap V3 SwapRouter (deadline'lı) ve SwapRouter02 (deadline'sız):
    exactInputSingle (tuple) ve exactInput (packed path).

Yön (BUY/SELL) WETH'e göre belirlenir: WETH girişi → BUY (token alımı),
WETH çıkışı → SELL. USD tahmini için her zaman WETH tarafındaki miktar
kullanılır (amountIn ya da amountOutMinimum).
"""
from __future__ import annotations

import logging
from typing import Optional

from lob_microstructure.models import PendingTx, DecodedSwap, Side

log = logging.getLogger("decode")

# Mainnet adresleri (lowercase)
UNISWAP_V2_ROUTER = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"
UNISWAP_V3_ROUTER = "0xe592427a0aece92de3edee1f18e0157c05861564"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

try:
    from eth_abi import decode as abi_decode  # type: ignore
except Exception:  # pragma: no cover
    abi_decode = None

# ---- V2: selector -> (dex, method, side, arg_types, path_arg_index) ----
V2_SELECTORS: dict[str, dict] = {
    "0x7ff36ab5": {  # swapExactETHForTokens(uint,address[],address,uint)
        "method": "swapExactETHForTokens", "side": Side.BUY,
        "types": ["uint256", "address[]", "address", "uint256"], "path_idx": 1,
    },
    "0x18cbafe5": {  # swapExactTokensForETH(uint,uint,address[],address,uint)
        "method": "swapExactTokensForETH", "side": Side.SELL,
        "types": ["uint256", "uint256", "address[]", "address", "uint256"], "path_idx": 2,
    },
    "0x38ed1739": {  # swapExactTokensForTokens(uint,uint,address[],address,uint)
        "method": "swapExactTokensForTokens", "side": Side.UNKNOWN,
        "types": ["uint256", "uint256", "address[]", "address", "uint256"], "path_idx": 2,
    },
}

# ---- V3 exactInputSingle (tuple) ----
# SwapRouter:   (tokenIn,tokenOut,fee,recipient,deadline,amountIn,amountOutMin,sqrtLimit)
# SwapRouter02: (tokenIn,tokenOut,fee,recipient,amountIn,amountOutMin,sqrtLimit)
V3_SINGLE: dict[str, dict] = {
    "0x414bf389": {
        "method": "exactInputSingle",
        "tuple": "(address,address,uint24,address,uint256,uint256,uint256,uint160)",
        "amount_in_idx": 5, "amount_out_idx": 6,
    },
    "0x04e45aaf": {
        "method": "exactInputSingle",
        "tuple": "(address,address,uint24,address,uint256,uint256,uint160)",
        "amount_in_idx": 4, "amount_out_idx": 5,
    },
}

# ---- V3 exactInput (packed path) ----
# SwapRouter:   (path,recipient,deadline,amountIn,amountOutMin)
# SwapRouter02: (path,recipient,amountIn,amountOutMin)
V3_PATH: dict[str, dict] = {
    "0xc04b8d59": {
        "method": "exactInput",
        "tuple": "(bytes,address,uint256,uint256,uint256)",
        "amount_in_idx": 3, "amount_out_idx": 4,
    },
    "0xb858183f": {
        "method": "exactInput",
        "tuple": "(bytes,address,uint256,uint256)",
        "amount_in_idx": 2, "amount_out_idx": 3,
    },
}

# ---- Universal Router (V2/V3/V4 birleşik, komut-tabanlı) ----
# execute(bytes commands, bytes[] inputs, uint256 deadline) | execute(bytes,bytes[])
UR_EXECUTE = {"0x3593564c", "0x24856bc3"}
# Komut kodları (yüksek bit 0x80 = allow_revert bayrağı; 0x3f ile maskelenir)
CMD_V3_SWAP_EXACT_IN = 0x00
CMD_V3_SWAP_EXACT_OUT = 0x01
CMD_V2_SWAP_EXACT_IN = 0x08
CMD_V2_SWAP_EXACT_OUT = 0x09
CMD_V4_SWAP = 0x10


def decode_tx(tx: PendingTx) -> Optional[DecodedSwap]:
    """Bir router swap'i ise DecodedSwap döndürür, değilse None."""
    data = (tx.input_data or "").lower()
    if not data.startswith("0x") or len(data) < 10:
        return None
    selector = data[:10]

    if selector in V2_SELECTORS:
        return _decode_v2(tx, selector, data)
    if selector in V3_SINGLE:
        return _decode_v3_single(tx, selector, data)
    if selector in V3_PATH:
        return _decode_v3_path(tx, selector, data)
    if selector in UR_EXECUTE:
        return _decode_universal(tx, selector, data)
    return None


# ---------------- Universal Router ----------------
def _decode_universal(tx, selector, data) -> Optional[DecodedSwap]:
    """execute()'i çöz, ilk swap komutunu bul ve DecodedSwap üret."""
    if abi_decode is None:
        return None
    try:
        body = bytes.fromhex(data[10:])
        if selector == "0x3593564c":  # commands, inputs, deadline
            commands, inputs, _ = abi_decode(["bytes", "bytes[]", "uint256"], body)
        else:                          # commands, inputs
            commands, inputs = abi_decode(["bytes", "bytes[]"], body)
    except Exception as e:
        log.debug("UR execute decode başarısız: %s", e)
        return None

    for i, raw_cmd in enumerate(commands):
        cmd = raw_cmd & 0x3f
        if i >= len(inputs):
            break
        if cmd in (CMD_V2_SWAP_EXACT_IN, CMD_V2_SWAP_EXACT_OUT):
            return _ur_v2(tx, cmd, inputs[i])
        if cmd in (CMD_V3_SWAP_EXACT_IN, CMD_V3_SWAP_EXACT_OUT):
            return _ur_v3(tx, cmd, inputs[i])
        if cmd == CMD_V4_SWAP:
            # V4 aksiyonları daha karmaşık; şimdilik komutu tanı, yönü best-effort bırak
            return DecodedSwap(tx=tx, dex="UniswapV4", method="V4_SWAP",
                               side=Side.UNKNOWN, amount_in_wei=tx.value_wei or None)
    return None


def _ur_v2(tx, cmd, inp: bytes) -> DecodedSwap:
    # (recipient, amountIn/Out, amountMin/Max, address[] path, payerIsUser)
    method = "V2_SWAP_EXACT_IN" if cmd == CMD_V2_SWAP_EXACT_IN else "V2_SWAP_EXACT_OUT"
    token_in = token_out = None
    weth_amount = tx.value_wei or None
    try:
        _, amt, _, path, _ = abi_decode(
            ["address", "uint256", "uint256", "address[]", "bool"], inp)
        if path:
            token_in, token_out = _addr(path[0]), _addr(path[-1])
        weth_amount = _ur_weth_amount(token_in, token_out, int(amt), tx.value_wei)
    except Exception as e:
        log.debug("UR V2 input decode başarısız: %s", e)
    return DecodedSwap(tx=tx, dex="UniswapV2(UR)", method=method,
                       side=_side_from_weth(token_in, token_out),
                       token_in=token_in, token_out=token_out, amount_in_wei=weth_amount)


def _ur_v3(tx, cmd, inp: bytes) -> DecodedSwap:
    # (recipient, amountIn/Out, amountMin/Max, bytes path, payerIsUser)
    exact_in = cmd == CMD_V3_SWAP_EXACT_IN
    method = "V3_SWAP_EXACT_IN" if exact_in else "V3_SWAP_EXACT_OUT"
    token_in = token_out = None
    weth_amount = tx.value_wei or None
    try:
        _, amt, _, path, _ = abi_decode(
            ["address", "uint256", "uint256", "bytes", "bool"], inp)
        a, b = _parse_path(path)
        # exact_out yolu ters kodlanır: ilk eleman tokenOut'tur
        token_in, token_out = (a, b) if exact_in else (b, a)
        weth_amount = _ur_weth_amount(token_in, token_out, int(amt), tx.value_wei)
    except Exception as e:
        log.debug("UR V3 input decode başarısız: %s", e)
    return DecodedSwap(tx=tx, dex="UniswapV3(UR)", method=method,
                       side=_side_from_weth(token_in, token_out),
                       token_in=token_in, token_out=token_out, amount_in_wei=weth_amount)


def _ur_weth_amount(token_in, token_out, amt, value_wei):
    if token_in == WETH or token_out == WETH:
        return amt or value_wei
    return value_wei or None


# ---------------- V2 ----------------
def _decode_v2(tx, selector, data) -> DecodedSwap:
    spec = V2_SELECTORS[selector]
    side = spec["side"]
    token_in = token_out = None
    amount_in = tx.value_wei if side == Side.BUY else None

    if abi_decode is not None:
        try:
            decoded = abi_decode(spec["types"], bytes.fromhex(data[10:]))
            path = decoded[spec["path_idx"]]
            if path:
                token_in = _addr(path[0])
                token_out = _addr(path[-1])
            if side != Side.BUY:  # token girişi: amountIn ilk argümanda
                amount_in = int(decoded[0])
        except Exception as e:
            log.debug("V2 decode başarısız (%s): %s", spec["method"], e)

    return DecodedSwap(tx=tx, dex="UniswapV2", method=spec["method"], side=side,
                       token_in=token_in, token_out=token_out, amount_in_wei=amount_in)


# ---------------- V3 exactInputSingle ----------------
def _decode_v3_single(tx, selector, data) -> DecodedSwap:
    spec = V3_SINGLE[selector]
    token_in = token_out = None
    weth_amount = tx.value_wei or None
    if abi_decode is not None:
        try:
            (t,) = abi_decode([spec["tuple"]], bytes.fromhex(data[10:]))
            token_in, token_out = _addr(t[0]), _addr(t[1])
            amount_in = int(t[spec["amount_in_idx"]])
            amount_out = int(t[spec["amount_out_idx"]])
            weth_amount = _weth_amount(token_in, token_out, amount_in, amount_out, tx.value_wei)
        except Exception as e:
            log.debug("V3 single decode başarısız: %s", e)
    side = _side_from_weth(token_in, token_out)
    return DecodedSwap(tx=tx, dex="UniswapV3", method=spec["method"], side=side,
                       token_in=token_in, token_out=token_out, amount_in_wei=weth_amount)


# ---------------- V3 exactInput (packed path) ----------------
def _decode_v3_path(tx, selector, data) -> DecodedSwap:
    spec = V3_PATH[selector]
    token_in = token_out = None
    weth_amount = tx.value_wei or None
    if abi_decode is not None:
        try:
            t = abi_decode([spec["tuple"]], bytes.fromhex(data[10:]))[0]
            path: bytes = t[0]
            token_in, token_out = _parse_path(path)
            amount_in = int(t[spec["amount_in_idx"]])
            amount_out = int(t[spec["amount_out_idx"]])
            weth_amount = _weth_amount(token_in, token_out, amount_in, amount_out, tx.value_wei)
        except Exception as e:
            log.debug("V3 path decode başarısız: %s", e)
    side = _side_from_weth(token_in, token_out)
    return DecodedSwap(tx=tx, dex="UniswapV3", method=spec["method"], side=side,
                       token_in=token_in, token_out=token_out, amount_in_wei=weth_amount)


# ---------------- yardımcılar ----------------
def _parse_path(path: bytes) -> tuple[Optional[str], Optional[str]]:
    """Packed path: token(20) + fee(3) + token(20) [+ fee(3)+token(20)...]."""
    if not path or len(path) < 20:
        return None, None
    token_in = "0x" + path[:20].hex()
    token_out = "0x" + path[-20:].hex()
    return token_in, token_out


def _side_from_weth(token_in, token_out) -> Side:
    if token_in == WETH:
        return Side.BUY
    if token_out == WETH:
        return Side.SELL
    return Side.UNKNOWN


def _weth_amount(token_in, token_out, amount_in, amount_out, value_wei):
    """USD tahmini için WETH tarafındaki miktarı döndür."""
    if token_in == WETH:
        return amount_in or value_wei
    if token_out == WETH:
        return amount_out
    return value_wei or None


def _addr(x) -> str:
    if isinstance(x, bytes):
        return "0x" + x.hex()
    return str(x).lower()
