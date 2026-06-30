"""Atomik (döngüsel) arbitraj tespiti — tek işlemdeki token-transfer grafı.

Sandwich ve JIT'i tamamlayan üçüncü büyük MEV türü: bir searcher tek tx içinde
birden çok havuzdan geçerek (A→B→C→A) başladığı token'a kârla döner. Hiç dış
sermaye bırakmaz; net pozisyon yalnızca base token'da (genelde WETH) pozitiftir,
diğer tüm tokenlarda ~0 (kapalı döngü).

Tespit: işlemin token-transfer'larından aktörün net bakiyesini çıkar.
  * base token net > 0  ve
  * diğer tokenların net'i ~0  →  atomik arbitraj.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"


@dataclass
class ArbResult:
    actor: str
    base_token: str
    profit_wei: int
    tokens_touched: int
    is_arb: bool


def net_positions(transfers: list[dict], actor: str) -> dict[str, int]:
    """Aktörün token bazında net değişimi (giren − çıkan)."""
    a = actor.lower()
    net: dict[str, int] = defaultdict(int)
    for t in transfers:
        tok = (t.get("token") or "").lower()
        amt = int(t.get("amount", 0) or 0)
        frm = (t.get("from") or "").lower()
        to = (t.get("to") or "").lower()
        if to == a:
            net[tok] += amt
        if frm == a:
            net[tok] -= amt
    return dict(net)


def detect_atomic_arb(transfers: list[dict], actor: str,
                      base_token: str = WETH, dust: int = 10**12) -> ArbResult:
    """dust: 'sıfır kabul edilen' tolerans (rounding). base profit > dust ve
    diğer tokenlar |net| <= dust ise arbitraj."""
    base = base_token.lower()
    net = net_positions(transfers, actor)
    base_profit = net.get(base, 0)
    others_closed = all(abs(v) <= dust for k, v in net.items() if k != base)
    is_arb = base_profit > dust and others_closed
    return ArbResult(
        actor=actor.lower(), base_token=base, profit_wei=int(base_profit),
        tokens_touched=len([k for k, v in net.items() if abs(v) > dust or k == base]),
        is_arb=is_arb,
    )
