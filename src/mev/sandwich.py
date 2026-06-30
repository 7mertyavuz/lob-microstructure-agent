"""Blok-içi pair-based sandwich tespiti (graf/clustering yaklaşımının
pratik orta adımı).

Tek-tx gas sezgisi zayıftır; literatür ön/arka bacakları AYNI aktöre
bağlamayı önerir. Burada onaylanmış bir bloktaki swap'lar tx_index sırasına
göre incelenir: aynı adres aynı token'da önce BUY sonra SELL yapıyor ve
arada başka adres(ler)in aynı token işlemi varsa → sandwich şüphesi.

Girdi: her biri {address, token, side, tx_index, est_value_usd} olan swap
listesi (decode edilmiş blok). Çıktı: tespit edilen sandwich kayıtları.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Sandwich:
    attacker: str
    token: str
    front_idx: int
    back_idx: int
    victim_indices: list[int]
    victim_volume_usd: float


def detect_sandwiches(swaps: list[dict]) -> list[Sandwich]:
    """swaps: confirmed blok swap'ları. tx_index'e göre sıralanmış kabul edilmez,
    fonksiyon kendi sıralar."""
    out: list[Sandwich] = []
    # (attacker, token) -> kronolojik [(idx, side, usd)]
    by_actor: dict[tuple, list] = defaultdict(list)
    all_by_token: dict[str, list] = defaultdict(list)

    for s in swaps:
        addr = (s.get("address") or "").lower()
        tok = (s.get("token") or "").lower()
        side = s.get("side")
        idx = s.get("tx_index")
        usd = float(s.get("est_value_usd") or 0)
        if not addr or not tok or idx is None or side not in ("BUY", "SELL"):
            continue
        by_actor[(addr, tok)].append((int(idx), side, usd))
        all_by_token[tok].append((int(idx), addr, side, usd))

    for (addr, tok), legs in by_actor.items():
        legs.sort()
        # BUY(front) ... SELL(back) deseni ara
        for i in range(len(legs)):
            f_idx, f_side, _ = legs[i]
            if f_side != "BUY":
                continue
            for j in range(i + 1, len(legs)):
                b_idx, b_side, _ = legs[j]
                if b_side != "SELL":
                    continue
                # arada başka adresin aynı token işlemi (kurban) var mı?
                victims = [
                    (vi, vu) for (vi, va, vs, vu) in all_by_token[tok]
                    if f_idx < vi < b_idx and va != addr
                ]
                if victims:
                    out.append(Sandwich(
                        attacker=addr, token=tok, front_idx=f_idx, back_idx=b_idx,
                        victim_indices=[v[0] for v in victims],
                        victim_volume_usd=round(sum(v[1] for v in victims), 2),
                    ))
                break  # bu front için ilk uygun back yeterli
    return out
