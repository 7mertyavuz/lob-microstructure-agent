"""JIT (Just-In-Time) likidite tespiti — Uniswap V3/V4 konsantre likidite.

MEV botu, büyük bir swap gelmeden hemen önce dar bir tick aralığına devasa
likidite ekler (Mint), swap olur, hemen sonra likiditeyi çeker (Burn) — böylece
işlem ücretinin büyük kısmını sömürür ve mevcut LP'leri seyreltir (~%85).

Tespit deseni (literatür): aynı havuzda ardışık 3 işlem —
  T1: Mint (saldırgan)
  T2: Swap (kurban, farklı adres)
  T3: Burn (aynı saldırgan)
Bu, anlık tick derinliği değişimini "toksik akış / balina geliyor" sinyaline
çevirir.

Girdi: blok olayları [{address, pool, kind, tx_index, liquidity_usd?}],
kind ∈ {MINT, SWAP, BURN}.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class JITEvent:
    provider: str
    pool: str
    mint_idx: int
    swap_idx: int
    burn_idx: int
    victim: str
    liquidity_usd: float


def detect_jit(events: list[dict], max_gap: int = 3) -> list[JITEvent]:
    """max_gap: Mint ile Burn arası izin verilen tx_index mesafesi (ardışıklık)."""
    by_pool: dict[str, list] = defaultdict(list)
    for e in events:
        pool = (e.get("pool") or "").lower()
        if pool:
            by_pool[pool].append(e)

    out: list[JITEvent] = []
    for pool, evs in by_pool.items():
        evs = sorted(evs, key=lambda x: x.get("tx_index", 0))
        for i, m in enumerate(evs):
            if (m.get("kind") or "").upper() != "MINT":
                continue
            prov = (m.get("address") or "").lower()
            m_idx = m.get("tx_index", 0)
            # aynı provider'ın yakın Burn'ünü ara
            for b in evs[i + 1:]:
                if b.get("tx_index", 0) - m_idx > max_gap:
                    break
                if (b.get("kind") or "").upper() != "BURN":
                    continue
                if (b.get("address") or "").lower() != prov:
                    continue
                b_idx = b.get("tx_index", 0)
                # arada başka adresin SWAP'i (kurban) var mı?
                victim = None
                for s in evs:
                    if (s.get("kind") or "").upper() != "SWAP":
                        continue
                    si = s.get("tx_index", 0)
                    if m_idx < si < b_idx and (s.get("address") or "").lower() != prov:
                        victim = (s.get("address") or "").lower()
                        break
                if victim:
                    out.append(JITEvent(
                        provider=prov, pool=pool, mint_idx=m_idx,
                        swap_idx=si, burn_idx=b_idx, victim=victim,
                        liquidity_usd=float(m.get("liquidity_usd") or 0),
                    ))
                break
    return out
