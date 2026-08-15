"""CAS entegrasyonu Katman 2 -- ajan davranis sablonu (veri, kod degil).

`cas-market-simulator`'in sentetik aktorleri (WHALE/MEV_BOT/RETAIL) icin
parametrik profil verir. Bu modul davranis calistirmaz; sadece
simulatorun kendi ajan mantiginda kullanacagi sabit veri sozlesmesidir.

Parametreler `lob_microstructure/actor/classifier.py`'deki mevcut esiklerle tutarlidir:
  * WHALE   -- `CONFIG.whale_usd_threshold` (varsayilan $100k) ve uzeri
    islem buyuklugu / omur boyu hacim taraniyor; profildeki `size_usd`
    araligi bu esigin altindan (partial/testing) ustune (whale confirmed)
    kadar makul bir bant kapsar.
  * MEV_BOT -- `CONFIG.mev_gas_multiplier` (varsayilan 2.0x) ve `coinbase.
    transfer` sinyaliyle tutarli: yuksek agresiflik (gas/oncelik ucreti).
  * RETAIL  -- esik altinda kalan sirada islem hacimleri.

Sozlesme: `docs/00-ORTAK-SOZLESME.md`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    """Bir aktor tipinin tipik davranis parametreleri (salt veri).

    size_usd:   tipik emir buyuklugu araligi (usd, min, max)
    freq:       islem sikligi -- "low" | "mid" | "high"
    aggression: 0..1, gas/oncelik ucreti agresifligi (1.0 = en agresif)
    trigger:    tetikleyici kosul ifadesi (serbest metin, simulator yorumlar)
    """
    size_usd: tuple[float, float]
    freq: str
    aggression: float
    trigger: str


WHALE = AgentProfile((50_000, 5_000_000), "low", 0.3, "onchain_flow")
MEV_BOT = AgentProfile((1_000, 200_000), "high", 1.0, "victim_in_mempool")
RETAIL = AgentProfile((50, 5_000), "mid", 0.5, "sentiment|momentum")

PROFILES: dict[str, AgentProfile] = {
    "WHALE": WHALE,
    "MEV_BOT": MEV_BOT,
    "RETAIL": RETAIL,
}
