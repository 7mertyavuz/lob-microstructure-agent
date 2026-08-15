"""CAS entegrasyonu Katman 2 -- MEV karar fonksiyonlari icin tek giris noktasi.

`lob_microstructure/mev/{sandwich,jit_liquidity,arbitrage,builder_tip}.py` icindeki
`detect_*` fonksiyonlari zaten yan etkisiz ve deterministiktir (global
durum okumaz/yazmaz, ayni girdi -> ayni cikti). Bu modul onlari
`cas-market-simulator`nin bir MEV ajaninin "karar fonksiyonu" olarak
dogrudan cagirabilecegi ince, isimlendirilmis bir cephe (facade) halinde
sunar -- hesaplama mantigini degistirmez, sadece simulator-dostu bir
giris noktasi ekler.

Sozlesme: `docs/00-ORTAK-SOZLESME.md`. Her fonksiyon: saf, deterministik,
yan etkisiz -- ayni girdi icin her zaman ayni cikti.
"""
from __future__ import annotations

from lob_microstructure.mev.sandwich import detect_sandwiches, Sandwich
from lob_microstructure.mev.jit_liquidity import detect_jit, JITEvent
from lob_microstructure.mev.arbitrage import detect_atomic_arb, ArbResult, WETH as ARB_BASE_TOKEN
from lob_microstructure.mev.builder_tip import builder_payment, mev_score_from_payment, BuilderPayment


def decide_sandwich(swaps: list[dict]) -> list[Sandwich]:
    """Girdi: onaylanmis blok swap'lari [{address, token, side, tx_index,
    est_value_usd}, ...]. Cikti: tespit edilen sandwich kayitlari (olabilir
    bos liste). Saf -- `swaps` degistirilmez, dis durum yazilmaz."""
    return detect_sandwiches(swaps)


def decide_jit(events: list[dict], max_gap: int = 3) -> list[JITEvent]:
    """Girdi: blok olaylari [{address, pool, kind, tx_index, liquidity_usd?}],
    kind in {MINT, SWAP, BURN}. Cikti: tespit edilen JIT likidite kayitlari.
    Saf -- ayni girdi + max_gap -> ayni cikti."""
    return detect_jit(events, max_gap=max_gap)


def decide_arbitrage(transfers: list[dict], actor: str,
                      base_token: str = ARB_BASE_TOKEN,
                      dust: int = 10**12) -> ArbResult:
    """Girdi: tek islemin token-transfer listesi + aktor adresi. Cikti:
    `ArbResult` (is_arb=True/False + detaylar). Saf -- deterministik."""
    return detect_atomic_arb(transfers, actor, base_token=base_token, dust=dust)


def decide_builder_tip(gas_used: int, gas_price_wei: int, base_fee_wei: int,
                        trace: list[dict] | None = None,
                        fee_recipient: str | None = None) -> tuple[BuilderPayment, float]:
    """Girdi: gas/trace bilgisi. Cikti: (BuilderPayment, mev_score 0..1).
    Saf -- ayni girdi -> ayni cikti; MEV ajaninin "builder odemesi yapayim
    mi" kararinda dogrudan kullanilabilir skor uretir."""
    payment = builder_payment(gas_used, gas_price_wei, base_fee_wei,
                               trace=trace, fee_recipient=fee_recipient)
    score = mev_score_from_payment(payment, base_fee_wei)
    return payment, score
