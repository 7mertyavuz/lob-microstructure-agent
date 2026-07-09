"""Katman 3 — Actor Layer (Kimlik).

Çözülmüş swap + cüzdan profili + gas davranışını birleştirip aktöre
WHALE / MEV_BOT / RETAIL etiketi ve bir güven skoru yapıştırır.

Kural tabanlı bir skorlayıcı (rule-based) olarak tasarlandı; çünkü PoC
aşamasında şeffaf ve denetlenebilir olması, kara-kutu modelden önemli.
Üretimde bu skorlar bir gradient-boosting / nöral modele feature olarak
verilebilir — arayüz (sınıf çıktısı) aynı kalır.

Heuristikler:
  MEV_BOT  → gas, blok baz ücretinin MEV_GAS_MULTIPLIER katından fazla;
             VEYA çok yüksek tx_count + düşük bakiye (relayer deseni).
  WHALE    → işlem USD değeri whale eşiğini aşıyor; VEYA cüzdan ömür hacmi
             ve bakiyesi çok yüksek.
  RETAIL   → diğer her şey.
"""
from __future__ import annotations

from src.config import CONFIG
from src.models import DecodedSwap, ActorSignal, ActorLabel, Side
from src.actor.wallet_profiler import get_profile, ETH_USD


def classify(swap: DecodedSwap, base_fee_wei: int = 20 * 10**9,
             coinbase_wei: float = 0.0) -> ActorSignal:
    """coinbase_wei: onaylanmış blok trace'inden gelen doğrudan builder ödemesi
    (coinbase.transfer). >0 ise neredeyse kesin MEV/searcher imzasıdır."""
    tx = swap.tx
    profile = get_profile(tx.from_addr)
    reasons: list[str] = []

    # İşlem USD değeri (ETH girişi varsa value, yoksa amount_in tahmini)
    eth_in = (swap.amount_in_wei or tx.value_wei) / 10**18
    est_usd = eth_in * ETH_USD

    gas_ratio = tx.gas_price_wei / base_fee_wei if base_fee_wei else 1.0

    # --- MEV skoru ---
    mev_score = 0.0
    if coinbase_wei and coinbase_wei > 0:
        mev_score += 0.8
        reasons.append(f"coinbase.transfer → builder ödemesi {coinbase_wei/1e18:.4f} ETH (kesin searcher imzası)")
    if gas_ratio >= CONFIG.mev_gas_multiplier:
        mev_score += 0.6
        reasons.append(f"gas baz ücretin {gas_ratio:.1f}x üzerinde (front-run şüphesi)")
    if profile.tx_count > 10_000 and profile.balance_eth < 5:
        mev_score += 0.3
        reasons.append("yüksek tx sayısı + düşük bakiye (bot/relayer deseni)")
    if profile.age_days < CONFIG.new_wallet_max_age_days and gas_ratio >= 1.5:
        mev_score += 0.2
        reasons.append("yeni cüzdan + agresif gas")

    # --- Whale skoru ---
    whale_score = 0.0
    if est_usd >= CONFIG.whale_usd_threshold:
        whale_score += 0.6
        reasons.append(f"işlem büyüklüğü ~${est_usd:,.0f} (eşik ${CONFIG.whale_usd_threshold:,.0f})")
    if profile.lifetime_volume_usd >= 5 * CONFIG.whale_usd_threshold:
        whale_score += 0.25
        reasons.append(f"ömür boyu hacim ~${profile.lifetime_volume_usd:,.0f}")
    if profile.balance_eth >= 200:
        whale_score += 0.15
        reasons.append(f"yüksek bakiye {profile.balance_eth:.0f} ETH")

    # --- Karar ---
    if mev_score >= whale_score and mev_score >= 0.5:
        label, confidence = ActorLabel.MEV_BOT, min(mev_score, 0.99)
    elif whale_score >= 0.5:
        label, confidence = ActorLabel.WHALE, min(whale_score, 0.99)
    else:
        label = ActorLabel.RETAIL
        confidence = round(1 - max(mev_score, whale_score), 2)
        if not reasons:
            reasons.append("eşik altı hacim ve normal gas")

    return ActorSignal(
        tx_hash=tx.tx_hash,
        address=tx.from_addr,
        label=label,
        side=swap.side if swap.side != Side.UNKNOWN else Side.UNKNOWN,
        dex=swap.dex,
        method=swap.method,
        est_value_usd=round(est_usd, 2),
        confidence=round(confidence, 2),
        reasons=reasons,
    )
