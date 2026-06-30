"""Katman 3 yardımcısı — cüzdan zenginleştirme.

Bir adres için yaş, işlem sayısı, bakiye ve ömür boyu hacim döndürür.
Gerçek dağıtımda kaynaklar:
  * on-chain: ilk tx zamanı (etherscan/erigon), nonce (tx_count), balance.
  * etiket API'leri: Nansen / Arkham / Etherscan labels.
LRU cache ile aynı adrese tekrar tekrar RPC atılması engellenir.

Bu PoC sürümü deterministik bir sahte profil üretir (adres hash'inden),
böylece node/anahtar olmadan sınıflandırıcı test edilebilir.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

from src.models import WalletProfile

# ETH/USD — gerçekte bir oracle/price feed'den gelir.
ETH_USD = 3000.0


@lru_cache(maxsize=10_000)
def get_profile(address: str) -> WalletProfile:
    """Adres profili (cache'li). PoC: adres hash'inden deterministik üretim."""
    h = int(hashlib.sha256(address.lower().encode()).hexdigest(), 16)
    age_days = (h % 1500)                 # 0..1500 gün
    tx_count = (h >> 8) % 50_000
    balance_eth = ((h >> 16) % 100_000) / 100.0    # 0..1000 ETH
    lifetime_volume_usd = balance_eth * ETH_USD * (1 + (h % 7))
    return WalletProfile(
        address=address.lower(),
        age_days=float(age_days),
        tx_count=int(tx_count),
        balance_eth=round(balance_eth, 3),
        lifetime_volume_usd=round(lifetime_volume_usd, 2),
    )
