"""Merkezi konfigürasyon. .env dosyasından okur, makul varsayılanlara düşer."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv yoksa sorun değil, os.environ kullanılır
    pass


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Config:
    wss_url: str = os.getenv("WSS_URL", "").strip()
    bus_backend: str = os.getenv("BUS_BACKEND", "memory").strip().lower()
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    kafka_topic: str = os.getenv("KAFKA_TOPIC", "dex.signals")

    whale_usd_threshold: float = _f("WHALE_USD_THRESHOLD", 100_000)
    new_wallet_max_age_days: float = _f("NEW_WALLET_MAX_AGE_DAYS", 7)
    mev_gas_multiplier: float = _f("MEV_GAS_MULTIPLIER", 2.0)

    @property
    def simulation_mode(self) -> bool:
        """WSS_URL yoksa gerçek node'a bağlanmadan simülasyon çalışır."""
        return not self.wss_url


CONFIG = Config()
