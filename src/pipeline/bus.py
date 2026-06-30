"""Pipeline çıkış katmanı — etiketlenmiş sinyalleri downstream'e basar.

Tek arayüz (SignalBus), üç backend:
  * memory : stdout'a yazar (PoC / geliştirme). Bağımlılık yok.
  * redis  : Redis Pub/Sub kanalına basar.   (pip install redis)
  * kafka  : Kafka topic'ine basar.           (pip install aiokafka)

Katman 4/5 (feature + ONNX sınıflandırma) bu kuyruğu tüketir.
"""
from __future__ import annotations

import json
import logging

from config import CONFIG
from src.models import ActorSignal

log = logging.getLogger("bus")


class SignalBus:
    async def start(self) -> None: ...
    async def publish(self, sig: ActorSignal) -> None: ...
    async def close(self) -> None: ...


class MemoryBus(SignalBus):
    """Bağımlılıksız PoC backend: konsola basar, sayaç tutar."""
    def __init__(self):
        self.count = 0

    async def publish(self, sig: ActorSignal) -> None:
        self.count += 1
        icon = {"WHALE": "🐋", "MEV_BOT": "🤖", "RETAIL": "👤"}.get(sig.label.value, "?")
        print(
            f"{icon} {sig.label.value:<7} | {sig.side.value:<7} | {sig.dex}:{sig.method} "
            f"| ~${sig.est_value_usd:,.0f} | güven %{sig.confidence*100:.0f} "
            f"| {sig.address[:10]}... | {'; '.join(sig.reasons)}"
        )


class RedisBus(SignalBus):
    def __init__(self):
        self._r = None

    async def start(self) -> None:
        import redis.asyncio as redis  # type: ignore
        self._r = redis.from_url(CONFIG.redis_url)

    async def publish(self, sig: ActorSignal) -> None:
        await self._r.publish(CONFIG.kafka_topic, json.dumps(sig.to_dict()))

    async def close(self) -> None:
        if self._r:
            await self._r.aclose()


class KafkaBus(SignalBus):
    def __init__(self):
        self._p = None

    async def start(self) -> None:
        from aiokafka import AIOKafkaProducer  # type: ignore
        self._p = AIOKafkaProducer(bootstrap_servers=CONFIG.kafka_bootstrap)
        await self._p.start()

    async def publish(self, sig: ActorSignal) -> None:
        await self._p.send_and_wait(
            CONFIG.kafka_topic, json.dumps(sig.to_dict()).encode()
        )

    async def close(self) -> None:
        if self._p:
            await self._p.stop()


def make_bus() -> SignalBus:
    backend = CONFIG.bus_backend
    if backend == "redis":
        return RedisBus()
    if backend == "kafka":
        return KafkaBus()
    if backend != "memory":
        log.warning("Bilinmeyen BUS_BACKEND=%s → memory kullanılıyor", backend)
    return MemoryBus()
