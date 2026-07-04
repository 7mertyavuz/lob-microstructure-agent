"""Dashboard yayın merkezi (Hub).

Bağlı tüm WebSocket istemcilerine JSON mesaj broadcast eder. Pipeline
worker'ları sinyalleri, predictor tahminleri buraya iter; tarayıcı paneli
ws ile bağlanıp canlı alır. `websockets` dışında bağımlılık yok.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

log = logging.getLogger("hub")


class Hub:
    def __init__(self) -> None:
        self._clients: set = set()
        self._lock = asyncio.Lock()

    async def register(self, ws) -> None:
        async with self._lock:
            self._clients.add(ws)
        log.info("İstemci bağlandı (toplam %d)", len(self._clients))

    async def unregister(self, ws) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.info("İstemci ayrıldı (toplam %d)", len(self._clients))

    async def broadcast(self, kind: str, payload: dict[str, Any]) -> None:
        """kind: 'signal' | 'prediction'. Kopuk istemcileri otomatik temizler."""
        if not self._clients:
            return
        msg = json.dumps({"kind": kind, "data": payload})
        async with self._lock:
            targets = list(self._clients)
        dead = []
        for ws in targets:
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)
