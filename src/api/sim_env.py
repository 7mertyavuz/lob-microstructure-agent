"""Katman 2 köprüsü — `SimEnvironment` (enjekte edilebilir sim çevresi).

`cas-market-simulator`'ın `Environment` katmanından dışarıdan ajan emri
kabul eder ve akış metriklerinin buna tepki verdiği bir arayüz sunar.

`step(orders, token)`: verilen `AgentOrder` listesini decode→classify→
`RollingFlow` yoluna sokar (bkz. `src/ingest/mempool_listener.py::
order_to_pending_tx`, `src/decode/tx_decoder.py::decode_tx`,
`src/actor/classifier.py::classify`) ve **güncellenmiş** `FlowState`'i
döndürür. `FlowFeed`'in autonomous sentetik üretimi burada tetiklenmez —
`SimEnvironment` yalnızca enjekte edilen emirlerle beslenir (driven mod);
bu da belirliliği (determinism) garanti eder: aynı emir dizisi + aynı
seed → aynı `FlowState` dizisi (replay).

Sözleşme: `docs/00-ORTAK-SOZLESME.md`.
"""
from __future__ import annotations

from src.models import AgentOrder, FlowState
from src.api.flow_feed import FlowFeed
from src.ingest.mempool_listener import order_to_pending_tx
from src.decode.tx_decoder import decode_tx
from src.actor.classifier import classify


class SimEnvironment:
    """`step(orders) -> FlowState` — simülatör dostu, tamamen enjekte
    edilebilir sim çevresi. Autonomous mod (mevcut `FlowFeed`/
    `MempoolListener` davranışı) bundan etkilenmez/değişmez; bu sınıf ek
    bir "driven" arayüzdür."""

    def __init__(self, seed: int | None = None, window_sec: float = 60.0):
        # mode="simulation" olarak kullanılır ama _feed_synthetic() hiç
        # çağrılmaz (yalnızca _read_state ile okuruz) — bu yüzden sim'in
        # autonomous gürültüsü SimEnvironment'a karışmaz.
        self._feed = FlowFeed(mode="simulation", seed=seed, window_sec=window_sec)

    def inject(self, order: AgentOrder) -> None:
        """Tek bir ajan emrini decode→classify→RollingFlow yoluna sokar."""
        tx = order_to_pending_tx(order)
        swap = decode_tx(tx)
        if swap is None:
            return  # router swap'i değil → main.py worker'ıyla tutarlı şekilde ele
        signal = classify(swap)
        self._feed.flow.add(signal)

    def step(self, orders: list[AgentOrder], token: str) -> FlowState:
        """`orders` listesini sırayla enjekte eder, ardından güncellenmiş
        `FlowState`'i döndürür (autonomous sentetik akış tetiklenmeden)."""
        for order in orders:
            self.inject(order)
        return self._feed._read_state(token)
