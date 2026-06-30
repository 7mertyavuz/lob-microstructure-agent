"""Uçtan uca wiring — Katman 1 → 2 → 3 → 4 → 5 → pipeline.

Mempool → decode → classify → flow feature → yön tahmini + bus.

Çalıştırma:
    python main.py            # WSS_URL yoksa simülasyon
    BUS_BACKEND=redis python main.py
"""
from __future__ import annotations

import asyncio
import logging

from config import CONFIG
from src.models import PendingTx
from src.ingest.mempool_listener import MempoolListener
from src.decode.tx_decoder import decode_tx
from src.actor.classifier import classify
from src.features.window import RollingFlow
from src.predict.direction import predict
from src.pipeline.bus import make_bus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

NUM_WORKERS = 4
PREDICT_EVERY_SEC = 5.0       # tahminleri kaç saniyede bir yayınla
FLOW_WINDOW_SEC = 60.0        # akış imbalance pencere boyu


async def worker(name: str, q: "asyncio.Queue[PendingTx]", bus, flow: RollingFlow) -> None:
    while True:
        tx = await q.get()
        try:
            swap = decode_tx(tx)          # Katman 2
            if swap is None:
                continue                  # router swap'i değil → ele
            signal = classify(swap)       # Katman 3
            flow.add(signal)              # Katman 4 besle
            await bus.publish(signal)     # pipeline
        except Exception:
            log.exception("worker %s işlem hatası", name)
        finally:
            q.task_done()


async def predictor(flow: RollingFlow) -> None:
    """Katman 5 — periyodik yön tahmini yayınlar."""
    while True:
        await asyncio.sleep(PREDICT_EVERY_SEC)
        for token in flow.tokens():
            pred = predict(flow.features(token))
            arrow = {"YUKARI": "📈", "AŞAĞI": "📉", "NÖTR": "➡️"}[pred.direction]
            print(
                f"  {arrow} TAHMİN [{pred.token}] {pred.direction} "
                f"| P(yukarı)=%{pred.prob_up*100:.0f} "
                f"| imbalance={pred.flow_imbalance:+.2f} "
                f"| balina net=${pred.whale_net_usd:,.0f} "
                f"| n={pred.sample_count}/{pred.window_sec:.0f}s"
            )


async def main() -> None:
    log.info("Başlatılıyor — backend=%s, mod=%s",
             CONFIG.bus_backend,
             "SIMÜLASYON" if CONFIG.simulation_mode else "CANLI")

    q: "asyncio.Queue[PendingTx]" = asyncio.Queue(maxsize=10_000)
    bus = make_bus()
    await bus.start()

    flow = RollingFlow(window_sec=FLOW_WINDOW_SEC)
    listener = MempoolListener(q)
    workers = [asyncio.create_task(worker(f"w{i}", q, bus, flow)) for i in range(NUM_WORKERS)]
    pred_task = asyncio.create_task(predictor(flow))
    ingest_task = asyncio.create_task(listener.run())

    try:
        await ingest_task
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        listener.stop()
        for w in workers:
            w.cancel()
        pred_task.cancel()
        await bus.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDurduruldu.")
