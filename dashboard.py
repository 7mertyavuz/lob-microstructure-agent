"""Canlı dashboard runner — pipeline + WebSocket sunucusu tek event loop'ta.

main.py ile aynı boru hattını çalıştırır; ek olarak her ActorSignal ve
PricePrediction'ı bağlı tarayıcı istemcilerine broadcast eder.

Çalıştırma:
    python dashboard.py
    # sonra dashboard.html'i tarayıcıda aç (ws://localhost:8765'e bağlanır)

WSS_URL boşsa simülasyon verisiyle çalışır.
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
from src.dashboard.hub import Hub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dashboard")

import websockets  # zorunlu: dashboard WS sunucusu için

WS_HOST = "0.0.0.0"
WS_PORT = 8765
NUM_WORKERS = 4
PREDICT_EVERY_SEC = 3.0
FLOW_WINDOW_SEC = 60.0


async def ws_handler(ws, hub: Hub):
    await hub.register(ws)
    try:
        async for _ in ws:   # istemciden mesaj beklemiyoruz, sadece bağlı tut
            pass
    except Exception:
        pass
    finally:
        await hub.unregister(ws)


async def worker(name, q, bus, flow: RollingFlow, hub: Hub):
    while True:
        tx = await q.get()
        try:
            swap = decode_tx(tx)
            if swap is None:
                continue
            signal = classify(swap)
            flow.add(signal)
            await bus.publish(signal)
            await hub.broadcast("signal", signal.to_dict())
        except Exception:
            log.exception("worker %s hata", name)
        finally:
            q.task_done()


async def predictor(flow: RollingFlow, hub: Hub):
    while True:
        await asyncio.sleep(PREDICT_EVERY_SEC)
        for token in flow.tokens():
            pred = predict(flow.features(token))
            d = pred.to_dict()
            d["direction"] = pred.direction
            await hub.broadcast("prediction", d)


async def main():
    log.info("Dashboard başlatılıyor — mod=%s, ws://%s:%d",
             "SIMÜLASYON" if CONFIG.simulation_mode else "CANLI", WS_HOST, WS_PORT)

    q: "asyncio.Queue[PendingTx]" = asyncio.Queue(maxsize=10_000)
    bus = make_bus()
    await bus.start()
    flow = RollingFlow(window_sec=FLOW_WINDOW_SEC)
    hub = Hub()

    async def handler(ws):
        await ws_handler(ws, hub)

    server = await websockets.serve(handler, WS_HOST, WS_PORT)
    listener = MempoolListener(q)
    workers = [asyncio.create_task(worker(f"w{i}", q, bus, flow, hub))
               for i in range(NUM_WORKERS)]
    pred_task = asyncio.create_task(predictor(flow, hub))
    ingest_task = asyncio.create_task(listener.run())

    log.info("Hazır. dashboard.html'i tarayıcıda aç.")
    try:
        await ingest_task
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        listener.stop()
        for w in workers:
            w.cancel()
        pred_task.cancel()
        server.close()
        await server.wait_closed()
        await bus.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDurduruldu.")
