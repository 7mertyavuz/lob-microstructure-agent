"""Uçtan uca wiring — Katman 1 → 2 → 3 → 4 → 5 → pipeline.

Mempool → decode → classify → flow feature → yön tahmini + bus.

Faz 1e (CAS entegrasyonu, opsiyonel iyileştirme): akış besleme/okuma artık
`lob_microstructure.api.FlowFeed` üzerinden tek bir yoldan geçiyor (`feed.feed_live_signal()`
/ `feed.flow`), ham `RollingFlow`'u main.py'de ayrıca yönetmek yerine. Bu,
CAS köprüsünün (FlowFeed) canlı modda da gerçek besleme yolunu kullandığını
garanti eder ve tek doğruluk kaynağı sağlar. Ekrana basılan çıktı ve akış
DAVRANIŞ OLARAK DEĞİŞMEDİ — sadece veri yolu FlowFeed'e taşındı.

Not: burada `feed.latest()` DEĞİL `feed.flow` doğrudan okunuyor, çünkü
`latest()` simülasyon modunda ek sentetik trafik enjekte eder (bkz.
`FlowFeed._feed_synthetic`); main.py'nin kendi ingest'i (MempoolListener,
gerekirse zaten kendi simülasyon modunda) tek gerçek kaynak olmalı — çift
sentetik üretim yaşanmasın diye.

Çalıştırma:
    python main.py            # WSS_URL yoksa simülasyon
    BUS_BACKEND=redis python main.py
"""
from __future__ import annotations

import asyncio
import logging

from lob_microstructure.config import CONFIG
from lob_microstructure.models import PendingTx
from lob_microstructure.ingest.mempool_listener import MempoolListener
from lob_microstructure.decode.tx_decoder import decode_tx
from lob_microstructure.actor.classifier import classify
from lob_microstructure.api import FlowFeed
from lob_microstructure.predict.direction import predict
from lob_microstructure.pipeline.bus import make_bus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

NUM_WORKERS = 4
PREDICT_EVERY_SEC = 5.0       # tahminleri kaç saniyede bir yayınla
FLOW_WINDOW_SEC = 60.0        # akış imbalance pencere boyu


async def worker(name: str, q: "asyncio.Queue[PendingTx]", bus, feed: FlowFeed) -> None:
    while True:
        tx = await q.get()
        try:
            swap = decode_tx(tx)          # Katman 2
            if swap is None:
                continue                  # router swap'i değil → ele
            signal = classify(swap)       # Katman 3
            feed.feed_live_signal(signal) # Katman 4 besle (FlowFeed üzerinden)
            await bus.publish(signal)     # pipeline
        except Exception:
            log.exception("worker %s işlem hatası", name)
        finally:
            q.task_done()


async def predictor(feed: FlowFeed) -> None:
    """Katman 5 — periyodik yön tahmini yayınlar."""
    while True:
        await asyncio.sleep(PREDICT_EVERY_SEC)
        for token in feed.flow.tokens():
            pred = predict(feed.flow.features(token))
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

    feed = FlowFeed(mode="live", window_sec=FLOW_WINDOW_SEC)
    listener = MempoolListener(q)
    workers = [asyncio.create_task(worker(f"w{i}", q, bus, feed)) for i in range(NUM_WORKERS)]
    pred_task = asyncio.create_task(predictor(feed))
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
