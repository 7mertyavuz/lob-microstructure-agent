"""zeromev REST client — gerçek MEV etiketi (ground truth).

data.zeromev.org anahtarsız, blok/işlem seviyesinde MEV özetini verir
(sandwich, arb, frontrun, liquidation...). Bizim heuristik MEV_BOT
etiketimizi GERÇEK etiketle karşılaştırıp sınıflandırıcıyı kalibre etmek
ve eğitim verisini etiketlemek için kullanılır.

Bağımlılık eklemez: stdlib urllib'i asyncio.to_thread ile sarar.
mevBlock yanıtı bir işlem listesidir; her öğede en az:
    block_number, tx_index, mev_type, protocol,
    user_loss_usd, extractor_profit_usd, user_swap_volume_usd
Şema alanları zamanla değişebileceğinden erişim savunmacıdır (.get).
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Any, Optional

log = logging.getLogger("zeromev")

BASE = "https://data.zeromev.org/v1"

# zeromev mev_type değerleri → bizim ActorLabel dünyamızla eşleme ipucu
MEV_TYPES = {"swap", "sandwich", "frontrun", "backrun", "arb", "liquid"}


def _get_json(url: str, timeout: float = 10.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "microstructure-analyzer"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


async def get_block_mev(block_number: int, count: int = 1) -> list[dict]:
    """Bir bloğun (veya count blok) MEV işlem listesini döndür."""
    url = f"{BASE}/mevBlock?block_number={int(block_number)}&count={int(count)}"
    try:
        data = await asyncio.to_thread(_get_json, url)
    except Exception as e:
        log.warning("zeromev mevBlock hatası (blok %s): %s", block_number, e)
        return []
    return data if isinstance(data, list) else []


def index_by_tx(rows: list[dict]) -> dict[int, dict]:
    """tx_index → satır eşlemesi (hızlı arama için)."""
    out: dict[int, dict] = {}
    for r in rows:
        idx = r.get("tx_index")
        if idx is not None:
            out[int(idx)] = r
    return out


def label_for_tx(rows_by_idx: dict[int, dict], tx_index: int) -> Optional[str]:
    """Belirli bir tx için gerçek mev_type (yoksa None)."""
    row = rows_by_idx.get(int(tx_index))
    return row.get("mev_type") if row else None


_TOXIC = {"sandwich", "sandwich_attack", "frontrun", "backrun", "arb", "liquid"}


def is_mev(mev_type: Optional[str]) -> bool:
    """Sıradan swap dışı (toksik) MEV mı?"""
    if not mev_type:
        return False
    return mev_type.lower() in _TOXIC
