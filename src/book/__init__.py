"""Emir defteri okuma paketi (L2 yeniden inşa + L3-türevi okumalar).

Plan: `DEFTER-L3-OKUMA-PLANI.md`. Sözleşme: `docs/00-ORTAK-SOZLESME.md`
(BookState bölümü).

- `state`   : BookState çıktı sözleşmesi + ham defter/tape struct'ları
- `sim`     : SimBookFeed — deterministik, rejim-anahtarlamalı sentetik L2
- `keeper`  : BookKeeper — seviye→miktar defter tutucu (snapshot+diff)
- `features`: saf okuma fonksiyonları (D2 çekirdek + D3 L3-türevi)
- `feed`    : BookFeed — `latest(symbol) -> BookState` okuma arayüzü

İlke (analist rolü): defter ham/temiz metrik verir, ağırlık kararı vermez.
"""
from src.book.state import BookState, RawBook, BookLevel, Trade, OrderEvent
from src.book.feed import BookFeed
from src.book.sim import SimBookFeed
from src.book.dex_virtual_book import DexVirtualBook

__all__ = [
    "BookState",
    "RawBook",
    "BookLevel",
    "Trade",
    "OrderEvent",
    "BookFeed",
    "SimBookFeed",
    "DexVirtualBook",
]
