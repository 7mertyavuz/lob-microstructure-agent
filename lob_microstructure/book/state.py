"""Emir defteri veri modelleri — ham defter/tape + `BookState` çıktı sözleşmesi.

`FlowState`'ten (lob_microstructure/models.py) tamamen ayrı bir struct'tır: akış (flow) ve
defter (book) sorumlulukları ayrık tutulur; çift sayım riski motor tarafında
yönetilir (bkz. docs/00-ORTAK-SOZLESME.md). Bu modül saf veri + küçük
yardımcılardır; hesaplama `features.py`'de, üretim `sim.py`/`keeper.py`'dedir.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import NamedTuple


class BookLevel(NamedTuple):
    """Tek defter seviyesi: fiyat ve o fiyattaki toplam pasif miktar."""
    price: float
    qty: float


class Trade(NamedTuple):
    """Tape (işlem bandı) kaydı. `side` = agresörün yönü.

    BUY  → alıcı agresör en iyi ask'i kaldırdı (yukarı baskı).
    SELL → satıcı agresör en iyi bid'e vurdu (aşağı baskı).
    """
    price: float
    qty: float
    side: str          # "BUY" | "SELL"
    ts: float          # enjekte edilebilir saat (determinizm)


class OrderEvent(NamedTuple):
    """L3-türevi okumalar için sadeleştirilmiş emir-yaşam-döngüsü olayı.

    Ham L3 verimiz yok; bu olaylar diff+tape kesişiminden çıkarılan
    (veya simülatörde bilerek üretilen) gözlemlerdir.

    kind:
      "add"    → yeni pasif emir belirdi
      "cancel" → pasif emir işlem görmeden iptal edildi
      "trade"  → seviyede işlem gerçekleşti (agresör doldu)
    """
    kind: str          # "add" | "cancel" | "trade"
    side: str          # "BUY" (bid tarafı) | "SELL" (ask tarafı)
    price: float
    qty: float
    ts: float
    dist_bps: float = 0.0   # mid'e uzaklık (bps); spoof tespitinde kullanılır


@dataclass
class RawBook:
    """Ham L2 defter anlık görüntüsü + son işlem bandı.

    `bids` en iyiden kötüye (fiyat azalan), `asks` en iyiden kötüye
    (fiyat artan) sıralı olmalıdır. `trades` en yeni penceredeki tape,
    `events` ise L3-türevi okumalar için emir-yaşam-döngüsü olaylarıdır.
    """
    symbol: str
    ts: datetime
    bids: list[BookLevel]
    asks: list[BookLevel]
    trades: list[Trade] = field(default_factory=list)
    events: list[OrderEvent] = field(default_factory=list)
    # Gerçek forceOrder olaylarından / OI-funding tahmininden likidasyon kümeleri:
    # (price, notional_usd) — fiyat mid üstündeyse short, altındaysa long küme.
    liquidations: list[tuple[float, float]] = field(default_factory=list)

    @property
    def best_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> float:
        """Orta fiyat. Tek taraf boşsa var olan tarafın fiyatı, ikisi de
        boşsa 0.0."""
        bb, ba = self.best_bid, self.best_ask
        if bb and ba:
            return (bb.price + ba.price) / 2.0
        if bb:
            return bb.price
        if ba:
            return ba.price
        return 0.0

    @property
    def spread(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb and ba:
            return ba.price - bb.price
        return 0.0


@dataclass
class BookState:
    """Defter okuma çıktı sözleşmesi (docs/00-ORTAK-SOZLESME.md BookState).

    Analist çıktısıdır: ham/temiz metrik verir, ağırlık kararı vermez. Motor
    (cas-market-simulator) bu alanları kendi ağırlıklandırma mantığında
    kullanır. `iceberg_score`/`spoof_score` **kanıt değil şüphedir** — tek
    başına yön oyu vermez, güven çarpanı olarak kullanılır.
    """
    symbol: str
    spread_bps: float           # ≥0, en iyi alış-satış farkı (bps)
    microprice: float           # >0, derinlik-ağırlıklı adil fiyat (Stoikov)
    depth_imbalance: float      # [-1,1], çok-seviyeli mesafe-ağırlıklı denge
    ofi: float                  # serbest, event-bazlı order flow imbalance
    queue_imbalance: float      # [-1,1], en iyi seviye kuyruk dengesizliği
    book_slope: float           # ≥0, defter eğimi (esneklik)
    kyle_lambda: float          # ≥0, hacim başına fiyat etkisi
    # `mid` olmadan tüketici `microprice_dev = (microprice-mid)/mid` sapmasını
    # HESAPLAYAMAZ. Uzun süre verilmiyordu ve cas-market-simulator bu yüzden
    # sapmayı her zaman tam 0.0 olarak üretiyordu (sessiz ölü yol).
    mid: float = 0.0            # >0, en iyi alış-satış orta noktası
    iceberg_score: float = 0.0  # [0,1], gizli likidite şüphesi
    spoof_score: float = 0.0    # [0,1], yanıltıcı katmanlama şüphesi
    absorption: float = 0.0     # [-1,1], + = satış baskısı emiliyor (bid güçlü)
    liq_map_skew: float = 0.0   # [-1,1], likidasyon yoğunluğu üstte(+)/altta(-)
    # --- daha önce hesaplanıp yüzeye çıkarılmayan metrikler ---
    sweep: float = 0.0          # [-1,1], çok-seviyeli süpürme (işaretli)
    liquidity_gaps: float = 0.0  # [0,1], ince defter boşluk skoru
    spread_z: float = 0.0       # spread'in kendi geçmişine göre z-skoru
    # --- Fourier HFT gürültü filtresi ---
    noise_ratio: float = 0.0    # [0,1], yüksek frekans enerji payı (HFT churn)
    # --- Lee-Mykland sıçrama testi (Heimdall Bates üretecini besler) ---
    lambda_jump: float = 0.0    # ≥0, bar başına sıçrama yoğunluğu
    jump_mean: float = 0.0      # sıçrama büyüklüğü ortalaması (log)
    jump_std: float = 0.0       # sıçrama büyüklüğü standart sapması
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d
