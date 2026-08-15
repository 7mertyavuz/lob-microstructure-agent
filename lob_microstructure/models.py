"""Tüm katmanlar arasında dolaşan normalize veri modelleri."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"


class ActorLabel(str, Enum):
    WHALE = "WHALE"          # Büyük hacimli gerçek oyuncu
    MEV_BOT = "MEV_BOT"      # Front-run / sandwich / arbitraj botu
    RETAIL = "RETAIL"        # Sıradan kullanıcı
    UNKNOWN = "UNKNOWN"


@dataclass
class PendingTx:
    """Katman 1 çıktısı: mempool'dan yakalanan ham işlem (normalize)."""
    tx_hash: str
    from_addr: str
    to_addr: Optional[str]
    value_wei: int
    gas_price_wei: int          # legacy gasPrice ya da maxFeePerGas
    input_data: str             # hex calldata
    ts_ns: int = field(default_factory=lambda: time.time_ns())


@dataclass
class DecodedSwap:
    """Katman 2 çıktısı: çözülmüş router çağrısı."""
    tx: PendingTx
    dex: str                    # ör. "UniswapV2", "UniswapV3"
    method: str                 # ör. "swapExactETHForTokens"
    side: Side
    token_in: Optional[str] = None
    token_out: Optional[str] = None
    amount_in_wei: Optional[int] = None
    # Calldata'dan çözülemeyen miktarlar için value_wei fallback olabilir


@dataclass
class WalletProfile:
    """Katman 3 yardımcı verisi: cüzdan zenginleştirme."""
    address: str
    age_days: float
    tx_count: int
    balance_eth: float
    lifetime_volume_usd: float


@dataclass
class ActorSignal:
    """Katman 3 nihai çıktısı: pipeline'a basılan etiketlenmiş sinyal."""
    tx_hash: str
    address: str
    label: ActorLabel
    side: Side
    dex: str
    method: str
    est_value_usd: float
    confidence: float           # 0..1
    reasons: list[str] = field(default_factory=list)
    ts_ns: int = field(default_factory=lambda: time.time_ns())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label.value
        d["side"] = self.side.value
        return d


@dataclass
class PricePrediction:
    """Katman 5 çıktısı: bir token için kısa vadeli yön tahmini."""
    token: str
    prob_up: float              # 0..1, fiyatın yükselme olasılığı
    flow_imbalance: float       # -1..+1, net alış/satış baskısı
    window_sec: float
    sample_count: int
    whale_net_usd: float        # pencere içi balina net alış (USD, + alış)
    toxicity: float = 0.0       # 0..1 VPIN akış toksisitesi (yüksek = volatil/riskli)
    ts_ns: int = field(default_factory=lambda: time.time_ns())

    @property
    def direction(self) -> str:
        if self.prob_up >= 0.58:
            return "YUKARI"
        if self.prob_up <= 0.42:
            return "AŞAĞI"
        return "NÖTR"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlowState:
    """CAS entegrasyonu Katman 1 çıktı sözleşmesi (docs/00-ORTAK-SOZLESME.md).

    FlowFeed.latest(token) tarafından üretilir. Ham/temiz metrik verir,
    ağırlık kararı vermez -- cas-market-simulator motoru bu alanları
    kendi ağırlıklandırma mantığında kullanır. Bu tip, mevcut
    FlowFeatures/PricePrediction hesaplamalarını değiştirmez; onları
    tek bir dışa-dönük struct'ta paketler.

    NOT: flow_imbalance/whale_net_usd gibi alanlar motorun kendi
    order_flow/onchain_flow faktörleriyle kavramsal olarak örtüşebilir
    (çift sayım riski) -- bkz. ortak sözleşme dokümanı.
    """
    token: str
    flow_imbalance: float        # -1..+1, harmanlı (fast+slow) net alış/satış baskısı
    vpin_toxicity: float         # 0..1, VPIN akış toksisitesi
    whale_net_usd: float         # pencere içi balina net alış (USD, + alış)
    actor_mix: dict[str, float]  # {"WHALE","MEV_BOT","RETAIL"} payları, toplam ~= 1.0
    direction_prob_up: float     # 0..1, fiyatın yükselme olasılığı
    lead_lag_spread: float       # CEX-DEX gecikme-düzeltmeli spread (canlı) / sentetik (sim)
    regime: str                  # "normal" | "toxic" | "highvol"
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d


@dataclass
class AgentOrder:
    """CAS entegrasyonu Katman 2 giriş sözleşmesi (docs/00-ORTAK-SOZLESME.md).

    Simülatörün (cas-market-simulator) sentetik bir ajanın verdiği emri
    bu repoya beslemesi için kullanılır. PendingTx ile eşdeğer bilgiyi
    taşır ki MempoolListener.inject() üzerinden aynı
    decode->classify->feature yolundan geçebilsin.
    """
    token: str
    side: str            # "BUY" | "SELL"
    size_usd: float
    actor_label: str      # "WHALE" | "MEV_BOT" | "RETAIL"
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
