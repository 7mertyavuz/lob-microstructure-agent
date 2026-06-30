"""Tüm katmanlar arasında dolaşan normalize veri modelleri."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
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
