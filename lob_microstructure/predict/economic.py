"""Ekonomik (maliyet-farkında) etiketleme + latency-arb fizibilite.

İki şey:
  1) Meta-model için DOĞRU hedef: ham yön doğruluğu değil, işlem
     maliyetinden (gas + havuz ücreti + slippage) SONRA net kâr. López de
     Prado'nun "ekonomik etiket" mantığı: sizing'i gerçekçi yapar.
  2) Latency-arb fizibilite filtresi: lead-lag spread'i anlık gas/ücret ile
     birleştirip "bu fırsat maliyet sonrası kârlı mı?" kararını üretir →
     sinyali aksiyona çevirir.
"""
from __future__ import annotations

from dataclasses import dataclass


def net_pnl_usd(direction: int, ret_frac: float, notional_usd: float,
                gas_cost_usd: float, fee_bps: float = 30.0,
                slippage_bps: float = 5.0) -> float:
    """direction: +1 long, -1 short. ret_frac: gerçekleşen fiyat değişimi
    (kesir). notional_usd: pozisyon büyüklüğü. Maliyetler düşülür."""
    gross = direction * ret_frac * notional_usd
    cost = gas_cost_usd + (fee_bps + slippage_bps) / 1e4 * notional_usd
    return gross - cost


def economic_label(direction: int, ret_frac: float, notional_usd: float,
                   gas_cost_usd: float, **kw) -> int:
    """Meta-etiket: işlem maliyet sonrası kâr ettiyse 1, yoksa 0."""
    return int(net_pnl_usd(direction, ret_frac, notional_usd, gas_cost_usd, **kw) > 0)


@dataclass
class ArbDecision:
    profitable: bool
    gross_usd: float
    cost_usd: float
    net_usd: float


class ArbFeasibility:
    """CEX-DEX spread fırsatının gas+ücret sonrası kârlılığını değerlendirir."""

    def __init__(self, fee_bps: float = 30.0, slippage_bps: float = 5.0):
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

    def evaluate(self, spread: float, notional_usd: float,
                 gas_cost_usd: float) -> ArbDecision:
        gross = abs(spread) * notional_usd
        # iki bacak (al+sat) ücret + slippage
        cost = gas_cost_usd + 2 * (self.fee_bps + self.slippage_bps) / 1e4 * notional_usd
        net = gross - cost
        return ArbDecision(profitable=net > 0, gross_usd=round(gross, 2),
                           cost_usd=round(cost, 2), net_usd=round(net, 2))


def gas_cost_usd(gas_used: int, gas_price_wei: int, eth_usd: float = 3000.0) -> float:
    """İşlem gas maliyetini USD'ye çevir."""
    return gas_used * gas_price_wei / 1e18 * eth_usd
