"""Builder ödeme / coinbase.transfer analizi — kesin MEV etiketi.

Searcher'lar bir işlemi öne almak için builder'a iki yoldan öder:
  1) Priority fee (maxPriorityFeePerGas): mempool'da görünür, kaba sinyal.
  2) coinbase.transfer(): işlem trace'inde fee_recipient'e doğrudan ETH
     transferi. Bu, sıradan kullanıcıların yapmadığı bir şeydir → güçlü/kesin
     MEV göstergesi (sandwich/frontrun searcher imzası).

Mempool aşamasında trace yoktur; coinbase analizi onaylanmış blok trace'i
(trace_block / debug_traceTransaction) ile yapılır. Priority-fee tahmini
mempool'da hemen kullanılabilir.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BuilderPayment:
    priority_fee_wei: int          # (gasPrice - baseFee) * gasUsed
    coinbase_transfer_wei: int     # trace'ten doğrudan transferler
    direct_builder_payment: bool   # coinbase_transfer > 0
    total_wei: int


def priority_fee_per_gas(gas_price_wei: int, base_fee_wei: int) -> int:
    """İşlemin gas başına öncelik ücreti (tip). Negatifse 0."""
    return max(0, int(gas_price_wei) - int(base_fee_wei))


def coinbase_transfer_wei(trace: list[dict], fee_recipient: str) -> int:
    """Trace içindeki internal call'lardan fee_recipient'e giden value toplamı.

    trace: [{to, value, ...}, ...] (debug_traceTransaction callTracer düzleştirilmiş).
    """
    fr = (fee_recipient or "").lower()
    total = 0
    for call in trace or []:
        to = (call.get("to") or "").lower()
        val = int(call.get("value", 0) or 0)
        if to == fr and val > 0:
            total += val
    return total


def builder_payment(gas_used: int, gas_price_wei: int, base_fee_wei: int,
                    trace: list[dict] | None = None,
                    fee_recipient: str | None = None) -> BuilderPayment:
    pri = priority_fee_per_gas(gas_price_wei, base_fee_wei) * int(gas_used)
    cb = coinbase_transfer_wei(trace or [], fee_recipient or "") if fee_recipient else 0
    return BuilderPayment(
        priority_fee_wei=pri, coinbase_transfer_wei=cb,
        direct_builder_payment=cb > 0, total_wei=pri + cb,
    )


def mev_score_from_payment(pay: BuilderPayment, base_fee_wei: int) -> float:
    """0..1 MEV güveni. coinbase.transfer → neredeyse kesin; yüksek priority
    fee çarpanı → şüphe."""
    score = 0.0
    if pay.direct_builder_payment:
        score += 0.8                      # coinbase.transfer = searcher imzası
    # priority fee, baz ücretin çok üstündeyse ek şüphe
    if base_fee_wei > 0:
        # priority_fee_wei toplam ödemedir; gas başına oranı bilmiyorsak
        # coinbase olmadan da büyük toplam ödeme şüphe yaratır
        if pay.coinbase_transfer_wei == 0 and pay.priority_fee_wei > 5 * base_fee_wei * 21000:
            score += 0.3
    return min(score, 0.99)
