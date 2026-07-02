"""CAS entegrasyon arayuzleri -- cas-market-simulator icin kopru katmani.

Bu paket mevcut 5 katmanli analist cekirdegini (ingest->decode->actor->
features->predict) degistirmez; onun uzerine iki ince, test edilebilir
arayuz ekler:

  * FlowFeed       -- Katman 1 koprusu: okuma arayuzu, FlowState dondurur.
  * SimEnvironment -- Katman 2 koprusu: enjekte edilebilir sim cevresi.

Sozlesme: docs/00-ORTAK-SOZLESME.md.
"""
from __future__ import annotations

from src.models import FlowState, AgentOrder
from src.api.flow_feed import FlowFeed
from src.api.sim_env import SimEnvironment

__all__ = ["FlowState", "AgentOrder", "FlowFeed", "SimEnvironment"]
