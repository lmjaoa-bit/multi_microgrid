from __future__ import annotations

from typing import Dict, TypedDict


class DecodedAction(TypedDict, total=False):
    battery: float
    sharing_offer_ratio: float


class CoordinationHistoryEntry(TypedDict):
    p2p_net_kwh: float
    grid_net_kwh: float
    ess_p2p_offer_kwh: float
    ess_p2p_accepted_kwh: float
    ess_p2p_unmatched_kwh: float


class CoordinationHistory(TypedDict):
    total_p2p_kwh: float
    total_buy_demand_kwh: float
    total_sell_offer_kwh: float
    clearing_ratio: float
    by_microgrid: Dict[str, CoordinationHistoryEntry]
