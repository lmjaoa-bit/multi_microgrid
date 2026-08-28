from __future__ import annotations

from typing import Any, Dict, TypedDict


class DecodedAction(TypedDict):
    battery: float
    generator_commitment: float
    generator_setpoint: float
    p2p_bid_action: float
    p2p_bid_price: float


class LocalPlan(TypedDict, total=False):
    microgrid_id: str
    load_kw: float
    pv_kw: float
    wind_kw: float
    renewable_kw: float
    battery_action: float
    battery_charge_requested_kw: float
    battery_discharge_requested_kw: float
    generator_commitment_action: float
    generator_setpoint_action: float
    generator_power_requested_kw: float
    renewable_to_load_kw: float
    renewable_to_battery_kw: float
    renewable_external_surplus_kw: float
    controllable_to_load_kw: float
    generator_to_load_kw: float
    battery_to_load_kw: float
    controllable_to_battery_kw: float
    load_external_demand_kw: float
    battery_external_charge_demand_kw: float
    battery_unserved_local_charge_request_kw: float
    controllable_external_surplus_kw: float
    generator_external_surplus_kw: float
    battery_external_surplus_kw: float
    external_demand_kw: float
    external_surplus_kw: float
    net_external_kw: float


class GridSettlementResult(TypedDict, total=False):
    p2p_buy_kw: float
    p2p_sell_kw: float
    p2p_to_load_kw: float
    p2p_to_battery_kw: float
    renewable_p2p_sell_kw: float
    generator_p2p_sell_kw: float
    battery_p2p_sell_kw: float
    grid_buy_kw: float
    grid_sell_kw: float
    grid_to_load_kw: float
    grid_to_battery_kw: float
    renewable_grid_sell_kw: float
    generator_grid_sell_kw: float
    battery_grid_sell_kw: float
    unmet_load_kw: float
    unserved_battery_charge_kw: float
    unserved_local_battery_charge_kw: float
    renewable_curtailment_kw: float
    pv_curtailment_kw: float
    wind_curtailment_kw: float
    controllable_dump_kw: float
    generator_dump_kw: float
    battery_dump_kw: float
    battery_unexecuted_discharge_kw: float
    power_balance_residual_kw: float


class MarketHistoryEntry(TypedDict):
    bid_price: float
    trade_kwh: float
    role: float


class MarketHistory(TypedDict):
    weighted_average_price: float
    total_trade_kwh: float
    submitted_buy_kwh: float
    submitted_sell_kwh: float
    cleared_buy_kwh: float
    cleared_sell_kwh: float
    by_microgrid: Dict[str, MarketHistoryEntry]


DeviceCommit = Dict[str, Dict[str, Any]]
