from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import pandas as pd

from components.battery import BatteryModel
from components.generator import GeneratorModel
from core.profiles import HeterogeneousProfileModel
from core.types import DeviceCommit, LocalPlan


@dataclass
class MicrogridCore:
    """Physical core for one participant in the heterogeneous multi-microgrid market."""

    config: Dict[str, Any]
    delta_t: float
    profiles: HeterogeneousProfileModel

    def __post_init__(self) -> None:
        self.microgrid_id = str(self.config["id"])
        self.battery = BatteryModel.from_config(self.config["battery"], self.delta_t)
        self.generator = GeneratorModel.from_config(self.config["generator"], self.delta_t)

    def reset(self) -> None:
        self.battery.reset()
        self.generator.reset()

    def exogenous(self, row: pd.Series) -> Dict[str, float]:
        return self.profiles.exogenous(self.microgrid_id, row)

    def plan_local(self, row: pd.Series, action: Mapping[str, Any]) -> LocalPlan:
        ex = self.exogenous(row)
        load = ex["load_kw"]
        renewable = ex["renewable_kw"]

        battery_map = self.battery.map_action(float(action["battery"]))
        charge_req = float(battery_map["charge_requested_kw"])
        discharge_req = float(battery_map["discharge_requested_kw"])

        gen_plan = self.generator.plan(
            float(action["generator_commitment"]),
            float(action["generator_setpoint"]),
        )
        gen_kw = float(gen_plan["power_kw"])

        # Battery charging is restricted to local PV/wind surplus. The generator,
        # P2P market, and main grid are never allowed to charge the battery.
        ren_to_load = min(renewable, load)
        load_left = max(0.0, load - ren_to_load)
        ren_left = max(0.0, renewable - ren_to_load)

        ren_to_battery = min(ren_left, charge_req)
        ren_external_surplus = max(0.0, ren_left - ren_to_battery)
        unserved_local_charge = max(0.0, charge_req - ren_to_battery)

        controllable_supply = gen_kw + discharge_req
        controllable_to_load = min(controllable_supply, load_left)
        load_external_demand = max(0.0, load_left - controllable_to_load)
        controllable_left = max(0.0, controllable_supply - controllable_to_load)

        controllable_to_battery = 0.0
        battery_external_charge_demand = 0.0
        controllable_external_surplus = controllable_left

        external_demand = load_external_demand
        external_surplus = ren_external_surplus + controllable_external_surplus
        if external_demand > 1e-8 and external_surplus > 1e-8:
            raise RuntimeError(f"{self.microgrid_id}: simultaneous buy and sell position.")

        if controllable_supply > 1e-9:
            gen_share = gen_kw / controllable_supply
            bat_share = discharge_req / controllable_supply
        else:
            gen_share = bat_share = 0.0

        return {
            "microgrid_id": self.microgrid_id,
            "load_kw": ex["load_kw"],
            "pv_kw": ex["pv_kw"],
            "wind_kw": ex["wind_kw"],
            "renewable_kw": ex["renewable_kw"],
            "battery_action": float(action["battery"]),
            "battery_charge_requested_kw": charge_req,
            "battery_discharge_requested_kw": discharge_req,
            "generator_commitment_action": float(action["generator_commitment"]),
            "generator_setpoint_action": float(action["generator_setpoint"]),
            "generator_power_requested_kw": gen_kw,
            "renewable_to_load_kw": ren_to_load,
            "renewable_to_battery_kw": ren_to_battery,
            "renewable_external_surplus_kw": ren_external_surplus,
            "controllable_to_load_kw": controllable_to_load,
            "generator_to_load_kw": controllable_to_load * gen_share,
            "battery_to_load_kw": controllable_to_load * bat_share,
            "controllable_to_battery_kw": controllable_to_battery,
            "load_external_demand_kw": load_external_demand,
            "battery_external_charge_demand_kw": battery_external_charge_demand,
            "battery_unserved_local_charge_request_kw": unserved_local_charge,
            "controllable_external_surplus_kw": controllable_external_surplus,
            "generator_external_surplus_kw": controllable_external_surplus * gen_share,
            "battery_external_surplus_kw": controllable_external_surplus * bat_share,
            "external_demand_kw": external_demand,
            "external_surplus_kw": external_surplus,
            "net_external_kw": external_surplus - external_demand,
        }

    def commit(self, plan: Mapping[str, Any], settlement: Mapping[str, Any]) -> DeviceCommit:
        if float(plan["controllable_to_battery_kw"]) > 1e-9:
            raise RuntimeError("Generator-to-battery charging is prohibited.")
        if float(settlement["p2p_to_battery_kw"]) > 1e-9:
            raise RuntimeError("P2P-to-battery charging is prohibited.")
        if float(settlement["grid_to_battery_kw"]) > 1e-9:
            raise RuntimeError("Grid-to-battery charging is prohibited.")
        actual_charge = float(plan["renewable_to_battery_kw"])
        actual_discharge = (
            float(plan.get("battery_to_load_kw", 0.0))
            + float(settlement.get("battery_p2p_sell_kw", 0.0))
        )
        battery = self.battery.commit(actual_charge, actual_discharge)
        generator = self.generator.commit(float(plan["generator_power_requested_kw"]))
        return {"battery": battery, "generator": generator}
