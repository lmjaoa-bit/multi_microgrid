from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.types import GridSettlementResult


@dataclass
class GridSettlement:
    import_limit_kw: float
    export_limit_kw: float
    delta_t: float = 1.0
    tolerance: float = 1e-9

    def settle(self, plan: Mapping[str, Any], market: Mapping[str, float]) -> GridSettlementResult:
        p2p_buy = float(market.get("p2p_buy_kwh", 0.0)) / self.delta_t
        p2p_sell = float(market.get("p2p_sell_kwh", 0.0)) / self.delta_t
        demand = float(plan["external_demand_kw"])
        surplus = float(plan["external_surplus_kw"])
        if p2p_buy > demand + 1e-7 or p2p_sell > surplus + 1e-7:
            raise ValueError("P2P clearing exceeded submitted order quantity.")

        # Buyer settlement: P2P first, then grid, exclusively for local load.
        # Battery charging is restricted to local PV/wind surplus in MicrogridCore.
        load_need = float(plan["load_external_demand_kw"])
        charge_need = float(plan["battery_external_charge_demand_kw"])
        if charge_need > self.tolerance:
            raise ValueError("External battery charging demand is prohibited.")
        p2p_to_load = min(p2p_buy, load_need)
        p2p_to_battery = 0.0
        load_left = max(0.0, load_need - p2p_to_load)

        grid_to_load = min(load_left, float(self.import_limit_kw))
        grid_to_battery = 0.0
        unmet_load = max(0.0, load_left - grid_to_load)
        unserved_charge = float(plan.get("battery_unserved_local_charge_request_kw", 0.0))
        grid_buy = grid_to_load

        # Seller settlement: attribute the pooled P2P trade proportionally to the
        # submitted source mix. Grid export may use renewable and generator output,
        # but battery discharge is restricted to local load or P2P sale.
        ren_surplus = float(plan["renewable_external_surplus_kw"])
        gen_surplus = float(plan.get("generator_external_surplus_kw", 0.0))
        bat_surplus = float(plan.get("battery_external_surplus_kw", 0.0))
        total_source_surplus = ren_surplus + gen_surplus + bat_surplus
        if total_source_surplus > self.tolerance:
            p2p_ren = p2p_sell * ren_surplus / total_source_surplus
            p2p_gen = p2p_sell * gen_surplus / total_source_surplus
            p2p_bat = max(0.0, p2p_sell - p2p_ren - p2p_gen)
        else:
            p2p_ren = p2p_gen = p2p_bat = 0.0
        ren_left = max(0.0, ren_surplus - p2p_ren)
        gen_left = max(0.0, gen_surplus - p2p_gen)
        bat_left = max(0.0, bat_surplus - p2p_bat)

        grid_sell = min(float(self.export_limit_kw), ren_left + gen_left)
        grid_ren = min(grid_sell, ren_left)
        grid_gen = max(0.0, grid_sell - grid_ren)
        renewable_curtailment = max(0.0, ren_left - grid_ren)
        generator_dump = max(0.0, gen_left - grid_gen)
        battery_unexecuted_discharge = bat_left
        controllable_dump = generator_dump

        pv = float(plan["pv_kw"])
        wind = float(plan["wind_kw"])
        ren = pv + wind
        pv_ratio = pv / ren if ren > self.tolerance else 0.0
        wind_ratio = wind / ren if ren > self.tolerance else 0.0

        actual_charge = (
            float(plan["renewable_to_battery_kw"])
            + float(plan["controllable_to_battery_kw"])
            + p2p_to_battery
            + grid_to_battery
        )
        residual = (
            ren
            + float(plan["generator_power_requested_kw"])
            + float(plan["battery_discharge_requested_kw"])
            + p2p_buy
            + grid_buy
            + unmet_load
            - float(plan["load_kw"])
            - actual_charge
            - p2p_sell
            - grid_sell
            - renewable_curtailment
            - controllable_dump
            - battery_unexecuted_discharge
        )

        return {
            "p2p_buy_kw": p2p_buy,
            "p2p_sell_kw": p2p_sell,
            "p2p_to_load_kw": p2p_to_load,
            "p2p_to_battery_kw": p2p_to_battery,
            "renewable_p2p_sell_kw": p2p_ren,
            "generator_p2p_sell_kw": p2p_gen,
            "battery_p2p_sell_kw": p2p_bat,
            "grid_buy_kw": grid_buy,
            "grid_sell_kw": grid_sell,
            "grid_to_load_kw": grid_to_load,
            "grid_to_battery_kw": grid_to_battery,
            "renewable_grid_sell_kw": grid_ren,
            "generator_grid_sell_kw": grid_gen,
            "battery_grid_sell_kw": 0.0,
            "unmet_load_kw": unmet_load,
            "unserved_battery_charge_kw": unserved_charge,
            "unserved_local_battery_charge_kw": unserved_charge,
            "renewable_curtailment_kw": renewable_curtailment,
            "pv_curtailment_kw": renewable_curtailment * pv_ratio,
            "wind_curtailment_kw": renewable_curtailment * wind_ratio,
            "controllable_dump_kw": controllable_dump,
            "generator_dump_kw": generator_dump,
            "battery_dump_kw": 0.0,
            "battery_unexecuted_discharge_kw": battery_unexecuted_discharge,
            "power_balance_residual_kw": residual,
        }
