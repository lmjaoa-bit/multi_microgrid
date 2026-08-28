from __future__ import annotations

from typing import Any, Dict, Mapping


class CostAccounting:
    """Transparent per-microgrid accounting with linear reward scaling only."""

    def __init__(
        self,
        carbon_price_per_kg: float = 0.0,
        unmet_load_cost_per_kwh: float = 20.0,
        dump_cost_per_kwh: float = 1.0,
        reward_scale: float = 100.0,
    ) -> None:
        self.carbon_price_per_kg = float(carbon_price_per_kg)
        self.unmet_load_cost_per_kwh = float(unmet_load_cost_per_kwh)
        self.dump_cost_per_kwh = float(dump_cost_per_kwh)
        self.reward_scale = float(reward_scale)
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")

    def calculate(
        self,
        row: Mapping[str, Any],
        p2p: Mapping[str, float],
        settlement: Mapping[str, float],
        device: Mapping[str, Any],
        delta_t: float,
    ) -> Dict[str, float]:
        buy_price = float(row["buy_price_per_kwh"])
        sell_price = float(row["sell_price_per_kwh"])
        grid_buy_kwh = float(settlement["grid_buy_kw"]) * delta_t
        grid_sell_kwh = float(settlement["grid_sell_kw"]) * delta_t
        grid_cost = grid_buy_kwh * buy_price - grid_sell_kwh * sell_price
        p2p_net_cost = float(p2p.get("p2p_net_cost", 0.0))
        generator_cost = float(device["generator"]["cost"])
        battery_cost = float(device["battery"]["degradation_cost"])

        grid_carbon = grid_buy_kwh * float(row.get("grid_carbon_factor_kg_per_kwh", 0.0))
        generator_carbon = float(device["generator"]["carbon_kg"])
        carbon_kg = grid_carbon + generator_carbon
        carbon_cost = carbon_kg * self.carbon_price_per_kg

        unmet_cost = float(settlement["unmet_load_kw"]) * delta_t * self.unmet_load_cost_per_kwh
        dump_cost = float(settlement["controllable_dump_kw"]) * delta_t * self.dump_cost_per_kwh
        total = grid_cost + p2p_net_cost + generator_cost + battery_cost + carbon_cost + unmet_cost + dump_cost
        return {
            "grid_cost": grid_cost,
            "p2p_net_cost": p2p_net_cost,
            "generator_cost": generator_cost,
            "battery_degradation_cost": battery_cost,
            "grid_carbon_kg": grid_carbon,
            "generator_carbon_kg": generator_carbon,
            "carbon_kg": carbon_kg,
            "carbon_cost": carbon_cost,
            "unmet_load_cost": unmet_cost,
            "controllable_dump_cost": dump_cost,
            "total_cost": total,
            "reward_scale": self.reward_scale,
            "reward": -total / self.reward_scale,
        }
