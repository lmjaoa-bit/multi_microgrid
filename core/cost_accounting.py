from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CostAccounting:
    unmet_load_cost_per_kwh: float
    curtailment_cost_per_kwh: float
    reward_scale: float

    def calculate(self, buy_price: float, sell_price: float, settlement: Mapping[str, float],
                  degradation_cost: float, delta_t: float) -> dict[str, float]:
        grid_import_cost = settlement["grid_import_kw"] * delta_t * buy_price
        grid_export_revenue = settlement["grid_export_kw"] * delta_t * sell_price
        unmet_cost = settlement["unmet_load_kw"] * delta_t * self.unmet_load_cost_per_kwh
        curtailment_cost = settlement["renewable_curtailment_kw"] * delta_t * self.curtailment_cost_per_kwh
        profit = grid_export_revenue - grid_import_cost - degradation_cost - unmet_cost - curtailment_cost
        return {
            "grid_import_cost": float(grid_import_cost),
            "grid_export_revenue": float(grid_export_revenue),
            "battery_degradation_cost": float(degradation_cost),
            "unmet_load_cost": float(unmet_cost),
            "curtailment_cost": float(curtailment_cost),
            "physical_profit": float(profit),
        }
