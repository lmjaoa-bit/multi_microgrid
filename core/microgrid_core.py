from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from components.battery import BatteryModel
from core.profiles import HeterogeneousProfileModel


@dataclass
class MicrogridCore:
    """RES + ESS + load physical model for one microgrid."""

    config: dict[str, Any]
    delta_t: float
    profiles: HeterogeneousProfileModel

    def __post_init__(self) -> None:
        self.microgrid_id = str(self.config["id"])
        self.battery = BatteryModel.from_config(self.config["battery"], self.delta_t)

    def reset(self) -> None:
        self.battery.reset()

    def exogenous(self, row: pd.Series) -> dict[str, float]:
        return self.profiles.exogenous(self.microgrid_id, row)

    def plan_local(
        self,
        row: pd.Series,
        action: Mapping[str, Any],
        exogenous: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        current = dict(exogenous) if exogenous is not None else self.exogenous(row)
        battery = self.battery.map_action(float(action["battery"]))
        charge = float(battery["charge_requested_kw"])
        discharge = float(battery["discharge_requested_kw"])
        net = current["renewable_kw"] + discharge - current["load_kw"] - charge
        return {
            "microgrid_id": self.microgrid_id,
            **current,
            "battery_action": float(action["battery"]),
            "battery_charge_kw": charge,
            "battery_discharge_kw": discharge,
            "net_power_before_p2p_kw": float(net),
            "feasible_min_kw": float(battery["feasible_min_kw"]),
            "feasible_max_kw": float(battery["feasible_max_kw"]),
        }

    def commit(self, plan: Mapping[str, float]) -> dict[str, float]:
        return self.battery.commit(plan["battery_charge_kw"], plan["battery_discharge_kw"])
