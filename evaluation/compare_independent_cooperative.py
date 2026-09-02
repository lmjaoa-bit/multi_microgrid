from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from optimization.cooperative_dispatch import solve_cooperative_dispatch
from optimization.independent_dispatch import solve_independent_dispatch


def compare_dispatches(config: str | Path | Mapping[str, Any], split: str = "test",
                       day: str | None = None) -> tuple[dict[str, float], pd.DataFrame]:
    independent = solve_independent_dispatch(config, split, day)
    cooperative = solve_cooperative_dispatch(config, split, day)
    local = independent.trajectories.rename(columns={
        "battery_charge_kw": "local_battery_charge_kw", "battery_discharge_kw": "local_battery_discharge_kw",
        "soc": "local_soc", "grid_import_kw": "local_grid_import_kw", "grid_export_kw": "local_grid_export_kw"})
    coop = cooperative.trajectories.rename(columns={
        "battery_charge_kw": "cooperative_battery_charge_kw",
        "battery_discharge_kw": "cooperative_battery_discharge_kw", "soc": "cooperative_soc",
        "grid_import_kw": "cooperative_grid_import_kw", "grid_export_kw": "cooperative_grid_export_kw"})
    columns = ["timestamp", "microgrid", "cooperative_battery_charge_kw",
               "cooperative_battery_discharge_kw", "cooperative_soc", "cooperative_grid_import_kw",
               "cooperative_grid_export_kw", "p2p_import_kw", "p2p_export_kw"]
    local_columns = ["timestamp", "microgrid", "local_battery_charge_kw",
                     "local_battery_discharge_kw", "local_soc", "local_grid_import_kw",
                     "local_grid_export_kw"]
    comparison = local[local_columns].merge(coop[columns], on=["timestamp", "microgrid"])
    comparison["coordination_battery_action_delta_kw"] = (
        comparison["cooperative_battery_discharge_kw"] - comparison["cooperative_battery_charge_kw"]
        - comparison["local_battery_discharge_kw"] + comparison["local_battery_charge_kw"])
    return {"independent_total_profit": independent.total_profit,
            "cooperative_total_profit": cooperative.total_profit,
            "cooperation_gain": cooperative.total_profit - independent.total_profit}, comparison
