"""Reconstruct MILP device schedules through the repository's physical models.

This validator checks industrial/HVAC/EV and ESS state transitions only.
It deliberately does NOT certify P2P market execution or equal actionability
of the oracle's joint energy flows in the local-first RL environment.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from components.battery import BatteryModel
from components.controllable_load import build_controllable_load
from optimization.common_constraints import DispatchData


def validate_device_trajectories(
    config: Mapping[str, Any],
    data: DispatchData,
    hourly: pd.DataFrame,
    *,
    atol: float = 1e-4,
) -> dict[str, Any]:
    if data.frame is None or len(data.frame) != data.horizon:
        raise ValueError("Replay requires the exact complete episode frame")
    dt = float(config["time"]["delta_t"])
    mismatches: list[dict[str, Any]] = []
    checked = 0

    def check(mg: str, t: int, variable: str, actual: float, expected: float) -> None:
        nonlocal checked
        checked += 1
        if not (np.isfinite(actual) and np.isfinite(expected)) or abs(actual - expected) > atol:
            mismatches.append({
                "microgrid": mg, "hour": t, "field": variable,
                "physical_replay": float(actual), "oracle": float(expected),
            })

    configs = {str(item["id"]): item for item in config["microgrids"]}
    for mg in data.agent_ids:
        rows = hourly.loc[hourly.microgrid == mg].sort_values("timestamp")
        if len(rows) != data.horizon:
            raise ValueError(f"Replay requires exactly {data.horizon} rows for {mg}")
        dr = build_controllable_load(
            configs[mg]["dr"], delta_t=dt, episode_length=data.horizon,
        )
        dr.reset(data.frame, np.asarray(data.load_kw[mg], dtype=float))
        battery = BatteryModel.from_config(configs[mg]["battery"], dt)
        battery.reset()
        kind = str(configs[mg]["dr"]["type"])
        for t, (_, row) in enumerate(rows.iterrows()):
            target_ch = float(row["battery_charge_kw"])
            target_dis = float(row["battery_discharge_kw"])
            check(mg, t, "ESS_simultaneous_charge_discharge", target_ch*target_dis, 0.0)
            ch_limit = battery.available_charge_kw()
            dis_limit = battery.available_discharge_kw()
            raw_ess = (target_dis / dis_limit if target_dis > 1e-10 and dis_limit > 0.0
                       else -target_ch / ch_limit if target_ch > 1e-10 and ch_limit > 0.0
                       else 0.0)
            requested = battery.map_action(raw_ess)
            check(mg, t, "ESS_charge_kW",
                  float(requested["charge_requested_kw"]), target_ch)
            check(mg, t, "ESS_discharge_kW",
                  float(requested["discharge_requested_kw"]), target_dis)
            committed = battery.commit(
                float(requested["charge_requested_kw"]),
                float(requested["discharge_requested_kw"]),
            )
            check(mg, t, "ESS_SOC", float(committed["soc"]), float(row["soc"]))

            if kind == "industrial_mill":
                duty = float(row["mill_duty"])
                raw_dr = 2.0*duty-1.0
            elif kind == "commercial_hvac":
                heat = float(row["hvac_heating_kw"])
                cool = float(row["hvac_cooling_kw"])
                raw_dr = (heat / dr.heat_max_kw if heat > 0.0
                          else -cool / dr.cool_max_kw if cool > 0.0 else 0.0)
                check(mg, t, "HVAC_simultaneous_heat_cool", heat*cool, 0.0)
            elif kind == "ev_charging":
                raw_dr = 2.0*float(row["flexible_load_kw"])/dr.max_charge_kw-1.0
            else:
                raise ValueError(f"Unknown oracle DR type: {kind}")
            plan = dr.plan(float(raw_dr), t)
            check(mg, t, "DR_power_kW", float(plan["flexible_load_kw"]),
                  float(row["flexible_load_kw"]))
            if float(row["unmet_load_kw"]) > atol:
                mismatches.append({"microgrid": mg, "hour": t,
                    "field": "unmet_load_requires_delivered_fraction_replay"})
            dr.commit(plan, delivered_fraction=1.0)
            if kind == "industrial_mill":
                check(mg, t, "silo_t", float(dr.silo_t), float(row["silo_after_t"]))
            elif kind == "commercial_hvac":
                check(mg, t, "HVAC_air", float(dr.t_air), float(row["hvac_air_after_c"]))
                check(mg, t, "HVAC_mass", float(dr.t_mass), float(row["hvac_mass_after_c"]))
            elif kind == "ev_charging":
                check(mg, t, "EV_SOC", float(dr.soc), float(row["ev_soc_after"]))

    return {
        "device_replay_passed": not mismatches,
        "compared_values": checked,
        "max_listed_mismatches": 20,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:20],
        "p2p_and_grid_clearing_replayed": False,
    }
