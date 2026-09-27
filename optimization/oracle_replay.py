"""Reconstruct MILP device schedules through the repository's physical models.

This validator checks industrial/HVAC/EV and ESS state transitions only.
It deliberately does NOT certify P2P market execution or equal actionability
of the oracle's free-allocation joint energy flows in the deterministic
renewable-first proportional-clearing RL environment.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from components.battery import BatteryModel
from components.controllable_load import build_controllable_load
from optimization.common_constraints import DispatchData, DispatchResult


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
                target_power = float(row["flexible_load_kw"])
                low, high = dr.feasible_bounds(t)
                if high > low + 1e-12:
                    raw_dr = 2.0 * (target_power - low) / (high - low) - 1.0
                else:
                    raw_dr = -1.0
                raw_dr = float(np.clip(raw_dr, -1.0, 1.0))
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


def compare_oracle_market_replay(
    oracle_result: DispatchResult,
    replay_hourly_rows,
    *,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Compare source/use market flows with a saved environment replay.

    The full-information oracle deliberately retains free allocation inside the
    coalition.  This function therefore reports a measurable allocation replay
    gap rather than forcing the oracle flow pattern to match the deterministic
    proportional environment clearing.
    """
    if isinstance(replay_hourly_rows, pd.DataFrame):
        replay_rows = replay_hourly_rows.to_dict(orient="records")
    else:
        replay_rows = [dict(row) for row in replay_hourly_rows]
    replay_by_time = {
        str(row.get("timestamp")): row for row in replay_rows
        if row.get("timestamp") is not None
    }
    if not replay_by_time:
        raise ValueError("Replay comparison requires timestamped hourly rows")

    oracle = oracle_result.trajectories
    if oracle.empty:
        raise ValueError("Oracle result contains no trajectories")

    mappings = {
        "renewable_to_local_load_kw": "renewable_to_local_load_kw",
        "renewable_to_local_charge_kw": "renewable_to_battery_kw",
        "ess_to_local_load_kw": "ess_to_local_load_kw",
        "grid_to_load_kw": "grid_to_load_kw",
        "grid_to_charge_kw": "grid_to_battery_kw",
        "renewable_peer_load_export_kw": "renewable_peer_load_export_kw",
        "renewable_peer_charge_export_kw": "renewable_peer_charge_export_kw",
        "ess_peer_load_export_kw": "ess_p2p_sell_kw",
        "grid_import_kw": "grid_import_kw",
        "grid_export_kw": "grid_export_kw",
    }
    errors: dict[str, list[float]] = {name: [] for name in mappings}
    missing_fields: set[str] = set()
    compared = 0
    for _, oracle_row in oracle.iterrows():
        timestamp = str(oracle_row["timestamp"])
        mg = str(oracle_row["microgrid"])
        replay_row = replay_by_time.get(timestamp)
        if replay_row is None:
            missing_fields.add(f"timestamp:{timestamp}")
            continue
        for oracle_field, replay_suffix in mappings.items():
            replay_field = f"{mg}_{replay_suffix}"
            if oracle_field not in oracle_row or replay_field not in replay_row:
                missing_fields.add(replay_field)
                continue
            expected = float(oracle_row[oracle_field])
            actual = float(replay_row[replay_field])
            if not (np.isfinite(expected) and np.isfinite(actual)):
                missing_fields.add(replay_field)
                continue
            errors[oracle_field].append(actual - expected)
            compared += 1

    field_errors: dict[str, dict[str, float | int | None]] = {}
    max_abs = 0.0
    for field, values in errors.items():
        array = np.asarray(values, dtype=np.float64)
        if array.size:
            field_max = float(np.max(np.abs(array)))
            max_abs = max(max_abs, field_max)
            field_errors[field] = {
                "count": int(array.size),
                "mean_error_kw": float(array.mean()),
                "mean_abs_error_kw": float(np.mean(np.abs(array))),
                "max_abs_error_kw": field_max,
            }
        else:
            field_errors[field] = {
                "count": 0,
                "mean_error_kw": None,
                "mean_abs_error_kw": None,
                "max_abs_error_kw": None,
            }

    replay_profit = 0.0
    profit_fields_found = 0
    for row in replay_rows:
        for key, value in row.items():
            if str(key).endswith("_hourly_profit"):
                replay_profit += float(value)
                profit_fields_found += 1
    profit_gap = (
        float(replay_profit - oracle_result.total_profit)
        if profit_fields_found else None
    )
    exactly = bool(
        not missing_fields
        and compared > 0
        and max_abs <= float(atol)
        and profit_gap is not None
        and abs(profit_gap) <= float(atol)
    )
    return {
        "allocation_mode": (
            (oracle_result.metadata or {}).get("allocation")
            or "unknown"
        ),
        "compared_values": int(compared),
        "missing_fields": sorted(missing_fields),
        "source_flow_errors": field_errors,
        "allocation_replay_gap_max_abs_kw": float(max_abs),
        "oracle_profit": float(oracle_result.total_profit),
        "replay_profit": float(replay_profit) if profit_fields_found else None,
        "profit_gap": profit_gap,
        "exactly_replayable": exactly,
        "status": (
            "exact_replay"
            if exactly
            else "oracle_not_exactly_replayable"
        ),
    }
