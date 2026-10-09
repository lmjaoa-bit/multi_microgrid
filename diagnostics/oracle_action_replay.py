"""Evaluation-only inverse device mapping and actual two-action environment replay.

Replay profit is an attainable value, not a certified action-space upper bound.
Oracle market flows and device states are comparison targets only: they are
never injected into the environment.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from envs.multi_microgrid_env import MultiMicrogridEnv
from utils.metrics_collector import EpisodeMetricsCollector


TOLERANCE = 1e-6


def _power(row: Mapping[str, Any], name: str) -> float:
    value = float(row[name])
    if not np.isfinite(value) or value < -TOLERANCE:
        raise ValueError(f"Invalid nonnegative oracle target {name}: {value}")
    return max(0., value)


def inverse_oracle_actions(
    env: MultiMicrogridEnv,
    targets: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Map targets using the live battery limits and each DR model's raw mapping."""
    if env.action_dim != 2 or not env.dr_enabled:
        raise ValueError("Oracle action replay requires single_dispatch_dr two-action controls")
    if set(targets) != set(env.agent_ids):
        raise ValueError("Targets must contain exactly the environment's agent IDs")
    actions, diagnostics = {}, []
    for mg in env.agent_ids:
        row = targets[mg]
        battery = env.cores[mg].battery
        charge = _power(row, "battery_charge_kw")
        discharge = _power(row, "battery_discharge_kw")
        if charge > TOLERANCE and discharge > TOLERANCE:
            raise ValueError("Oracle battery cannot charge and discharge simultaneously")
        charge_limit = battery.available_charge_kw()
        discharge_limit = battery.available_discharge_kw()
        if discharge > TOLERANCE:
            dispatch = discharge / discharge_limit if discharge_limit > 1e-12 else 1.
        elif charge > TOLERANCE:
            dispatch = -charge / charge_limit if charge_limit > 1e-12 else -1.
        else:
            dispatch = 0.
        dispatch = float(np.clip(dispatch, -1., 1.))
        requested = battery.map_action(dispatch)
        device = env.dr_devices[mg]
        kind = str(env.cfg_by_id[mg]["dr"]["type"])
        if kind == "industrial_mill":
            duty = float(row["mill_duty"])
            if not np.isfinite(duty) or not -TOLERANCE <= duty <= 1. + TOLERANCE:
                raise ValueError("Invalid oracle mill duty")
            raw_dr = 2. * duty - 1.
            target_dr = float(device.rated_power_kw) * max(0., duty)
        elif kind == "commercial_hvac":
            heat, cool = _power(row, "hvac_heating_kw"), _power(row, "hvac_cooling_kw")
            if heat > TOLERANCE and cool > TOLERANCE:
                raise ValueError("Oracle HVAC cannot heat and cool simultaneously")
            raw_dr = heat / device.heat_max_kw if heat > TOLERANCE else -cool / device.cool_max_kw
            target_dr = heat + cool
        elif kind == "ev_charging":
            target_dr = _power(row, "flexible_load_kw")
            if device.cohort_mode:
                low, high = device.feasible_bounds(env.t)
                raw_dr = 2. * (target_dr - low) / (high - low) - 1. if high > low + 1e-12 else -1.
            else:
                # Non-cohort EV uses nameplate scaling followed by clipping,
                # whereas cohort EV maps the full current feasible interval.
                raw_dr = 2. * target_dr / device.max_charge_kw - 1.
        else:
            raise ValueError(f"Unsupported DR type: {kind}")
        raw_dr = float(np.clip(raw_dr, -1., 1.))
        actions[mg] = np.array([dispatch, raw_dr], dtype=np.float64)
        requested_net = float(requested["discharge_requested_kw"] - requested["charge_requested_kw"])
        error = requested_net - (discharge - charge)
        diagnostics.append({
            "microgrid": mg, "timestep": int(env.t),
            "soc_before": float(battery.soc),
            "available_charge_kw": float(charge_limit),
            "available_discharge_kw": float(discharge_limit),
            "oracle_charge_kw": charge, "oracle_discharge_kw": discharge,
            "oracle_flexible_load_kw": target_dr,
            "raw_dispatch_action": dispatch, "raw_dr_action": raw_dr,
            "requested_charge_kw": float(requested["charge_requested_kw"]),
            "requested_discharge_kw": float(requested["discharge_requested_kw"]),
            "dispatch_request_error_kw": error,
            "dispatch_target_clipped": bool(abs(error) > TOLERANCE),
        })
    return actions, diagnostics


def replay_oracle_trajectory(
    env: MultiMicrogridEnv,
    trajectory: pd.DataFrame,
    *,
    day: str,
    seed: int,
    forecast_noise: bool,
) -> dict[str, Any]:
    """Execute one complete natural day with no oracle-state restoration."""
    required = {"timestamp", "microgrid", "base_load_kw", "renewable_kw",
                "battery_charge_kw", "battery_discharge_kw", "soc", "flexible_load_kw"}
    if not required.issubset(trajectory.columns):
        raise ValueError(f"Oracle trajectory missing fields: {sorted(required - set(trajectory))}")
    targets = trajectory.copy()
    targets["timestamp"] = pd.to_datetime(targets["timestamp"])
    if targets.duplicated(["timestamp", "microgrid"]).any():
        raise ValueError("Duplicate oracle timestamp/agent targets")
    env.reset(day=day, seed=seed, deterministic=True, forecast_noise=forecast_noise)
    times = pd.to_datetime(env.episode_df["timestamp"])
    expected = {(stamp, mg) for stamp in times for mg in env.agent_ids}
    actual = set(zip(targets["timestamp"], targets["microgrid"]))
    if len(targets) != len(expected) or actual != expected:
        raise ValueError("Oracle targets must cover exactly the complete environment day")
    by_key = targets.set_index(["timestamp", "microgrid"])
    # Compare all 24 hours before taking the first step, including exogenous
    # demand and renewables; forecast noise affects observations, not physics.
    for t, stamp in enumerate(times):
        exogenous = env._exogenous_at(t)
        for mg in env.agent_ids:
            row = by_key.loc[(stamp, mg)]
            for source, target in (("base_load_kw", "load_kw"), ("renewable_kw", "renewable_kw")):
                value = float(row[source])
                if not np.isfinite(value) or abs(value - float(exogenous[mg][target])) > TOLERANCE:
                    raise ValueError(f"Oracle scenario mismatch: {stamp} {mg} {source}")
    collector = EpisodeMetricsCollector(env, capture_hourly=True)
    action_rows = []
    for t, stamp in enumerate(times):
        rows = {mg: by_key.loc[(stamp, mg)].to_dict() for mg in env.agent_ids}
        actions, diagnostics = inverse_oracle_actions(env, rows)
        _, rewards, done, info = env.step(actions)
        collector.record_step(actions, info, rewards)
        if bool(done) != (t == len(times) - 1):
            raise RuntimeError("Unexpected replay episode termination")
        hourly = collector.hourly_rows[-1]
        for item in diagnostics:
            mg = item["microgrid"]
            row = rows[mg]
            item.update({
                "timestamp": str(stamp),
                "actual_charge_kw": float(hourly[f"{mg}_battery_charge_kw"]),
                "actual_discharge_kw": float(hourly[f"{mg}_battery_discharge_kw"]),
                "actual_soc": float(hourly[f"{mg}_soc"]),
                "oracle_soc": float(row["soc"]),
                "soc_error": float(hourly[f"{mg}_soc"]) - float(row["soc"]),
                "flexible_load_error_kw": float(hourly[f"{mg}_flexible_load_kw"]) - float(row["flexible_load_kw"]),
                "charge_error_kw": float(hourly[f"{mg}_battery_charge_kw"]) - float(row["battery_charge_kw"]),
                "discharge_error_kw": float(hourly[f"{mg}_battery_discharge_kw"]) - float(row["battery_discharge_kw"]),
            })
            # Cohort allocation is internal to the existing DR rule. Record
            # deviations instead of injecting per-group oracle decisions.
            for key, value in row.items():
                if key.startswith("ev_") and key.endswith("_charge_kw") and pd.notna(value):
                    replay_key = f"{mg}_{key.removesuffix('_charge_kw')}_delivered_kw"
                    if replay_key in hourly:
                        item[f"{key}_error"] = float(hourly[replay_key]) - float(value)
        action_rows.extend(diagnostics)
    metrics = collector.finalize_episode()
    metrics.update(date=day, test_seed=int(seed))
    hourly_profit = sum(float(row[f"{mg}_hourly_profit"])
                        for row in collector.hourly_rows for mg in env.agent_ids)
    if abs(hourly_profit - float(metrics["team_profit"])) > TOLERANCE:
        raise RuntimeError("Replay hourly and daily accounting profits do not reconcile")
    if not metrics["constraint_statistics"]["physical_feasible_episode"]:
        raise RuntimeError("Oracle action replay violated physical constraints")
    return {"metrics": metrics, "hourly_rows": collector.hourly_rows,
            "action_diagnostics": action_rows, "state_injection": False,
            "certified_action_space_upper_bound": False}
