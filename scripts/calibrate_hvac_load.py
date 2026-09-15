from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.commercial_hvac import CommercialHVACDR
from core.data_manager import DataManager
from core.profiles import HeterogeneousProfileModel


def _state_is_occupied(timestamp: pd.Timestamp, occupied_hours: set[int]) -> bool:
    return timestamp.dayofweek < 5 and timestamp.hour in occupied_hours


def _solve_reference_day(
    dr_cfg: dict[str, Any],
    frame: pd.DataFrame,
    base_load_kw: np.ndarray,
    *,
    delta_t: float,
    episode_length: int,
) -> dict[str, Any]:
    """Solve one no-price reference thermostat day as a linear program.

    The calibration controller uses the approved narrow reference bands
    (22--24 C occupied, 18--28 C unoccupied), the same 2R2C dynamics and heat
    pump limits as the RL environment, and the same terminal thermal band. Its
    objective is only minimum HVAC electrical energy; there is no tariff signal,
    so the result is a reproducible physical-load calibration rather than a DR
    schedule. One LP per natural day is much faster than replaying the online
    terminal-reachability projector at every hour.
    """
    import highspy

    hvac = CommercialHVACDR(dr_cfg, delta_t=delta_t, episode_length=episode_length)
    hvac.reset(frame, base_load_kw)
    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    continuous = highspy.HighsVarType.kContinuous

    air: dict[int, Any] = {}
    mass: dict[int, Any] = {}
    heat: dict[int, Any] = {}
    cool: dict[int, Any] = {}
    air[0] = model.addVariable(
        lb=hvac.initial_t_air, ub=hvac.initial_t_air, type=continuous, name="t_air[0]"
    )
    mass[0] = model.addVariable(
        lb=hvac.initial_t_mass, ub=hvac.initial_t_mass, type=continuous, name="t_mass[0]"
    )

    for t in range(episode_length):
        air[t + 1] = model.addVariable(lb=-100.0, ub=100.0, type=continuous, name=f"t_air[{t+1}]")
        mass[t + 1] = model.addVariable(lb=-100.0, ub=100.0, type=continuous, name=f"t_mass[{t+1}]")
        heat[t] = model.addVariable(
            lb=0.0,
            ub=hvac.heat_max_kw,
            obj=delta_t,
            type=continuous,
            name=f"heat_kw[{t}]",
        )
        cool[t] = model.addVariable(
            lb=0.0,
            ub=hvac.cool_max_kw,
            obj=delta_t,
            type=continuous,
            name=f"cool_kw[{t}]",
        )

        timestamp = pd.Timestamp(frame.iloc[t]["timestamp"])
        t_out = float(frame.iloc[t][hvac.temperature_column])
        ghi = max(0.0, float(frame.iloc[t][hvac.ghi_column]))
        interval_occupied = _state_is_occupied(timestamp, hvac.occupied_hours)
        q_solar = hvac.solar_gain * ghi
        q_air = hvac.solar_air_fraction * q_solar
        q_mass = (1.0 - hvac.solar_air_fraction) * q_solar
        q_internal = float(base_load_kw[t]) * (
            hvac.internal_occ_fraction if interval_occupied else hvac.internal_unocc_fraction
        )

        a_air = 1.0 - delta_t / hvac.c_air * (1.0 / hvac.r_oa + 1.0 / hvac.r_am)
        b_mass = delta_t / hvac.c_air * (1.0 / hvac.r_am)
        constant_air = delta_t / hvac.c_air * (
            t_out / hvac.r_oa + q_internal + q_air
        )
        mass_from_air = delta_t / hvac.c_mass * (1.0 / hvac.r_am)
        mass_self = 1.0 - delta_t / hvac.c_mass * (1.0 / hvac.r_am)
        constant_mass = delta_t / hvac.c_mass * q_mass

        model.addConstr(
            air[t + 1]
            == a_air * air[t]
            + b_mass * mass[t]
            + delta_t / hvac.c_air * (
                hvac.cop_heat * heat[t] - hvac.cop_cool * cool[t]
            )
            + constant_air
        )
        model.addConstr(
            mass[t + 1]
            == mass_from_air * air[t]
            + mass_self * mass[t]
            + constant_mass
        )

        next_timestamp = timestamp + pd.Timedelta(hours=1)
        occupied = _state_is_occupied(next_timestamp, hvac.occupied_hours)
        reference_low, reference_high = (22.0, 24.0) if occupied else (18.0, 28.0)
        model.addConstr(air[t + 1] >= reference_low)
        model.addConstr(air[t + 1] <= reference_high)

    terminal = episode_length
    model.addConstr(air[terminal] >= hvac.initial_t_air - hvac.terminal_air_tolerance)
    model.addConstr(air[terminal] <= hvac.initial_t_air + hvac.terminal_air_tolerance)
    model.addConstr(mass[terminal] >= hvac.initial_t_mass - hvac.terminal_mass_tolerance)
    model.addConstr(mass[terminal] <= hvac.initial_t_mass + hvac.terminal_mass_tolerance)
    model.minimize()
    solution = model.getSolution()
    if not solution.value_valid:
        return {
            "feasible": False,
            "initial_t_air_c": hvac.initial_t_air,
            "initial_t_mass_c": hvac.initial_t_mass,
        }

    heating = np.asarray([float(model.val(heat[t])) for t in range(episode_length)])
    cooling = np.asarray([float(model.val(cool[t])) for t in range(episode_length)])
    air_values = np.asarray([float(model.val(air[t])) for t in range(episode_length + 1)])
    mass_values = np.asarray([float(model.val(mass[t])) for t in range(episode_length + 1)])
    if np.any((heating > 1e-7) & (cooling > 1e-7)):
        raise RuntimeError("Reference LP produced simultaneous heating and cooling")
    return {
        "feasible": True,
        "heating_kw": heating,
        "cooling_kw": cooling,
        "hvac_kw": heating + cooling,
        "air_c": air_values,
        "mass_c": mass_values,
        "initial_t_air_c": hvac.initial_t_air,
        "initial_t_mass_c": hvac.initial_t_mass,
    }


def evaluate_target(config: dict, target_base_kw: float) -> dict[str, float | int | bool]:
    cfg = deepcopy(config)
    mg2 = next(item for item in cfg["microgrids"] if str(item["id"]) == "mg2")
    mg2["load_profile"]["target_mean_kw"] = float(target_base_kw)
    delta_t = float(cfg["time"].get("delta_t", 1.0))
    episode_length = int(cfg["time"].get("episode_length", 24))

    managers = {
        split: DataManager(cfg, split=split, seed=int(cfg.get("env", {}).get("seed", 42)))
        for split in ("train", "val", "test")
    }
    profiles = HeterogeneousProfileModel(cfg, managers["train"].df)
    dr_cfg = deepcopy(mg2["dr"])

    base_kwh = 0.0
    hvac_kwh = 0.0
    heating_kwh = 0.0
    cooling_kwh = 0.0
    cap_hit_hours = 0
    hours = 0
    days = 0
    infeasible_days = 0
    max_hvac_kw = 0.0
    hvac_samples: list[float] = []

    for manager in managers.values():
        for day in manager.valid_days:
            frame = manager.get_day(day)
            base = profiles.load_series("mg2", frame)
            result = _solve_reference_day(
                dr_cfg,
                frame,
                base,
                delta_t=delta_t,
                episode_length=episode_length,
            )
            days += 1
            hours += episode_length
            base_kwh += float(np.sum(base) * delta_t)
            if not bool(result["feasible"]):
                infeasible_days += 1
                continue
            heating = np.asarray(result["heating_kw"], dtype=np.float64)
            cooling = np.asarray(result["cooling_kw"], dtype=np.float64)
            hvac_power = np.asarray(result["hvac_kw"], dtype=np.float64)
            heating_kwh += float(np.sum(heating) * delta_t)
            cooling_kwh += float(np.sum(cooling) * delta_t)
            hvac_kwh += float(np.sum(hvac_power) * delta_t)
            hvac_samples.extend(hvac_power.tolist())
            max_hvac_kw = max(max_hvac_kw, float(np.max(hvac_power, initial=0.0)))
            cap_hit_hours += int(np.count_nonzero(
                hvac_power >= max(float(dr_cfg["heating_max_kw"]), float(dr_cfg["cooling_max_kw"])) - 1e-7
            ))

    total_kwh = base_kwh + hvac_kwh
    array = np.asarray(hvac_samples, dtype=np.float64)
    valid_hours = max(hours - infeasible_days * episode_length, 1)
    result: dict[str, float | int | bool] = {
        "target_base_mean_kw": float(target_base_kw),
        "days": int(days),
        "hours": int(hours),
        "reference_infeasible_days": int(infeasible_days),
        "reference_feasible": bool(infeasible_days == 0),
        "base_energy_mwh": float(base_kwh / 1000.0),
        "hvac_energy_mwh": float(hvac_kwh / 1000.0),
        "heating_energy_mwh": float(heating_kwh / 1000.0),
        "cooling_energy_mwh": float(cooling_kwh / 1000.0),
        "total_energy_mwh": float(total_kwh / 1000.0),
        "annualized_base_mean_kw": float(base_kwh / max(hours, 1)),
        "annualized_hvac_mean_kw": float(hvac_kwh / valid_hours),
        "annualized_total_mean_kw": float(total_kwh / max(hours, 1)),
        "hvac_p95_kw": float(np.percentile(array, 95)) if array.size else 0.0,
        "hvac_max_kw": float(max_hvac_kw),
        "hvac_cap_hit_hours": int(cap_hit_hours),
    }
    return result


def solve_base_target(config: dict, desired_total_kw: float, initial: float) -> dict:
    target = float(initial)
    history = []
    for _ in range(5):
        result = evaluate_target(config, target)
        history.append(result)
        if not bool(result["reference_feasible"]):
            break
        error = float(desired_total_kw) - float(result["annualized_total_mean_kw"])
        if abs(error) <= 0.05:
            break
        target = max(1.0, target + error)
    return {
        "desired_total_mean_kw": float(desired_total_kw),
        "recommended_base_target_kw": float(target),
        "iterations": history,
        "final": history[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate MG2 non-HVAC base load against real T2M")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/cooperative.yaml")
    parser.add_argument("--desired-total-kw", type=float, default=80.0)
    parser.add_argument("--solve", action="store_true", help="Iterate base target to the desired annual total")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_project_root"] = str(config_path.parent.parent)
    mg2 = next(item for item in config["microgrids"] if str(item["id"]) == "mg2")
    initial = float(mg2["load_profile"]["target_mean_kw"])
    result = (
        solve_base_target(config, args.desired_total_kw, initial)
        if args.solve
        else evaluate_target(config, initial)
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if isinstance(result, dict) and not bool(
        result.get("final", result).get("reference_feasible", True)
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
