from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from components.commercial_hvac import CommercialHVACDR
from core.data_manager import DataManager
from core.profiles import HeterogeneousProfileModel


def _reference_raw_action(
    hvac: CommercialHVACDR,
    frame: pd.DataFrame,
    t: int,
    *,
    occupied_band: tuple[float, float] = (22.0, 24.0),
    unoccupied_band: tuple[float, float] = (18.0, 28.0),
) -> float:
    """Greedy no-price thermostat request before hard feasibility projection.

    Internal gains and weather are taken from interval ``t``.  The thermostat
    request targets the band that applies to the state at the next clock-time
    boundary, matching the environment's state-timing convention.  The device's
    own ``plan`` then projects this request onto hard comfort and daily terminal
    reachability constraints.
    """
    next_air, _ = hvac._next_state(hvac.t_air, hvac.t_mass, 0.0, t)
    next_ts = pd.Timestamp(frame.iloc[t]["timestamp"]) + pd.Timedelta(hours=1)
    occupied = next_ts.dayofweek < 5 and next_ts.hour in hvac.occupied_hours
    low, high = occupied_band if occupied else unoccupied_band
    if next_air < low:
        q = (low - next_air) * hvac.c_air / hvac.delta_t
    elif next_air > high:
        q = (high - next_air) * hvac.c_air / hvac.delta_t
    else:
        q = 0.0
    q = float(np.clip(q, hvac.q_min_kw, hvac.q_max_kw))
    power = hvac._q_to_signed_electric(q)
    denominator = hvac.heat_max_kw if power >= 0.0 else hvac.cool_max_kw
    return float(np.clip(power / denominator, -1.0, 1.0))


def evaluate_target(config: dict, target_base_kw: float) -> dict[str, float | int | bool]:
    cfg = deepcopy(config)
    mg2 = next(item for item in cfg["microgrids"] if str(item["id"]) == "mg2")
    mg2["load_profile"]["target_mean_kw"] = float(target_base_kw)

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
    occupied_hours = 0
    calibration_band_violation_hours = 0
    hard_comfort_infeasible_hours = 0
    terminal_infeasible_days = 0
    cap_hit_hours = 0
    hours = 0
    days = 0
    max_hvac_kw = 0.0
    hvac_samples: list[float] = []

    for split, manager in managers.items():
        for day in manager.valid_days:
            frame = manager.get_day(day)
            base = profiles.load_series("mg2", frame)
            hvac = CommercialHVACDR(
                dr_cfg,
                delta_t=float(cfg["time"].get("delta_t", 1.0)),
                episode_length=int(cfg["time"].get("episode_length", 24)),
            )
            hvac.reset(frame, base)
            day_terminal_flag = False
            for t in range(24):
                raw = _reference_raw_action(hvac, frame, t)
                plan = hvac.plan(raw, t)
                power = float(plan["flexible_load_kw"])
                heating = float(plan["heating_power_kw"])
                cooling = float(plan["cooling_power_kw"])
                base_kwh += float(base[t]) * hvac.delta_t
                hvac_kwh += power * hvac.delta_t
                heating_kwh += heating * hvac.delta_t
                cooling_kwh += cooling * hvac.delta_t
                max_hvac_kw = max(max_hvac_kw, power)
                hvac_samples.append(power)
                cap_hit_hours += int(power >= max(hvac.heat_max_kw, hvac.cool_max_kw) - 1e-7)
                hard_comfort_infeasible_hours += int(bool(plan["comfort_infeasible"]))
                hvac.commit(plan)

                state_ts = pd.Timestamp(frame.iloc[t]["timestamp"]) + pd.Timedelta(hours=1)
                if state_ts.hour != 0 or t < 23:
                    occupied = state_ts.dayofweek < 5 and state_ts.hour in hvac.occupied_hours
                    low, high = (22.0, 24.0) if occupied else (18.0, 28.0)
                    occupied_hours += int(occupied)
                    calibration_band_violation_hours += int(
                        hvac.t_air < low - 1e-6 or hvac.t_air > high + 1e-6
                    )
                day_terminal_flag |= bool(plan["terminal_thermal_infeasible"])
                hours += 1
            day_terminal_flag |= bool(hvac.terminal_thermal_infeasible)
            day_terminal_flag |= bool(
                abs(hvac.t_air - hvac.initial_t_air) > hvac.terminal_air_tolerance + 1e-6
                or abs(hvac.t_mass - hvac.initial_t_mass) > hvac.terminal_mass_tolerance + 1e-6
            )
            terminal_infeasible_days += int(day_terminal_flag)
            days += 1

    total_kwh = base_kwh + hvac_kwh
    array = np.asarray(hvac_samples, dtype=np.float64)
    return {
        "target_base_mean_kw": float(target_base_kw),
        "days": int(days),
        "hours": int(hours),
        "base_energy_mwh": float(base_kwh / 1000.0),
        "hvac_energy_mwh": float(hvac_kwh / 1000.0),
        "heating_energy_mwh": float(heating_kwh / 1000.0),
        "cooling_energy_mwh": float(cooling_kwh / 1000.0),
        "total_energy_mwh": float(total_kwh / 1000.0),
        "annualized_base_mean_kw": float(base_kwh / max(hours, 1)),
        "annualized_hvac_mean_kw": float(hvac_kwh / max(hours, 1)),
        "annualized_total_mean_kw": float(total_kwh / max(hours, 1)),
        "hvac_p95_kw": float(np.percentile(array, 95)) if array.size else 0.0,
        "hvac_max_kw": float(max_hvac_kw),
        "hvac_cap_hit_hours": int(cap_hit_hours),
        "hard_comfort_infeasible_hours": int(hard_comfort_infeasible_hours),
        "terminal_infeasible_days": int(terminal_infeasible_days),
        "calibration_band_violation_hours": int(calibration_band_violation_hours),
        "occupied_hours": int(occupied_hours),
    }


def solve_base_target(config: dict, desired_total_kw: float, initial: float) -> dict:
    target = float(initial)
    history = []
    for _ in range(4):
        result = evaluate_target(config, target)
        history.append(result)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
