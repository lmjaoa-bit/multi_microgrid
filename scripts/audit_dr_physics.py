from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.multi_microgrid_env import MultiMicrogridEnv


HARD_TOL = 1e-6
BALANCE_TOL = 1e-7


@dataclass(frozen=True)
class DayCandidate:
    split: str
    day: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "day", pd.Timestamp(self.day).normalize())


def select_monthly_representatives(
    candidates: Iterable[DayCandidate], *, year: int
) -> list[DayCandidate]:
    """Choose one deterministic representative day per month.

    Odd months prefer a weekday and even months prefer a weekend.  Within that
    class the day closest to the 15th is selected.  This gives all-season
    coverage while deliberately exercising both HVAC occupancy regimes.
    """
    pool = [item for item in candidates if item.day.year == int(year)]
    result: list[DayCandidate] = []
    for month in range(1, 13):
        options = [item for item in pool if item.day.month == month]
        if not options:
            raise ValueError(f"No valid natural-day episode is available for month {month}")
        prefer_weekend = month % 2 == 0
        preferred = [
            item for item in options
            if bool(item.day.dayofweek >= 5) == prefer_weekend
        ]
        usable = preferred or options
        target = pd.Timestamp(year=year, month=month, day=15)
        usable.sort(
            key=lambda item: (
                abs((item.day - target).days),
                item.day,
                {"train": 0, "val": 1, "test": 2}.get(item.split, 9),
            )
        )
        result.append(usable[0])
    return result


def build_stress_actions(
    agent_ids: Sequence[str], *, action_dim: int, seed: int, step: int
) -> dict[str, np.ndarray]:
    """Seeded random actions with deterministic corner probes at five hours."""
    if int(action_dim) <= 0:
        raise ValueError("action_dim must be positive")
    sequence = np.random.SeedSequence([int(seed), int(step)])
    rng = np.random.default_rng(sequence)
    result = {
        str(mg): rng.uniform(-1.0, 1.0, size=int(action_dim)).astype(np.float64)
        for mg in agent_ids
    }
    corner_steps = (0, 6, 12, 18, 23)
    if int(step) in corner_steps:
        patterns = ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0))
        phase = corner_steps.index(int(step))
        for index, mg in enumerate(agent_ids):
            values = result[str(mg)]
            pattern = patterns[(phase + index) % len(patterns)]
            for dimension in range(min(2, int(action_dim))):
                values[dimension] = pattern[dimension]
    return result


def require_clean_audit(summary: Mapping[str, Any]) -> None:
    violations = int(summary.get("hard_violation_count", 0))
    if violations:
        raise RuntimeError(f"DR physics audit found {violations} hard violations")
    flags = list(summary.get("data_quality_flags", []))
    if flags:
        raise RuntimeError(f"DR physics audit detected data quality flags: {flags}")
    if not bool(summary.get("projection_exercised", False)):
        raise RuntimeError("DR physics audit did not exercise projection paths")


def _add_violation(day: dict[str, Any], kind: str, **context: Any) -> None:
    day["hard_violation_count"] += 1
    if len(day["hard_violations"]) < 100:
        clean = {"kind": kind}
        for key, value in context.items():
            if isinstance(value, np.generic):
                value = value.item()
            clean[str(key)] = value
        day["hard_violations"].append(clean)


def _comfort_bounds(env: MultiMicrogridEnv, state_index: int) -> tuple[float, float]:
    cfg = env.cfg_by_id["mg2"]["dr"]
    occupied = False
    if 0 <= int(state_index) < env.episode_length:
        timestamp = pd.Timestamp(env.episode_df.iloc[int(state_index)]["timestamp"])
        occupied = timestamp.dayofweek < 5 and timestamp.hour in {
            int(x) for x in cfg["occupied_hours"]
        }
    bounds = cfg["occupied_comfort_c"] if occupied else cfg["unoccupied_comfort_c"]
    return float(bounds[0]), float(bounds[1])


def _pair_usage_kwh(
    flows: Mapping[str, float], first: str, second: str
) -> float:
    return float(flows.get(f"{first}->{second}", 0.0)) + float(
        flows.get(f"{second}->{first}", 0.0)
    )


def audit_day(
    env: MultiMicrogridEnv,
    candidate: DayCandidate,
    *,
    seed: int,
) -> dict[str, Any]:
    observations, reset_info = env.reset(
        day=str(candidate.day.date()),
        seed=int(seed),
        deterministic=True,
        forecast_noise=False,
    )
    day: dict[str, Any] = {
        "split": candidate.split,
        "day": str(candidate.day.date()),
        "is_weekend": bool(candidate.day.dayofweek >= 5),
        "hard_violation_count": 0,
        "hard_violations": [],
        "ess_projection_steps": 0,
        "dr_projection_steps": 0,
        "battery_charge_curtailed_kwh": 0.0,
        "battery_charge_curtailed_steps": 0,
        "p2p_saturated_pair_steps": 0,
        "max_power_balance_residual_kw": 0.0,
        "max_p2p_pair_usage_kw": 0.0,
        "max_reward_spread": 0.0,
        "max_transfer_payment_imbalance": 0.0,
        "unmet_load_kwh": 0.0,
    }
    if reset_info.get("data_quality_flags"):
        _add_violation(day, "data_quality", flags=list(reset_info["data_quality_flags"]))
    if not all(np.isfinite(value).all() for value in observations.values()):
        _add_violation(day, "nonfinite_reset_observation")

    initial_hvac = env.dr_devices["mg2"].state(0)
    initial_t_air = float(initial_hvac["t_air_c"])
    initial_t_mass = float(initial_hvac["t_mass_c"])
    pair_capacity_kw = float(env.config["p2p"]["pair_capacity_kw"])
    dt = float(env.delta_t)
    ids = list(env.agent_ids)
    done = False
    last_info: dict[str, Any] | None = None

    for step in range(env.episode_length):
        actions = build_stress_actions(
            ids,
            action_dim=env.action_dim,
            seed=int(seed),
            step=step,
        )
        observations, rewards, done, info = env.step(actions)
        last_info = info
        if not all(np.isfinite(value).all() for value in observations.values()):
            _add_violation(day, "nonfinite_observation", step=step)
        reward_values = np.asarray(list(rewards.values()), dtype=np.float64)
        if not np.isfinite(reward_values).all():
            _add_violation(day, "nonfinite_reward", step=step)
        reward_spread = float(np.ptp(reward_values)) if reward_values.size else 0.0
        day["max_reward_spread"] = max(day["max_reward_spread"], reward_spread)
        if reward_spread > HARD_TOL:
            _add_violation(day, "team_reward_mismatch", step=step, spread=reward_spread)

        transfer_error = abs(float(info["p2p"]["transfer_payment_sum"]))
        day["max_transfer_payment_imbalance"] = max(
            day["max_transfer_payment_imbalance"], transfer_error
        )
        if transfer_error > BALANCE_TOL:
            _add_violation(day, "p2p_transfer_imbalance", step=step, error=transfer_error)

        flows = info["p2p"]["pair_flows_kwh"]
        for i, first in enumerate(ids):
            for second in ids[i + 1:]:
                energy = _pair_usage_kwh(flows, first, second)
                usage_kw = energy / max(dt, 1e-12)
                day["max_p2p_pair_usage_kw"] = max(day["max_p2p_pair_usage_kw"], usage_kw)
                if usage_kw >= pair_capacity_kw - BALANCE_TOL:
                    day["p2p_saturated_pair_steps"] += 1
                if energy > pair_capacity_kw * dt + BALANCE_TOL:
                    _add_violation(
                        day,
                        "p2p_pair_capacity",
                        step=step,
                        pair=f"{first}-{second}",
                        usage_kw=usage_kw,
                    )

        for mg in ids:
            plan = info["plans"][mg]
            settlement = info["settlements"][mg]
            device = info["devices"][mg]
            residual = abs(float(settlement["power_balance_residual_kw"]))
            day["max_power_balance_residual_kw"] = max(
                day["max_power_balance_residual_kw"], residual
            )
            if residual > BALANCE_TOL:
                _add_violation(day, "power_balance", step=step, microgrid=mg, residual_kw=residual)
            unmet_kw = float(settlement["unmet_load_kw"])
            day["unmet_load_kwh"] += unmet_kw * dt
            if unmet_kw > BALANCE_TOL or bool(plan.get("dr_supply_infeasible", False)):
                _add_violation(day, "unmet_actual_load", step=step, microgrid=mg, unmet_kw=unmet_kw)
            if abs(float(plan["actual_load_kw"]) - float(plan["base_load_kw"]) - float(plan["flexible_load_kw"])) > BALANCE_TOL:
                _add_violation(day, "load_decomposition", step=step, microgrid=mg)

            battery_cfg = env.cfg_by_id[mg]["battery"]
            soc = float(device["soc"])
            if soc < float(battery_cfg["soc_min"]) - HARD_TOL or soc > float(battery_cfg["soc_max"]) + HARD_TOL:
                _add_violation(day, "ess_soc_bounds", step=step, microgrid=mg, soc=soc)
            if bool(plan.get("ess_projected", False)):
                day["ess_projection_steps"] += 1
            if bool(plan.get("dr_projected", False)):
                day["dr_projection_steps"] += 1
            curtailed_kw = float(plan.get("battery_charge_curtailed_kw", 0.0))
            day["battery_charge_curtailed_kwh"] += curtailed_kw * dt
            if curtailed_kw > BALANCE_TOL:
                day["battery_charge_curtailed_steps"] += 1
            p2p_sell_kw = float(info["p2p"]["by_microgrid"][mg]["p2p_sell_kw"])
            if (
                not env.allow_grid_resale
                and float(settlement["grid_import_kw"]) > BALANCE_TOL
                and p2p_sell_kw > BALANCE_TOL
            ):
                _add_violation(day, "grid_resale", step=step, microgrid=mg)

        state_index = step + 1
        dr_state = info["dr_state_after"]
        mg1_cfg = env.cfg_by_id["mg1"]["dr"]
        silo = float(dr_state["mg1"]["silo_level_t"])
        if silo < float(mg1_cfg["silo_min_t"]) - HARD_TOL or silo > float(mg1_cfg["silo_max_t"]) + HARD_TOL:
            _add_violation(day, "mg1_silo_bounds", step=step, silo_t=silo)
        if bool(dr_state["mg1"].get("terminal_infeasible", 0.0)):
            _add_violation(day, "mg1_terminal_reachability", step=step)

        t_air = float(dr_state["mg2"]["t_air_c"])
        comfort_low, comfort_high = _comfort_bounds(env, state_index)
        if t_air < comfort_low - HARD_TOL or t_air > comfort_high + HARD_TOL:
            _add_violation(
                day,
                "mg2_comfort",
                step=step,
                t_air_c=t_air,
                lower_c=comfort_low,
                upper_c=comfort_high,
            )
        if bool(dr_state["mg2"].get("comfort_infeasible", 0.0)):
            _add_violation(day, "mg2_comfort_infeasible", step=step)
        if bool(dr_state["mg2"].get("terminal_thermal_infeasible", 0.0)):
            _add_violation(day, "mg2_terminal_reachability", step=step)

        mg3_cfg = env.cfg_by_id["mg3"]["dr"]
        ev_soc = float(dr_state["mg3"]["ev_soc"])
        if ev_soc > float(mg3_cfg["max_soc"]) + HARD_TOL:
            _add_violation(day, "mg3_ev_max_soc", step=step, ev_soc=ev_soc)
        if bool(dr_state["mg3"].get("target_missed", 0.0)):
            _add_violation(day, "mg3_ev_target_missed", step=step)
        if state_index == int(mg3_cfg["departure_hour"]):
            target_soc = float(mg3_cfg["target_soc"])
            if abs(ev_soc - target_soc) > HARD_TOL:
                _add_violation(day, "mg3_departure_soc", step=step, ev_soc=ev_soc, target_soc=target_soc)

    if not done or last_info is None:
        _add_violation(day, "episode_not_terminated")
        return day

    for mg, infeasible in last_info["terminal_soc_infeasible"].items():
        if bool(infeasible):
            _add_violation(day, "ess_terminal_soc", microgrid=mg)

    final_dr = last_info["dr_state_after"]
    mg1_cfg = env.cfg_by_id["mg1"]["dr"]
    final_silo = float(final_dr["mg1"]["silo_level_t"])
    silo_gap = abs(final_silo - float(mg1_cfg["silo_initial_t"]))
    day["mg1_terminal_silo_gap_t"] = silo_gap
    if silo_gap > float(mg1_cfg["terminal_tolerance_t"]) + HARD_TOL:
        _add_violation(day, "mg1_terminal_silo", gap_t=silo_gap)

    mg2_cfg = env.cfg_by_id["mg2"]["dr"]
    final_t_air = float(final_dr["mg2"]["t_air_c"])
    final_t_mass = float(final_dr["mg2"]["t_mass_c"])
    air_gap = abs(final_t_air - initial_t_air)
    mass_gap = abs(final_t_mass - initial_t_mass)
    day["mg2_terminal_air_gap_c"] = air_gap
    day["mg2_terminal_mass_gap_c"] = mass_gap
    if air_gap > float(mg2_cfg["terminal_air_tolerance_c"]) + HARD_TOL:
        _add_violation(day, "mg2_terminal_air", gap_c=air_gap)
    if mass_gap > float(mg2_cfg["terminal_mass_tolerance_c"]) + HARD_TOL:
        _add_violation(day, "mg2_terminal_mass", gap_c=mass_gap)

    mg3_cfg = env.cfg_by_id["mg3"]["dr"]
    final_ev_soc = float(final_dr["mg3"]["ev_soc"])
    day["mg3_final_ev_soc"] = final_ev_soc
    if abs(final_ev_soc - float(mg3_cfg["target_soc"])) > HARD_TOL:
        _add_violation(day, "mg3_final_ev_soc", ev_soc=final_ev_soc)
    return day


def audit_seasonal_physics(
    config_path: str | Path,
    *,
    year: int = 2018,
    seed: int = 20260915,
) -> dict[str, Any]:
    envs = {
        split: MultiMicrogridEnv(config_path, split=split)
        for split in ("train", "val", "test")
    }
    candidates = [
        DayCandidate(split=split, day=day)
        for split, env in envs.items()
        for day in env.data.valid_days
    ]
    selected = select_monthly_representatives(candidates, year=int(year))
    day_results: list[dict[str, Any]] = []
    flags: list[str] = []
    for env in envs.values():
        flags.extend(str(flag) for flag in env.data.data_quality_flags)
    flags = sorted(set(flags))

    for index, candidate in enumerate(selected):
        day_seed = int(seed) + index * 1009 + int(candidate.day.dayofyear)
        day_results.append(
            audit_day(envs[candidate.split], candidate, seed=day_seed)
        )

    summary: dict[str, Any] = {
        "year": int(year),
        "seed": int(seed),
        "representative_days": [
            {"split": item.split, "day": str(item.day.date()), "weekday": item.day.day_name()}
            for item in selected
        ],
        "day_count": len(day_results),
        "hard_violation_count": int(sum(item["hard_violation_count"] for item in day_results)),
        "data_quality_flags": flags,
        "ess_projection_steps": int(sum(item["ess_projection_steps"] for item in day_results)),
        "dr_projection_steps": int(sum(item["dr_projection_steps"] for item in day_results)),
        "battery_charge_curtailed_kwh": float(
            sum(item["battery_charge_curtailed_kwh"] for item in day_results)
        ),
        "battery_charge_curtailed_steps": int(
            sum(item["battery_charge_curtailed_steps"] for item in day_results)
        ),
        "p2p_saturated_pair_steps": int(
            sum(item["p2p_saturated_pair_steps"] for item in day_results)
        ),
        "max_power_balance_residual_kw": float(
            max((item["max_power_balance_residual_kw"] for item in day_results), default=0.0)
        ),
        "max_p2p_pair_usage_kw": float(
            max((item["max_p2p_pair_usage_kw"] for item in day_results), default=0.0)
        ),
        "max_reward_spread": float(
            max((item["max_reward_spread"] for item in day_results), default=0.0)
        ),
        "max_transfer_payment_imbalance": float(
            max((item["max_transfer_payment_imbalance"] for item in day_results), default=0.0)
        ),
        "unmet_load_kwh": float(sum(item["unmet_load_kwh"] for item in day_results)),
        "projection_exercised": bool(
            sum(item["ess_projection_steps"] for item in day_results) > 0
            and sum(item["dr_projection_steps"] for item in day_results) > 0
        ),
        "days": day_results,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit heterogeneous DR physics across 12 seasonal days")
    parser.add_argument("--config", default="configs/cooperative.yaml")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    summary = audit_seasonal_physics(
        args.config,
        year=args.year,
        seed=args.seed,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    print(payload, flush=True)
    require_clean_audit(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
