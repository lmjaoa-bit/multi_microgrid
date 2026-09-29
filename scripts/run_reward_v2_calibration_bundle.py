from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.registry import get_algorithm_spec
from core.reward_v2 import resolve_carbon_factors, resolve_reward_v2_config
from core.tariffs import prices_from_row
from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.reward_calibration import collect_calibration


def _select_days(env: MultiMicrogridEnv, count: int) -> list[str]:
    values = [str(day.date()) for day in env.data.valid_days]
    if count <= 0:
        raise ValueError("day count must be positive")
    if count >= len(values):
        return values
    indices = np.linspace(0, len(values) - 1, num=count, dtype=int)
    return [values[int(index)] for index in indices]


def _finite_training_diagnostics(run_dir: Path, gamma: float) -> dict[str, Any]:
    history_path = run_dir / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    fields = (
        "critic_loss",
        "actor_loss",
        "critic_grad_norm",
        "actor_grad_norm",
        "q1_mean",
        "q2_mean",
        "target_q_mean",
        "td_error_mean",
        "td_error_std",
    )
    stats: dict[str, Any] = {}
    nonfinite: list[str] = []
    for field in fields:
        values = []
        for index, row in enumerate(history):
            value = row.get(field)
            if value is None:
                continue
            parsed = float(value)
            if not math.isfinite(parsed):
                nonfinite.append(f"episode={index + 1}:{field}={value}")
                continue
            values.append(parsed)
        stats[field] = {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "max_abs": max((abs(v) for v in values), default=None),
        }
    explosion_threshold = 1.0e6
    exploded = [
        field
        for field, item in stats.items()
        if item["max_abs"] is not None and float(item["max_abs"]) > explosion_threshold
    ]
    return {
        "gamma": float(gamma),
        "episodes": len(history),
        "nonfinite_fields": nonfinite,
        "explosion_threshold_abs": explosion_threshold,
        "exploded_fields": exploded,
        "stable_for_short_diagnostic": not nonfinite and not exploded,
        "field_ranges": stats,
        "note": (
            "A stable short diagnostic only establishes finite numerics; "
            "it is not evidence of convergence or policy quality."
        ),
    }


def _selected_frame(env: MultiMicrogridEnv, days: list[str]) -> pd.DataFrame:
    frame = env.data.frame.copy()
    dates = pd.to_datetime(frame["timestamp"]).dt.date.astype(str)
    return frame.loc[dates.isin(days)].reset_index(drop=True)


def _marginal_analysis(
    config: Mapping[str, Any],
    env: MultiMicrogridEnv,
    days: list[str],
) -> dict[str, Any]:
    reward = resolve_reward_v2_config(config)
    if reward is None:
        raise ValueError("Reward V2 is required")
    frame = _selected_frame(env, days)
    carbon = resolve_carbon_factors(config, frame)
    max_carbon = float(np.max(carbon)) if len(carbon) else 0.0
    max_buy = 0.0
    for _, row in frame.iterrows():
        for mg in env.agent_ids:
            buy, _ = prices_from_row(row, mg)
            max_buy = max(max_buy, float(buy))

    ev_cfg = next(
        mg["dr"] for mg in config["microgrids"]
        if str(mg.get("dr", {}).get("type")) == "ev_charging"
    )
    eta = float(ev_cfg["eta_charge"])
    grid_energy_per_battery_kwh = 1.0 / eta
    purchase_margin = max_buy * grid_energy_per_battery_kwh
    carbon_margin = (
        float(reward["carbon_price_per_kg"])
        * max_carbon
        * grid_energy_per_battery_kwh
    )
    # For one battery-side kWh that must be delivered within one 1 h peak
    # interval, the corresponding power increment is 1/eta kW.
    peak_margin = (
        float(reward["peak_price_per_kw"])
        * grid_energy_per_battery_kwh
    )
    worst_case_ev_service_cost = purchase_margin + carbon_margin + peak_margin

    return {
        "selected_validation_days": list(days),
        "max_observed_buy_price_per_kwh": max_buy,
        "max_observed_carbon_factor_kg_per_kwh": max_carbon,
        "ev_eta_charge": eta,
        "ev_battery_kwh_grid_energy_kwh": grid_energy_per_battery_kwh,
        "ev_marginal_purchase_cost_per_battery_kwh": purchase_margin,
        "ev_marginal_carbon_cost_per_battery_kwh": carbon_margin,
        "ev_one_hour_peak_cost_per_battery_kwh": peak_margin,
        "ev_worst_case_one_hour_service_cost_per_battery_kwh": (
            worst_case_ev_service_cost
        ),
        "configured_ev_shortfall_price_per_kwh": float(
            reward["ev_shortfall_price_per_kwh"]
        ),
        "ev_penalty_covers_worst_case_one_hour_service": bool(
            float(reward["ev_shortfall_price_per_kwh"])
            > worst_case_ev_service_cost
        ),
        "comfort_tradeoff_note": (
            "Comfort price is assessed together with observed comfort degree-hours "
            "and HVAC energy/peak behavior from idle, service-priority, and trained "
            "policies; it is not auto-derived by numerical equalization."
        ),
        "curtailment_tradeoff_note": (
            "Curtailment price is an explicit marginal value per curtailed kWh. "
            "Avoiding curtailment may also avoid later grid purchase, so it is not "
            "calibrated by forcing its daily contribution to match grid cost."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the combined Reward V2 calibration evidence bundle"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--algorithm", default="matd3")
    parser.add_argument("--algorithm-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--val-days", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_project_root"] = str(config_path.parent.parent)
    resolved = resolve_reward_v2_config(config)
    if resolved is None:
        raise ValueError("Reward V2 config is required")
    if str(resolved["calibration_status"]) != "diagnostic":
        raise ValueError("Calibration bundle requires calibration_status=diagnostic")

    run_dir = args.run_dir.resolve()
    diagnostics_dir = run_dir / "diagnostics"
    if not diagnostics_dir.is_dir():
        raise FileNotFoundError(
            "run-dir must be an existing diagnostic run with diagnostics/"
        )
    checkpoint = run_dir / "checkpoints" / "final_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    train_env = MultiMicrogridEnv(config, split="train")
    val_env = MultiMicrogridEnv(config, split="val")
    train_days = _select_days(train_env, int(args.train_days))
    val_days = _select_days(val_env, int(args.val_days))

    spec = get_algorithm_spec(args.algorithm)
    algorithm_config = yaml.safe_load(
        args.algorithm_config.resolve().read_text(encoding="utf-8")
    )
    model = spec.create(
        val_env, algorithm_config, device="cpu", seed=int(args.seed)
    )
    model.load(checkpoint)

    def trained_policy(observations):
        return model.select_actions(observations, noise_std=0.0)

    trained_policy.calibration_policy_name = "short_trained_matd3"  # type: ignore[attr-defined]

    reports = {
        "train_idle": collect_calibration(
            config,
            split="train",
            days=train_days,
            policy="idle",
            seed=int(args.seed),
        ),
        "train_service_priority": collect_calibration(
            config,
            split="train",
            days=train_days,
            policy="service_priority",
            seed=int(args.seed),
        ),
        "val_idle": collect_calibration(
            config,
            split="val",
            days=val_days,
            policy="idle",
            seed=int(args.seed),
        ),
        "val_service_priority": collect_calibration(
            config,
            split="val",
            days=val_days,
            policy="service_priority",
            seed=int(args.seed),
        ),
        "val_short_trained": collect_calibration(
            config,
            split="val",
            days=val_days,
            policy=trained_policy,
            seed=int(args.seed),
        ),
    }

    training_numerics = _finite_training_diagnostics(
        run_dir, gamma=float(algorithm_config["algorithm"]["gamma"])
    )
    marginal = _marginal_analysis(config, val_env, val_days)
    bundle = {
        "schema_version": "reward_calibration_bundle_v1",
        "run_directory": str(run_dir),
        "seed": int(args.seed),
        "reward_semantics_version": resolved["semantics_version"],
        "calibration_status": resolved["calibration_status"],
        "diagnostic_coefficients": {
            key: float(resolved[key])
            for key in (
                "curtailment_price_per_kwh",
                "carbon_price_per_kg",
                "peak_price_per_kw",
                "mill_unserved_price_per_t",
                "comfort_price_per_degree_hour",
                "ev_shortfall_price_per_kwh",
                "reward_scale",
            )
        },
        "reports": reports,
        "training_numerics": training_numerics,
        "marginal_tradeoffs": marginal,
        "lock_status": "not_locked",
        "lock_note": (
            "This file is diagnostic evidence. Formal coefficients must be "
            "chosen from these train/validation results before switching the "
            "shared configuration to calibration_status=locked."
        ),
    }
    destination = diagnostics_dir / "reward_calibration.json"
    destination.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
