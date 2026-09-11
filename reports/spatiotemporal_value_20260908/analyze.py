from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.data_manager import DataManager
from optimization.common_constraints import build_dispatch_data, load_config, solve_dispatch


CONFIG = ROOT / "configs" / "cooperative.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent


def disable_storage(config: dict) -> dict:
    result = deepcopy(config)
    for microgrid in result["microgrids"]:
        battery = microgrid["battery"]
        battery["max_charge_kw"] = 0.0
        battery["max_discharge_kw"] = 0.0
        battery["initial_soc"] = float(battery["soc_min"])
    return result


def energy_metrics(result) -> dict[str, float]:
    frame = result.trajectories
    return {
        "grid_import_kwh": float(frame["grid_import_kw"].sum()),
        "grid_export_kwh": float(frame["grid_export_kw"].sum()),
        "curtailment_kwh": float(frame["curtailment_kw"].sum()),
        "battery_throughput_kwh": float(
            frame["battery_charge_kw"].sum() + frame["battery_discharge_kw"].sum()
        ),
        "p2p_energy_kwh": float(frame["p2p_export_kw"].sum()),
    }


def solve_independent(config: dict, data):
    results = [
        solve_dispatch(config, data, [microgrid], cooperative=False)
        for microgrid in data.agent_ids
    ]
    return {
        "cost": float(sum(result.total_cost for result in results)),
        "grid_import_kwh": float(
            sum(result.trajectories["grid_import_kw"].sum() for result in results)
        ),
        "grid_export_kwh": float(
            sum(result.trajectories["grid_export_kw"].sum() for result in results)
        ),
        "curtailment_kwh": float(
            sum(result.trajectories["curtailment_kw"].sum() for result in results)
        ),
        "battery_throughput_kwh": float(
            sum(
                result.trajectories["battery_charge_kw"].sum()
                + result.trajectories["battery_discharge_kw"].sum()
                for result in results
            )
        ),
        "p2p_energy_kwh": 0.0,
    }


def main() -> None:
    config = load_config(CONFIG)
    no_storage = disable_storage(config)
    manager = DataManager(config, split="test", seed=int(config["env"]["seed"]))
    days = [str(day.date()) for day in manager.valid_days]
    rows: list[dict[str, float | str]] = []

    for day in days:
        _, data = build_dispatch_data(config, split="test", day=day)
        none = solve_independent(no_storage, data)
        temporal = solve_independent(config, data)
        spatial_result = solve_dispatch(no_storage, data, data.agent_ids, cooperative=True)
        full_result = solve_dispatch(config, data, data.agent_ids, cooperative=True)
        spatial = {"cost": float(spatial_result.total_cost), **energy_metrics(spatial_result)}
        full = {"cost": float(full_result.total_cost), **energy_metrics(full_result)}

        c00 = none["cost"]
        c10 = temporal["cost"]
        c01 = spatial["cost"]
        c11 = full["cost"]
        temporal_standalone = c00 - c10
        spatial_standalone = c00 - c01
        temporal_after_space = c01 - c11
        spatial_after_time = c10 - c11
        interaction = c10 + c01 - c00 - c11
        temporal_shapley = 0.5 * (temporal_standalone + temporal_after_space)
        spatial_shapley = 0.5 * (spatial_standalone + spatial_after_time)

        row: dict[str, float | str] = {
            "day": day,
            "cost_no_storage_no_p2p": c00,
            "cost_temporal_only": c10,
            "cost_spatial_only": c01,
            "cost_spatiotemporal": c11,
            "temporal_gain_standalone": temporal_standalone,
            "spatial_gain_standalone": spatial_standalone,
            "temporal_gain_after_space": temporal_after_space,
            "spatial_gain_after_time": spatial_after_time,
            "interaction_gain": interaction,
            "temporal_shapley_gain": temporal_shapley,
            "spatial_shapley_gain": spatial_shapley,
            "total_spatiotemporal_gain": c00 - c11,
        }
        for scenario, values in (
            ("none", none),
            ("temporal", temporal),
            ("spatial", spatial),
            ("full", full),
        ):
            for metric, value in values.items():
                if metric != "cost":
                    row[f"{scenario}_{metric}"] = value
        rows.append(row)
        print(f"{day}: {len(rows)}/{len(days)}")

    daily = pd.DataFrame(rows)
    daily.to_csv(OUTPUT_DIR / "daily_counterfactuals.csv", index=False)

    numeric = daily.select_dtypes(include=[np.number])
    summary = {
        "config": str(CONFIG),
        "split": "test",
        "days": len(daily),
        "first_day": str(daily["day"].min()),
        "last_day": str(daily["day"].max()),
        "definition": {
            "no_storage_no_p2p": "Independent dispatch with storage power disabled",
            "temporal_only": "Independent dispatch with each microgrid's local storage enabled",
            "spatial_only": "Cooperative same-hour P2P dispatch with storage power disabled",
            "spatiotemporal": "Cooperative P2P dispatch with all local storage enabled",
            "factor_shapley": "Two-factor Shapley decomposition over temporal and spatial flexibility",
        },
        "mean": {column: float(numeric[column].mean()) for column in numeric.columns},
        "median": {column: float(numeric[column].median()) for column in numeric.columns},
        "total": {column: float(numeric[column].sum()) for column in numeric.columns},
        "p05": {column: float(numeric[column].quantile(0.05)) for column in numeric.columns},
        "p95": {column: float(numeric[column].quantile(0.95)) for column in numeric.columns},
        "day_rates": {
            "temporal_shapley_gt_spatial": float(
                (daily["temporal_shapley_gain"] > daily["spatial_shapley_gain"]).mean()
            ),
            "spatial_shapley_gt_temporal": float(
                (daily["spatial_shapley_gain"] > daily["temporal_shapley_gain"]).mean()
            ),
            "positive_interaction": float((daily["interaction_gain"] > 1e-7).mean()),
            "negative_interaction": float((daily["interaction_gain"] < -1e-7).mean()),
        },
    }
    total_gain = summary["mean"]["total_spatiotemporal_gain"]
    summary["factor_shares"] = {
        "temporal": summary["mean"]["temporal_shapley_gain"] / total_gain,
        "spatial": summary["mean"]["spatial_shapley_gain"] / total_gain,
    }
    summary["cost_reduction_pct"] = {
        "temporal_only_vs_none": summary["mean"]["temporal_gain_standalone"]
        / summary["mean"]["cost_no_storage_no_p2p"],
        "spatial_only_vs_none": summary["mean"]["spatial_gain_standalone"]
        / summary["mean"]["cost_no_storage_no_p2p"],
        "full_vs_none": total_gain / summary["mean"]["cost_no_storage_no_p2p"],
    }

    rng = np.random.default_rng(42)
    bootstrap_index = rng.integers(0, len(daily), size=(20_000, len(daily)))
    summary["bootstrap_95_ci"] = {}
    for column in ("temporal_shapley_gain", "spatial_shapley_gain"):
        bootstrap_means = daily[column].to_numpy()[bootstrap_index].mean(axis=1)
        summary["bootstrap_95_ci"][column] = [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ]
    factor_difference = (
        daily["spatial_shapley_gain"] - daily["temporal_shapley_gain"]
    ).to_numpy()
    difference_means = factor_difference[bootstrap_index].mean(axis=1)
    summary["bootstrap_95_ci"]["spatial_minus_temporal"] = [
        float(np.quantile(difference_means, 0.025)),
        float(np.quantile(difference_means, 0.975)),
    ]

    latest_pointer = ROOT / "outputs" / "matd3_dispatch" / "latest_run.json"
    if latest_pointer.exists():
        latest = json.loads(latest_pointer.read_text(encoding="utf-8"))
        run_dir = Path(latest["run_directory"])
        test_summary = json.loads((run_dir / "test_summary.json").read_text(encoding="utf-8"))
        algorithm = test_summary["algorithm"]
        current_cost = -float(algorithm["team_profit_mean"])
        current_throughput = float(sum(algorithm["ess"]["throughput_mean_kwh"].values()))
        current_p2p = float(algorithm["p2p"]["total_energy_mean_kwh"])
        current_import = float(sum(algorithm["grid"]["import_mean_kwh"].values()))
        current_export = float(sum(algorithm["grid"]["export_mean_kwh"].values()))
        current_curtailment = float(test_summary["joint_curtailment_mean_kwh"])
        summary["current_policy"] = {
            "run_id": latest["run_id"],
            "mean_cost": current_cost,
            "gap_to_full_oracle": current_cost - summary["mean"]["cost_spatiotemporal"],
            "total_flexibility_value_capture": (
                summary["mean"]["cost_no_storage_no_p2p"] - current_cost
            ) / total_gain,
            "battery_throughput_kwh": current_throughput,
            "battery_throughput_vs_full_oracle": (
                current_throughput / summary["mean"]["full_battery_throughput_kwh"]
            ),
            "p2p_energy_kwh": current_p2p,
            "p2p_energy_vs_full_oracle": (
                current_p2p / summary["mean"]["full_p2p_energy_kwh"]
            ),
            "grid_import_kwh": current_import,
            "grid_import_gap_to_full_oracle_kwh": (
                current_import - summary["mean"]["full_grid_import_kwh"]
            ),
            "grid_export_kwh": current_export,
            "grid_export_gap_to_full_oracle_kwh": (
                current_export - summary["mean"]["full_grid_export_kwh"]
            ),
            "curtailment_kwh": current_curtailment,
            "curtailment_gap_to_full_oracle_kwh": (
                current_curtailment - summary["mean"]["full_curtailment_kwh"]
            ),
            "ess_offer_acceptance_rate": float(
                algorithm["p2p"]["ess_offer"]["energy_acceptance_rate"]
            ),
        }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
