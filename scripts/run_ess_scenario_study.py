from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.registry import get_algorithm_spec
from diagnostics.scenario_diagnostic import diagnose_scenario
from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.evaluate_matd3 import evaluate_policy_on_days
from utils.metrics_collector import distribution_statistics, write_csv, write_json

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired MILP sensitivity scenarios for ESS cross-grid exports"
    )
    parser.add_argument(
        "--study-config",
        type=Path,
        default=Path("configs/scenario_studies/ess_cross_grid.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/scenario_studies/ess_cross_grid"),
    )
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenario_ids",
        help="Run only the named scenario; repeat the option to select several scenarios.",
    )
    return parser


def _deep_merge(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            nested = dict(target[key])
            _deep_merge(nested, value)
            target[key] = nested
        else:
            target[key] = deepcopy(value)


def resolve_scenario(
    base_config: Mapping[str, Any], scenario: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Apply one study scenario without mutating the canonical environment config."""
    result = deepcopy(dict(base_config))
    global_overrides = scenario.get("global", {})
    renewable_scale = float(global_overrides.get("renewable_capacity_scale", 1.0))
    battery_scale = float(global_overrides.get("battery_capacity_scale", 1.0))
    if renewable_scale <= 0.0:
        raise ValueError("renewable_capacity_scale must be positive")
    if battery_scale <= 0.0:
        raise ValueError("battery_capacity_scale must be positive")
    for microgrid in result["microgrids"]:
        if "grid_import_limit_kw" in global_overrides:
            microgrid["grid"]["import_limit_kw"] = float(
                global_overrides["grid_import_limit_kw"]
            )
        if "grid_export_limit_kw" in global_overrides:
            microgrid["grid"]["export_limit_kw"] = float(
                global_overrides["grid_export_limit_kw"]
            )
        if renewable_scale != 1.0:
            renewable = microgrid["renewables"]
            renewable["pv_capacity_kw"] = float(renewable["pv_capacity_kw"]) * renewable_scale
            renewable["wind_capacity_kw"] = (
                float(renewable["wind_capacity_kw"]) * renewable_scale
            )
        if battery_scale != 1.0:
            battery = microgrid["battery"]
            battery["capacity_kwh"] = float(battery["capacity_kwh"]) * battery_scale

    by_id = {str(item["id"]): item for item in result["microgrids"]}
    for microgrid_id, updates in scenario.get("microgrids", {}).items():
        if str(microgrid_id) not in by_id:
            raise KeyError(f"Unknown microgrid in scenario: {microgrid_id}")
        _deep_merge(by_id[str(microgrid_id)], updates)
    result["_project_root"] = str(project_root.resolve())
    return result


def _sum_section(day: Mapping[str, Any], section: str, field: str) -> float:
    return float(sum(float(item[field]) for item in day[section].values()))


def _energy(day: Mapping[str, Any], field: str) -> float:
    return _sum_section(day, "energy_statistics_by_microgrid", field)


def _economics(day: Mapping[str, Any], field: str) -> float:
    return _sum_section(day, "economic_breakdown_by_microgrid", field)


def _evaluate_controller(
    config: Mapping[str, Any], algorithm_slug: str, split: str, days: Sequence[str], seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = get_algorithm_spec(algorithm_slug)
    effective_config = spec.build_environment_config(config)
    env = MultiMicrogridEnv(effective_config, split=split)
    algorithm_config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    controller = spec.create(env, algorithm_config, device="cpu", seed=seed)
    summary, daily = evaluate_policy_on_days(
        env,
        list(days),
        lambda observations: controller.select_actions(observations, noise_std=0.0),
        seed=seed,
        deterministic_environment=True,
        capture_hourly=False,
    )
    return summary, daily


def _scenario_rows(
    scenario_id: str,
    scenario_description: str,
    diagnostic: Mapping[str, Any],
    ess_daily: Sequence[Mapping[str, Any]],
    noess_daily: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(ess_daily) != len(noess_daily):
        raise ValueError(f"{scenario_id}: paired methods returned different day counts")
    daily_rows: list[dict[str, Any]] = []
    for ess, noess in zip(ess_daily, noess_daily):
        if str(ess["date"]) != str(noess["date"]):
            raise ValueError(f"{scenario_id}: paired dates do not match")
        gain = float(ess["team_profit"]) - float(noess["team_profit"])
        row = {
            "scenario": scenario_id,
            "description": scenario_description,
            "date": str(ess["date"]),
            "test_seed": int(ess["test_seed"]),
            "ess_profit": float(ess["team_profit"]),
            "noess_profit": float(noess["team_profit"]),
            "ess_cross_grid_value": gain,
            "ess_cross_grid_value_pct": (
                gain / abs(float(noess["team_profit"])) * 100.0
                if abs(float(noess["team_profit"])) > 1e-9 else None
            ),
            "ess_p2p_kwh": _energy(ess, "ess_p2p_sell_kwh"),
            "renewable_p2p_kwh_ess": _energy(ess, "renewable_p2p_sell_kwh"),
            "renewable_p2p_kwh_noess": _energy(noess, "renewable_p2p_sell_kwh"),
            "unmet_load_kwh_ess": _energy(ess, "unmet_load_kwh"),
            "unmet_load_kwh_noess": _energy(noess, "unmet_load_kwh"),
            "unmet_load_reduction_kwh": (
                _energy(noess, "unmet_load_kwh") - _energy(ess, "unmet_load_kwh")
            ),
            "curtailment_kwh_ess": _energy(ess, "renewable_curtailment_kwh"),
            "curtailment_kwh_noess": _energy(noess, "renewable_curtailment_kwh"),
            "curtailment_reduction_kwh": (
                _energy(noess, "renewable_curtailment_kwh")
                - _energy(ess, "renewable_curtailment_kwh")
            ),
            "grid_import_kwh_ess": _energy(ess, "grid_import_kwh"),
            "grid_import_kwh_noess": _energy(noess, "grid_import_kwh"),
            "grid_export_kwh_ess": _energy(ess, "grid_export_kwh"),
            "grid_export_kwh_noess": _energy(noess, "grid_export_kwh"),
            "renewable_available_kwh": _energy(ess, "renewable_available_kwh"),
            "load_kwh": _energy(ess, "load_kwh"),
            "renewable_utilization_pct_ess": float(ess["renewable_utilization_rate"]) * 100.0,
            "renewable_utilization_pct_noess": float(noess["renewable_utilization_rate"]) * 100.0,
            "avoided_grid_import_cost": (
                _economics(noess, "grid_import_cost") - _economics(ess, "grid_import_cost")
            ),
            "export_revenue_change": (
                _economics(ess, "grid_export_revenue")
                - _economics(noess, "grid_export_revenue")
            ),
            "avoided_degradation_cost": (
                _economics(noess, "battery_degradation_cost")
                - _economics(ess, "battery_degradation_cost")
            ),
            "avoided_unmet_load_cost": (
                _economics(noess, "unmet_load_cost") - _economics(ess, "unmet_load_cost")
            ),
            "avoided_curtailment_cost": (
                _economics(noess, "renewable_curtailment_cost")
                - _economics(ess, "renewable_curtailment_cost")
            ),
            "avoided_terminal_soc_penalty": (
                _economics(noess, "terminal_soc_penalty")
                - _economics(ess, "terminal_soc_penalty")
            ),
            "ess_feasible": bool(ess["constraint_statistics"]["feasible_episode"]),
            "noess_feasible": bool(noess["constraint_statistics"]["feasible_episode"]),
            "ess_metric_checks_passed": bool(ess["sanity_checks"]["passed"]),
            "noess_metric_checks_passed": bool(noess["sanity_checks"]["passed"]),
        }
        row["decomposition_residual"] = gain - sum(
            float(row[field]) for field in (
                "avoided_grid_import_cost",
                "export_revenue_change",
                "avoided_degradation_cost",
                "avoided_unmet_load_cost",
                "avoided_curtailment_cost",
                "avoided_terminal_soc_penalty",
            )
        )
        daily_rows.append(row)

    gains = np.asarray([float(row["ess_cross_grid_value"]) for row in daily_rows])
    noess_profit = np.asarray([float(row["noess_profit"]) for row in daily_rows])
    value_pct = gains.mean() / max(abs(noess_profit.mean()), 1e-9) * 100.0
    gain_stats = distribution_statistics(gains)
    standard_error = float(np.std(gains) / np.sqrt(max(1, len(gains))))

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in daily_rows]))

    scenario_row = {
        "scenario": scenario_id,
        "description": scenario_description,
        "days": len(daily_rows),
        "system_res_load_ratio": float(
            sum(row["renewable_available_kwh"] for row in daily_rows)
            / max(sum(row["load_kwh"] for row in daily_rows), 1e-9)
        ),
        "mixed_role_hour_ratio": float(
            diagnostic["system"]["simultaneous_surplus_deficit_hour_ratio"]
        ),
        "spatiotemporal_opportunity_day_ratio": float(
            diagnostic["system"]["spatiotemporal_opportunity_day_ratio"]
        ),
        "ess_profit_mean": mean("ess_profit"),
        "noess_profit_mean": mean("noess_profit"),
        "ess_cross_grid_value_mean": float(gain_stats["mean"]),
        "ess_cross_grid_value_median": float(gain_stats["median"]),
        "ess_cross_grid_value_std": float(gain_stats["std"]),
        "ess_cross_grid_value_p05": float(gain_stats["p05"]),
        "ess_cross_grid_value_p95": float(gain_stats["p95"]),
        "ess_cross_grid_value_ci95_low": float(gain_stats["mean"]) - 1.96 * standard_error,
        "ess_cross_grid_value_ci95_high": float(gain_stats["mean"]) + 1.96 * standard_error,
        "ess_cross_grid_value_pct_of_noess_profit": float(value_pct),
        "win_rate": float(np.mean(gains > 1e-8)),
        "loss_rate": float(np.mean(gains < -1e-8)),
        "ess_p2p_kwh_mean": mean("ess_p2p_kwh"),
        "unmet_load_kwh_ess_mean": mean("unmet_load_kwh_ess"),
        "unmet_load_kwh_noess_mean": mean("unmet_load_kwh_noess"),
        "unmet_load_reduction_kwh_mean": mean("unmet_load_reduction_kwh"),
        "curtailment_kwh_ess_mean": mean("curtailment_kwh_ess"),
        "curtailment_kwh_noess_mean": mean("curtailment_kwh_noess"),
        "curtailment_reduction_kwh_mean": mean("curtailment_reduction_kwh"),
        "grid_import_kwh_ess_mean": mean("grid_import_kwh_ess"),
        "grid_import_kwh_noess_mean": mean("grid_import_kwh_noess"),
        "renewable_utilization_pct_ess_mean": mean("renewable_utilization_pct_ess"),
        "renewable_utilization_pct_noess_mean": mean("renewable_utilization_pct_noess"),
        "avoided_grid_import_cost_mean": mean("avoided_grid_import_cost"),
        "export_revenue_change_mean": mean("export_revenue_change"),
        "avoided_degradation_cost_mean": mean("avoided_degradation_cost"),
        "avoided_unmet_load_cost_mean": mean("avoided_unmet_load_cost"),
        "avoided_curtailment_cost_mean": mean("avoided_curtailment_cost"),
        "avoided_terminal_soc_penalty_mean": mean("avoided_terminal_soc_penalty"),
        "decomposition_residual_mean": mean("decomposition_residual"),
        "all_days_feasible": all(
            bool(row["ess_feasible"] and row["noess_feasible"]) for row in daily_rows
        ),
        "all_metric_checks_passed": all(
            bool(row["ess_metric_checks_passed"] and row["noess_metric_checks_passed"])
            for row in daily_rows
        ),
    }
    return scenario_row, daily_rows


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    study_path = Path(args.study_config).resolve()
    manifest = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    settings = manifest["study"]
    base_path = (PROJECT_ROOT / str(settings["base_config"])).resolve()
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    split = str(settings.get("split", "test"))
    seed = int(args.seed if args.seed is not None else settings.get("seed", 42))
    algorithm_slugs = settings["algorithms"]

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / "runs" / f"ESS_CROSS_GRID_{stamp}_seed{seed}"
    config_dir = run_dir / "resolved_configs"
    config_dir.mkdir(parents=True, exist_ok=False)

    scenarios = list(manifest["scenarios"])
    if args.scenario_ids:
        requested = set(args.scenario_ids)
        known = {str(item["id"]) for item in scenarios}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown scenario(s): {', '.join(unknown)}")
        scenarios = [item for item in scenarios if str(item["id"]) in requested]

    scenario_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    write_json(output_root / "latest_run.json", {"run_id": run_dir.name, "run_dir": str(run_dir)})
    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = str(scenario["id"])
        description = str(scenario.get("description", ""))
        print(f"[{index}/{len(scenarios)}] {scenario_id}: resolving scenario", flush=True)
        config = resolve_scenario(base_config, scenario)
        snapshot_path = config_dir / f"{scenario_id}.yaml"
        snapshot = deepcopy(config)
        snapshot.pop("_project_root", None)
        snapshot_path.write_text(
            yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        probe_spec = get_algorithm_spec(str(algorithm_slugs["ess_enabled"]))
        probe_env = MultiMicrogridEnv(probe_spec.build_environment_config(config), split=split)
        days = [str(day.date()) for day in probe_env.data.valid_days]
        if args.max_days is not None:
            days = days[: max(0, int(args.max_days))]
        if not days:
            raise ValueError("Scenario study requires at least one complete day")
        diagnostic = diagnose_scenario(config, split=split)

        print(f"[{index}/{len(scenarios)}] {scenario_id}: ESS enabled", flush=True)
        _, ess_daily = _evaluate_controller(
            config, str(algorithm_slugs["ess_enabled"]), split, days, seed
        )
        print(f"[{index}/{len(scenarios)}] {scenario_id}: ESS disabled", flush=True)
        _, noess_daily = _evaluate_controller(
            config, str(algorithm_slugs["ess_disabled"]), split, days, seed
        )
        scenario_row, paired_rows = _scenario_rows(
            scenario_id, description, diagnostic, ess_daily, noess_daily
        )
        scenario_rows.append(scenario_row)
        daily_rows.extend(paired_rows)
        write_csv(run_dir / "scenario_summary.csv", scenario_rows)
        write_csv(run_dir / "daily_paired_results.csv", daily_rows)
        print(
            f"[{index}/{len(scenarios)}] {scenario_id}: "
            f"value={scenario_row['ess_cross_grid_value_mean']:.3f}/day",
            flush=True,
        )

    write_csv(run_dir / "scenario_summary.csv", scenario_rows)
    write_csv(run_dir / "daily_paired_results.csv", daily_rows)
    result = {
        "study": str(settings["name"]),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "base_config": str(base_path),
        "study_config": str(study_path),
        "split": split,
        "seed": seed,
        "max_days": args.max_days,
        "algorithms": dict(algorithm_slugs),
        "requested_scenarios": list(args.scenario_ids or []),
        "scenario_count": len(scenario_rows),
        "scenarios": scenario_rows,
    }
    write_json(run_dir / "summary.json", result)
    write_json(output_root / "latest_run.json", {"run_id": run_dir.name, "run_dir": str(run_dir)})
    print(f"Study completed: {run_dir}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    run_study(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
