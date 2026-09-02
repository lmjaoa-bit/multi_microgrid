from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.coalition_evaluator import (
    CoalitionEvaluation,
    dispatch_energy_metrics,
    evaluate_day_coalitions,
)
from evaluation.settlement import settle_test_day
from utils.metrics_collector import (
    EpisodeMetricsCollector,
    aggregate_episode_metrics,
    distribution_statistics,
    safe_ratio,
    write_csv,
    write_json,
)


def _legacy_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_episode_metrics(episodes)
    ids = list(episodes[0]["profit_by_microgrid"]) if episodes else []
    scale = max(1, len(episodes))

    def mean_mg(section: str, field: str) -> dict[str, float]:
        return {
            mg: float(sum(day[section][mg][field] for day in episodes) / scale)
            for mg in ids
        }

    pair_keys = list(episodes[0]["p2p_statistics"]["pair_energy_kwh"]) if episodes else []
    return {
        **aggregate,
        "physical_profit_mean_by_microgrid": aggregate["profit_by_microgrid_mean"],
        "p2p": {
            "total_energy_mean_kwh": aggregate["p2p_total_energy_kwh"],
            "pair_energy_mean_kwh": {
                pair: float(sum(day["p2p_statistics"]["pair_energy_kwh"][pair]
                                for day in episodes) / scale)
                for pair in pair_keys
            },
            "utilization": None,
        },
        "ess": {
            "throughput_mean_kwh": mean_mg("energy_statistics_by_microgrid", "battery_throughput_kwh"),
            "terminal_soc_mean": {
                mg: float(sum(day["soc_statistics_by_microgrid"][mg]["terminal_soc"]
                              for day in episodes) / scale)
                for mg in ids
            },
        },
        "grid": {
            "import_mean_kwh": mean_mg("energy_statistics_by_microgrid", "grid_import_kwh"),
            "export_mean_kwh": mean_mg("energy_statistics_by_microgrid", "grid_export_kwh"),
            "net_cost_mean": {
                mg: float(sum(
                    day["economic_breakdown_by_microgrid"][mg]["grid_import_cost"]
                    - day["economic_breakdown_by_microgrid"][mg]["grid_export_revenue"]
                    for day in episodes
                ) / scale) for mg in ids
            },
        },
    }


def evaluate_policy_on_days(
    env: MultiMicrogridEnv,
    days: list[str],
    policy_fn: Any,
    *,
    seed: int = 42,
    deterministic_environment: bool = True,
    capture_hourly: bool = False,
) -> Any:
    """Evaluate fixed day/seed pairs so policies receive common random numbers."""
    daily_metrics: list[dict[str, Any]] = []
    hourly_by_day: dict[str, list[dict[str, Any]]] = {}
    for index, day in enumerate(days):
        observations, _ = env.reset(
            day=day,
            seed=int(seed) + index,
            deterministic=bool(deterministic_environment),
        )
        collector = EpisodeMetricsCollector(env, capture_hourly=capture_hourly)
        done = False
        while not done:
            actions = policy_fn(observations)
            observations, rewards, done, info = env.step(actions)
            collector.record_step(actions, info, rewards)
        metrics = collector.finalize_episode()
        metrics["date"] = day
        metrics["test_seed"] = int(seed) + index
        daily_metrics.append(metrics)
        if capture_hourly:
            hourly_by_day[day] = collector.hourly_rows

    summary = _legacy_summary(daily_metrics)
    if capture_hourly:
        return summary, daily_metrics, hourly_by_day
    return summary, daily_metrics


def _paired_comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    label: str,
    candidate_label: str = "MATD3",
) -> dict[str, Any]:
    valid = np.isfinite(candidate) & np.isfinite(baseline)
    delta = candidate[valid] - baseline[valid]
    percent = np.asarray([
        safe_ratio(float(a - b), abs(float(b))) * 100.0
        for a, b in zip(candidate[valid], baseline[valid])
    ], dtype=np.float64)
    formula = f"{candidate_label} profit - {label} profit; positive means {candidate_label} is better"
    if delta.size == 0:
        return {
            "formula": formula,
            "gain_mean_abs": None, "gain_mean_pct": None, "gain_std": None,
            "gain_median": None, "win_count": 0, "tie_count": 0, "loss_count": 0,
            "win_rate": 0.0, "tie_rate": 0.0, "loss_rate": 0.0,
        }
    tolerance = 1e-8
    wins = int(np.sum(delta > tolerance))
    ties = int(np.sum(np.abs(delta) <= tolerance))
    losses = int(np.sum(delta < -tolerance))
    return {
        "formula": formula,
        "gain_mean_abs": float(np.mean(delta)),
        "gain_mean_pct": float(np.mean(percent)),
        "gain_std": float(np.std(delta)),
        "gain_median": float(np.median(delta)),
        "win_count": wins, "tie_count": ties, "loss_count": losses,
        "win_rate": wins / int(delta.size), "tie_rate": ties / int(delta.size),
        "loss_rate": losses / int(delta.size),
    }


def _daily_row(
    day: Mapping[str, Any],
    idle_day: Mapping[str, Any],
    coalition: CoalitionEvaluation | None,
    *,
    algorithm_slug: str = "matd3",
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    matd3_profit = float(day["team_profit"])
    idle_profit = float(idle_day["team_profit"])
    energies = day["energy_statistics_by_microgrid"]
    constraints = day["constraint_statistics"]
    available = float(sum(x["renewable_available_kwh"] for x in energies.values()))
    used = float(sum(x["renewable_used_kwh"] for x in energies.values()))
    curtailed = float(sum(x["renewable_curtailment_kwh"] for x in energies.values()))
    row: dict[str, Any] = {
        "date": day["date"],
        "test_seed": day["test_seed"],
        "matd3_team_profit": matd3_profit,
        "matd3_joint_profit": matd3_profit,
        "algorithm_slug": algorithm_slug,
        "algorithm_team_profit": matd3_profit,
        "algorithm_joint_profit": matd3_profit,
        "idle_team_profit": idle_profit,
        "gain_vs_idle_abs": matd3_profit - idle_profit,
        "gain_vs_idle_pct": (
            safe_ratio(matd3_profit - idle_profit, abs(idle_profit)) * 100.0
            if abs(idle_profit) > tolerance else None
        ),
        "renewable_available_kwh": available,
        "renewable_used_kwh": used,
        "renewable_curtailment_kwh": curtailed,
        "renewable_utilization_rate": float(day["renewable_utilization_rate"]),
        "res_utilization_pct": safe_ratio(used, available) * 100.0 if available > tolerance else None,
        "grid_import_kwh": sum(x["grid_import_kwh"] for x in energies.values()),
        "grid_export_kwh": sum(x["grid_export_kwh"] for x in energies.values()),
        "p2p_total_energy_kwh": day["p2p_statistics"]["total_energy_kwh"],
        "battery_charge_kwh": sum(x["battery_charge_kwh"] for x in energies.values()),
        "battery_discharge_kwh": sum(x["battery_discharge_kwh"] for x in energies.values()),
        "feasible_day": bool(constraints["feasible_episode"]),
        "soc_violation_day": constraints["soc_violation"]["count"] > 0,
        "power_balance_violation_day": constraints["power_balance_violation"]["count"] > 0,
        "p2p_balance_violation_day": constraints["p2p_balance_violation"]["count"] > 0,
    }
    for mg, value in day["profit_by_microgrid"].items():
        row[f"{mg}_profit"] = value
        row[f"physical_profit_{mg}"] = value
        energy = energies[mg]
        row[f"battery_charge_{mg}_kwh"] = energy["battery_charge_kwh"]
        row[f"battery_discharge_{mg}_kwh"] = energy["battery_discharge_kwh"]
        row[f"battery_throughput_{mg}_kwh"] = energy["battery_throughput_kwh"]
        row[f"terminal_soc_{mg}"] = day["soc_statistics_by_microgrid"][mg]["terminal_soc"]
    for pair, value in day["p2p_statistics"]["pair_energy_kwh"].items():
        row[f"p2p_energy_{pair}_kwh"] = value

    if coalition is None:
        return row
    row.update(coalition.values)
    row.update(coalition.flat_status_fields())
    row["coalition_solve_failure_count"] = coalition.failure_count
    row["coalition_all_solved"] = coalition.valid
    settlement = settle_test_day(
        coalition.values,
        matd3_joint_profit=matd3_profit,
        physical_profit_by_microgrid=day["profit_by_microgrid"],
        tolerance=tolerance,
    )
    row.update(settlement)
    row["algorithm_cooperation_surplus"] = row.get("matd3_cooperation_surplus")
    row["algorithm_cooperation_gain_pct"] = row.get("matd3_cooperation_gain_pct")
    row["algorithm_cooperation_acceptable"] = row.get("matd3_cooperation_acceptable")
    independent_metrics = [
        dispatch_energy_metrics(coalition.results.get(f"v_mg{index}")) for index in (1, 2, 3)
    ]
    if all(item["renewable_curtailment_kwh"] is not None for item in independent_metrics):
        ind_available = float(sum(float(item["renewable_available_kwh"]) for item in independent_metrics))
        ind_used = float(sum(float(item["renewable_used_kwh"]) for item in independent_metrics))
        ind_curtailment = float(sum(float(item["renewable_curtailment_kwh"]) for item in independent_metrics))
        row["independent_renewable_utilization_pct"] = (
            ind_used / ind_available * 100.0 if ind_available > tolerance else None
        )
        row["independent_curtailment_kwh"] = ind_curtailment
        row["curtailment_reduction_kwh"] = ind_curtailment - curtailed
        row["curtailment_reduction_pct"] = (
            (ind_curtailment - curtailed) / ind_curtailment * 100.0
            if ind_curtailment > tolerance else None
        )
    grand_metrics = dispatch_energy_metrics(coalition.grand_result)
    row["cooperative_milp_res_utilization_pct"] = grand_metrics["renewable_utilization_pct"]
    row["cooperative_milp_curtailment_kwh"] = grand_metrics["renewable_curtailment_kwh"]
    independent_profit = row.get("independent_total_profit")
    grand_profit = row.get("v_grand_coalition")
    if independent_profit is not None and grand_profit is not None:
        milp_surplus = float(grand_profit) - float(independent_profit)
        row["cooperative_milp_profit"] = float(grand_profit)
        row["cooperative_milp_cooperation_surplus"] = milp_surplus
        row["cooperative_milp_cooperation_gain_pct"] = (
            milp_surplus / abs(float(independent_profit)) * 100.0
            if abs(float(independent_profit)) > tolerance else None
        )
        row["cooperative_value_capture_ratio"] = (
            (matd3_profit - float(independent_profit)) / milp_surplus
            if milp_surplus > tolerance else None
        )
    return row


def _column_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None or isinstance(value, bool):
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            values.append(numeric)
    return values


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = _column_values(rows, field)
    return float(np.mean(values)) if values else None


def _std(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = _column_values(rows, field)
    return float(np.std(values)) if values else None


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return float(np.mean([bool(value) for value in values])) if values else None


def _by_microgrid(rows: Sequence[Mapping[str, Any]], prefix: str, statistic: str = "mean") -> dict[str, float | None]:
    function = _mean if statistic == "mean" else _std
    return {f"mg{index}": function(rows, f"{prefix}mg{index}") for index in (1, 2, 3)}


def _build_joint_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    surplus_stats = distribution_statistics(_column_values(rows, "algorithm_cooperation_surplus"))
    gain_stats = distribution_statistics(_column_values(rows, "algorithm_cooperation_gain_pct"))
    team_stats = distribution_statistics(_column_values(rows, "algorithm_joint_profit"))
    result: dict[str, Any] = {
        "joint_team_profit_mean": team_stats["mean"],
        "joint_team_profit_std": team_stats["std"],
        "cooperation_surplus_mean": surplus_stats["mean"],
        "cooperation_surplus_std": surplus_stats["std"],
        "cooperation_surplus_median": surplus_stats["median"],
        "cooperation_gain_pct_mean": gain_stats["mean"],
        "cooperation_gain_pct_std": gain_stats["std"],
        "matd3_cooperation_surplus_mean": surplus_stats["mean"],
        "matd3_cooperation_surplus_std": surplus_stats["std"],
        "matd3_cooperation_surplus_median": surplus_stats["median"],
        "matd3_cooperation_gain_pct_mean": gain_stats["mean"],
        "matd3_cooperation_gain_pct_std": gain_stats["std"],
        "positive_cooperation_surplus_rate": _rate(rows, "positive_cooperation_surplus"),
        "cooperative_value_capture_ratio": _mean(rows, "cooperative_value_capture_ratio"),
        "res_utilization_mean_pct": _mean(rows, "res_utilization_pct"),
        "res_utilization_std_pct": _std(rows, "res_utilization_pct"),
        "independent_res_utilization_mean_pct": _mean(rows, "independent_renewable_utilization_pct"),
        "cooperative_milp_res_utilization_mean_pct": _mean(rows, "cooperative_milp_res_utilization_pct"),
        "independent_curtailment_mean_kwh": _mean(rows, "independent_curtailment_kwh"),
        "joint_curtailment_mean_kwh": _mean(rows, "renewable_curtailment_kwh"),
        "cooperative_milp_curtailment_mean_kwh": _mean(rows, "cooperative_milp_curtailment_kwh"),
        "curtailment_reduction_mean_kwh": _mean(rows, "curtailment_reduction_kwh"),
        "curtailment_reduction_mean_pct": _mean(rows, "curtailment_reduction_pct"),
        "feasible_day_rate": _rate(rows, "feasible_day"),
        "soc_violation_rate": _rate(rows, "soc_violation_day"),
        "power_balance_violation_rate": _rate(rows, "power_balance_violation_day"),
        "p2p_balance_violation_rate": _rate(rows, "p2p_balance_violation_day"),
        "milp_failure_count": int(sum(int(row.get("coalition_solve_failure_count", 0)) for row in rows)),
        "shapley_invalid_day_count": int(sum(not bool(row.get("shapley_allocation_valid", False)) for row in rows)),
        "shapley_valid_day_count": int(sum(bool(row.get("shapley_allocation_valid", False)) for row in rows)),
        "physical_profit_mean_by_microgrid": _by_microgrid(rows, "physical_profit_"),
        "physical_profit_std_by_microgrid": _by_microgrid(rows, "physical_profit_", "std"),
        "independent_profit_mean_by_microgrid": {
            f"mg{index}": _mean(rows, f"v_mg{index}") for index in (1, 2, 3)
        },
        "shapley_weight_mean_by_microgrid": _by_microgrid(rows, "shapley_weight_"),
        "shapley_weight_std_by_microgrid": _by_microgrid(rows, "shapley_weight_", "std"),
        "shapley_allocated_surplus_mean_by_microgrid": _by_microgrid(rows, "shapley_allocated_surplus_"),
        "shapley_settled_profit_mean_by_microgrid": _by_microgrid(rows, "shapley_settled_profit_"),
        "shapley_transfer_payment_mean_by_microgrid": _by_microgrid(rows, "shapley_transfer_payment_"),
        "shapley_transfer_payment_std_by_microgrid": _by_microgrid(rows, "shapley_transfer_payment_", "std"),
        "equal_settled_profit_mean_by_microgrid": _by_microgrid(rows, "equal_settled_profit_"),
        "equal_transfer_payment_mean_by_microgrid": _by_microgrid(rows, "equal_transfer_"),
        "equal_transfer_payment_std_by_microgrid": _by_microgrid(rows, "equal_transfer_", "std"),
    }
    result["shapley_ir_rate_by_microgrid"] = {
        f"mg{index}": _rate(rows, f"shapley_ir_satisfied_mg{index}") for index in (1, 2, 3)
    }
    result["equal_ir_rate_by_microgrid"] = {
        f"mg{index}": _rate(rows, f"equal_ir_satisfied_mg{index}") for index in (1, 2, 3)
    }
    for index in (1, 2, 3):
        result[f"shapley_ir_rate_mg{index}"] = result["shapley_ir_rate_by_microgrid"][f"mg{index}"]
    result["shapley_all_ir_satisfaction_rate"] = _rate(rows, "shapley_all_ir_satisfied")
    result["equal_all_ir_satisfaction_rate"] = _rate(rows, "equal_all_ir_satisfied")
    budget_errors = [abs(value) for value in _column_values(rows, "shapley_budget_balance_error")]
    result["shapley_budget_balance_max_error"] = max(budget_errors, default=None)
    result["cooperative_milp_profit_mean"] = _mean(rows, "cooperative_milp_profit")
    result["independent_milp_profit_mean"] = _mean(rows, "independent_total_profit")
    result["cooperative_milp_cooperation_surplus_mean"] = _mean(rows, "cooperative_milp_cooperation_surplus")
    result["cooperative_milp_cooperation_surplus_std"] = _std(rows, "cooperative_milp_cooperation_surplus")
    result["cooperative_milp_cooperation_gain_pct_mean"] = _mean(rows, "cooperative_milp_cooperation_gain_pct")
    result["cooperative_milp_cooperation_gain_pct_std"] = _std(rows, "cooperative_milp_cooperation_gain_pct")
    return result


def evaluate(
    config: Path | str,
    algorithm_config: Path | str,
    checkpoint: Path | str,
    split: str = "test",
    run_milp: bool = True,
    *,
    output_dir: Path | str | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    test_seed: int = 42,
    max_days: int | None = None,
    device: str = "cpu",
    algorithm_name: str = "matd3",
) -> dict[str, Any]:
    spec = get_algorithm_spec(algorithm_name)
    cfg_path, alg_path, chk_path = Path(config), Path(algorithm_config), Path(checkpoint).resolve()
    metadata = dict(checkpoint_metadata or {})
    base_env_config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base_env_config["_project_root"] = str(cfg_path.resolve().parent.parent)
    env_config = spec.build_environment_config(base_env_config)
    env = MultiMicrogridEnv(env_config, split=split)
    loaded_algorithm_config = yaml.safe_load(alg_path.read_text(encoding="utf-8"))
    algorithm = spec.create(
        env,
        loaded_algorithm_config,
        device=device,
        seed=int(test_seed),
    )
    algorithm.load(chk_path)
    days = [str(day.date()) for day in env.data.valid_days]
    if max_days is not None:
        days = days[:max(0, int(max_days))]

    print(f"Testing checkpoint:\n{chk_path}\n")
    print(f"Selected from episode:\n{metadata.get('best_episode')}\n")
    print(f"Selection metric:\n{metadata.get('selection_metric', 'validation_team_profit_mean')}\n")

    matd3_policy = lambda obs: algorithm.select_actions(obs, noise_std=0.0)
    matd3_summary, matd3_daily, hourly_by_day = evaluate_policy_on_days(
        env, days, matd3_policy, seed=test_seed, deterministic_environment=True, capture_hourly=True,
    )
    idle_env = MultiMicrogridEnv(env_config, split=split)
    idle_policy = lambda obs: {
        mg: np.zeros(idle_env.action_dim, dtype=np.float32) for mg in idle_env.agent_ids
    }
    idle_summary, idle_daily = evaluate_policy_on_days(
        idle_env, days, idle_policy, seed=test_seed, deterministic_environment=True,
    )

    matd3_values = np.asarray([day["team_profit"] for day in matd3_daily], dtype=np.float64)
    idle_values = np.asarray([day["team_profit"] for day in idle_daily], dtype=np.float64)
    comparisons = {
        "vs_idle": _paired_comparison(matd3_values, idle_values, "Idle", spec.display_name),
    }

    daily_rows: list[dict[str, Any]] = []
    paired_days: list[dict[str, Any]] = []
    for index, day in enumerate(matd3_daily):
        coalition: CoalitionEvaluation | None = None
        if run_milp:
            coalition = evaluate_day_coalitions(cfg_path, split=split, day=str(day["date"]))
            if not coalition.valid:
                warnings.warn(
                    f"Coalition MILP failure on {day['date']}; Shapley allocation marked invalid: "
                    f"{coalition.statuses}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        row = _daily_row(day, idle_daily[index], coalition, algorithm_slug=spec.slug)
        daily_rows.append(row)
        paired = {
            "day": day["date"], "matd3_profit": day["team_profit"],
            "idle_profit": idle_daily[index]["team_profit"],
            "matd3_terminal_soc": {mg: values["terminal_soc"] for mg, values in
                                    day["soc_statistics_by_microgrid"].items()},
        }
        if run_milp:
            paired.update({
                "independent_milp_profit": row.get("independent_total_profit"),
                "cooperative_milp_profit": row.get("cooperative_milp_profit"),
                "matd3_cooperation_surplus": row.get("matd3_cooperation_surplus"),
                "shapley_allocation_valid": row.get("shapley_allocation_valid"),
            })
        paired_days.append(paired)

    if run_milp:
        independent_values = np.asarray([
            np.nan if row.get("independent_total_profit") is None else row["independent_total_profit"]
            for row in daily_rows
        ], dtype=np.float64)
        cooperative_values = np.asarray([
            np.nan if row.get("cooperative_milp_profit") is None else row["cooperative_milp_profit"]
            for row in daily_rows
        ], dtype=np.float64)
        comparisons["vs_independent_milp"] = _paired_comparison(
            matd3_values, independent_values, "Independent MILP", spec.display_name,
        )
        comparisons["vs_cooperative_milp"] = _paired_comparison(
            matd3_values, cooperative_values, "Cooperative MILP", spec.display_name,
        )

    legacy_comparison = {
        "algorithm_slug": spec.slug,
        "algorithm_profit_mean": matd3_summary["team_profit_mean"],
        "algorithm_vs_idle_gain_mean": comparisons["vs_idle"]["gain_mean_abs"],
        "algorithm_vs_idle_win_rate": comparisons["vs_idle"]["win_rate"],
        "cooperative_matd3_profit_mean": matd3_summary["team_profit_mean"],
        "idle_profit_mean": idle_summary["team_profit_mean"],
        "matd3_beats_idle": bool(matd3_summary["team_profit_mean"] > idle_summary["team_profit_mean"]),
        "matd3_vs_idle_gain_mean": comparisons["vs_idle"]["gain_mean_abs"],
        "matd3_vs_idle_win_rate": comparisons["vs_idle"]["win_rate"],
        "independent_milp_profit_mean": _mean(daily_rows, "independent_total_profit"),
        "cooperative_milp_profit_mean": _mean(daily_rows, "cooperative_milp_profit"),
    }
    joint_summary = _build_joint_summary(daily_rows)
    result = {
        "split": split, "test_days_count": len(days), "test_seed": int(test_seed),
        "algorithm_slug": spec.slug,
        "algorithm_display_name": spec.display_name,
        "method_interface": {
            "action_dim": env.action_dim,
            "observation_dims": env.observation_dims,
            "global_state_dim": env.global_state_dim,
            "ess_sharing_control_mode": env.ess_sharing_control_mode,
            "ess_sharing_observation_mode": env.ess_sharing_observation_mode,
            "ess_p2p_enabled": env.ess_p2p_enabled,
            "mpc_forecast_horizon": loaded_algorithm_config.get("mpc", {}).get("forecast_horizon"),
            "environment_overrides": dict(spec.environment_overrides),
        },
        "deterministic_policy": True, "deterministic_environment": True,
        "common_random_numbers": {
            "enabled": True,
            "definition": "same dates, initial SOC configuration, and per-day environment seeds",
        },
        "checkpoint_used": str(chk_path),
        "checkpoint_episode": metadata.get("best_episode"),
        "checkpoint_selection_metric": metadata.get("selection_metric", "validation_team_profit_mean"),
        "coalition_milp_enabled": bool(run_milp),
        "coalition_evaluation_granularity": "one complete 24-hour test day",
        "team_profit_statistics": matd3_summary["team_profit_statistics"],
        "profit_by_microgrid_mean": matd3_summary["profit_by_microgrid_mean"],
        "renewable_utilization_rate": matd3_summary["renewable_utilization_rate"],
        "p2p_total_energy_kwh_mean": matd3_summary["p2p_total_energy_kwh"],
        "feasible_day_rate": matd3_summary["feasible_rate"],
        "comparisons": comparisons,
        "comparison": legacy_comparison,
        "algorithm": matd3_summary,
        "matd3": matd3_summary,
        "idle": idle_summary, "paired_daily": paired_days,
        **joint_summary,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "test_summary.json", result)
        write_json(destination / "test_evaluation.json", result)
        write_csv(destination / "test_daily_metrics.csv", daily_rows)
        trajectory_dir = destination / "test_hourly_trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for day, rows in hourly_by_day.items():
            write_csv(trajectory_dir / f"{day}.csv", rows)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a registered cooperative algorithm")
    parser.add_argument("--algorithm", default="matd3")
    parser.add_argument("--config", type=Path, default=Path("configs/cooperative_v1.yaml"))
    parser.add_argument("--algorithm-config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--no-milp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test-seed", type=int, default=42)
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    spec = get_algorithm_spec(args.algorithm)
    algorithm_config = args.algorithm_config or spec.default_config_path
    result = evaluate(
        args.config, algorithm_config, args.checkpoint, args.split,
        run_milp=not args.no_milp, output_dir=args.output,
        test_seed=args.test_seed, max_days=args.max_days, device=args.device,
        algorithm_name=spec.slug,
    )
    print(json.dumps({
        "test_days_count": result["test_days_count"],
        "team_profit_statistics": result["team_profit_statistics"],
        "comparisons": result["comparisons"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
