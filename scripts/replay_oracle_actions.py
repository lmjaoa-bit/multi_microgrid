"""Replay full-information MILP targets as existing two-dimensional actions.

This evaluation entry point never loads or trains a neural network. Original
completed runs are read-only; each evaluation writes a new study directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from diagnostics.oracle_action_replay import replay_oracle_trajectory
from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.service_metrics import SERVICE_METRICS_SCHEMA_VERSION, summarize_day_services
from optimization.common_constraints import DispatchResult, build_dispatch_data, solve_dispatch
from optimization.oracle_replay import compare_oracle_market_replay
from utils.metrics_collector import write_json
from experiments.layout import artifact_directory


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run_inputs(run: Path) -> tuple[dict, pd.DataFrame, dict]:
    run = artifact_directory(run)
    status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETED":
        raise ValueError(f"Source run must be COMPLETED: {run}")
    cfg = yaml.safe_load((run / "config_snapshot.yaml").read_text(encoding="utf-8"))["environment"]
    if cfg.get("action_design") != "single_dispatch_dr":
        raise ValueError("Source run must use single_dispatch_dr")
    cfg["_project_root"] = str(ROOT)
    daily = pd.read_csv(run / "test_daily_metrics.csv")
    required = {"date", "test_seed", "algorithm_team_profit"}
    if not required.issubset(daily.columns) or daily.empty:
        raise ValueError("Source run has no complete daily accounting metrics")
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    if daily["date"].duplicated().any():
        raise ValueError("Duplicate test dates in source run")
    if not np.isfinite(daily[["test_seed", "algorithm_team_profit"]].to_numpy(float)).all():
        raise ValueError("Non-finite source run daily metrics")
    summary = json.loads((run / "test_summary.json").read_text(encoding="utf-8"))
    return cfg, daily, summary


def validate_paired_runs(cfg: dict, reference: pd.DataFrame,
                         other_cfg: dict, other: pd.DataFrame) -> pd.DataFrame:
    if other_cfg != cfg:
        raise ValueError("Comparison environment snapshot differs")
    if set(reference["date"]) != set(other["date"]):
        raise ValueError("Comparison test dates differ")
    aligned = other.set_index("date").loc[reference["date"]].reset_index()
    if not np.array_equal(aligned["test_seed"].to_numpy(), reference["test_seed"].to_numpy()):
        raise ValueError("Comparison daily seeds differ")
    for column in ("generalized_reference_scenario_fingerprint",
                   "generalized_reference_initial_state_fingerprint"):
        if column not in reference or column not in aligned:
            raise ValueError(f"Missing comparison fingerprint: {column}")
        for frame in (reference, aligned):
            if frame[column].isna().any() or not frame[column].astype(str).str.fullmatch("[0-9a-f]{64}").all():
                raise ValueError(f"Invalid comparison fingerprint: {column}")
        if not np.array_equal(reference[column].to_numpy(), aligned[column].to_numpy()):
            raise ValueError(f"Comparison fingerprint differs: {column}")
    return aligned


def validate_study_output(run: Path, output: Path) -> Path:
    run, output = run.resolve(), output.resolve()
    working_root = (ROOT / "outputs/working_runs").resolve()
    if (not run.is_relative_to(working_root) or run.parent.name != "runs"
            or run.parents[2] != working_root):
        raise ValueError("Source run must follow working_runs/<slug>/runs/ layout; output belongs in studies/")
    study_root = (run.parents[1] / "studies").resolve()
    if output == study_root or not output.is_relative_to(study_root):
        raise ValueError(f"Output must be a new child of the selected algorithm's studies directory: {study_root}")
    if output.exists():
        raise ValueError("Output studies directory already exists")
    return output


def _cached_generalized(run: Path, row: pd.Series, accounting: DispatchResult) -> DispatchResult:
    path = run / "generalized_reference_trajectories" / f"{row['date']}.csv"
    if (str(row["generalized_reference_status"]) != "kOptimal"
            or str(row["generalized_reference_proven_optimal"]).lower() != "true"):
        raise ValueError(f"Saved generalized reference not proven optimal: {row['date']}")
    for name in ("scenario_fingerprint", "initial_state_fingerprint"):
        if str(row[f"generalized_reference_{name}"]) != str(accounting.metadata[name]):
            raise ValueError(f"Saved generalized reference mismatch: {name} {row['date']}")
    cost, profit = float(row["generalized_reference_cost"]), float(row["generalized_reference_accounting_profit"])
    if not np.isfinite([cost, profit]).all():
        raise ValueError("Non-finite saved generalized reference objective")
    return DispatchResult(
        mode="cooperative_full_information_relaxed", status="kOptimal",
        total_cost=cost, total_profit=profit, profit_by_microgrid={},
        trajectories=pd.read_csv(path), pair_flows_kwh={}, metadata={
            "objective_kind": "generalized_cost", "allocation": "source_priority_free_allocation",
            "perfect_foresight": True, "proven_optimal": True,
            "solver_mip_gap": float(row["generalized_reference_mip_gap"]),
            "scenario_fingerprint": str(row["generalized_reference_scenario_fingerprint"]),
            "initial_state_fingerprint": str(row["generalized_reference_initial_state_fingerprint"]),
            "reused_saved_trajectory": True,
        },
    )


def _day_metrics(replay: dict, oracle: DispatchResult, cfg: dict) -> dict[str, Any]:
    metrics, hourly = replay["metrics"], pd.DataFrame(replay["hourly_rows"])
    actions = pd.DataFrame(replay["action_diagnostics"])
    service = summarize_day_services(replay["hourly_rows"], cfg, True, SERVICE_METRICS_SCHEMA_VERSION)
    ids = [str(m["id"]) for m in cfg["microgrids"]]
    dt = float(cfg["time"]["delta_t"])
    market = compare_oracle_market_replay(oracle, replay["hourly_rows"])
    return {
        "date": metrics["date"], "test_seed": metrics["test_seed"],
        "oracle_profit": float(oracle.total_profit), "oracle_cost": float(oracle.total_cost),
        "replay_profit": float(metrics["team_profit"]),
        "replay_generalized_cost": float(metrics["generalized_system_cost"]),
        "profit_loss_from_mapping": float(oracle.total_profit - metrics["team_profit"]),
        "solver_status": oracle.status, "solver_mip_gap": oracle.metadata.get("solver_mip_gap"),
        "physical_feasible": bool(metrics["constraint_statistics"]["physical_feasible_episode"]),
        "strict_feasible": bool(service["strict_feasible_day"]),
        "hvac_comfort_success_rate": float(service["hvac"]["all_boundary_comfort_rate"]),
        "hvac_comfort_degree_hours": float(service["hvac"]["comfort_violation_degree_hours"]),
        "ev_departure_shortfall_kwh": float(service["ev"]["departure_shortfall_kwh"]),
        "grid_import_kwh": float(sum(hourly[f"{mg}_grid_import_kw"].sum() for mg in ids) * dt),
        "curtailment_kwh": float(sum(hourly[f"{mg}_renewable_curtailment_kw"].sum() for mg in ids) * dt),
        "p2p_sell_kwh": float(sum(hourly[f"{mg}_p2p_sell_kw"].sum() for mg in ids) * dt),
        "dispatch_clipped_count": int(actions["dispatch_target_clipped"].sum()),
        "dispatch_requested_error_max_abs_kw": float(actions["dispatch_request_error_kw"].abs().max()),
        "charge_error_max_abs_kw": float(actions["charge_error_kw"].abs().max()),
        "discharge_error_max_abs_kw": float(actions["discharge_error_kw"].abs().max()),
        "dr_error_max_abs_kw": float(actions["flexible_load_error_kw"].abs().max()),
        "soc_error_max_abs": float(actions["soc_error"].abs().max()),
        "flow_error_max_abs_kw": float(market["allocation_replay_gap_max_abs_kw"]),
        "flow_exactly_replayable": bool(market["exactly_replayable"]),
        "action_rows": len(actions), "hourly_rows": len(hourly),
    }


def run_study(run: Path, compare_runs: list[Path], output: Path, max_days: int | None = None) -> dict:
    start = time.perf_counter()
    output = validate_study_output(run, output)
    cfg, reference, test_summary = load_run_inputs(run)
    comparisons = []
    for path in [run, *compare_runs]:
        other_cfg, daily, summary = load_run_inputs(path)
        aligned = validate_paired_runs(cfg, reference, other_cfg, daily)
        comparisons.append((path, str(summary["algorithm_slug"]), aligned))
    if max_days is not None:
        if max_days <= 0:
            raise ValueError("max_days must be positive")
        reference = reference.iloc[:max_days].copy()
    output.mkdir(parents=True, exist_ok=False)
    cfg_path = output / "environment_snapshot.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    source_files = []
    for path, _, _ in comparisons:
        for name in ("config_snapshot.yaml", "test_daily_metrics.csv", "test_summary.json", "run_status.json"):
            source_files.append({"path": str(artifact_directory(path) / name), "sha256": _sha256(artifact_directory(path) / name)})
    for name in ("diagnostics/oracle_action_replay.py", "scripts/replay_oracle_actions.py"):
        source_files.append({"path": str(ROOT / name), "sha256": _sha256(ROOT / name)})
    metadata = {
        "evaluation_only": True, "training_started": False, "source_run": str(run),
        "comparison_runs": [str(p) for p in compare_runs],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
        "dates": reference["date"].tolist(), "days": len(reference),
        "actual_environment_step": True, "state_injection": False,
        "paired_scenario_and_initial_fingerprints_verified": True,
        "forecast_noise": bool(cfg["observation"].get("evaluation_forecast_noise", False)),
        "source_files": source_files,
        "mapping": "Live-SOC signed ESS power; mill duty; signed HVAC power; live-feasible cohort EV total power",
        "oracle_objectives": ["accounting_profit", "generalized_cost"],
        "certified_action_space_upper_bound": False,
        "limitations": [
            "Mapped replay is an attainable value, not a global optimum over existing raw actions.",
            "Oracle uses exact future information; learned policies receive noisy forecasts.",
            "MILP permits free source-flow and EV-group allocation; existing environment rules execute the mapped controls.",
            "Accounting profit does not price soft HVAC comfort, carbon or peak demand; generalized cost does.",
            "Results diagnose the selected 55-day scenario set; only one trained seed is compared.",
        ],
    }
    write_json(output / "metadata.json", metadata)
    env = MultiMicrogridEnv(cfg, split="test")
    records = []
    for index, (_, row) in enumerate(reference.iterrows()):
        date, seed = str(row["date"]), int(row["test_seed"])
        _, data = build_dispatch_data(cfg, split="test", day=date)
        accounting = solve_dispatch(cfg, data, data.agent_ids, cooperative=True,
                                    include_dr=True, objective_kind="accounting_profit")
        if accounting.status != "kOptimal":
            raise RuntimeError(f"Accounting oracle failed: {date} {accounting.status}")
        if abs(accounting.total_profit - float(row["cooperative_milp_profit"])) > 1e-3:
            raise ValueError(f"Recomputed accounting reference differs from completed run: {date}")
        generalized = _cached_generalized(run, row, accounting)
        source_path = run / "generalized_reference_trajectories" / f"{date}.csv"
        source_files.append({"path": str(source_path), "sha256": _sha256(source_path)})
        for objective, oracle in (("accounting_profit", accounting), ("generalized_cost", generalized)):
            destination = output / objective
            for folder in ("oracle_hourly", "replay_hourly", "action_diagnostics", "day_evidence"):
                (destination / folder).mkdir(parents=True, exist_ok=True)
            oracle.trajectories.to_csv(destination / "oracle_hourly" / f"{date}.csv", index=False)
            replay = replay_oracle_trajectory(env, oracle.trajectories, day=date, seed=seed,
                                             forecast_noise=metadata["forecast_noise"])
            pd.DataFrame(replay["hourly_rows"]).to_csv(destination / "replay_hourly" / f"{date}.csv", index=False)
            pd.DataFrame(replay["action_diagnostics"]).to_csv(destination / "action_diagnostics" / f"{date}.csv", index=False)
            market = compare_oracle_market_replay(oracle, replay["hourly_rows"])
            write_json(destination / "day_evidence" / f"{date}.json", {
                "oracle": oracle.summary(), "replay_metrics": replay["metrics"], "flow_comparison": market,
            })
            record = {"objective": objective, **_day_metrics(replay, oracle, cfg)}
            for _, slug, baseline in comparisons:
                baseline_row = baseline.loc[baseline["date"] == date].iloc[0]
                record[f"{slug}_profit"] = float(baseline_row["algorithm_team_profit"])
                record[f"gain_vs_{slug}"] = float(record["replay_profit"] - baseline_row["algorithm_team_profit"])
            records.append(record)
        # Persist each complete paired day so a failed evaluation is inspectable.
        pd.DataFrame(records).to_csv(output / "daily_comparison.csv", index=False)
        print(f"[{index+1}/{len(reference)}] {date}: profit-optimal replay={records[-2]['replay_profit']:.3f}, reward-optimal replay={records[-1]['replay_profit']:.3f}", flush=True)
    frame = pd.DataFrame(records)
    modes = {}
    for objective, rows in frame.groupby("objective", sort=False):
        modes[objective] = {
            "days": len(rows),
            **{f"{name}_mean": float(rows[name].mean()) for name in (
                "oracle_profit", "replay_profit", "profit_loss_from_mapping", "replay_generalized_cost",
                "hvac_comfort_success_rate", "hvac_comfort_degree_hours", "ev_departure_shortfall_kwh",
                "grid_import_kwh", "curtailment_kwh", "p2p_sell_kwh")},
            "physical_feasible_days": int(rows["physical_feasible"].sum()),
            "strict_feasible_days": int(rows["strict_feasible"].sum()),
            "dispatch_clipped_count": int(rows["dispatch_clipped_count"].sum()),
            "flow_exactly_replayable_days": int(rows["flow_exactly_replayable"].sum()),
            **{name: float(rows[name].max()) for name in (
                "dispatch_requested_error_max_abs_kw", "charge_error_max_abs_kw", "discharge_error_max_abs_kw",
                "dr_error_max_abs_kw", "soc_error_max_abs", "flow_error_max_abs_kw")},
            "comparison": {slug: {
                "trained_profit_mean": float(rows[f"{slug}_profit"].mean()),
                "replay_gain_mean": float(rows[f"gain_vs_{slug}"].mean()),
                "replay_winning_days": int((rows[f"gain_vs_{slug}"] > 1e-6).sum()),
            } for _, slug, _ in comparisons},
        }
    best = frame.groupby("date")["replay_profit"].max()
    summary = {
        "status": "complete", "days": len(reference), "evaluation_seconds": time.perf_counter() - start,
        "modes": modes, "best_of_two_mapped_schedules_profit_mean": float(best.mean()),
        "certified_action_space_upper_bound": False,
        "conclusion_scope": "Feasible full-information replay values; no claim of global two-action optimality",
    }
    # Verify source artifacts stayed byte-for-byte unchanged.
    for source in source_files:
        if _sha256(Path(source["path"])) != source["sha256"]:
            raise RuntimeError(f"Source artifact changed during evaluation: {source['path']}")
    metadata["source_files"] = source_files
    metadata["source_artifacts_unchanged"] = True
    write_json(output / "metadata.json", metadata)
    write_json(output / "summary.json", summary)
    lines = ["# MILP 两维动作真实环境回放", "", f"共 {len(reference)} 个相同测试日；只做评估，没有启动训练。", "",
             "| 轨迹目标 | MILP 日均利润 | 两维动作回放日均利润 | 映射损失 | 广义成本 | HVAC 舒适率 | EV 日均欠充 kWh |", "|---|---:|---:|---:|---:|---:|---:|"]
    for objective, label in (("accounting_profit", "利润最优"), ("generalized_cost", "当前奖励最优")):
        m = modes[objective]
        lines.append(f"| {label} | {m['oracle_profit_mean']:.3f} | {m['replay_profit_mean']:.3f} | {m['profit_loss_from_mapping_mean']:.3f} | {m['replay_generalized_cost_mean']:.3f} | {100*m['hvac_comfort_success_rate_mean']:.2f}% | {m['ev_departure_shortfall_kwh_mean']:.6f} |")
    lines += ["", "原始 MILP 的设备功率目标经现有动作映射进入 `env.step`，使用同一 DR、可再生能源优先和比例交易分配规则；24 小时内不恢复或注入 MILP 状态。", "",
              "本结果是现有动作与规则下**已实现的可行值**，不是穷举或优化全部动作后的全局上限。即使回放接近 -800，也不能据此断言任何网络都无法改善。", "",
              "利润最优目标没有给 HVAC 舒适、碳排放和峰值计价；当前训练优化的是含这些成本的广义奖励，应优先参考当前奖励最优轨迹的回放。", "",
              "逐日结果：`daily_comparison.csv`；每种目标目录保存 MILP/回放逐小时轨迹、两维动作及状态误差、流分配比较与服务指标。来源和哈希见 `metadata.json`。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed source run with saved generalized MILP trajectories")
    parser.add_argument("--compare-run", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-days", type=int, help="Limit evaluation days for a short smoke; never trains")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    output = args.output.resolve() if args.output else run.parents[1] / "studies" / (
        "oracle_action_replay_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    summary = run_study(run, [p.resolve() for p in args.compare_run], output, args.max_days)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
