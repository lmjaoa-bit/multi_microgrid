#!/usr/bin/env python3
"""Saved-result-only acceptance gate checker for Feasible DR Transformer V2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

PRESERVATION_LIMITS = {
    "mg2_projection_rate_max": 0.01,
    "mg3_projection_rate_max": 0.01,
    "ev_success_rate_min": 1.00,
    "ev_shortfall_kwh_per_day_max": 0.05,
    "hvac_comfort_rate_min": 0.995,
    "strict_feasible_rate_min": 1.00,
    "unmet_dr_energy_kwh_per_day_max": 0.0,
    "mg1_production_rate_min": 0.995,
}

HVAC_LIMITS = {
    "terminal_joint_success_rate_min": 0.4545,
    "terminal_joint_success_rate_target": 0.55,
    "terminal_task_penalty_max": 35.25,
}

EV_LIMITS = {
    "ev_valley_energy_share_min": 0.95,
    "mean_peak_load_reduction_pct_min": -4.07,
}

COMBINED_LIMITS = {
    "total_system_cost_max": 874.63,
    "nonterminal_system_cost_max": 839.38,
    "terminal_task_penalty_max": 35.25,
    "ev_valley_energy_share_min": 0.95,
    "mean_peak_grid_import_kwh_max": 283.02,
    "mean_peak_load_reduction_pct_min": -4.07,
    "mean_peak_load_reduction_pct_target": 0.0,
    "p2p_energy_kwh_per_day_min": 1049.78,
    "p2p_offer_acceptance_rate_min": 0.3149,
    "ess_equivalent_cycles_per_day_max": 1.1665,
    "renewable_utilization_rate_min": 0.9661,
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def mean_peak_import(
    rows: list[dict[str, Any]],
    start_hour: int = 16,
    end_hour: int = 21,
) -> float:
    """Calculate mean 16:00-21:00 grid import energy across episodes."""
    if not rows:
        return 0.0

    episode_sums: dict[Any, float] = {}
    for row in rows:
        hour = int(row["hour"])
        if start_hour <= hour <= end_hour:
            ep_id = row.get("episode", row.get("date", "default"))
            val: float
            if "grid_import_kwh" in row:
                val = float(row["grid_import_kwh"])
            elif "grid_import_kw" in row:
                val = float(row["grid_import_kw"])
            else:
                # Sum individual microgrids
                mg_vals = [
                    float(v)
                    for k, v in row.items()
                    if k.endswith("_grid_import_kw") or k.endswith("_grid_import_kwh")
                ]
                val = sum(mg_vals)

            if not math.isfinite(val):
                raise ValueError(f"Non-finite grid import value found: {val}")
            episode_sums[ep_id] = episode_sums.get(ep_id, 0.0) + val

    if not episode_sums:
        return 0.0
    return float(sum(episode_sums.values()) / len(episode_sums))


def read_run_artifacts(
    run_dir: Path,
    expected_slug_prefix: str = "matd3_hierarchical_transformer_feasible_dr_v2",
) -> dict[str, Any]:
    """Read saved artifacts from a run directory without importing simulation logic."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    required_files = [
        "paper_metrics.json",
        "test_summary.json",
        "dr_shift_metrics.csv",
        "run_metadata.json",
        "training_summary.json",
    ]
    hashes: dict[str, str] = {}
    for filename in required_files:
        p = run_dir / filename
        if not p.is_file():
            raise FileNotFoundError(f"Missing required artifact: {p}")
        hashes[filename] = compute_sha256(p)

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    slug = metadata.get("algorithm_slug", "")
    if expected_slug_prefix and not slug.startswith(expected_slug_prefix):
        raise ValueError(
            f"Invalid algorithm_slug in metadata: expected {expected_slug_prefix}*, got '{slug}'"
        )

    test_summary = json.loads((run_dir / "test_summary.json").read_text(encoding="utf-8"))
    ckpt = test_summary.get("checkpoint_used", "")
    if not str(ckpt).replace("\\", "/").endswith("best_model.pt"):
        raise ValueError(
            f"Invalid checkpoint_used in test_summary: expected best_model.pt, got '{ckpt}'"
        )

    paper_metrics_raw = json.loads((run_dir / "paper_metrics.json").read_text(encoding="utf-8"))
    paper_metrics_dict: dict[str, float] = {}
    for item in paper_metrics_raw.get("metrics", []):
        mid = item.get("metric_id")
        val = float(item.get("value", 0.0))
        if not math.isfinite(val):
            raise ValueError(f"Non-finite value in paper_metrics for {mid}: {val}")
        paper_metrics_dict[mid] = val

    # Read dr_shift_metrics.csv
    peak_reductions: list[float] = []
    with (run_dir / "dr_shift_metrics.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val = float(r["peak_load_reduction_pct"])
            if not math.isfinite(val):
                raise ValueError(f"Non-finite value in dr_shift_metrics: {val}")
            peak_reductions.append(val)
    mean_peak_load_reduction = float(sum(peak_reductions) / len(peak_reductions)) if peak_reductions else 0.0

    # Read test_hourly_trajectories
    hourly_dir = run_dir / "test_hourly_trajectories"
    hourly_rows: list[dict[str, Any]] = []
    if hourly_dir.is_dir():
        for csv_file in sorted(hourly_dir.glob("*.csv")):
            hashes[f"test_hourly_trajectories/{csv_file.name}"] = compute_sha256(csv_file)
            with csv_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    hourly_rows.append(r)

    mean_peak_imp = mean_peak_import(hourly_rows, start_hour=16, end_hour=21)

    return {
        "run_dir": str(run_dir),
        "hashes": hashes,
        "metadata": metadata,
        "test_summary": test_summary,
        "paper_metrics": paper_metrics_dict,
        "mean_peak_load_reduction_pct": mean_peak_load_reduction,
        "mean_peak_grid_import_kwh": mean_peak_imp,
    }


def evaluate_v2_gate(
    run_dir: Path,
    baseline_dir: Path | None = None,
    mode: Literal["hvac", "ev", "combined"] = "combined",
) -> dict[str, Any]:
    """Evaluate V2 gate on saved run artifacts."""
    slug_prefix = "matd3_hierarchical_transformer_feasible_dr_v2"
    data = read_run_artifacts(run_dir, expected_slug_prefix=slug_prefix)
    pm = data["paper_metrics"]
    ts = data["test_summary"]

    # Normalize percentage metrics in paper_metrics to fractions (0-1) where appropriate
    actuals: dict[str, float] = {
        "mg2_projection_rate_max": pm.get("mg2_dr_projection_rate", 0.0) / 100.0,
        "mg3_projection_rate_max": pm.get("mg3_dr_projection_rate", 0.0) / 100.0,
        "ev_success_rate_min": pm.get("ev_departure_target_success_rate", 0.0) / 100.0,
        "ev_shortfall_kwh_per_day_max": pm.get("ev_departure_shortfall_kwh_day", 0.0),
        "hvac_comfort_rate_min": pm.get("hvac_occupied_comfort_rate", 0.0) / 100.0,
        "strict_feasible_rate_min": float(ts.get("strict_feasible_day_rate", 0.0)),
        "unmet_dr_energy_kwh_per_day_max": float(ts.get("unmet_energy_mean_kwh", 0.0)),
        "mg1_production_rate_min": pm.get("mill_production_completion_rate", 0.0) / 100.0,
    }

    # Add mode-specific actuals
    if mode == "hvac":
        actuals["terminal_joint_success_rate_min"] = pm.get("hvac_terminal_joint_success_rate", 0.0) / 100.0
        actuals["terminal_joint_success_rate_target"] = pm.get("hvac_terminal_joint_success_rate", 0.0) / 100.0
        actuals["terminal_task_penalty_max"] = pm.get("terminal_task_penalty_day", 0.0)
    elif mode == "ev":
        actuals["ev_valley_energy_share_min"] = pm.get("ev_valley_charge_share", 0.0) / 100.0
        actuals["mean_peak_load_reduction_pct_min"] = data["mean_peak_load_reduction_pct"]
    elif mode == "combined":
        actuals["total_system_cost_max"] = pm.get("total_operating_cost_day", 0.0)
        actuals["nonterminal_system_cost_max"] = pm.get("nonterminal_operating_cost_day", 0.0)
        actuals["terminal_task_penalty_max"] = pm.get("terminal_task_penalty_day", 0.0)
        actuals["ev_valley_energy_share_min"] = pm.get("ev_valley_charge_share", 0.0) / 100.0
        actuals["mean_peak_grid_import_kwh_max"] = data["mean_peak_grid_import_kwh"]
        actuals["mean_peak_load_reduction_pct_min"] = data["mean_peak_load_reduction_pct"]
        actuals["mean_peak_load_reduction_pct_target"] = data["mean_peak_load_reduction_pct"]
        actuals["p2p_energy_kwh_per_day_min"] = float(ts.get("p2p_total_energy_kwh_mean", 0.0))
        # acceptance rate
        p2p_stats = ts.get("algorithm", {}).get("p2p", {}).get("ess_offer", {})
        actuals["p2p_offer_acceptance_rate_min"] = float(p2p_stats.get("energy_acceptance_rate", 0.0))
        actuals["ess_equivalent_cycles_per_day_max"] = pm.get("ess_equivalent_cycles_day", 0.0)
        actuals["renewable_utilization_rate_min"] = float(ts.get("renewable_utilization_rate", 0.0))

    gates: dict[str, Any] = {}
    all_passed = True

    # Check preservation limits
    for k, limit in PRESERVATION_LIMITS.items():
        actual = actuals[k]
        passed = (actual <= limit) if k.endswith("_max") else (actual >= limit)
        if not passed:
            all_passed = False
        gates[k] = {
            "actual": actual,
            "threshold": limit,
            "passed": passed,
            "is_stretch": False,
        }

    # Check mode limits
    mode_limits = HVAC_LIMITS if mode == "hvac" else (EV_LIMITS if mode == "ev" else COMBINED_LIMITS)
    for k, limit in mode_limits.items():
        actual = actuals[k]
        is_target = "target" in k
        passed = (actual <= limit) if k.endswith("_max") else (actual >= limit)
        if not is_target and not passed:
            all_passed = False
        gates[k] = {
            "actual": actual,
            "threshold": limit,
            "passed": passed,
            "is_stretch": is_target,
        }

    # Multi-seed eligibility check for combined mode
    eligible_for_multiseed = all_passed
    mill_silo_success = pm.get("mill_terminal_success_rate", 0.0) / 100.0
    silo_warning = False
    if mode == "combined":
        if mill_silo_success < 0.35:
            eligible_for_multiseed = False
            silo_warning = True

    return {
        "run_directory": str(run_dir),
        "baseline_directory": str(baseline_dir) if baseline_dir else None,
        "mode": mode,
        "overall_status": "PASS" if all_passed else "FAIL",
        "eligible_for_multiseed": eligible_for_multiseed,
        "mill_terminal_silo_success": mill_silo_success,
        "silo_regression_warning": silo_warning,
        "source_hashes": data["hashes"],
        "gates": gates,
    }


def evaluate_v2_multiseed_gate(
    run_dirs: list[Path],
    baseline_dirs: list[Path],
) -> dict[str, Any]:
    """Evaluate multi-seed gate across 3 runs and 3 baselines."""
    if len(run_dirs) != 3 or len(baseline_dirs) != 3:
        raise ValueError(
            f"Multi-seed evaluation requires exactly 3 runs and 3 baselines, got {len(run_dirs)} runs and {len(baseline_dirs)} baselines"
        )

    runs_by_seed: dict[int, dict[str, Any]] = {}
    for rd in run_dirs:
        data = read_run_artifacts(rd)
        seed = int(data["metadata"]["seed"])
        if seed in runs_by_seed:
            raise ValueError(f"Duplicate seed {seed} found in runs")
        runs_by_seed[seed] = data

    baselines_by_seed: dict[int, dict[str, Any]] = {}
    for bd in baseline_dirs:
        data = read_run_artifacts(bd, expected_slug_prefix="matd3_hierarchical_transformer")
        seed = int(data["metadata"]["seed"])
        if seed in baselines_by_seed:
            raise ValueError(f"Duplicate seed {seed} found in baselines")
        baselines_by_seed[seed] = data

    if set(runs_by_seed.keys()) != set(baselines_by_seed.keys()):
        raise ValueError(
            f"Seed mismatch between runs {set(runs_by_seed.keys())} and baselines {set(baselines_by_seed.keys())}"
        )

    seeds = sorted(runs_by_seed.keys())
    per_seed_results: dict[int, Any] = {}
    conditions: dict[str, Any] = {}
    all_passed = True

    # Extract arrays
    mg2_projs = [runs_by_seed[s]["paper_metrics"]["mg2_dr_projection_rate"] / 100.0 for s in seeds]
    mg3_projs = [runs_by_seed[s]["paper_metrics"]["mg3_dr_projection_rate"] / 100.0 for s in seeds]
    ev_succs = [runs_by_seed[s]["paper_metrics"]["ev_departure_target_success_rate"] / 100.0 for s in seeds]
    hvac_comfs = [runs_by_seed[s]["paper_metrics"]["hvac_occupied_comfort_rate"] / 100.0 for s in seeds]
    hvac_joints = [runs_by_seed[s]["paper_metrics"]["hvac_terminal_joint_success_rate"] / 100.0 for s in seeds]
    ev_valleys = [runs_by_seed[s]["paper_metrics"]["ev_valley_charge_share"] / 100.0 for s in seeds]
    costs = [runs_by_seed[s]["paper_metrics"]["total_operating_cost_day"] for s in seeds]
    peak_imports = [runs_by_seed[s]["mean_peak_grid_import_kwh"] for s in seeds]

    base_costs = [baselines_by_seed[s]["paper_metrics"]["total_operating_cost_day"] for s in seeds]
    base_peak_imports = [baselines_by_seed[s]["mean_peak_grid_import_kwh"] for s in seeds]

    def _stats(arr: list[float]) -> dict[str, float]:
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        return {
            "mean": float(np.mean(arr)),
            "median": float(q50),
            "std": float(np.std(arr, ddof=0)),
            "iqr": float(q75 - q25),
        }

    # Condition 1: each seed projection <= 1%
    c1 = all(p <= 0.01 for p in mg2_projs) and all(p <= 0.01 for p in mg3_projs)
    conditions["mg2_mg3_projection_le_1pct"] = {"passed": c1, "mg2": mg2_projs, "mg3": mg3_projs}
    if not c1:
        all_passed = False

    # Condition 2: each seed 100% departure success
    c2 = all(s >= 1.00 for s in ev_succs)
    conditions["ev_departure_success_100pct"] = {"passed": c2, "values": ev_succs}
    if not c2:
        all_passed = False

    # Condition 3: each seed >= 99.5% comfort
    c3 = all(c >= 0.995 for c in hvac_comfs)
    conditions["hvac_occupied_comfort_ge_99_5pct"] = {"passed": c3, "values": hvac_comfs}
    if not c3:
        all_passed = False

    # Condition 4: median MG2 joint thermal >= 45.45%, no seed < 40%
    joint_stats = _stats(hvac_joints)
    c4 = joint_stats["median"] >= 0.4545 and all(j >= 0.40 for j in hvac_joints)
    conditions["hvac_terminal_joint_success_ge_45_45pct"] = {
        "passed": c4,
        "median": joint_stats["median"],
        "min": min(hvac_joints),
        "values": hvac_joints,
    }
    if not c4:
        all_passed = False

    # Condition 5: median EV valley share >= 95%
    valley_stats = _stats(ev_valleys)
    c5 = valley_stats["median"] >= 0.95
    conditions["ev_valley_share_ge_95pct"] = {"passed": c5, "median": valley_stats["median"], "values": ev_valleys}
    if not c5:
        all_passed = False

    # Condition 6: median cost <= baseline median cost
    cost_stats = _stats(costs)
    base_cost_stats = _stats(base_costs)
    c6 = cost_stats["median"] <= base_cost_stats["median"]
    conditions["cost_le_baseline"] = {
        "passed": c6,
        "run_median": cost_stats["median"],
        "baseline_median": base_cost_stats["median"],
    }
    if not c6:
        all_passed = False

    # Condition 7: median peak import <= 1.05 * baseline median
    import_stats = _stats(peak_imports)
    base_import_stats = _stats(base_peak_imports)
    import_threshold = 1.05 * base_import_stats["median"]
    c7 = import_stats["median"] <= import_threshold
    conditions["peak_import_le_105pct_baseline"] = {
        "passed": c7,
        "run_median": import_stats["median"],
        "threshold": import_threshold,
        "baseline_median": base_import_stats["median"],
    }
    if not c7:
        all_passed = False

    return {
        "seeds": seeds,
        "overall_status": "PASS" if all_passed else "FAIL",
        "conditions": conditions,
        "metrics_summary": {
            "total_operating_cost": cost_stats,
            "baseline_total_operating_cost": base_cost_stats,
            "peak_grid_import": import_stats,
            "baseline_peak_grid_import": base_import_stats,
            "hvac_terminal_joint_success": joint_stats,
            "ev_valley_charge_share": valley_stats,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Feasible DR V2 saved-result acceptance gate.")
    parser.add_argument("--run", type=Path, help="Path to single V2 run directory")
    parser.add_argument("--baseline", type=Path, help="Path to baseline run directory")
    parser.add_argument("--mode", choices=["hvac", "ev", "combined"], default="combined")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting gate report")

    parser.add_argument("--multiseed-run", action="append", type=Path, help="Multi-seed run directory (pass 3 times)")
    parser.add_argument("--multiseed-baseline", action="append", type=Path, help="Multi-seed baseline directory (pass 3 times)")
    parser.add_argument("--multiseed-output", type=Path, help="Output path for multi-seed report JSON")

    args = parser.parse_args()

    # Multi-seed mode
    if args.multiseed_run or args.multiseed_baseline or args.multiseed_output:
        if args.run:
            sys.stderr.write("Error: Cannot combine --run with multi-seed options.\n")
            return 2
        if not args.multiseed_output:
            sys.stderr.write("Error: --multiseed-output is required in multi-seed mode.\n")
            return 2
        if len(args.multiseed_run or []) != 3 or len(args.multiseed_baseline or []) != 3:
            sys.stderr.write("Error: Exactly 3 --multiseed-run and 3 --multiseed-baseline arguments required.\n")
            return 2
        if args.multiseed_output.exists() and not args.overwrite:
            sys.stderr.write(f"Error: Output file {args.multiseed_output} exists. Pass --overwrite to overwrite.\n")
            return 2

        try:
            report = evaluate_v2_multiseed_gate(args.multiseed_run, args.multiseed_baseline)
        except Exception as exc:
            sys.stderr.write(f"Error evaluating multi-seed gate: {exc}\n")
            return 2

        args.multiseed_output.parent.mkdir(parents=True, exist_ok=True)
        args.multiseed_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Multi-seed gate status: {report['overall_status']}")
        return 0 if report["overall_status"] == "PASS" else 1

    # Single-run mode
    if not args.run:
        sys.stderr.write("Error: Either --run or multi-seed arguments must be provided.\n")
        return 2

    out_file = args.run / "v2_gate_report.json"
    if out_file.exists() and not args.overwrite:
        sys.stderr.write(f"Error: Report {out_file} already exists. Pass --overwrite to overwrite.\n")
        return 2

    try:
        report = evaluate_v2_gate(args.run, baseline_dir=args.baseline, mode=args.mode)
    except Exception as exc:
        sys.stderr.write(f"Error evaluating gate: {exc}\n")
        return 2

    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Gate evaluation: {report['overall_status']}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
