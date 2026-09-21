#!/usr/bin/env python3
"""Saved-result-only acceptance gate checker for Predictive HVAC DR Transformer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BASELINE_RUN = (
    Path("outputs/working_runs/matd3_hierarchical_transformer/runs")
    / "MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42"
)

HARD_GATES = {
    "hvac_terminal_joint_success_rate_min": 0.4545454545,
    "terminal_task_penalty_day_max": 35.25,
    "hvac_occupied_comfort_rate_min": 0.995,
    "strict_feasible_day_rate_min": 1.0,
    "unmet_energy_mean_kwh_max": 0.0,
    "mill_terminal_success_rate_min": 0.995,
    "ev_departure_target_success_rate_min": 1.0,
    "mg2_dr_projection_rate_max": 0.0,
    "mg3_dr_projection_rate_max": 0.0,
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
    expected_slug_prefix: str = "matd3_hierarchical_transformer_predictive_dr",
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

    peak_reductions: list[float] = []
    with (run_dir / "dr_shift_metrics.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val = float(r["peak_load_reduction_pct"])
            if not math.isfinite(val):
                raise ValueError(f"Non-finite value in dr_shift_metrics: {val}")
            peak_reductions.append(val)
    mean_peak_load_reduction = (
        float(sum(peak_reductions) / len(peak_reductions)) if peak_reductions else 0.0
    )

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

    training_summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))

    return {
        "run_dir": str(run_dir),
        "hashes": hashes,
        "metadata": metadata,
        "test_summary": test_summary,
        "training_summary": training_summary,
        "paper_metrics": paper_metrics_dict,
        "mean_peak_load_reduction_pct": mean_peak_load_reduction,
        "mean_peak_grid_import_kwh": mean_peak_imp,
    }


def evaluate_predictive_hvac_dr_gate(
    run_dir: Path,
    baseline_dir: Path | None = None,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Evaluate Predictive HVAC DR gate on saved run artifacts."""
    candidate_slug_prefix = "matd3_hierarchical_transformer_predictive_dr"
    candidate = read_run_artifacts(run_dir, expected_slug_prefix=candidate_slug_prefix)
    pm = candidate["paper_metrics"]
    ts = candidate["test_summary"]

    baseline_path = baseline_dir or DEFAULT_BASELINE_RUN
    baseline = read_run_artifacts(baseline_path, expected_slug_prefix="")

    conditions: dict[str, dict[str, Any]] = {}
    all_passed = True

    # 0. Checkpoint selection constraint eligibility (Spec Section 15.3)
    ckpt_sel = (
        candidate.get("training_summary", {}).get("checkpoint_selection")
        or ts.get("checkpoint_metadata")
        or ts.get("checkpoint_selection")
        or {}
    )
    selection_mode = ckpt_sel.get("selection_mode", "unknown")
    constraint_eligible = ckpt_sel.get("constraint_eligible", True)
    c0 = bool(constraint_eligible)
    conditions["validation_checkpoint_eligible"] = {
        "passed": c0,
        "selection_mode": selection_mode,
        "constraint_eligible": constraint_eligible,
        "constraint_failures": ckpt_sel.get("constraint_failures", []),
    }
    if not c0:
        all_passed = False

    # 1. HVAC joint terminal thermal success rate >= 45.45%
    hvac_joint = pm.get("hvac_terminal_joint_success_rate", 0.0) / 100.0
    c1 = hvac_joint >= HARD_GATES["hvac_terminal_joint_success_rate_min"] - atol
    conditions["hvac_terminal_joint_success_ge_45_45pct"] = {
        "passed": c1,
        "actual": hvac_joint,
        "threshold": HARD_GATES["hvac_terminal_joint_success_rate_min"],
    }
    if not c1:
        all_passed = False

    # 2. HVAC terminal task penalty <= 35.25
    term_penalty = pm.get("terminal_task_penalty_day", 0.0)
    c2 = term_penalty <= HARD_GATES["terminal_task_penalty_day_max"] + atol
    conditions["terminal_task_penalty_le_35_25"] = {
        "passed": c2,
        "actual": term_penalty,
        "threshold": HARD_GATES["terminal_task_penalty_day_max"],
    }
    if not c2:
        all_passed = False

    # 3. HVAC occupied comfort compliance rate >= 99.5%
    comfort = pm.get("hvac_occupied_comfort_rate", 0.0) / 100.0
    c3 = comfort >= HARD_GATES["hvac_occupied_comfort_rate_min"] - atol
    conditions["hvac_comfort_ge_99_5pct"] = {
        "passed": c3,
        "actual": comfort,
        "threshold": HARD_GATES["hvac_occupied_comfort_rate_min"],
    }
    if not c3:
        all_passed = False

    # 4. Strict feasible day rate == 100%
    strict_feasible = float(ts.get("strict_feasible_day_rate", 0.0))
    c4 = strict_feasible >= HARD_GATES["strict_feasible_day_rate_min"] - atol
    conditions["strict_feasible_rate_100pct"] = {
        "passed": c4,
        "actual": strict_feasible,
        "threshold": HARD_GATES["strict_feasible_day_rate_min"],
    }
    if not c4:
        all_passed = False

    # 5. Unmet energy == 0 kWh/day
    unmet_energy = float(ts.get("unmet_energy_mean_kwh", 0.0))
    c5 = unmet_energy <= HARD_GATES["unmet_energy_mean_kwh_max"] + atol
    conditions["unmet_energy_eq_0_kwh"] = {
        "passed": c5,
        "actual": unmet_energy,
        "threshold": HARD_GATES["unmet_energy_mean_kwh_max"],
    }
    if not c5:
        all_passed = False

    # 6. MG1 Mill terminal success rate >= 99.5% (or production completion >= 99.5%)
    mill_success = pm.get("mill_terminal_success_rate", 0.0) / 100.0
    mill_prod = pm.get("mill_production_completion_rate", 0.0) / 100.0
    c6 = (mill_success >= HARD_GATES["mill_terminal_success_rate_min"] - atol) or (
        mill_prod >= HARD_GATES["mill_terminal_success_rate_min"] - atol
    )
    conditions["mill_terminal_or_production_ge_99_5pct"] = {
        "passed": c6,
        "mill_terminal_success_rate": mill_success,
        "mill_production_completion_rate": mill_prod,
        "threshold": HARD_GATES["mill_terminal_success_rate_min"],
    }
    if not c6:
        all_passed = False

    # 7. MG3 EV departure success rate == 100%
    ev_success = pm.get("ev_departure_target_success_rate", 0.0) / 100.0
    c7 = ev_success >= HARD_GATES["ev_departure_target_success_rate_min"] - atol
    conditions["ev_departure_success_100pct"] = {
        "passed": c7,
        "actual": ev_success,
        "threshold": HARD_GATES["ev_departure_target_success_rate_min"],
    }
    if not c7:
        all_passed = False

    # 8. MG2 DR projection rate == 0
    mg2_proj = pm.get("mg2_dr_projection_rate", 0.0) / 100.0
    c8 = mg2_proj <= HARD_GATES["mg2_dr_projection_rate_max"] + atol
    conditions["mg2_projection_rate_0"] = {
        "passed": c8,
        "actual": mg2_proj,
        "threshold": HARD_GATES["mg2_dr_projection_rate_max"],
    }
    if not c8:
        all_passed = False

    # 9. MG3 DR projection rate == 0
    mg3_proj = pm.get("mg3_dr_projection_rate", 0.0) / 100.0
    c9 = mg3_proj <= HARD_GATES["mg3_dr_projection_rate_max"] + atol
    conditions["mg3_projection_rate_0"] = {
        "passed": c9,
        "actual": mg3_proj,
        "threshold": HARD_GATES["mg3_dr_projection_rate_max"],
    }
    if not c9:
        all_passed = False

    # Economic Guardrails vs baseline (Spec Section 19.2)
    cand_cost = pm.get("total_operating_cost_day", 0.0)
    base_cost = baseline["paper_metrics"].get("total_operating_cost_day", 0.0)
    cost_threshold = base_cost * 1.02
    e1 = cand_cost <= cost_threshold + atol
    conditions["economic_cost_le_102pct_baseline"] = {
        "passed": e1,
        "actual": cand_cost,
        "baseline": base_cost,
        "threshold": cost_threshold,
    }
    if not e1:
        all_passed = False

    cand_peak = candidate["mean_peak_grid_import_kwh"]
    base_peak = baseline["mean_peak_grid_import_kwh"]
    peak_threshold = base_peak * 1.05
    e2 = cand_peak <= peak_threshold + atol
    conditions["economic_peak_import_le_105pct_baseline"] = {
        "passed": e2,
        "actual": cand_peak,
        "baseline": base_peak,
        "threshold": peak_threshold,
    }
    if not e2:
        all_passed = False

    cand_res = float(ts.get("renewable_utilization_rate", 0.0))
    base_res = float(baseline["test_summary"].get("renewable_utilization_rate", 0.0))
    res_threshold = base_res - 0.005
    e3 = cand_res >= res_threshold - atol
    conditions["economic_res_utilization_within_0_5pct"] = {
        "passed": e3,
        "actual": cand_res,
        "baseline": base_res,
        "threshold": res_threshold,
    }
    if not e3:
        all_passed = False

    cand_p2p = float(
        ts.get("algorithm", {})
        .get("p2p", {})
        .get("ess_offer", {})
        .get("energy_acceptance_rate", 0.0)
    )
    base_p2p = float(
        baseline["test_summary"]
        .get("algorithm", {})
        .get("p2p", {})
        .get("ess_offer", {})
        .get("energy_acceptance_rate", 0.0)
    )
    p2p_threshold = base_p2p - 0.05
    e4 = cand_p2p >= p2p_threshold - atol
    conditions["economic_p2p_acceptance_within_5pct"] = {
        "passed": e4,
        "actual": cand_p2p,
        "baseline": base_p2p,
        "threshold": p2p_threshold,
    }
    if not e4:
        all_passed = False

    return {
        "candidate_run": str(run_dir),
        "baseline_run": str(baseline_path),
        "baseline_hashes": baseline["hashes"],
        "overall_status": "PASS" if all_passed else "FAIL",
        "conditions": conditions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Predictive HVAC DR Transformer acceptance gate."
    )
    parser.add_argument("--run", type=Path, required=True, help="Candidate run directory")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_RUN,
        help="Frozen parent baseline run directory",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    try:
        report = evaluate_predictive_hvac_dr_gate(args.run, args.baseline)
    except Exception as exc:
        sys.stderr.write(f"Gate check error: {exc}\n")
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 70)
        print("PREDICTIVE HVAC DR TRANSFORMER GATE REPORT")
        print(f"Candidate : {report['candidate_run']}")
        print(f"Baseline  : {report['baseline_run']}")
        print(f"Status    : {report['overall_status']}")
        print("-" * 70)
        for cond, res in report["conditions"].items():
            status = "PASS" if res.get("passed") else "FAIL"
            print(f"[{status}] {cond}: {res}")
        print("=" * 70)

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
