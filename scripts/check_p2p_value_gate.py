#!/usr/bin/env python3
"""Saved-result-only acceptance gate checker for P2P Value Auxiliary Head Transformer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import sys
from pathlib import Path
from typing import Any

DEFAULT_REFERENCE_RUN = (
    Path("outputs/working_runs/matd3_hierarchical_transformer/runs")
    / "MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42"
)
DEFAULT_PREDICTIVE_RUN = (
    Path("outputs/working_runs/matd3_hierarchical_transformer_predictive_dr/runs")
    / "MATD3HIERARCHICALTRANSFORMERPREDICTIVEDR_2026-09-21_13-01-58_seed42"
)


def _require_finite_number(mapping: dict[str, Any], key: str, source: str) -> float:
    if key not in mapping:
        raise ValueError(f"{source} requires numeric {key}")
    raw = mapping[key]
    if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
        raise ValueError(f"{source} requires numeric {key}")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{source} has non-finite {key}")
    return value


def _require_bool(mapping: dict[str, Any], key: str, source: str) -> bool:
    if key not in mapping or not isinstance(mapping[key], bool):
        raise ValueError(f"{source} requires boolean {key}")
    return bool(mapping[key])


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
    expected_slug_prefix: str = "matd3_hierarchical_transformer_p2p_value",
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
        if not isinstance(mid, str) or not mid:
            raise ValueError("paper_metrics contains an invalid metric_id")
        val = _require_finite_number(item, "value", f"paper_metrics.{mid}")
        paper_metrics_dict[mid] = val

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

    # Extract P2P offer metrics
    p2p_info = test_summary.get("algorithm", {}).get("p2p", {})
    ess_offer = p2p_info.get("ess_offer", {})
    if not ess_offer:
        raise ValueError(f"Missing algorithm.p2p.ess_offer in test_summary for {run_dir}")

    if "offer_mean_kwh" not in ess_offer or "accepted_mean_kwh" not in ess_offer:
        raise ValueError(f"Missing offer_mean_kwh or accepted_mean_kwh for {run_dir}")
    offer_means = ess_offer["offer_mean_kwh"]
    accepted_means = ess_offer["accepted_mean_kwh"]
    if isinstance(offer_means, dict):
        total_offer_kwh = sum(
            _require_finite_number(offer_means, key, "algorithm.p2p.ess_offer.offer_mean_kwh")
            for key in offer_means
        )
    else:
        total_offer_kwh = _require_finite_number(
            {"offer_mean_kwh": offer_means}, "offer_mean_kwh", "algorithm.p2p.ess_offer"
        )

    if isinstance(accepted_means, dict):
        total_accepted_kwh = sum(
            _require_finite_number(
                accepted_means, key, "algorithm.p2p.ess_offer.accepted_mean_kwh"
            )
            for key in accepted_means
        )
    else:
        total_accepted_kwh = _require_finite_number(
            {"accepted_mean_kwh": accepted_means},
            "accepted_mean_kwh",
            "algorithm.p2p.ess_offer",
        )

    acceptance_rate = _require_finite_number(
        ess_offer, "energy_acceptance_rate", "algorithm.p2p.ess_offer"
    )
    if "total_energy_mean_kwh" in p2p_info:
        total_p2p_kwh = _require_finite_number(
            p2p_info, "total_energy_mean_kwh", "algorithm.p2p"
        )
    else:
        total_p2p_kwh = _require_finite_number(
            test_summary, "p2p_total_energy_kwh_mean", "test_summary"
        )

    return {
        "run_dir": str(run_dir),
        "hashes": hashes,
        "metadata": metadata,
        "test_summary": test_summary,
        "training_summary": training_summary,
        "paper_metrics": paper_metrics_dict,
        "mean_peak_grid_import_kwh": mean_peak_imp,
        "p2p_acceptance_rate": acceptance_rate,
        "p2p_offer_kwh": total_offer_kwh,
        "p2p_accepted_kwh": total_accepted_kwh,
        "p2p_total_energy_kwh": total_p2p_kwh,
    }


def evaluate_p2p_value_gate(
    run_dir: Path,
    reference_dir: Path | None = None,
    predictive_dir: Path | None = None,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Evaluate P2P value auxiliary head acceptance and safety gates."""
    candidate_slug_prefix = "matd3_hierarchical_transformer_p2p_value"
    candidate = read_run_artifacts(run_dir, expected_slug_prefix=candidate_slug_prefix)
    pm = candidate["paper_metrics"]
    ts = candidate["test_summary"]

    ref_path = reference_dir or DEFAULT_REFERENCE_RUN
    reference = read_run_artifacts(ref_path, expected_slug_prefix="")

    pred_path = predictive_dir or DEFAULT_PREDICTIVE_RUN
    predictive = read_run_artifacts(pred_path, expected_slug_prefix="")

    conditions: dict[str, dict[str, Any]] = {}
    all_passed = True

    # 0. Checkpoint selection constraint eligibility
    ckpt_sel = (
        candidate.get("training_summary", {}).get("checkpoint_selection")
        or ts.get("checkpoint_metadata")
        or ts.get("checkpoint_selection")
        or {}
    )
    selection_mode = ckpt_sel.get("selection_mode", "unknown")
    constraint_eligible = _require_bool(
        ckpt_sel, "constraint_eligible", "checkpoint_selection"
    )
    c0 = constraint_eligible
    conditions["validation_checkpoint_eligible"] = {
        "passed": c0,
        "selection_mode": selection_mode,
        "constraint_eligible": constraint_eligible,
        "constraint_failures": ckpt_sel.get("constraint_failures", []),
    }
    if not c0:
        all_passed = False

    # 1. Primary P2P Gates (Spec Section 16.1)
    # ESS offer energy acceptance rate >= 36.49%
    cand_accept = candidate["p2p_acceptance_rate"]
    c1 = cand_accept >= 0.3649 - atol
    conditions["p2p_acceptance_ge_36_49pct"] = {
        "passed": c1,
        "actual": cand_accept,
        "threshold": 0.3649,
    }
    if not c1:
        all_passed = False

    # Improvement over predictive exploratory run >= 5 percentage points (0.05)
    pred_accept = predictive["p2p_acceptance_rate"]
    accept_diff = cand_accept - pred_accept
    c2 = accept_diff >= 0.05 - atol
    conditions["p2p_acceptance_improvement_ge_5pp"] = {
        "passed": c2,
        "actual": cand_accept,
        "predictive": pred_accept,
        "improvement": accept_diff,
        "threshold": 0.05,
    }
    if not c2:
        all_passed = False

    # ESS offer volume >= 80% of frozen reference
    cand_offer = candidate["p2p_offer_kwh"]
    ref_offer = reference["p2p_offer_kwh"]
    offer_threshold = ref_offer * 0.80
    c3 = cand_offer >= offer_threshold - atol
    conditions["p2p_offer_volume_ge_80pct_reference"] = {
        "passed": c3,
        "actual": cand_offer,
        "reference": ref_offer,
        "threshold": offer_threshold,
    }
    if not c3:
        all_passed = False

    # ESS accepted energy >= 95% of frozen reference
    cand_accepted = candidate["p2p_accepted_kwh"]
    ref_accepted = reference["p2p_accepted_kwh"]
    accepted_threshold = ref_accepted * 0.95
    c4 = cand_accepted >= accepted_threshold - atol
    conditions["p2p_accepted_volume_ge_95pct_reference"] = {
        "passed": c4,
        "actual": cand_accepted,
        "reference": ref_accepted,
        "threshold": accepted_threshold,
    }
    if not c4:
        all_passed = False

    # Total P2P energy >= 95% of frozen reference
    cand_total_p2p = candidate["p2p_total_energy_kwh"]
    ref_total_p2p = reference["p2p_total_energy_kwh"]
    total_p2p_threshold = ref_total_p2p * 0.95
    c5 = cand_total_p2p >= total_p2p_threshold - atol
    conditions["total_p2p_ge_95pct_reference"] = {
        "passed": c5,
        "actual": cand_total_p2p,
        "reference": ref_total_p2p,
        "threshold": total_p2p_threshold,
    }
    if not c5:
        all_passed = False

    # 2. Inherited Safety Gates (Spec Section 16.2)
    strict_feasible = _require_finite_number(ts, "strict_feasible_day_rate", "test_summary")
    s1 = strict_feasible >= 1.0 - atol
    conditions["strict_feasible_rate_100pct"] = {
        "passed": s1,
        "actual": strict_feasible,
        "threshold": 1.0,
    }
    if not s1:
        all_passed = False

    unmet_energy = _require_finite_number(ts, "unmet_energy_mean_kwh", "test_summary")
    s2 = unmet_energy <= 0.0 + atol
    conditions["unmet_energy_eq_0_kwh"] = {
        "passed": s2,
        "actual": unmet_energy,
        "threshold": 0.0,
    }
    if not s2:
        all_passed = False

    mg2_proj = _require_finite_number(pm, "mg2_dr_projection_rate", "paper_metrics") / 100.0
    s3 = mg2_proj <= 0.0 + atol
    conditions["mg2_projection_rate_0"] = {
        "passed": s3,
        "actual": mg2_proj,
        "threshold": 0.0,
    }
    if not s3:
        all_passed = False

    mg3_proj = _require_finite_number(pm, "mg3_dr_projection_rate", "paper_metrics") / 100.0
    s4 = mg3_proj <= 0.0 + atol
    conditions["mg3_projection_rate_0"] = {
        "passed": s4,
        "actual": mg3_proj,
        "threshold": 0.0,
    }
    if not s4:
        all_passed = False

    ev_success = _require_finite_number(
        pm, "ev_departure_target_success_rate", "paper_metrics"
    ) / 100.0
    s5 = ev_success >= 1.0 - atol
    conditions["ev_departure_success_100pct"] = {
        "passed": s5,
        "actual": ev_success,
        "threshold": 1.0,
    }
    if not s5:
        all_passed = False

    comfort = _require_finite_number(
        pm, "hvac_occupied_comfort_rate", "paper_metrics"
    ) / 100.0
    s6 = comfort >= 0.995 - atol
    conditions["hvac_comfort_ge_99_5pct"] = {
        "passed": s6,
        "actual": comfort,
        "threshold": 0.995,
    }
    if not s6:
        all_passed = False

    cand_hvac_joint = _require_finite_number(
        pm, "hvac_terminal_joint_success_rate", "candidate paper_metrics"
    ) / 100.0
    pred_hvac_joint = _require_finite_number(
        predictive["paper_metrics"],
        "hvac_terminal_joint_success_rate",
        "predictive paper_metrics",
    ) / 100.0
    hvac_joint_threshold = pred_hvac_joint - 0.05
    s7 = cand_hvac_joint >= hvac_joint_threshold - atol
    conditions["hvac_joint_no_worse_than_predictive_5pp"] = {
        "passed": s7,
        "actual": cand_hvac_joint,
        "predictive": pred_hvac_joint,
        "threshold": hvac_joint_threshold,
    }
    if not s7:
        all_passed = False

    cand_late_power = _require_finite_number(
        pm, "hvac_late_power_p95_kw", "candidate paper_metrics"
    )
    pred_late_power = _require_finite_number(
        predictive["paper_metrics"], "hvac_late_power_p95_kw", "predictive paper_metrics"
    )
    late_power_threshold = pred_late_power * 1.05 if pred_late_power > 0.0 else 35.0
    s8 = cand_late_power <= late_power_threshold + atol
    conditions["hvac_late_power_no_worse_than_predictive_5pct"] = {
        "passed": s8,
        "actual": cand_late_power,
        "predictive": pred_late_power,
        "threshold": late_power_threshold,
    }
    if not s8:
        all_passed = False

    mill_success = _require_finite_number(
        pm, "mill_terminal_success_rate", "paper_metrics"
    ) / 100.0
    mill_prod = _require_finite_number(
        pm, "mill_production_completion_rate", "paper_metrics"
    ) / 100.0
    s9 = (mill_success >= 0.995 - atol) or (mill_prod >= 0.995 - atol)
    conditions["mill_terminal_or_production_ge_99_5pct"] = {
        "passed": s9,
        "mill_terminal_success_rate": mill_success,
        "mill_production_completion_rate": mill_prod,
        "threshold": 0.995,
    }
    if not s9:
        all_passed = False

    # 3. Economic Guardrails (Spec Section 16.3)
    cand_cost = _require_finite_number(pm, "total_operating_cost_day", "candidate paper_metrics")
    ref_cost = _require_finite_number(
        reference["paper_metrics"], "total_operating_cost_day", "reference paper_metrics"
    )
    cost_threshold = ref_cost * 1.02
    e1 = cand_cost <= cost_threshold + atol
    conditions["economic_cost_le_102pct_reference"] = {
        "passed": e1,
        "actual": cand_cost,
        "reference": ref_cost,
        "threshold": cost_threshold,
    }
    if not e1:
        all_passed = False

    cand_peak = candidate["mean_peak_grid_import_kwh"]
    ref_peak = reference["mean_peak_grid_import_kwh"]
    peak_threshold = ref_peak * 1.05
    e2 = cand_peak <= peak_threshold + atol
    conditions["economic_peak_import_le_105pct_reference"] = {
        "passed": e2,
        "actual": cand_peak,
        "reference": ref_peak,
        "threshold": peak_threshold,
    }
    if not e2:
        all_passed = False

    cand_res = _require_finite_number(ts, "renewable_utilization_rate", "candidate test_summary")
    ref_res = _require_finite_number(
        reference["test_summary"], "renewable_utilization_rate", "reference test_summary"
    )
    res_threshold = ref_res - 0.005
    e3 = cand_res >= res_threshold - atol
    conditions["economic_res_utilization_within_0_5pct"] = {
        "passed": e3,
        "actual": cand_res,
        "reference": ref_res,
        "threshold": res_threshold,
    }
    if not e3:
        all_passed = False

    return {
        "candidate_run": str(run_dir),
        "reference_run": str(ref_path),
        "predictive_run": str(pred_path),
        "reference_hashes": reference["hashes"],
        "predictive_hashes": predictive["hashes"],
        "overall_status": "PASS" if all_passed else "FAIL",
        "conditions": conditions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check P2P Value Auxiliary Head acceptance gate."
    )
    parser.add_argument("--run", type=Path, required=True, help="Candidate run directory")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_RUN,
        help="Frozen parent baseline run directory",
    )
    parser.add_argument(
        "--predictive",
        type=Path,
        default=DEFAULT_PREDICTIVE_RUN,
        help="Predictive parent exploratory run directory",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    try:
        report = evaluate_p2p_value_gate(args.run, args.reference, args.predictive)
    except Exception as exc:
        sys.stderr.write(f"Gate check error: {exc}\n")
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 70)
        print("P2P VALUE AUXILIARY HEAD GATE REPORT")
        print(f"Candidate  : {report['candidate_run']}")
        print(f"Reference  : {report['reference_run']}")
        print(f"Predictive : {report['predictive_run']}")
        print(f"Status     : {report['overall_status']}")
        print("-" * 70)
        for cond, res in report["conditions"].items():
            status = "PASS" if res.get("passed") else "FAIL"
            print(f"[{status}] {cond}: {res}")
        print("=" * 70)

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
