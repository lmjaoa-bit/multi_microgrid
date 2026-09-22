#!/usr/bin/env python3
"""Fail-closed, saved-result-only acceptance gate for peak-risk Transformer.

This script does not import algorithms, load checkpoints, or execute a rollout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CANDIDATE_SLUG = "matd3_hierarchical_transformer_peak_risk"
PARENT_SLUG = "matd3_hierarchical_transformer_predictive_dr"


def require_number(obj: Mapping[str, Any], key: str, source: str) -> float:
    value = obj.get(key)
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise ValueError(f"{source}.{key} requires a numeric value")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{source}.{key} requires a finite value")
    return value


def require_bool(obj: Mapping[str, Any], key: str, source: str) -> bool:
    value = obj.get(key)
    if type(value) is not bool:
        raise ValueError(f"{source}.{key} requires a boolean value")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} must contain at least one row")
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        date = row.get("date")
        if not date or date in index:
            raise ValueError(f"{path} missing/duplicate date {date}")
        index[date] = row
    return index


def _csv_number(row: Mapping[str, str], field: str, source: str) -> float:
    if field not in row:
        raise ValueError(f"{source} missing {field}")
    raw = row[field]
    if not raw or raw.lower() in {"true", "false"}:
        raise ValueError(f"{source}.{field} is not a finite number")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{source}.{field} is not numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source}.{field} is not finite")
    return value



def _read_hourly_peaks(hourly_dir: Path, dates: set[str]) -> dict[str, dict[str, float]]:
    """Reconstruct physical daily maxima from 24 saved rows, not DR-reference differences."""
    result: dict[str, dict[str, float]] = {}
    actual_fields = tuple(f"{mg}_actual_load_kw" for mg in ("mg1", "mg2", "mg3"))
    grid_fields = tuple(f"{mg}_grid_import_kw" for mg in ("mg1", "mg2", "mg3"))
    for day in sorted(dates):
        source = hourly_dir / f"{day}.csv"
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        hours: dict[int, tuple[float, float]] = {}
        for row in rows:
            if row.get("date") != day:
                raise ValueError(f"{source}: hourly date mismatch")
            hour_raw = _csv_number(row, "hour", str(source))
            if not hour_raw.is_integer() or int(hour_raw) not in range(24):
                raise ValueError(f"{source}: invalid hour {hour_raw}")
            hour = int(hour_raw)
            if hour in hours:
                raise ValueError(f"{source}: duplicate hour {hour}")
            actual_kw = sum(_csv_number(row, col, str(source)) for col in actual_fields)
            import_kw = sum(_csv_number(row, col, str(source)) for col in grid_fields)
            if actual_kw < -1e-6 or import_kw < -1e-6:
                raise ValueError(f"{source}: negative physical system load/import")
            hours[hour] = (actual_kw, import_kw)
        if set(hours) != set(range(24)):
            raise ValueError(f"{source}: expected exactly 24 complete hourly rows")
        result[day] = {
            "actual_load_peak_kw": max(values[0] for values in hours.values()),
            "grid_import_peak_kw": max(values[1] for values in hours.values()),
        }
    return result


def _load_run(directory: Path, slug: str) -> dict[str, Any]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    required = (
        "run_metadata.json", "run_status.json", "training_summary.json",
        "test_summary.json", "paper_metrics.json",
        "dr_shift_metrics.csv", "test_daily_metrics.csv",
    )
    hashes = {name: _hash(directory / name) for name in required}
    metadata = _load_json(directory / "run_metadata.json")
    if metadata.get("algorithm_slug") != slug:
        raise ValueError(f"{directory}: algorithm_slug must be {slug}")
    status = _load_json(directory / "run_status.json")
    if status.get("status") != "COMPLETED":
        raise ValueError(f"{directory}: run_status must be COMPLETED")
    summary = _load_json(directory / "test_summary.json")
    if not str(summary.get("checkpoint_used", "")).replace("\\", "/").endswith("/best_model.pt"):
        raise ValueError(f"{directory}: formal test must use checkpoints/best_model.pt")
    training = _load_json(directory / "training_summary.json")
    selection = training.get("checkpoint_selection")
    if not isinstance(selection, dict) or not require_bool(
        selection, "constraint_eligible", f"{directory}/training_summary.checkpoint_selection"
    ):
        raise ValueError(f"{directory}: checkpoint_selection.constraint_eligible must be true")
    paper = _load_json(directory / "paper_metrics.json")
    items = paper.get("metrics")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{directory}: paper_metrics.metrics must be nonempty")
    paper_values = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("metric_id"), str):
            raise ValueError(f"{directory}: invalid paper_metrics item")
        field = item["metric_id"]
        if field in paper_values:
            raise ValueError(f"{directory}: duplicate paper metric {field}")
        paper_values[field] = require_number(item, "value", f"{directory}/paper_metrics.{field}")
    shift = _read_csv(directory / "dr_shift_metrics.csv")
    daily = _read_csv(directory / "test_daily_metrics.csv")
    if set(daily) != set(shift):
        raise ValueError(f"{directory}: dr_shift_metrics/test_daily_metrics date sets differ")
    for date, row in shift.items():
        for field in ("peak_load_reduction_kw", "shed_energy_kwh", "rebound_energy_kwh"):
            _csv_number(row, field, f"{directory}/dr_shift_metrics[{date}]")
    hourly = directory / "test_hourly_trajectories"
    if not hourly.is_dir():
        raise FileNotFoundError(f"{directory}: missing test_hourly_trajectories")
    saved_days = {file.stem for file in hourly.glob("*.csv")}
    if saved_days != set(daily):
        raise ValueError(f"{directory}: saved hourly/daily date sets differ")
    hourly_peaks = _read_hourly_peaks(hourly, set(daily))
    for file in sorted(hourly.glob("*.csv")):
        hashes[f"test_hourly_trajectories/{file.name}"] = _hash(file)
    return {
        "path": str(directory), "metadata": metadata,
        "test": summary, "training": training, "paper": paper_values,
        "shift": shift, "daily": daily, "hourly_peaks": hourly_peaks,
        "hashes": hashes,
    }


def _require_runtime(report: Mapping[str, Any], key: str) -> float:
    return require_number(report, key, "runtime_report")


def evaluate_peak_risk_gate(candidate_dir: Path, parent_dir: Path,
                            runtime_json: Path, *, rebound_margin: float = 0.0) -> dict[str, Any]:
    """Compare aligned formal-test dates, explicit safety metrics, and measured runtime."""
    if not math.isfinite(rebound_margin) or rebound_margin < 0:
        raise ValueError("rebound_margin must be finite and nonnegative")
    candidate = _load_run(Path(candidate_dir), CANDIDATE_SLUG)
    parent = _load_run(Path(parent_dir), PARENT_SLUG)
    dates = sorted(candidate["daily"])
    if set(dates) != set(parent["daily"]):
        raise ValueError("candidate/parent formal-test date sets differ")
    runtime_path = Path(runtime_json).resolve()
    runtime = _load_json(runtime_path)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, actual: float, limit: float, op: str,
              *, reference: float | None = None) -> None:
        if op == ">":
            passed = actual > limit
        elif op == ">=":
            passed = actual >= limit
        elif op == "<=":
            passed = actual <= limit
        else:
            raise ValueError(op)
        checks[name] = {
            "actual": actual, "threshold": limit, "operator": op,
            "reference": reference, "passed": bool(passed),
        }

    reduction = np.asarray([
        parent["hourly_peaks"][date]["actual_load_peak_kw"]
        - candidate["hourly_peaks"][date]["actual_load_peak_kw"]
        for date in dates
    ], dtype=float)
    grid_peak_delta = np.asarray([
        parent["hourly_peaks"][date]["grid_import_peak_kw"]
        - candidate["hourly_peaks"][date]["grid_import_peak_kw"]
        for date in dates
    ], dtype=float)
    # Independently reconstruct the rule-reference peak implied by each
    # dr_shift_metrics row. A different reference invalidates the paired study.
    for date in dates:
        parent_reference = (
            parent["hourly_peaks"][date]["actual_load_peak_kw"]
            + _csv_number(parent["shift"][date], "peak_load_reduction_kw", date)
        )
        candidate_reference = (
            candidate["hourly_peaks"][date]["actual_load_peak_kw"]
            + _csv_number(candidate["shift"][date], "peak_load_reduction_kw", date)
        )
        if not math.isclose(parent_reference, candidate_reference, rel_tol=1e-6, abs_tol=1e-3):
            raise ValueError(f"{date}: DR-reference peak differs between parent and candidate")
    check("mean_actual_load_peak_reduction_kw", float(reduction.mean()), 0.0, ">")
    check("median_actual_load_peak_reduction_kw", float(np.median(reduction)), 0.0, ">")
    check("positive_peak_reduction_day_rate", float(np.mean(reduction > 0)), 0.60, ">=")
    for label, run in (("candidate", candidate), ("parent", parent)):
        shed = sum(_csv_number(row, "shed_energy_kwh", date)
                   for date, row in run["shift"].items())
        rebound = sum(_csv_number(row, "rebound_energy_kwh", date)
                      for date, row in run["shift"].items())
        if shed <= 0 or rebound < 0:
            raise ValueError(f"{label}: shed_energy_kwh must be positive and rebound nonnegative")
        run["rebound_to_shed_ratio"] = rebound / shed
    check("rebound_to_shed_ratio_reduction",
          parent["rebound_to_shed_ratio"] - candidate["rebound_to_shed_ratio"],
          rebound_margin, ">", reference=parent["rebound_to_shed_ratio"])
    profits = {}
    for label, run in (("candidate", candidate), ("parent", parent)):
        profits[label] = float(np.mean([
            _csv_number(run["daily"][date], "algorithm_team_profit", date) for date in dates
        ]))
    profit_ratio = (profits["candidate"] - profits["parent"]) / max(abs(profits["parent"]), 1e-12)
    check("mean_team_profit_relative_change", profit_ratio, -0.02, ">=",
          reference=profits["parent"])
    ts, pm = candidate["test"], candidate["paper"]
    check("strict_feasible_day_rate", require_number(ts, "strict_feasible_day_rate", "test_summary"),
          1.0, ">=")
    check("unmet_energy_mean_kwh", require_number(ts, "unmet_energy_mean_kwh", "test_summary"),
          1e-9, "<=")
    for field, minimum in (
        ("ev_departure_target_success_rate", 100.0),
        ("hvac_occupied_comfort_rate", 99.5),
    ):
        check(field, require_number(pm, field, "paper_metrics"), minimum, ">=")
    mill = max(
        require_number(pm, "mill_terminal_success_rate", "paper_metrics"),
        require_number(pm, "mill_production_completion_rate", "paper_metrics"),
    )
    check("mill_terminal_or_production_success_rate", mill, 99.5, ">=")
    hvac_joint = require_number(pm, "hvac_terminal_joint_success_rate", "paper_metrics")
    parent_hvac_joint = require_number(parent["paper"], "hvac_terminal_joint_success_rate", "parent paper_metrics")
    check("hvac_terminal_joint_success_rate", hvac_joint, max(45.45454545, parent_hvac_joint), ">=",
          reference=parent_hvac_joint)
    for field in ("mg2_dr_projection_rate", "mg3_dr_projection_rate"):
        check(field, require_number(pm, field, "paper_metrics"), 1e-9, "<=")

    # Speed numbers must come from a separately saved, controlled benchmark.
    for field, maximum in (
        ("median_update_overhead_pct", 20.0),
        ("end_to_end_training_overhead_pct", 15.0),
        ("recovery_snapshot_overhead_pct", 3.0),
        ("deterministic_evaluation_overhead_pct", 5.0),
        ("peak_critic_parameter_fraction", 0.40),
    ):
        check(field, _require_runtime(runtime, field), maximum, "<=")
    if runtime.get("device") == "cuda":
        check("gpu_peak_memory_overhead_pct",
              _require_runtime(runtime, "gpu_peak_memory_overhead_pct"), 15.0, "<=")
    if not require_bool(runtime, "projector_call_parity", "runtime_report"):
        raise ValueError("runtime_report.projector_call_parity must be true")
    if not require_bool(runtime, "replay_sample_parity", "runtime_report"):
        raise ValueError("runtime_report.replay_sample_parity must be true")
    if _require_runtime(runtime, "update_calls_per_repetition") < 1000:
        raise ValueError("runtime_report.update_calls_per_repetition must be >= 1000")
    if _require_runtime(runtime, "repetitions") < 3:
        raise ValueError("runtime_report.repetitions must be >= 3")

    return {
        "overall_status": "PASS" if all(row["passed"] for row in checks.values()) else "FAIL",
        "candidate_run": candidate["path"], "parent_run": parent["path"],
        "dates": dates, "mean_peak_load_reduction_kw": float(reduction.mean()),
        "median_peak_load_reduction_kw": float(np.median(reduction)),
        "p10_peak_load_reduction_kw": float(np.quantile(reduction, .10)),
        "worst_day_peak_increase_kw": float(max(0.0, -reduction.min())),
        "positive_peak_reduction_day_rate": float(np.mean(reduction > 0)),
        "mean_grid_import_peak_reduction_kw": float(grid_peak_delta.mean()),
        "candidate_mean_grid_import_peak_kw": float(np.mean([
            candidate["hourly_peaks"][day]["grid_import_peak_kw"] for day in dates
        ])),
        "parent_mean_grid_import_peak_kw": float(np.mean([
            parent["hourly_peaks"][day]["grid_import_peak_kw"] for day in dates
        ])),
        "rebound_to_shed_ratio_candidate": candidate["rebound_to_shed_ratio"],
        "rebound_to_shed_ratio_parent": parent["rebound_to_shed_ratio"],
        "mean_team_profit_candidate": profits["candidate"],
        "mean_team_profit_parent": profits["parent"],
        "checks": checks,
        "hashes": {
            "candidate": candidate["hashes"], "parent": parent["hashes"],
            "runtime_report": _hash(runtime_path),
        },
        "note": "rebound_margin is an explicit input; any broader materiality judgement requires review",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare saved Peak-Risk and corrected Predictive-DR results")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--runtime-json", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rebound-margin", type=float, default=0.0)
    args = parser.parse_args()
    try:
        result = evaluate_peak_risk_gate(
            args.candidate, args.parent, args.runtime_json,
            rebound_margin=args.rebound_margin)
    except (ValueError, FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Peak-risk gate input error: {exc}\n")
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
