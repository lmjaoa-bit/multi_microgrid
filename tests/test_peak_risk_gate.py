from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.check_peak_risk_gate import (
    CANDIDATE_SLUG, PARENT_SLUG, evaluate_peak_risk_gate,
)


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _saved_run(directory: Path, slug: str, *, peak_advantage: float = 0.0,
               profit: float = 100.0, rebound: float = 10.0) -> Path:
    directory.mkdir(parents=True)
    (directory / "run_status.json").write_text(json.dumps({"status": "COMPLETED"}))
    (directory / "run_metadata.json").write_text(json.dumps({"algorithm_slug": slug}))
    (directory / "training_summary.json").write_text(json.dumps({
        "checkpoint_selection": {"constraint_eligible": True}
    }))
    (directory / "test_summary.json").write_text(json.dumps({
        "checkpoint_used": "checkpoints/best_model.pt",
        "strict_feasible_day_rate": 1.0, "unmet_energy_mean_kwh": 0.0,
    }))
    metrics = {
        "ev_departure_target_success_rate": 100.0,
        "hvac_occupied_comfort_rate": 100.0,
        "mill_terminal_success_rate": 100.0,
        "mill_production_completion_rate": 100.0,
        "hvac_terminal_joint_success_rate": 60.0,
        "mg2_dr_projection_rate": 0.0, "mg3_dr_projection_rate": 0.0,
    }
    (directory / "paper_metrics.json").write_text(json.dumps({
        "metrics": [{"metric_id": key, "value": value} for key, value in metrics.items()]
    }))
    days = ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"]
    _write_csv(
        directory / "dr_shift_metrics.csv",
        ["date", "peak_load_reduction_kw", "shed_energy_kwh", "rebound_energy_kwh"],
        [{"date": day, "peak_load_reduction_kw": peak_advantage,
          "shed_energy_kwh": 20.0, "rebound_energy_kwh": rebound} for day in days],
    )
    _write_csv(
        directory / "test_daily_metrics.csv", ["date", "algorithm_team_profit"],
        [{"date": day, "algorithm_team_profit": profit} for day in days],
    )
    hourly = directory / "test_hourly_trajectories"
    hourly.mkdir()
    for day in days:
        # The physical system peak is 100 kW for the parent and 97 kW
        # for the candidate; the independently saved DR reference is 100 kW.
        actual_peak = 100.0 - peak_advantage
        rows = []
        for hour in range(24):
            total_actual = actual_peak if hour == 18 else 40.0
            total_import = 15.0 - peak_advantage if hour == 18 else 5.0
            rows.append({
                "date": day, "hour": hour,
                **{f"{mg}_actual_load_kw": total_actual / 3.0
                   for mg in ("mg1", "mg2", "mg3")},
                **{f"{mg}_grid_import_kw": total_import / 3.0
                   for mg in ("mg1", "mg2", "mg3")},
            })
        _write_csv(
            hourly / f"{day}.csv",
            ["date", "hour", *[f"{mg}_actual_load_kw" for mg in ("mg1", "mg2", "mg3")],
             *[f"{mg}_grid_import_kw" for mg in ("mg1", "mg2", "mg3")]],
            rows,
        )
    return directory


@pytest.fixture
def saved_runs(tmp_path: Path):
    candidate = _saved_run(tmp_path / "candidate", CANDIDATE_SLUG,
                           peak_advantage=3.0, rebound=5.0)
    parent = _saved_run(tmp_path / "parent", PARENT_SLUG)
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({
        "device": "cpu", "median_update_overhead_pct": 12.0,
        "end_to_end_training_overhead_pct": 10.0,
        "recovery_snapshot_overhead_pct": 1.0,
        "deterministic_evaluation_overhead_pct": 0.0,
        "peak_critic_parameter_fraction": 0.32,
        "projector_call_parity": True, "replay_sample_parity": True,
        "update_calls_per_repetition": 1000, "repetitions": 3,
    }))
    return candidate, parent, runtime


def test_saved_gate_passes_complete_aligned_finite_evidence(saved_runs):
    candidate, parent, runtime = saved_runs
    result = evaluate_peak_risk_gate(candidate, parent, runtime)
    assert result["overall_status"] == "PASS"
    assert result["mean_peak_load_reduction_kw"] == pytest.approx(3.0)
    assert result["positive_peak_reduction_day_rate"] == 1.0
    assert result["mean_grid_import_peak_reduction_kw"] == pytest.approx(3.0)
    assert result["hashes"]["candidate"]["paper_metrics.json"]


def test_saved_gate_uses_weighted_update_overhead_when_paired(saved_runs):
    candidate, parent, runtime = saved_runs
    report = json.loads(runtime.read_text())
    report["comparison_method"] = "median_of_paired_percent_overheads"
    report["weighted_update_overhead_pct"] = 21.0
    # The pooled median would misleadingly pass; paired weighted mean must fail.
    report["median_update_overhead_pct"] = 2.0
    runtime.write_text(json.dumps(report))
    result = evaluate_peak_risk_gate(candidate, parent, runtime)
    assert result["overall_status"] == "FAIL"

    report.pop("weighted_update_overhead_pct")
    runtime.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="weighted_update_overhead_pct"):
        evaluate_peak_risk_gate(candidate, parent, runtime)


def test_saved_gate_rejects_different_test_days(saved_runs):
    candidate, parent, runtime = saved_runs
    target = parent / "dr_shift_metrics.csv"
    target.write_text(target.read_text().replace("2018-01-05", "2018-01-06"))
    with pytest.raises(ValueError, match="date sets differ"):
        evaluate_peak_risk_gate(candidate, parent, runtime)


@pytest.mark.parametrize("field,value", [
    ("median_update_overhead_pct", 21),
    ("end_to_end_training_overhead_pct", 16),
    ("recovery_snapshot_overhead_pct", 4),
    ("deterministic_evaluation_overhead_pct", 6),
    ("peak_critic_parameter_fraction", 0.41),
])
def test_saved_gate_rejects_runtime_regression(saved_runs, field, value):
    candidate, parent, runtime = saved_runs
    report = json.loads(runtime.read_text())
    report[field] = value
    runtime.write_text(json.dumps(report))
    assert evaluate_peak_risk_gate(candidate, parent, runtime)["overall_status"] == "FAIL"


@pytest.mark.parametrize("field,value", [
    ("median_update_overhead_pct", None),
    ("median_update_overhead_pct", True),
    ("median_update_overhead_pct", "5"),
    ("median_update_overhead_pct", float("nan")),
    ("median_update_overhead_pct", float("inf")),
])
def test_saved_gate_fails_closed_nonfinite_or_missing_runtime(saved_runs, field, value):
    candidate, parent, runtime = saved_runs
    report = json.loads(runtime.read_text())
    report[field] = value
    runtime.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=field):
        evaluate_peak_risk_gate(candidate, parent, runtime)


def test_saved_gate_rejects_missing_safety_and_ineligible_selection(saved_runs):
    candidate, parent, runtime = saved_runs
    paper = json.loads((candidate / "paper_metrics.json").read_text())
    paper["metrics"] = [row for row in paper["metrics"] if row["metric_id"] != "mg2_dr_projection_rate"]
    (candidate / "paper_metrics.json").write_text(json.dumps(paper))
    with pytest.raises(ValueError, match="mg2_dr_projection_rate"):
        evaluate_peak_risk_gate(candidate, parent, runtime)
    paper["metrics"].append({"metric_id": "mg2_dr_projection_rate", "value": 0.0})
    (candidate / "paper_metrics.json").write_text(json.dumps(paper))
    training = json.loads((candidate / "training_summary.json").read_text())
    training["checkpoint_selection"]["constraint_eligible"] = False
    (candidate / "training_summary.json").write_text(json.dumps(training))
    with pytest.raises(ValueError, match="constraint_eligible"):
        evaluate_peak_risk_gate(candidate, parent, runtime)


def test_gate_rejects_divergent_saved_reference_peak(saved_runs):
    candidate, parent, runtime = saved_runs
    file = candidate / "dr_shift_metrics.csv"
    file.write_text(file.read_text().replace("3.0", "5.0"))
    with pytest.raises(ValueError, match="DR-reference peak differs"):
        evaluate_peak_risk_gate(candidate, parent, runtime)


def test_gate_rejects_incomplete_or_nonfinite_hourly_physics(saved_runs):
    candidate, parent, runtime = saved_runs
    file = next((candidate / "test_hourly_trajectories").glob("*.csv"))
    text = file.read_text()
    file.write_text(text.replace(",23,", ",bad,"))
    with pytest.raises(ValueError, match="hour"):
        evaluate_peak_risk_gate(candidate, parent, runtime)
    file.write_text(text.replace(",18,", ",18,nan,", 1))
    with pytest.raises(ValueError):
        evaluate_peak_risk_gate(candidate, parent, runtime)
