from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.check_feasible_dr_v2_gate import (
    evaluate_v2_gate,
    evaluate_v2_multiseed_gate,
    mean_peak_import,
)


def test_mean_peak_import_hand_checkable():
    rows = [
        {"episode": 0, "hour": 16, "grid_import_kwh": 100.0},
        {"episode": 0, "hour": 17, "grid_import_kwh": 200.0},
        {"episode": 1, "hour": 16, "grid_import_kwh": 300.0},
        {"episode": 1, "hour": 17, "grid_import_kwh": 400.0},
    ]
    assert mean_peak_import(rows, start_hour=16, end_hour=21) == 500.0


def test_mean_peak_import_microgrid_summed():
    rows = [
        {
            "date": "2018-01-01",
            "hour": 16,
            "mg1_grid_import_kw": 50.0,
            "mg2_grid_import_kw": 30.0,
            "mg3_grid_import_kw": 20.0,
        },
        {
            "date": "2018-01-01",
            "hour": 21,
            "mg1_grid_import_kw": 10.0,
            "mg2_grid_import_kw": 10.0,
            "mg3_grid_import_kw": 10.0,
        },
        {
            "date": "2018-01-02",
            "hour": 18,
            "mg1_grid_import_kw": 100.0,
            "mg2_grid_import_kw": 50.0,
            "mg3_grid_import_kw": 50.0,
        },
    ]
    # Day 1: hour 16 (100) + hour 21 (30) = 130
    # Day 2: hour 18 (200) = 200
    # Mean = (130 + 200) / 2 = 165.0
    assert mean_peak_import(rows, start_hour=16, end_hour=21) == 165.0


def _create_mock_run(
    run_dir: Path,
    slug: str = "matd3_hierarchical_transformer_feasible_dr_v2",
    seed: int = 42,
    checkpoint: str = "checkpoints/best_model.pt",
    overrides: dict[str, Any] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "best_model.pt").touch()

    meta = {
        "algorithm_slug": slug,
        "seed": seed,
        "run_id": f"{slug}_{seed}",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    test_summary = {
        "algorithm_slug": slug,
        "test_seed": seed,
        "checkpoint_used": checkpoint,
        "strict_feasible_day_rate": 1.0,
        "unmet_energy_mean_kwh": 0.0,
        "p2p_total_energy_kwh_mean": 1100.0,
        "renewable_utilization_rate": 0.98,
        "algorithm": {
            "p2p": {
                "ess_offer": {
                    "energy_acceptance_rate": 0.35,
                }
            }
        },
    }
    (run_dir / "test_summary.json").write_text(json.dumps(test_summary), encoding="utf-8")

    training_summary = {
        "episodes": 5000,
        "best_episode": 4200,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(training_summary), encoding="utf-8")

    paper_metrics_data = {
        "algorithm_display_name": "Mock V2",
        "metrics": [
            {"metric_id": "mg2_dr_projection_rate", "unit": "%", "value": 0.0},
            {"metric_id": "mg3_dr_projection_rate", "unit": "%", "value": 0.0},
            {"metric_id": "ev_departure_target_success_rate", "unit": "%", "value": 100.0},
            {"metric_id": "ev_departure_shortfall_kwh_day", "unit": "kWh/day", "value": 0.0},
            {"metric_id": "hvac_occupied_comfort_rate", "unit": "%", "value": 100.0},
            {"metric_id": "mill_production_completion_rate", "unit": "%", "value": 99.8},
            {"metric_id": "mill_terminal_success_rate", "unit": "%", "value": 42.0},
            {"metric_id": "hvac_terminal_joint_success_rate", "unit": "%", "value": 50.0},
            {"metric_id": "ev_valley_charge_share", "unit": "%", "value": 96.0},
            {"metric_id": "total_operating_cost_day", "unit": "currency/day", "value": 850.0},
            {"metric_id": "terminal_task_penalty_day", "unit": "currency/day", "value": 30.0},
            {"metric_id": "nonterminal_operating_cost_day", "unit": "currency/day", "value": 820.0},
            {"metric_id": "ess_equivalent_cycles_day", "unit": "cycles/day", "value": 1.0},
        ],
    }
    if overrides:
        for m in paper_metrics_data["metrics"]:
            if m["metric_id"] in overrides:
                m["value"] = overrides[m["metric_id"]]

    (run_dir / "paper_metrics.json").write_text(json.dumps(paper_metrics_data), encoding="utf-8")

    dr_shift_path = run_dir / "dr_shift_metrics.csv"
    with dr_shift_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "peak_load_reduction_pct"])
        writer.writerow(["2018-01-01", -2.0])
        writer.writerow(["2018-01-02", -1.0])

    hourly_dir = run_dir / "test_hourly_trajectories"
    hourly_dir.mkdir(exist_ok=True)
    h_csv = hourly_dir / "2018-01-01.csv"
    with h_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "hour", "mg1_grid_import_kw", "mg2_grid_import_kw", "mg3_grid_import_kw"])
        for h in range(24):
            # 16-21: 6 hours * 40 kW = 240 kWh peak import
            p = 40.0 if 16 <= h <= 21 else 10.0
            writer.writerow(["2018-01-01", h, p / 3.0, p / 3.0, p / 3.0])


def test_gate_passing_run(tmp_path):
    run_dir = tmp_path / "passing_run"
    baseline_dir = tmp_path / "baseline_run"
    _create_mock_run(run_dir)
    _create_mock_run(baseline_dir, slug="matd3_hierarchical_transformer")

    report = evaluate_v2_gate(run_dir, baseline_dir, mode="combined")
    assert report["overall_status"] == "PASS"
    assert report["eligible_for_multiseed"] is True


def test_gate_failing_run(tmp_path):
    run_dir = tmp_path / "failing_run"
    baseline_dir = tmp_path / "baseline_run"
    _create_mock_run(run_dir, overrides={"hvac_occupied_comfort_rate": 90.0})
    _create_mock_run(baseline_dir, slug="matd3_hierarchical_transformer")

    report = evaluate_v2_gate(run_dir, baseline_dir, mode="combined")
    assert report["overall_status"] == "FAIL"


def test_gate_rejection_invalid_inputs(tmp_path):
    run_dir = tmp_path / "bad_run"
    baseline_dir = tmp_path / "baseline_run"
    _create_mock_run(run_dir, slug="wrong_slug")
    _create_mock_run(baseline_dir, slug="matd3_hierarchical_transformer")

    with pytest.raises(ValueError, match="algorithm_slug"):
        evaluate_v2_gate(run_dir, baseline_dir, mode="combined")


def test_silo_regression_warning(tmp_path):
    run_dir = tmp_path / "silo_warn_run"
    baseline_dir = tmp_path / "baseline_run"
    _create_mock_run(run_dir, overrides={"mill_terminal_success_rate": 30.0})
    _create_mock_run(baseline_dir, slug="matd3_hierarchical_transformer")

    report = evaluate_v2_gate(run_dir, baseline_dir, mode="combined")
    assert report["silo_regression_warning"] is True
    assert report["eligible_for_multiseed"] is False


def test_modes_hvac_and_ev(tmp_path):
    hvac_dir = tmp_path / "hvac_run"
    _create_mock_run(hvac_dir, slug="matd3_hierarchical_transformer_feasible_dr_v2_hvac")
    report_hvac = evaluate_v2_gate(hvac_dir, mode="hvac")
    assert report_hvac["overall_status"] == "PASS"

    ev_dir = tmp_path / "ev_run"
    _create_mock_run(ev_dir, slug="matd3_hierarchical_transformer_feasible_dr_v2_ev")
    report_ev = evaluate_v2_gate(ev_dir, mode="ev")
    assert report_ev["overall_status"] == "PASS"


def test_cli_single_run_and_overwrite(tmp_path):
    run_dir = tmp_path / "cli_run"
    baseline_dir = tmp_path / "cli_base"
    _create_mock_run(run_dir)
    _create_mock_run(baseline_dir, slug="matd3_hierarchical_transformer")

    script = Path("scripts/check_feasible_dr_v2_gate.py").resolve()

    # Successful pass (exit code 0)
    cmd = [sys.executable, str(script), "--run", str(run_dir), "--baseline", str(baseline_dir), "--mode", "combined"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    report_file = run_dir / "v2_gate_report.json"
    assert report_file.exists()
    content1 = report_file.read_bytes()

    # Second invocation without --overwrite (exit code 2)
    res_no_over = subprocess.run(cmd, capture_output=True, text=True)
    assert res_no_over.returncode == 2
    assert report_file.read_bytes() == content1

    # With --overwrite (exit code 0)
    res_over = subprocess.run(cmd + ["--overwrite"], capture_output=True, text=True)
    assert res_over.returncode == 0


def test_cli_failing_run(tmp_path):
    run_dir = tmp_path / "cli_failing_run"
    _create_mock_run(run_dir, overrides={"ev_departure_target_success_rate": 80.0})

    script = Path("scripts/check_feasible_dr_v2_gate.py").resolve()
    cmd = [sys.executable, str(script), "--run", str(run_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1


def test_cli_multiseed(tmp_path):
    runs = []
    bases = []
    for s in (42, 43, 44):
        r_dir = tmp_path / f"run_seed_{s}"
        b_dir = tmp_path / f"base_seed_{s}"
        _create_mock_run(r_dir, seed=s)
        _create_mock_run(b_dir, slug="matd3_hierarchical_transformer", seed=s)
        runs.append(r_dir)
        bases.append(b_dir)

    out_file = tmp_path / "multi_report.json"
    script = Path("scripts/check_feasible_dr_v2_gate.py").resolve()

    cmd = [sys.executable, str(script)]
    for r in runs:
        cmd.extend(["--multiseed-run", str(r)])
    for b in bases:
        cmd.extend(["--multiseed-baseline", str(b)])
    cmd.extend(["--multiseed-output", str(out_file)])

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert out_file.exists()

    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["overall_status"] == "PASS"
    assert report["seeds"] == [42, 43, 44]
    assert "conditions" in report
    assert "metrics_summary" in report
    for metric_key in ("total_operating_cost", "peak_grid_import", "hvac_terminal_joint_success"):
        stats = report["metrics_summary"][metric_key]
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "iqr" in stats
