from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.check_predictive_hvac_dr_gate import (
    DEFAULT_BASELINE_RUN,
    evaluate_predictive_hvac_dr_gate,
    read_run_artifacts,
)


def _create_mock_run(
    tmp_path: Path,
    *,
    hvac_joint: float = 0.50,
    term_penalty: float = 30.0,
    comfort_rate: float = 0.998,
    strict_feasible: float = 1.0,
    unmet_energy: float = 0.0,
    mill_success: float = 1.0,
    mill_prod: float = 1.0,
    ev_success: float = 1.0,
    mg2_proj: float = 0.0,
    mg3_proj: float = 0.0,
    total_cost: float = 870.0,
    peak_import: float = 270.0,
    res_util: float = 0.970,
    p2p_acceptance: float = 0.360,
    constraint_eligible: bool = True,
    slug: str = "matd3_hierarchical_transformer_predictive_dr",
    checkpoint: str = "checkpoints/best_model.pt",
) -> Path:
    run_dir = tmp_path / "mock_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # run_metadata.json
    metadata = {"algorithm_slug": slug, "run_id": "mock_run_1"}
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    # training_summary.json
    training_summary = {
        "checkpoint_selection": {
            "selection_mode": "constrained_profit",
            "constraint_eligible": constraint_eligible,
            "best_checkpoint": checkpoint,
        }
    }
    (run_dir / "training_summary.json").write_text(json.dumps(training_summary), encoding="utf-8")

    # test_summary.json
    test_summary = {
        "checkpoint_used": checkpoint,
        "strict_feasible_day_rate": strict_feasible,
        "unmet_energy_mean_kwh": unmet_energy,
        "renewable_utilization_rate": res_util,
        "algorithm": {
            "p2p": {
                "ess_offer": {
                    "energy_acceptance_rate": p2p_acceptance,
                }
            }
        },
        "checkpoint_metadata": {
            "selection_mode": "constrained_profit",
            "constraint_eligible": constraint_eligible,
        },
    }
    (run_dir / "test_summary.json").write_text(json.dumps(test_summary), encoding="utf-8")

    # paper_metrics.json
    paper_metrics = {
        "metrics": [
            {"metric_id": "hvac_terminal_joint_success_rate", "value": hvac_joint * 100.0},
            {"metric_id": "terminal_task_penalty_day", "value": term_penalty},
            {"metric_id": "hvac_occupied_comfort_rate", "value": comfort_rate * 100.0},
            {"metric_id": "mill_terminal_success_rate", "value": mill_success * 100.0},
            {"metric_id": "mill_production_completion_rate", "value": mill_prod * 100.0},
            {"metric_id": "ev_departure_target_success_rate", "value": ev_success * 100.0},
            {"metric_id": "mg2_dr_projection_rate", "value": mg2_proj * 100.0},
            {"metric_id": "mg3_dr_projection_rate", "value": mg3_proj * 100.0},
            {"metric_id": "total_operating_cost_day", "value": total_cost},
        ]
    }
    (run_dir / "paper_metrics.json").write_text(json.dumps(paper_metrics), encoding="utf-8")

    # dr_shift_metrics.csv
    (run_dir / "dr_shift_metrics.csv").write_text(
        "date,peak_load_reduction_pct\n2018-04-27,5.0\n", encoding="utf-8"
    )

    # test_hourly_trajectories with peak grid import
    hourly_dir = run_dir / "test_hourly_trajectories"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    # Peak hours: 16 to 21 (6 hours). peak_import / 6.0 per hour
    rows = ["date,hour,grid_import_kwh\n"]
    for h in range(24):
        imp = (peak_import / 6.0) if 16 <= h <= 21 else 10.0
        rows.append(f"2018-04-27,{h},{imp}\n")
    (hourly_dir / "2018-04-27.csv").write_text("".join(rows), encoding="utf-8")

    return run_dir


def test_gate_checker_passes_on_valid_run(tmp_path: Path):
    if not DEFAULT_BASELINE_RUN.exists():
        pytest.skip("Frozen baseline directory not found")

    mock_dir = _create_mock_run(tmp_path)
    report = evaluate_predictive_hvac_dr_gate(mock_dir, DEFAULT_BASELINE_RUN)
    assert report["overall_status"] == "PASS"
    for cond, res in report["conditions"].items():
        assert res["passed"] is True, f"Condition {cond} failed: {res}"


def test_gate_checker_fails_on_low_joint_success(tmp_path: Path):
    if not DEFAULT_BASELINE_RUN.exists():
        pytest.skip("Frozen baseline directory not found")

    mock_dir = _create_mock_run(tmp_path, hvac_joint=0.40)  # below 45.45%
    report = evaluate_predictive_hvac_dr_gate(mock_dir, DEFAULT_BASELINE_RUN)
    assert report["overall_status"] == "FAIL"
    assert report["conditions"]["hvac_terminal_joint_success_ge_45_45pct"]["passed"] is False


def test_gate_checker_fails_on_ineligible_checkpoint(tmp_path: Path):
    if not DEFAULT_BASELINE_RUN.exists():
        pytest.skip("Frozen baseline directory not found")

    mock_dir = _create_mock_run(tmp_path, constraint_eligible=False)
    report = evaluate_predictive_hvac_dr_gate(mock_dir, DEFAULT_BASELINE_RUN)
    assert report["overall_status"] == "FAIL"
    assert report["conditions"]["validation_checkpoint_eligible"]["passed"] is False


def test_gate_checker_fails_on_cost_regression(tmp_path: Path):
    if not DEFAULT_BASELINE_RUN.exists():
        pytest.skip("Frozen baseline directory not found")

    # Baseline cost is ~874.63. 1.02 * 874.63 = 892.12. Set to 950.0
    mock_dir = _create_mock_run(tmp_path, total_cost=950.0)
    report = evaluate_predictive_hvac_dr_gate(mock_dir, DEFAULT_BASELINE_RUN)
    assert report["overall_status"] == "FAIL"
    assert report["conditions"]["economic_cost_le_102pct_baseline"]["passed"] is False
