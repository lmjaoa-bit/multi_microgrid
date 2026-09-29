from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from evaluation.plotting import _training_convergence_figure, generate_figures
from evaluation.service_reliability import augment_run_outputs


def _write_v2_saved_run(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame([{
        "date": "2018-01-01",
        "algorithm_team_profit": -120.0,
        "algorithm_joint_profit": -120.0,
        "algorithm_team_reward": -2.5,
        "algorithm_generalized_system_cost": 250.0,
        "idle_team_profit": -140.0,
        "renewable_available_kwh": 100.0,
        "renewable_used_kwh": 90.0,
        "renewable_curtailment_kwh": 10.0,
        "res_utilization_pct": 90.0,
        "grid_import_kwh": 80.0,
        "grid_export_kwh": 0.0,
        "p2p_total_energy_kwh": 20.0,
        "feasible_day": True,
        "soc_violation_day": False,
        "power_balance_violation_day": False,
        "p2p_balance_violation_day": False,
        "physical_profit_mg1": -40.0,
        "physical_profit_mg2": -40.0,
        "physical_profit_mg3": -40.0,
        "v_mg1": -50.0,
        "v_mg2": -50.0,
        "v_mg3": -50.0,
        "equal_settled_profit_mg1": -40.0,
        "equal_settled_profit_mg2": -40.0,
        "equal_settled_profit_mg3": -40.0,
        "shapley_settled_profit_mg1": -40.0,
        "shapley_settled_profit_mg2": -40.0,
        "shapley_settled_profit_mg3": -40.0,
        "equal_ir_satisfied_mg1": True,
        "equal_ir_satisfied_mg2": True,
        "equal_ir_satisfied_mg3": True,
        "equal_all_ir_satisfied": True,
        "shapley_ir_satisfied_mg1": True,
        "shapley_ir_satisfied_mg2": True,
        "shapley_ir_satisfied_mg3": True,
        "shapley_all_ir_satisfied": True,
        "battery_charge_mg1_kwh": 10.0,
        "battery_charge_mg2_kwh": 10.0,
        "battery_charge_mg3_kwh": 10.0,
        "battery_discharge_mg1_kwh": 8.0,
        "battery_discharge_mg2_kwh": 8.0,
        "battery_discharge_mg3_kwh": 8.0,
        "battery_throughput_mg1_kwh": 18.0,
        "battery_throughput_mg2_kwh": 18.0,
        "battery_throughput_mg3_kwh": 18.0,
        "p2p_energy_mg1_to_mg2_kwh": 5.0,
        "p2p_energy_mg1_to_mg3_kwh": 0.0,
        "p2p_energy_mg2_to_mg1_kwh": 0.0,
        "p2p_energy_mg2_to_mg3_kwh": 5.0,
        "p2p_energy_mg3_to_mg1_kwh": 5.0,
        "p2p_energy_mg3_to_mg2_kwh": 5.0,
    }])
    daily.to_csv(root / "test_daily_metrics.csv", index=False)

    summary = {
        "algorithm_display_name": "MATD3",
        "reward_semantics_version": "generalized_system_cost_v2",
        "generalized_system_cost_mean": 250.0,
        "joint_team_profit_mean": -120.0,
        "comparisons": {"vs_idle": {"gain_mean_abs": 20.0, "win_rate": 1.0}},
        "checkpoint_selection_metric": "team_reward_mean",
        "experiment_contract": {"service_metrics_schema_version": "legacy_unknown"},
    }
    (root / "test_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (root / "run_metadata.json").write_text(
        json.dumps({
            "algorithm_display_name": "MATD3",
            "reward_semantics_version": "generalized_system_cost_v2",
            "run_id": "v2-fixture",
            "seed": 42,
        }),
        encoding="utf-8",
    )
    (root / "config_snapshot.yaml").write_text(
        yaml.safe_dump({
            "environment": {
                "reward": {"semantics_version": "generalized_system_cost_v2"}
            },
            "algorithm": {"algorithm": {"warmup_steps": 24}},
        }),
        encoding="utf-8",
    )
    training = [
        {
            "episode": i,
            "total_steps": i * 24,
            "team_reward": -10.0 + i,
            "team_profit": -100.0 + i,
        }
        for i in range(1, 4)
    ]
    validation = [
        {
            "episode": 2,
            "team_reward_mean": -7.0,
            "team_profit_mean": -98.0,
            "generalized_system_cost_mean": 700.0,
        },
        {
            "episode": 3,
            "team_reward_mean": -6.0,
            "team_profit_mean": -97.0,
            "generalized_system_cost_mean": 600.0,
        },
    ]
    (root / "training_history.json").write_text(
        json.dumps(training), encoding="utf-8"
    )
    (root / "validation_history.json").write_text(
        json.dumps(validation), encoding="utf-8"
    )

    hourly_dir = root / "test_hourly_trajectories"
    hourly_dir.mkdir()
    hourly = []
    for hour in range(24):
        row = {
            "date": "2018-01-01",
            "hour": hour,
            "timestamp": f"2018-01-01T{hour:02d}:00:00",
        }
        for mg in ("mg1", "mg2", "mg3"):
            row[f"{mg}_unmet_load_kw"] = 0.0
            row[f"{mg}_dr_supply_infeasible"] = 0
        hourly.append(row)
    pd.DataFrame(hourly).to_csv(
        hourly_dir / "2018-01-01.csv", index=False
    )


def test_v2_training_labels_separate_generalized_return_and_accounting_profit():
    training = [
        {"episode": 1, "total_steps": 24, "team_reward": -5.0, "team_profit": -50.0},
        {"episode": 2, "total_steps": 48, "team_reward": -4.0, "team_profit": -45.0},
    ]
    validation = [
        {"episode": 2, "team_reward_mean": -3.5, "team_profit_mean": -44.0}
    ]
    fig = _training_convergence_figure(
        training,
        validation,
        warmup_steps=24,
        reward_semantics_version="generalized_system_cost_v2",
    )
    labels = [
        line.get_label()
        for axis in fig.axes
        for line in axis.lines
    ]
    assert "Training Generalized Return (raw)" in labels
    assert "Generalized Return MA100" in labels
    assert "Validation Mean Generalized Return" in labels
    assert "Accounting Profit MA100" in labels
    assert "Validation Mean Accounting Profit" in labels
    assert fig.axes[0].get_ylabel().startswith("Generalized Return")
    assert fig.axes[1].get_ylabel() == "Accounting Profit"


def test_saved_v2_figures_do_not_execute_models(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    _write_v2_saved_run(run)

    def boom(*args, **kwargs):
        raise AssertionError("saved-result plotting must not execute models or MILP")

    import envs.multi_microgrid_env as env_module
    import optimization.common_constraints as oracle_module
    import evaluation.evaluate_matd3 as evaluation_module

    monkeypatch.setattr(env_module.MultiMicrogridEnv, "step", boom)
    monkeypatch.setattr(oracle_module, "solve_dispatch", boom)
    monkeypatch.setattr(evaluation_module, "evaluate_policy_on_days", boom)

    paths = generate_figures(run, formats=("png", "pdf"))
    png = [path for path in paths if path.suffix == ".png"]
    pdf = [path for path in paths if path.suffix == ".pdf"]
    assert len(png) == 15
    assert len(pdf) == 15
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_service_augmentation_preserves_reward_and_selection(tmp_path: Path):
    run = tmp_path / "run"
    _write_v2_saved_run(run)
    before_daily = pd.read_csv(run / "test_daily_metrics.csv")
    before_summary = json.loads(
        (run / "test_summary.json").read_text(encoding="utf-8")
    )

    augment_run_outputs(run, generate_plot=False)

    after_daily = pd.read_csv(run / "test_daily_metrics.csv")
    after_summary = json.loads(
        (run / "test_summary.json").read_text(encoding="utf-8")
    )
    for field in (
        "algorithm_team_reward",
        "algorithm_generalized_system_cost",
        "algorithm_team_profit",
    ):
        assert after_daily[field].tolist() == before_daily[field].tolist()
    assert after_summary["reward_semantics_version"] == before_summary[
        "reward_semantics_version"
    ]
    assert after_summary["checkpoint_selection_metric"] == before_summary[
        "checkpoint_selection_metric"
    ]
    assert after_summary["generalized_system_cost_mean"] == before_summary[
        "generalized_system_cost_mean"
    ]
    # Service augmentation may add strict service metrics, but it must not
    # rewrite the saved physical-feasibility field.
    assert after_daily["feasible_day"].tolist() == before_daily["feasible_day"].tolist()
