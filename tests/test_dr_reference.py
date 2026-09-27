from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.dr_reference import (
    DeterministicDRReferencePolicy,
    evaluate_and_save_dr_reference,
    evaluate_dr_reference,
    save_dr_reference_results,
)
from evaluation.evaluate_matd3 import evaluate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"


def _environment_config() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["_project_root"] = str(ROOT)
    return config


def test_reference_policy_uses_task_preserving_device_rules() -> None:
    env = MultiMicrogridEnv(_environment_config(), split="test")
    observations, _ = env.reset(seed=42, deterministic=True)
    policy = DeterministicDRReferencePolicy(env)

    actions = policy(observations)

    mill = env.cfg_by_id["mg1"]["dr"]
    expected_mill_raw = 2.0 * (
        float(mill["downstream_consumption_tph"])
        / float(mill["rated_production_tph"])
    ) - 1.0
    assert actions["mg1"][0] == 0.0
    np.testing.assert_allclose(
        actions["mg1"][1], expected_mill_raw, atol=1e-7
    )
    assert actions["mg2"].tolist() == [0.0, 0.0]
    assert actions["mg3"].tolist() == [0.0, 1.0]

    # The cohort reference always requests the current feasible maximum.
    # Availability/departure windows are enforced by the EV device projector.
    assert policy(observations)["mg3"].tolist() == [0.0, 1.0]


def test_reference_evaluation_saves_complete_hourly_and_daily_results(tmp_path: Path) -> None:
    config = _environment_config()
    probe = MultiMicrogridEnv(config, split="test")
    day = str(probe.data.valid_days[0].date())

    summary, daily, hourly = evaluate_dr_reference(
        config,
        split="test",
        days=[day],
        seed=42,
        forecast_noise=False,
    )
    paths = save_dr_reference_results(tmp_path, summary, daily, hourly)

    assert summary["team_profit_statistics"]["count"] == 1
    assert len(hourly[day]) == 24
    frame = pd.DataFrame(hourly[day])
    for mg in ("mg1", "mg2", "mg3"):
        np.testing.assert_allclose(
            frame[f"{mg}_actual_load_kw"],
            frame[f"{mg}_base_load_kw"] + frame[f"{mg}_flexible_load_kw"],
            atol=1e-8,
        )
        assert (frame[f"{mg}_flexible_load_kw"] >= 0.0).all()
    assert abs(float(frame["mg1_silo_level_t"].iloc[-1]) - 10.53) < 1e-6
    assert float(frame.loc[frame["hour"] == 6, "mg3_ev_soc"].iloc[0]) >= 0.80 - 1e-7

    assert {path.name for path in paths} == {
        "dr_reference_summary.json",
        "dr_reference_daily_metrics.csv",
        "dr_reference_hourly_trajectories",
    }
    assert (tmp_path / "dr_reference_hourly_trajectories" / f"{day}.csv").is_file()
    daily_frame = pd.read_csv(tmp_path / "dr_reference_daily_metrics.csv")
    assert daily_frame.loc[0, "date"] == day
    assert daily_frame.loc[0, "reference_team_profit"] == daily[0]["team_profit"]


def test_reference_evaluation_and_save_is_one_formal_test_operation(tmp_path: Path) -> None:
    config = _environment_config()
    probe = MultiMicrogridEnv(config, split="test")
    day = str(probe.data.valid_days[0].date())

    summary, daily, hourly = evaluate_and_save_dr_reference(
        config,
        split="test",
        days=[day],
        seed=42,
        forecast_noise=False,
        output_dir=tmp_path,
    )

    assert summary["team_profit_statistics"]["count"] == 1
    assert daily[0]["date"] == day
    assert len(hourly[day]) == 24
    assert (tmp_path / "dr_reference_summary.json").is_file()
    assert (tmp_path / "dr_reference_daily_metrics.csv").is_file()
    assert (tmp_path / "dr_reference_hourly_trajectories" / f"{day}.csv").is_file()


def test_formal_evaluation_includes_saved_rule_reference(tmp_path: Path) -> None:
    spec = get_algorithm_spec("matd3_dispatch")
    config = _environment_config()
    env = MultiMicrogridEnv(spec.build_environment_config(config), split="test")
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "model.pt"
    spec.create(env, settings, device="cpu", seed=42).save(checkpoint)
    output = tmp_path / "evaluation"

    result = evaluate(
        CONFIG,
        spec.default_config_path,
        checkpoint,
        split="test",
        run_milp=False,
        output_dir=output,
        test_seed=42,
        max_days=1,
        device="cpu",
        algorithm_name=spec.slug,
    )

    assert result["dr_reference"]["policy_definition"]["name"] == (
        "deterministic_rule_based_dr_reference"
    )
    assert "vs_dr_reference" in result["comparisons"]
    daily = pd.read_csv(output / "test_daily_metrics.csv")
    assert daily["dr_reference_team_profit"].notna().all()
    assert (output / "dr_reference_summary.json").is_file()
    assert len(list((output / "dr_reference_hourly_trajectories").glob("*.csv"))) == 1
