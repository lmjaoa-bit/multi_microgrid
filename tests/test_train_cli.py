from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from algorithms.conditional_matd3 import (
    DEFAULT_ALGORITHM_CONFIG,
    ConditionalMATD3Trainer,
    load_runtime_config,
)
from envs.multi_microgrid_env import MultiMicrogridEnv
from scripts.train_conditional_matd3 import (
    _default_output,
    _maybe_update_best_model,
    _resolve_run_arguments,
    build_parser,
    run_deterministic_validation,
    run_training,
)
from scripts.training_dashboard import build_dashboard_data
from utils.config import load_algorithm_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_conditional_matd3.py"
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def test_default_output_uses_algorithm_name_and_datetime():
    output = _default_output()
    assert output.parent == ROOT / "outputs"
    assert re.fullmatch(
        r"conditional_matd3_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}",
        output.name,
    )


def test_training_defaults_are_loaded_from_independent_algorithm_config():
    args = build_parser().parse_args([])
    algorithm_document = load_algorithm_config(DEFAULT_ALGORITHM_CONFIG)
    resolved = _resolve_run_arguments(args, algorithm_document)
    assert resolved.episodes == algorithm_document["training"]["episodes"]
    assert resolved.validation_interval == algorithm_document["training"][
        "validation_interval"
    ]
    assert resolved.warmup_steps == algorithm_document["algorithm"]["warmup_steps"]
    assert resolved.no_live is (
        not algorithm_document["training"]["live_dashboard"]
    )


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _run_cli(output: Path, *, no_live: bool) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--episodes",
        "2",
        "--seed",
        "7",
        "--device",
        "cpu",
        "--output",
        str(output),
        "--warmup-steps",
        "0",
        "--checkpoint-interval",
        "1",
        "--validation-interval",
        "1",
        "--validation-episodes",
        "1",
        "--log-interval",
        "1",
        "--config",
        str(CONFIG),
    ]
    if no_live:
        command.append("--no-live")
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _assert_training_artifacts(output: Path) -> pd.DataFrame:
    log_path = output / "training_log.csv"
    run_config_path = output / "run_config.json"
    validation_log_path = output / "validation_log.csv"
    policy_stability_log_path = output / "policy_stability_log.csv"
    policy_stability_plot_path = output / "policy_action_change_vs_episode.png"
    best_path = output / "best_model.pt"
    last_path = output / "last_checkpoint.pt"
    assert log_path.is_file()
    assert run_config_path.is_file()
    assert validation_log_path.is_file()
    assert policy_stability_log_path.is_file()
    assert policy_stability_plot_path.is_file()
    assert best_path.is_file()
    assert last_path.is_file()
    assert (output / "configs" / "environment.yaml").is_file()
    assert (output / "configs" / "algorithm.yaml").is_file()
    assert (output / "configs" / "effective_runtime.yaml").is_file()
    reproduced_env = MultiMicrogridEnv(
        output / "configs" / "effective_runtime.yaml", split="train"
    )
    assert reproduced_env.config["algorithm"]["name"] == "conditional_matd3"
    assert not list(output.glob("checkpoint_ep*.pt"))
    assert not (output / "checkpoint_final.pt").exists()

    frame = pd.read_csv(log_path)
    required_columns = {
        "episode",
        "env_steps",
        "replay_size",
        "system_total_cost",
        "system_total_reward",
        "market_trade_kwh",
        "renewable_curtailment_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "MG1_reward",
        "MG1_cost",
        "MG1_p2p_kwh",
        "MG1_critic_loss",
        "MG1_commitment_loss",
        "MG1_common_loss",
        "MG1_generator_loss",
        "MG1_delta_q",
        "MG1_p_on",
        "MG1_q_scale_ema",
    }
    assert required_columns.issubset(frame.columns)
    assert frame["episode"].tolist() == [1, 2]
    assert frame["env_steps"].tolist() == [24, 48]
    assert frame["replay_size"].tolist() == [24, 48]
    assert frame["validation_available"].tolist() == [1, 1]
    assert frame["best_episode"].tolist() == [1, 1]
    assert np.isfinite(frame.to_numpy(dtype=np.float64)).all()

    validation_frame = pd.read_csv(validation_log_path)
    assert validation_frame["episode"].tolist() == [1, 2]
    assert validation_frame["is_best"].tolist() == [1, 0]
    assert np.isfinite(validation_frame.to_numpy(dtype=np.float64)).all()
    stability_frame = pd.read_csv(policy_stability_log_path)
    assert stability_frame["episode"].tolist() == [1, 2]
    assert {
        "MG1_policy_action_l2",
        "MG2_policy_action_l2",
        "MG3_policy_action_l2",
        "max_policy_action_l2",
    }.issubset(stability_frame.columns)
    assert np.isfinite(stability_frame.to_numpy(dtype=np.float64)).all()
    return frame


def test_live_and_no_live_cli_complete_with_identical_logs(tmp_path):
    no_live_output = tmp_path / "cli_no_live"
    live_output = tmp_path / "cli_live"
    no_live_result = _run_cli(no_live_output, no_live=True)
    live_result = _run_cli(live_output, no_live=False)
    assert no_live_result.returncode == 0, no_live_result.stdout + "\n" + no_live_result.stderr
    assert live_result.returncode == 0, live_result.stdout + "\n" + live_result.stderr
    assert "training_complete" in no_live_result.stdout
    assert "TRAINING COMPLETED" in live_result.stdout

    no_live_frame = _assert_training_artifacts(no_live_output)
    live_frame = _assert_training_artifacts(live_output)
    pd.testing.assert_frame_equal(live_frame, no_live_frame, check_exact=True)

    run_config = json.loads(
        (no_live_output / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config["training"]["episodes"] == 2
    assert Path(run_config["source_configs"]["algorithm"]) == DEFAULT_ALGORITHM_CONFIG
    assert run_config["training"]["seed"] == 7
    assert run_config["training"]["device_resolved"] == "cpu"
    assert run_config["training"]["warmup_steps"] == 0
    assert run_config["training"]["live_dashboard"] is False
    assert run_config["training"]["rolling_window"] == 20
    assert run_config["training"]["validation_split"] == "val"
    assert run_config["training"]["validation_deterministic"] is True
    assert len(run_config["training"]["validation_days"]) == 1
    assert run_config["training"]["policy_stability_probe_count"] == 24
    assert run_config["training"]["keep_periodic_checkpoints"] is False
    assert run_config["checkpoint_scope"]["replay_buffer"] is False
    assert run_config["resolved_environment_config"]["time"]["delta_t"] == 1.0

    checkpoint = _load_checkpoint(no_live_output / "last_checkpoint.pt")
    assert checkpoint["episode"] == 2
    assert checkpoint["env_steps"] == 48
    assert checkpoint["trainer_episodes"] == 2
    assert checkpoint["exact_resume_supported"] is False
    assert set(checkpoint["policy_probe_actions"]) == {"mg1", "mg2", "mg3"}
    assert set(checkpoint["policy_probe_observations"]) == {"mg1", "mg2", "mg3"}
    assert {
        "actors",
        "target_actors",
        "critics1",
        "critics2",
        "target_critics1",
        "target_critics2",
        "actor_optimizers",
        "critic_optimizers",
        "q_scale_ema",
        "update_counter",
    }.issubset(checkpoint["algorithm"])

    best_model = _load_checkpoint(no_live_output / "best_model.pt")
    assert best_model["checkpoint_type"] == "best_policy"
    assert best_model["episode"] == 1
    assert set(best_model["actors"]) == {"mg1", "mg2", "mg3"}
    assert "actor_optimizers" not in best_model
    assert "critic_optimizers" not in best_model
    assert "algorithm" not in best_model


def test_dashboard_data_handles_warmup_without_learning_metrics():
    row = {
        "episode": 1,
        "env_steps": 24,
        "replay_size": 24,
        "system_total_cost": 300.0,
        "system_total_reward": -3.0,
        "market_trade_kwh": 12.0,
        "renewable_curtailment_kwh": 4.0,
        "grid_import_kwh": 50.0,
        "grid_export_kwh": 2.0,
    }
    for index in range(1, 4):
        row[f"MG{index}_reward"] = -float(index)
        row[f"MG{index}_cost"] = 100.0 * index
        row[f"MG{index}_p2p_kwh"] = 2.0 * index

    data = build_dashboard_data(
        row,
        [row],
        ["mg1", "mg2", "mg3"],
        current_episode=1,
        total_episodes=300,
        start_episode=1,
        device="cpu",
        warmup_steps=1000,
        update_count=0,
        elapsed_seconds=2.0,
        rolling_window=20,
        available_metrics=set(),
    )
    assert set(data["microgrids"]) == {"MG1", "MG2", "MG3"}
    assert data["overview"]["warmup"] == "976 steps remaining"
    assert data["system"]["mean_critic_loss"]["current"] is None
    for metrics in data["microgrids"].values():
        assert metrics["p_on"] is None
        assert metrics["q_on"] is None
        assert metrics["critic_loss"] is None


def test_deterministic_validation_is_reproducible_and_training_state_isolated():
    train_env = MultiMicrogridEnv(load_runtime_config(), split="train")
    validation_env = MultiMicrogridEnv(load_runtime_config(), split="val")
    trainer = ConditionalMATD3Trainer.from_env(train_env, device="cpu")
    validation_days = [str(validation_env.data.valid_days[0].date())]
    replay_size = len(trainer.replay)
    update_counter = trainer.algorithm.update_counter
    q_scale = dict(trainer.algorithm.q_scale_ema)

    first = run_deterministic_validation(trainer, validation_env, validation_days, seed=999)
    second = run_deterministic_validation(trainer, validation_env, validation_days, seed=999)
    assert first == second
    assert len(trainer.replay) == replay_size
    assert trainer.algorithm.update_counter == update_counter
    assert trainer.algorithm.q_scale_ema == q_scale
    assert trainer.env_steps == 0


def test_best_model_overwrites_only_for_lower_validation_cost(tmp_path):
    env = MultiMicrogridEnv(load_runtime_config(), split="train")
    trainer = ConditionalMATD3Trainer.from_env(env, device="cpu")
    path = tmp_path / "best_model.pt"
    best_cost, best_episode = float("inf"), 0

    best_cost, best_episode, changed = _maybe_update_best_model(
        path,
        trainer,
        1,
        {"validation_system_cost_mean": 10.0},
        {},
        best_cost,
        best_episode,
    )
    assert changed and (best_cost, best_episode) == (10.0, 1)
    best_cost, best_episode, changed = _maybe_update_best_model(
        path,
        trainer,
        2,
        {"validation_system_cost_mean": 8.0},
        {},
        best_cost,
        best_episode,
    )
    assert changed and (best_cost, best_episode) == (8.0, 2)
    best_cost, best_episode, changed = _maybe_update_best_model(
        path,
        trainer,
        3,
        {"validation_system_cost_mean": 9.0},
        {},
        best_cost,
        best_episode,
    )
    assert not changed and (best_cost, best_episode) == (8.0, 2)
    assert _load_checkpoint(path)["episode"] == 2


def test_trainer_aggregates_actor_metrics_when_final_update_is_critic_only(monkeypatch):
    env = MultiMicrogridEnv(load_runtime_config(), split="train")
    trainer = ConditionalMATD3Trainer.from_env(env, device="cpu")
    trainer.warmup_steps = 0
    trainer.batch_size = 1
    calls = 0

    def fake_update(batch):
        nonlocal calls
        calls += 1
        trainer.algorithm.update_counter += 1
        logs = {
            f"{mg}/critic_loss": float(calls)
            for mg in env.agent_ids
        }
        logs.update({f"{mg}/q_mean": -float(calls) for mg in env.agent_ids})
        actor_updated = calls % 2 == 1
        if actor_updated:
            for mg in env.agent_ids:
                for metric in (
                    "actor_loss",
                    "commitment_loss",
                    "common_loss",
                    "generator_loss",
                    "q_on",
                    "q_off",
                    "delta_q",
                    "p_on",
                    "commit_target",
                    "q_scale",
                ):
                    logs[f"{mg}/{metric}"] = float(calls)
        logs["policy_updated"] = float(actor_updated)
        return logs

    monkeypatch.setattr(trainer.algorithm, "update", fake_update)
    result = trainer.run_episode(training=True, seed=123)
    metrics = result["last_update"]
    assert calls == 24
    assert metrics["critic_update_count"] == 24.0
    assert metrics["actor_update_count"] == 12.0
    assert metrics["policy_updated"] == 1.0
    assert metrics["mg1/actor_loss"] == 12.0
    assert metrics["mg1/q_on"] == 12.0
    assert metrics["mg1/delta_q"] == 12.0


def test_live_mode_does_not_swallow_nonfinite_failure(tmp_path, monkeypatch, capsys):
    def invalid_episode(self, *args, **kwargs):
        by_microgrid = {
            mg: {"reward": 0.0, "cost": 0.0, "p2p_kwh": 0.0}
            for mg in self.env.agent_ids
        }
        by_microgrid["mg1"]["reward"] = float("nan")
        return {
            "by_microgrid": by_microgrid,
            "market_trade_kwh": 0.0,
            "renewable_curtailment_kwh": 0.0,
            "grid_import_kwh": 0.0,
            "grid_export_kwh": 0.0,
            "env_steps": 0,
            "replay_size": 0,
            "last_update": {},
        }

    monkeypatch.setattr(
        "algorithms.conditional_matd3.trainer.ConditionalMATD3Trainer.run_episode",
        invalid_episode,
    )
    output = tmp_path / "nan_live"
    args = build_parser().parse_args(
        [
            "--episodes",
            "1",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--config",
            str(CONFIG),
        ]
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        run_training(args)
    captured = capsys.readouterr()
    assert "TRAINING ABORTED" in captured.err
    assert (output / "crash_checkpoint.pt").is_file()
