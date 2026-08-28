from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from algorithms.conditional_matd3 import (
    ConditionalMATD3,
    DEFAULT_ALGORITHM_CONFIG,
    load_runtime_config,
)
from envs.multi_microgrid_env import MultiMicrogridEnv
from scripts.evaluate_nash import build_parser, run_nash_verification
from utils.config import load_algorithm_config, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def test_nash_pipeline_smoke_uses_train_val_test_and_writes_outputs(tmp_path):
    config = copy.deepcopy(load_config(CONFIG))
    config.pop("_project_root", None)
    for split in ("train", "val", "test"):
        config["data"][f"{split}_path"] = str(
            ROOT / "data" / f"uk_microgrid_2018_{split}.csv"
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    algorithm_config = copy.deepcopy(load_algorithm_config(DEFAULT_ALGORITHM_CONFIG))
    algorithm_config.pop("_config_path", None)
    algorithm_config["algorithm"]["actor_hidden_dims"] = [16, 16]
    algorithm_config["algorithm"]["critic_hidden_dims"] = [16, 16]
    algorithm_config["algorithm"]["batch_size"] = 4
    algorithm_config["algorithm"]["replay_capacity"] = 256
    algorithm_config_path = tmp_path / "algorithm.yaml"
    algorithm_config_path.write_text(
        yaml.safe_dump(algorithm_config, sort_keys=False), encoding="utf-8"
    )

    env = MultiMicrogridEnv(
        load_runtime_config(config_path, algorithm_config_path), split="train"
    )
    algorithm = ConditionalMATD3(
        env.agent_ids,
        env.observation_dims,
        env.global_state_dim,
        env.config,
        device="cpu",
    )
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "checkpoint_type": "best_policy",
            "actors": algorithm.get_policy_state_dict()["actors"],
        },
        checkpoint_path,
    )
    output = tmp_path / "nash"
    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--algo-config",
            str(algorithm_config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output),
            "--focal-agents",
            "mg1",
            "--br-episodes",
            "2",
            "--br-restarts",
            "1",
            "--validation-interval",
            "1",
            "--validation-days",
            "1",
            "--test-days",
            "1",
            "--br-warmup-steps",
            "0",
            "--batch-size",
            "4",
            "--replay-capacity",
            "256",
            "--restart-init",
            "equilibrium",
            "--device",
            "cpu",
        ]
    )
    summary = run_nash_verification(args)

    expected = {
        "nash_summary.csv",
        "nash_summary.json",
        "baseline_test_metrics.csv",
        "br_test_metrics.csv",
        "br_training_log.csv",
        "br_mg1_best.pt",
        "nash_unilateral_gain.png",
        "nash_best_response_mg1.png",
    }
    assert expected <= {path.name for path in output.iterdir()}
    assert summary["train_split"] == "train"
    assert summary["validation_split"] == "val"
    assert summary["test_split"] == "test"
    assert summary["test_selection_used"] is False
    assert summary["focal_agents"] == ["mg1"]
    rows = pd.read_csv(output / "nash_summary.csv")
    assert rows["agent"].tolist() == ["mg1"]
    assert rows["br_restart_count"].tolist() == [1]
    loaded = json.loads((output / "nash_summary.json").read_text(encoding="utf-8"))
    assert loaded["epsilon_nash_pct"] == summary["epsilon_nash_pct"]
