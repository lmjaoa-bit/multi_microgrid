from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from algorithms.conditional_matd3 import (
    ConditionalMATD3,
    DEFAULT_ALGORITHM_CONFIG,
    load_runtime_config,
)
from envs.multi_microgrid_env import MultiMicrogridEnv
from scripts.evaluate_nash import (
    build_parser,
    train_best_response,
)
from utils.config import load_algorithm_config, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def test_br_retraining_retains_baseline_when_no_improvement(tmp_path):
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

    runtime_config = load_runtime_config(config_path, algorithm_config_path)
    train_env = MultiMicrogridEnv(runtime_config, split="train")
    val_env = MultiMicrogridEnv(runtime_config, split="val")
    val_days = [str(day.date()) for day in val_env.data.valid_days[:1]]

    algorithm = ConditionalMATD3(
        train_env.agent_ids,
        train_env.observation_dims,
        train_env.global_state_dim,
        train_env.config,
        device="cpu",
    )
    baseline_policy = algorithm.get_policy_state_dict()
    baseline_focal_weights = copy.deepcopy(baseline_policy["actors"]["mg1"])

    obs, _ = val_env.reset(day=val_days[0], deterministic=True)
    probe_obs = np.asarray([obs["mg1"]], dtype=np.float32)
    reference_actions = np.zeros((1, 4), dtype=np.float32)

    # Set an impossibly low baseline cost (-1e9) so retraining can never beat it
    baseline_val_cost = -1_000_000_000.0

    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "mock.pt"),
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
            "random",
            "--seed",
            "42",
            "--device",
            "cpu",
        ]
    )

    result = train_best_response(
        runtime_config=runtime_config,
        baseline_policy=baseline_policy,
        focal_agent="mg1",
        restart=0,
        args=args,
        device=torch.device("cpu"),
        validation_days=val_days,
        probe_observations=probe_obs,
        reference_probe_actions=reference_actions,
        baseline_validation_cost=baseline_val_cost,
        validation_seed_base=2_000_042,
    )

    assert result.best_episode == 0
    assert result.best_validation_cost == baseline_val_cost
    assert result.best_validation_cost <= baseline_val_cost

    for key, tensor in baseline_focal_weights.items():
        current_tensor = result.policy_state["actors"]["mg1"][key]
        assert torch.equal(tensor, current_tensor)
