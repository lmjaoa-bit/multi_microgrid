from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from algorithms.conditional_matd3 import (
    ConditionalMATD3,
    ConditionalMATD3Trainer,
    load_runtime_config,
)
from envs.multi_microgrid_env import MultiMicrogridEnv
from scripts.evaluate_nash import evaluate_policy


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_and_br_evaluation_use_identical_common_random_numbers(monkeypatch):
    runtime_config = load_runtime_config()
    test_env = MultiMicrogridEnv(copy.deepcopy(runtime_config), split="test")
    test_days = [str(day.date()) for day in test_env.data.valid_days[:3]]

    algorithm = ConditionalMATD3(
        test_env.agent_ids,
        test_env.observation_dims,
        test_env.global_state_dim,
        test_env.config,
        device="cpu",
    )

    recorded_calls: list[dict[str, Any]] = []
    original_run_episode = ConditionalMATD3Trainer.run_episode

    def spy_run_episode(self, **kwargs):
        recorded_calls.append(copy.deepcopy(kwargs))
        return original_run_episode(self, **kwargs)

    monkeypatch.setattr(ConditionalMATD3Trainer, "run_episode", spy_run_episode)

    test_seed_base = 3_000_042

    # Baseline test evaluation
    evaluate_policy(
        algorithm,
        test_env,
        test_days,
        test_seed_base,
        policy="baseline",
        environment_deterministic=False,
    )
    baseline_calls = copy.deepcopy(recorded_calls)
    recorded_calls.clear()

    # BR test evaluation
    evaluate_policy(
        algorithm,
        test_env,
        test_days,
        test_seed_base,
        policy="best_response",
        focal_agent="mg1",
        restart=0,
        environment_deterministic=False,
    )
    br_calls = copy.deepcopy(recorded_calls)

    assert len(baseline_calls) == len(test_days)
    assert len(br_calls) == len(test_days)

    for i in range(len(test_days)):
        b_call = baseline_calls[i]
        br_call = br_calls[i]
        assert b_call["day"] == br_call["day"] == test_days[i]
        assert b_call["seed"] == br_call["seed"] == test_seed_base + i
        assert b_call["training"] is False
        assert br_call["training"] is False
        assert b_call["environment_deterministic"] is False
        assert br_call["environment_deterministic"] is False
