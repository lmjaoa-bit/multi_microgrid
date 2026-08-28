from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from algorithms.conditional_matd3 import (
    ConditionalMATD3,
    ConditionalMATD3Trainer,
    load_runtime_config,
)
from algorithms.conditional_matd3.replay_buffer import JointReplayBuffer
from envs.multi_microgrid_env import MultiMicrogridEnv


def test_stochastic_nash_evaluation_decoupling():
    runtime_config = load_runtime_config()
    env = MultiMicrogridEnv(copy.deepcopy(runtime_config), split="val")
    day = str(env.data.valid_days[0].date())

    algorithm = ConditionalMATD3(
        env.agent_ids,
        env.observation_dims,
        env.global_state_dim,
        env.config,
        device="cpu",
    )
    replay = JointReplayBuffer(
        capacity=10,
        agent_ids=env.agent_ids,
        observation_dims=env.observation_dims,
        state_dim=env.global_state_dim,
        action_dim=env.action_dim,
        seed=42,
    )
    trainer = ConditionalMATD3Trainer(
        env=env,
        algorithm=algorithm,
        replay=replay,
        batch_size=4,
        warmup_steps=0,
    )

    # 1. Verify training=False, environment_deterministic=False sets env.forecasts.deterministic = False
    result_seed42_a = trainer.run_episode(
        day=day,
        training=False,
        deterministic_eval=False,
        environment_deterministic=False,
        seed=42,
    )
    assert env.forecasts.deterministic is False

    # 2. Same seed produces identical metrics (reproducibility)
    result_seed42_b = trainer.run_episode(
        day=day,
        training=False,
        deterministic_eval=False,
        environment_deterministic=False,
        seed=42,
    )
    for mg in env.agent_ids:
        assert np.isclose(
            result_seed42_a["by_microgrid"][mg]["cost"],
            result_seed42_b["by_microgrid"][mg]["cost"],
        )

    # 3. Different seed produces different realization under stochastic environment
    result_seed999 = trainer.run_episode(
        day=day,
        training=False,
        deterministic_eval=False,
        environment_deterministic=False,
        seed=999,
    )
    assert not all(
        np.isclose(
            result_seed42_a["by_microgrid"][mg]["cost"],
            result_seed999["by_microgrid"][mg]["cost"],
        )
        for mg in env.agent_ids
    )
