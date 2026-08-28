from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.conditional_matd3 import ConditionalMATD3, load_runtime_config
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def test_only_focal_agent_uses_warmup_and_exploration_randomness():
    env = MultiMicrogridEnv(load_runtime_config())
    algorithm = ConditionalMATD3(
        env.agent_ids,
        env.observation_dims,
        env.global_state_dim,
        env.config,
        device="cpu",
    )
    obs, _ = env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    deterministic, _ = algorithm.select_actions(obs, explore=False, warmup=False)
    samples = []
    for _ in range(6):
        actions, diagnostics = algorithm.select_actions(
            obs,
            explore=True,
            warmup=True,
            explore_agents={"mg1"},
            warmup_agents={"mg1"},
        )
        samples.append(actions["mg1"])
        for opponent in ("mg2", "mg3"):
            assert np.array_equal(actions[opponent], deterministic[opponent])
            assert diagnostics[opponent]["commitment_epsilon"] == 0.0
    assert any(not np.array_equal(samples[0], sample) for sample in samples[1:])
