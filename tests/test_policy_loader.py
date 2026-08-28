from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.conditional_matd3 import ConditionalMATD3, load_runtime_config
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def _algorithm(env: MultiMicrogridEnv) -> ConditionalMATD3:
    return ConditionalMATD3(
        env.agent_ids,
        env.observation_dims,
        env.global_state_dim,
        env.config,
        device="cpu",
    )


def test_lightweight_policy_loader_preserves_deterministic_actions_and_syncs_targets():
    env = MultiMicrogridEnv(load_runtime_config())
    source = _algorithm(env)
    restored = _algorithm(env)
    restored.load_policy_state_dict(source.get_policy_state_dict(), sync_targets=True)
    obs, _ = env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    expected, _ = source.select_actions(obs, explore=False)
    actual, _ = restored.select_actions(obs, explore=False)
    for mg in env.agent_ids:
        assert np.array_equal(actual[mg], expected[mg])
        for online, target in zip(
            restored.actors[mg].parameters(),
            restored.target_actors[mg].parameters(),
        ):
            assert np.array_equal(
                online.detach().numpy(),
                target.detach().numpy(),
            )
