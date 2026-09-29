from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


def _configs() -> tuple[dict, dict, dict]:
    environment = yaml.safe_load(
        Path("configs/cooperative.yaml").read_text(encoding="utf-8")
    )
    ordinary = yaml.safe_load(
        Path(
            "configs/algorithms/matd3_hierarchical_transformer.yaml"
        ).read_text(encoding="utf-8")
    )
    group_a = yaml.safe_load(
        Path(
            "configs/algorithms/matd3_forecast_context.yaml"
        ).read_text(encoding="utf-8")
    )
    # This test protects the historical 2x2 ablation boundary.  The current
    # ordinary Transformer has intentionally moved to Reward V2 + peer H8, so
    # reconstruct the pre-V2 Group-A contract instead of mutating the archived
    # forecast-context implementation to follow the new formal default.
    environment.pop("reward", None)
    ordinary["algorithm"]["gamma"] = group_a["algorithm"]["gamma"]
    ordinary["algorithm"]["network"]["include_peer_forecast"] = False
    ordinary["algorithm"]["network"]["peer_forecast_features"] = ["energy_balance"]
    ordinary["logging"]["diagnostic_interval"] = 0
    group_a["logging"]["diagnostic_interval"] = 0
    return environment, ordinary, group_a


def _assert_state_dict_close(
    left: torch.nn.Module,
    right: torch.nn.Module,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for name in left_state:
        torch.testing.assert_close(
            left_state[name],
            right_state[name],
            rtol=rtol,
            atol=atol,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_group_a_seeded_initialization_and_forward_equal_locked_ordinary_t():
    env_cfg, ordinary_cfg, group_a_cfg = _configs()
    old_spec = get_algorithm_spec("matd3_hierarchical_transformer")
    new_spec = get_algorithm_spec("matd3_forecast_context")

    old_env = MultiMicrogridEnv(
        old_spec.build_environment_config(copy.deepcopy(env_cfg)),
        split="train",
    )
    new_env = MultiMicrogridEnv(
        new_spec.build_environment_config(copy.deepcopy(env_cfg)),
        split="train",
    )
    old = old_spec.create(
        old_env,
        copy.deepcopy(ordinary_cfg),
        device="cpu",
        seed=42,
    )
    new = new_spec.create(
        new_env,
        copy.deepcopy(group_a_cfg),
        device="cpu",
        seed=42,
    )

    assert old.agent_ids == new.agent_ids
    for mg in old.agent_ids:
        _assert_state_dict_close(old.actors[mg], new.actors[mg])
        _assert_state_dict_close(
            old.target_actors[mg],
            new.target_actors[mg],
        )
        _assert_state_dict_close(old.critics[mg], new.critics[mg])
        _assert_state_dict_close(
            old.target_critics[mg],
            new.target_critics[mg],
        )

    old_obs, old_info = old_env.reset(seed=12345)
    new_obs, new_info = new_env.reset(seed=12345)
    assert old_obs.keys() == new_obs.keys()
    for mg in old.agent_ids:
        np.testing.assert_array_equal(old_obs[mg], new_obs[mg])

    old_actions = old.select_actions(old_obs, noise_std=0.0)
    new_actions = new.select_actions(new_obs, noise_std=0.0)
    for mg in old.agent_ids:
        np.testing.assert_allclose(
            old_actions[mg],
            new_actions[mg],
            rtol=1e-7,
            atol=1e-8,
        )

    raw = torch.as_tensor(
        np.stack([old_actions[mg] for mg in old.agent_ids], axis=0),
        dtype=torch.float32,
    ).unsqueeze(0)
    old_projected, _ = old.project_actions(
        torch.as_tensor(
            old_info["global_state"],
            dtype=torch.float32,
        ).unsqueeze(0),
        raw,
    )
    new_projected, _ = new.project_actions(
        torch.as_tensor(
            new_info["global_state"],
            dtype=torch.float32,
        ).unsqueeze(0),
        raw,
    )
    torch.testing.assert_close(
        old_projected,
        new_projected,
        rtol=1e-7,
        atol=1e-8,
    )


def test_group_a_uses_same_sampled_grouped_update_path_as_ordinary_t():
    env_cfg, ordinary_cfg, group_a_cfg = _configs()
    old_spec = get_algorithm_spec("matd3_hierarchical_transformer")
    new_spec = get_algorithm_spec("matd3_forecast_context")

    old_env = MultiMicrogridEnv(
        old_spec.build_environment_config(copy.deepcopy(env_cfg)),
        split="train",
    )
    new_env = MultiMicrogridEnv(
        new_spec.build_environment_config(copy.deepcopy(env_cfg)),
        split="train",
    )
    old = old_spec.create(
        old_env,
        copy.deepcopy(ordinary_cfg),
        device="cpu",
        seed=7,
    )
    new = new_spec.create(
        new_env,
        copy.deepcopy(group_a_cfg),
        device="cpu",
        seed=7,
    )

    assert old.update.__func__.__name__ == "fast_projected_update"
    assert new.update.__func__.__name__ == "fast_projected_update"
    assert old._diagnostic_plain_actor is False
    assert new._diagnostic_plain_actor is False
    assert old._diagnostic_variant == "matd3_hierarchical_transformer"
    assert new._diagnostic_variant == "matd3_forecast_context"

    # Generate identical deterministic transitions once and feed the same
    # transition values into both replay buffers.
    source_env = old_env
    for episode in range(2):
        observations, info = source_env.reset(seed=500 + episode)
        state = info["global_state"]
        done = False
        while not done:
            actions = {
                mg: np.zeros(source_env.action_dim, dtype=np.float32)
                for mg in source_env.agent_ids
            }
            next_observations, rewards, done, step_info = source_env.step(actions)
            team_reward = float(rewards[source_env.agent_ids[0]])
            for algorithm in (old, new):
                algorithm.replay.add(
                    observations,
                    state,
                    actions,
                    team_reward,
                    next_observations,
                    step_info["global_state_after"],
                    done,
                )
            observations = next_observations
            state = step_info["global_state_after"]

    assert len(old.replay) == len(new.replay) >= 32

    # Two calls include one delayed actor update when policy_delay=2.
    torch.manual_seed(20260928)
    old.update(16)
    old.update(16)
    torch.manual_seed(20260928)
    new.update(16)
    new.update(16)

    for mg in old.agent_ids:
        _assert_state_dict_close(
            old.actors[mg],
            new.actors[mg],
            rtol=2e-6,
            atol=2e-7,
        )
        _assert_state_dict_close(
            old.target_actors[mg],
            new.target_actors[mg],
            rtol=2e-6,
            atol=2e-7,
        )
        _assert_state_dict_close(
            old.critics[mg],
            new.critics[mg],
            rtol=2e-6,
            atol=2e-7,
        )
        _assert_state_dict_close(
            old.target_critics[mg],
            new.target_critics[mg],
            rtol=2e-6,
            atol=2e-7,
        )
