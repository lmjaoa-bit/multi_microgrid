from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


def _setup(*, learning_starts: int = 4, random_prob: float = 1.0):
    spec = get_algorithm_spec("matd3_dispatch_exploration")
    config = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    config["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(config), split="train")
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    settings["algorithm"].update(
        learning_starts=learning_starts,
        dispatch_random_action_prob=random_prob,
    )
    settings["logging"]["diagnostic_interval"] = 0
    algorithm = spec.create(env, settings, device="cpu", seed=17)
    return env, algorithm


def test_learning_starts_waits_for_replay_steps() -> None:
    env, algorithm = _setup()
    observations, info = env.reset(seed=17, deterministic=True)
    state = info["global_state"]
    for step in range(4):
        actions = algorithm.select_actions(observations, noise_std=0.0)
        next_observations, rewards, done, info = env.step(actions)
        algorithm.replay.add(
            observations, state, actions, rewards[env.agent_ids[0]],
            next_observations, info["global_state_after"], done,
        )
        result = algorithm.update(batch_size=2)
        assert algorithm.update_steps == (0 if step < 3 else 1)
        if step < 3:
            assert result == {}
        observations, state = next_observations, info["global_state_after"]


def test_dispatch_randomization_preserves_dr_and_deterministic_policy() -> None:
    env, algorithm = _setup(random_prob=1.0)
    observations, _ = env.reset(seed=17, deterministic=True)
    deterministic = algorithm.select_actions(observations, noise_std=0.0)
    assert all(np.allclose(deterministic[mg], Matd3Dispatch.select_actions(algorithm, observations, 0.0)[mg]) for mg in env.agent_ids)

    state = deepcopy(algorithm.rng.bit_generator.state)
    baseline = Matd3Dispatch.select_actions(algorithm, observations, noise_std=0.15)
    algorithm.rng.bit_generator.state = state
    explored = algorithm.select_actions(observations, noise_std=0.15)
    assert all(explored[mg][1] == baseline[mg][1] for mg in env.agent_ids)
    assert any(explored[mg][0] != baseline[mg][0] for mg in env.agent_ids)
    assert all(-1.0 <= explored[mg][0] <= 1.0 for mg in env.agent_ids)


def test_variant_checkpoint_is_its_own_algorithm(tmp_path: Path) -> None:
    env, algorithm = _setup()
    checkpoint = tmp_path / "variant.pt"
    algorithm.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["algorithm_slug"] == "matd3_dispatch_exploration"
    _, restored = _setup()
    restored.load(checkpoint)
    assert restored.update_steps == algorithm.update_steps
