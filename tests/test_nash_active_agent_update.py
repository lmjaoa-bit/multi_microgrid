from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from algorithms.conditional_matd3 import ConditionalMATD3, load_runtime_config
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def _clone_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _equal(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return all(
        torch.equal(old, current.detach())
        for old, current in zip(before, module.parameters())
    )


def _synthetic_batch(env: MultiMicrogridEnv, batch_size: int = 8):
    obs_dim = env.observation_dims["mg1"]
    actions = torch.empty(batch_size, 3, 4).uniform_(-1.0, 1.0)
    commitments = torch.where(
        torch.rand(batch_size, 3) >= 0.5,
        torch.ones(batch_size, 3),
        -torch.ones(batch_size, 3),
    )
    actions[:, :, 1] = commitments
    actions[:, :, 2] = torch.where(
        commitments > 0.0,
        actions[:, :, 2],
        -torch.ones_like(actions[:, :, 2]),
    )
    return {
        "state": torch.randn(batch_size, env.global_state_dim),
        "obs": torch.randn(batch_size, 3, obs_dim),
        "action": actions,
        "reward": torch.randn(batch_size, 3),
        "next_state": torch.randn(batch_size, env.global_state_dim),
        "next_obs": torch.randn(batch_size, 3, obs_dim),
        "done": torch.zeros(batch_size, 1),
    }


def test_active_agent_update_freezes_all_opponent_networks_and_targets():
    torch.manual_seed(101)
    env = MultiMicrogridEnv(load_runtime_config())
    config = copy.deepcopy(env.config)
    config["algorithm"]["actor_hidden_dims"] = [16, 16]
    config["algorithm"]["critic_hidden_dims"] = [16, 16]
    algorithm = ConditionalMATD3(
        env.agent_ids,
        env.observation_dims,
        env.global_state_dim,
        config,
        device="cpu",
    )
    batch = _synthetic_batch(env)
    modules = {}
    for mg in env.agent_ids:
        modules[mg] = {
            "actor": _clone_parameters(algorithm.actors[mg]),
            "critic1": _clone_parameters(algorithm.critics1[mg]),
            "critic2": _clone_parameters(algorithm.critics2[mg]),
            "target_actor": _clone_parameters(algorithm.target_actors[mg]),
            "target_critic1": _clone_parameters(algorithm.target_critics1[mg]),
            "target_critic2": _clone_parameters(algorithm.target_critics2[mg]),
        }

    algorithm.update(batch, active_agents={"mg1"})
    logs = algorithm.update(batch, active_agents={"mg1"})

    assert logs["policy_updated"] == 1.0
    assert not _equal(modules["mg1"]["actor"], algorithm.actors["mg1"])
    assert not _equal(modules["mg1"]["critic1"], algorithm.critics1["mg1"])
    assert not _equal(modules["mg1"]["target_actor"], algorithm.target_actors["mg1"])
    for opponent in ("mg2", "mg3"):
        assert _equal(modules[opponent]["actor"], algorithm.actors[opponent])
        assert _equal(modules[opponent]["critic1"], algorithm.critics1[opponent])
        assert _equal(modules[opponent]["critic2"], algorithm.critics2[opponent])
        assert _equal(
            modules[opponent]["target_actor"],
            algorithm.target_actors[opponent],
        )
        assert _equal(
            modules[opponent]["target_critic1"],
            algorithm.target_critics1[opponent],
        )
        assert _equal(
            modules[opponent]["target_critic2"],
            algorithm.target_critics2[opponent],
        )
        assert not any(key.startswith(f"{opponent}/") for key in logs)
