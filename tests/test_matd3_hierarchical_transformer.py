from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch
import yaml

from algorithms.matd3_hierarchical_transformer.algorithm import Matd3HierarchicalTransformer
import algorithms.matd3_hierarchical_transformer.plugin  # noqa: F401
from envs.multi_microgrid_env import MultiMicrogridEnv


@pytest.fixture
def env_and_config():
    config_path = Path("configs/cooperative.yaml")
    algo_config_path = Path("configs/algorithms/matd3_hierarchical_transformer.yaml")
    env_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    algo_cfg = yaml.safe_load(algo_config_path.read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(env_cfg, split="train")
    return env, algo_cfg


def test_actor_token_construction_and_shapes(env_and_config):
    env, algo_cfg = env_and_config
    algo = Matd3HierarchicalTransformer.from_env(env, algo_cfg, device="cpu", seed=42)

    assert len(algo.actors) == 3
    for mg in env.agent_ids:
        actor = algo.actors[mg]
        obs_dim = env.observation_dims[mg]
        dummy_obs = torch.randn(4, obs_dim)

        tokens, padding = actor._build_tokens(dummy_obs)
        # Expected: 1 ESS + 1 DR + 1 Supply + 1 Trade + 2 Peers + 8 Forecasts = 14 Tokens
        assert tokens.shape == (4, 14, 32)
        assert padding.shape == (4, 14)
        assert padding.dtype == torch.bool


def test_actor_forward_output_bounds(env_and_config):
    env, algo_cfg = env_and_config
    algo = Matd3HierarchicalTransformer.from_env(env, algo_cfg, device="cpu", seed=42)

    for mg in env.agent_ids:
        actor = algo.actors[mg]
        obs_dim = env.observation_dims[mg]
        dummy_obs = torch.randn(8, obs_dim)

        actions = actor(dummy_obs)
        assert actions.shape == (8, 2)
        assert (actions >= -1.0).all() and (actions <= 1.0).all()


def test_actor_gradient_flow_to_all_token_branches(env_and_config):
    env, algo_cfg = env_and_config
    algo = Matd3HierarchicalTransformer.from_env(env, algo_cfg, device="cpu", seed=42)

    for mg in env.agent_ids:
        actor = algo.actors[mg]
        obs_dim = env.observation_dims[mg]
        dummy_obs = torch.randn(4, obs_dim, requires_grad=True)

        actions = actor(dummy_obs)
        loss = actions.sum()
        loss.backward()

        # Verify gradient flow to all key submodules
        assert actor.ess_encoder[0].weight.grad is not None
        assert actor.dr_encoder[0].weight.grad is not None
        assert actor.supply_encoder[0].weight.grad is not None
        assert actor.trade_encoder[0].weight.grad is not None
        assert actor.peer_encoder[0].weight.grad is not None
        assert actor.forecast_encoder[0].weight.grad is not None
        assert actor.ess_head[0].weight.grad is not None
        assert actor.dr_head[0].weight.grad is not None
        assert actor.ess_linear_skip.weight.grad is not None
        assert actor.dr_linear_skip.weight.grad is not None


def test_select_actions_and_replay_add(env_and_config):
    env, algo_cfg = env_and_config
    algo = Matd3HierarchicalTransformer.from_env(env, algo_cfg, device="cpu", seed=42)

    obs, _ = env.reset(seed=42)
    actions = algo.select_actions(obs, noise_std=0.1)

    assert set(actions.keys()) == set(env.agent_ids)
    for mg in env.agent_ids:
        assert actions[mg].shape == (2,)
        assert -1.0 <= actions[mg][0] <= 1.0
        assert -1.0 <= actions[mg][1] <= 1.0

    next_obs, rewards, done, info = env.step(actions)
    assert set(next_obs.keys()) == set(env.agent_ids)


def test_update_step(env_and_config):
    env, algo_cfg = env_and_config
    algo = Matd3HierarchicalTransformer.from_env(env, algo_cfg, device="cpu", seed=42)

    obs, _ = env.reset(seed=42)
    state = env.global_state()
    for _ in range(10):
        actions = algo.select_actions(obs, noise_std=0.1)
        next_obs, rewards, done, info = env.step(actions)
        next_state = env.global_state()
        team_reward = sum(rewards.values())
        algo.replay.add(obs, state, actions, team_reward, next_obs, next_state, done)
        obs = next_obs
        state = next_state
        if done:
            obs, _ = env.reset()
            state = env.global_state()

    assert len(algo.replay) >= 10
    # Ordinary step (diagnostic not due) returns empty dict
    assert algo.update(batch_size=4) == {}
    # Trigger diagnostic step
    algo.diagnostic_schedule.next_boundary = 2
    metrics = algo.update(batch_size=4)
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics


def test_checkpoint_save_and_load(env_and_config):
    env, algo_cfg = env_and_config
    algo = Matd3HierarchicalTransformer.from_env(env, algo_cfg, device="cpu", seed=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "model.pt"
        algo.save(ckpt_path)
        assert ckpt_path.exists()

        algo_new = Matd3HierarchicalTransformer.from_env(env, algo_cfg, device="cpu", seed=42)
        algo_new.load(ckpt_path)

        for mg in env.agent_ids:
            for p1, p2 in zip(algo.actors[mg].parameters(), algo_new.actors[mg].parameters()):
                assert torch.allclose(p1, p2)
