from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from algorithms.matd3_hierarchical_transformer_predictive_dr.actor import PredictiveDRActor
from algorithms.matd3_hierarchical_transformer_peak_risk.actor import PeakRiskActor
from algorithms.matd3_hierarchical_transformer_peak_risk.critic import TwinPeakCritic
from algorithms.matd3_hierarchical_transformer_peak_risk.peak_risk import (
    PeakRiskFeatureExtractor, max_return_target, peak_actor_weight,
    system_actual_load_kw, validate_peak_risk_config,
)
from envs.multi_microgrid_env import MultiMicrogridEnv


def test_plugin_config_is_isolated():
    spec = get_algorithm_spec("matd3_hierarchical_transformer_peak_risk")
    cfg = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    assert cfg["algorithm"]["name"] == spec.slug
    assert "p2p_auxiliary" not in cfg["algorithm"]
    assert cfg["algorithm"]["peak_risk"]["contract_version"] == "1.0"


@pytest.mark.parametrize("field,value", [
    ("critic_update_interval", 0), ("critic_lr", 0), ("critic_max_grad_norm", -1),
    ("critic_hidden_dims", [64, 0]), ("actor_weight_final", float("nan")),
    ("actor_weight_ramp_updates", 0), ("use_discount", True),
    ("conservative_twin_mode", "min"), ("reuse_parent_projection", False),
    ("unsupported_option", 1),
])
def test_peak_config_fails_closed(field, value):
    with pytest.raises(ValueError, match="peak_risk"):
        validate_peak_risk_config({field: value})


def test_exact_max_return_without_gamma_and_single_row():
    x = torch.tensor([[0.7], [0.7], [0.7]])
    n1 = torch.tensor([[0.4], [0.8], [0.9]])
    n2 = torch.tensor([[0.5], [0.9], [0.8]])
    done = torch.tensor([[0.0], [0.0], [1.0]])
    assert torch.equal(max_return_target(x, n1, n2, done),
                       torch.tensor([[0.7], [0.9], [0.7]]))
    assert torch.equal(max_return_target(x[:1], n1[:1], n2[:1], done[:1]), x[:1])


def test_peak_weight_ramp():
    assert peak_actor_weight(4999, 5000, 15000, 0.02) == 0
    assert peak_actor_weight(12500, 5000, 15000, 0.02) == pytest.approx(0.01)
    assert peak_actor_weight(20000, 5000, 15000, 0.02) == pytest.approx(0.02)


def test_peak_cost_differentiable_and_nonnegative_twin():
    raw = torch.tensor([[1.0], [2.0]], requires_grad=True)
    plans = [{"actual_load_kw": raw * i} for i in (1, 2, 3)]
    cost = system_actual_load_kw(plans)
    assert cost.shape == (2, 1)
    assert torch.equal(cost, raw * 6)
    cost.sum().backward()
    assert torch.equal(raw.grad, torch.full_like(raw, 6.0))
    critic = TwinPeakCritic(11, 9)
    a, b = critic(torch.randn(2, 11), torch.randn(2, 9))
    assert a.shape == b.shape == (2, 1)
    assert a.min() >= 0 and b.min() >= 0


@pytest.fixture
def env_and_configs():
    env_cfg = yaml.safe_load(Path("configs/cooperative.yaml").read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(env_cfg, split="train")
    algo_cfg = yaml.safe_load(Path(
        "configs/algorithms/matd3_hierarchical_transformer_peak_risk.yaml"
    ).read_text(encoding="utf-8"))
    return env, algo_cfg


def test_peak_actor_token_features_use_only_observation(env_and_configs):
    env, config = env_and_configs
    observations, _ = env.reset(seed=42)
    algorithm = get_algorithm_spec(config["algorithm"]["name"]).create(
        env, config, device="cpu", seed=42)
    assert len(algorithm.actors) == 3
    assert isinstance(algorithm.peak_critic, TwinPeakCritic)
    for mg in env.agent_ids:
        actor = algorithm.actors[mg]
        assert type(actor) is PeakRiskActor
        tensor = torch.as_tensor(observations[mg], dtype=torch.float32).unsqueeze(0)
        features = actor.peak_features(tensor)
        assert features.shape == (1, 14)
        assert torch.isfinite(features).all()
        tokens, mask = actor._build_tokens(tensor)
        assert tokens.shape == (1, 15, actor.embed_dim)
        assert mask.shape == (1, 15)
        assert not mask[:, 4].any()
        assert actor(tensor).shape == (1, 2)


def test_all_invalid_forecast_is_finite_and_marked(env_and_configs):
    env, config = env_and_configs
    observations, _ = env.reset(seed=42)
    algorithm = get_algorithm_spec(config["algorithm"]["name"]).create(
        env, config, device="cpu", seed=42)
    for mg in env.agent_ids:
        actor = algorithm.actors[mg]
        tensor = torch.as_tensor(observations[mg], dtype=torch.float32).unsqueeze(0)
        for field, idx in actor.peak_features.indices.items():
            if field.startswith("forecast_valid_t+"):
                tensor[:, idx] = 0.0
        values = actor.peak_features(tensor)
        assert torch.isfinite(values).all()
        if any(field.startswith("forecast_valid_t+") for field in actor.peak_features.indices):
            assert values[0, 12] == 0.0


def test_disabled_peak_returns_exact_predictive_actor(env_and_configs):
    env, config = env_and_configs
    config["algorithm"]["peak_risk"].update(
        token_enabled=False, critic_enabled=False, actor_weight_final=0.0)
    algorithm = get_algorithm_spec(config["algorithm"]["name"]).create(
        env, config, device="cpu", seed=42)
    assert algorithm.peak_critic is None
    assert all(type(actor) is PredictiveDRActor for actor in algorithm.actors.values())
    assert all(actor.total_tokens == 14 for actor in algorithm.actors.values())


def test_checkpoint_roundtrip_and_strict_contract(env_and_configs, tmp_path):
    env, config = env_and_configs
    spec = get_algorithm_spec(config["algorithm"]["name"])
    original = spec.create(env, config, device="cpu", seed=42)
    target = tmp_path / "model.pt"
    original.save(target)
    restored = spec.create(env, config, device="cpu", seed=42)
    restored.load(target)
    assert all(torch.equal(a, b) for (a, b) in zip(
        original.peak_critic.state_dict().values(), restored.peak_critic.state_dict().values()))
    state = torch.load(target, weights_only=False)
    for name in ("algorithm_slug", "peak_risk_config_fingerprint",
                 "normalization_signature", "projection_signature", "state_schema"):
        altered = copy.deepcopy(state)
        altered[name] = "corrupted"
        bad = tmp_path / (name + ".pt")
        torch.save(altered, bad)
        with pytest.raises(ValueError, match=name):
            restored.load(bad)


def test_parent_projection_call_count_on_actor_due_and_not_due(env_and_configs):
    env, config = env_and_configs
    config["logging"]["diagnostic_interval"] = 100
    algorithm = get_algorithm_spec(config["algorithm"]["name"]).create(
        env, config, device="cpu", seed=42)
    observations, info = env.reset(seed=42)
    state = info["global_state"]
    for step in range(5):
        actions = {mg: torch.zeros(env.action_dim).numpy() for mg in env.agent_ids}
        next_obs, rewards, done, step_info = env.step(actions)
        algorithm.replay.add(observations, state, actions, float(rewards[env.agent_ids[0]]),
                             next_obs, step_info["global_state_after"], done)
        observations, state = next_obs, step_info["global_state_after"]
    calls = [0]
    original = algorithm.project_actions

    def tracked(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)
    algorithm.project_actions = tracked
    algorithm.update(batch_size=2)
    assert calls[0] == 2  # behavior, target
    calls[0] = 0
    algorithm.update(batch_size=2)
    assert calls[0] == 5  # behavior, target, one actor projection per agent
