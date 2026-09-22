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


def test_nonfinite_peak_guards_keep_cpu_fail_closed():
    from algorithms.matd3_hierarchical_transformer_peak_risk.peak_risk import assert_finite

    assert_finite(torch.ones(2, 1), "finite")
    with pytest.raises(FloatingPointError, match="invalid peak tensor"):
        assert_finite(torch.tensor([[float("nan")]]), "invalid peak tensor")
    with pytest.raises(FloatingPointError, match="peak target next2"):
        max_return_target(
            torch.ones(1, 1), torch.ones(1, 1),
            torch.tensor([[float("inf")]]), torch.zeros(1, 1),
        )
    with pytest.raises(FloatingPointError, match="Projected actual_load_kw"):
        system_actual_load_kw([
            {"actual_load_kw": torch.tensor([[1.0]])},
            {"actual_load_kw": torch.tensor([[float("nan")]])},
            {"actual_load_kw": torch.tensor([[2.0]])},
        ])


def test_peak_token_feature_check_rejects_nan_on_cpu(env_and_configs):
    env, config = env_and_configs
    observations, _ = env.reset(seed=42)
    algorithm = get_algorithm_spec(config["algorithm"]["name"]).create(
        env, config, device="cpu", seed=42)
    actor = algorithm.actors["mg1"]
    obs = torch.as_tensor(observations["mg1"], dtype=torch.float32).unsqueeze(0)
    actor.peak_features(obs)
    obs[:, actor.peak_features.indices["previous_system_buy_demand"]] = float("nan")
    with pytest.raises(FloatingPointError, match="peak-risk token features"):
        actor.peak_features(obs)


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


def test_token_and_critic_disabled_reuse_exact_parent_action_path(env_and_configs):
    from algorithms.matd3_hierarchical_transformer_predictive_dr.algorithm import (
        Matd3HierarchicalTransformerPredictiveDr,
    )

    env, child_config = env_and_configs
    parent_config = yaml.safe_load(Path(
        "configs/algorithms/matd3_hierarchical_transformer_predictive_dr_hvac.yaml"
    ).read_text(encoding="utf-8"))
    child_config["algorithm"]["peak_risk"].update(
        enabled=False, token_enabled=False, critic_enabled=False,
        actor_weight_final=0.0,
    )
    parent = Matd3HierarchicalTransformerPredictiveDr.from_env(
        env, parent_config, device="cpu", seed=42)
    child = get_algorithm_spec(child_config["algorithm"]["name"]).create(
        env, child_config, device="cpu", seed=42)
    child.actors.load_state_dict(parent.actors.state_dict())
    child.target_actors.load_state_dict(parent.target_actors.state_dict())
    observations, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        x = torch.as_tensor(observations[mg], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            assert torch.equal(child.actors[mg](x), parent.actors[mg](x))


def test_shared_peak_actor_gradient_and_recovery_state(env_and_configs, tmp_path):
    env, cfg = env_and_configs
    cfg["algorithm"]["peak_risk"].update(
        critic_warmup_updates=0, actor_start_update=0,
        actor_weight_ramp_updates=1, critic_update_interval=1,
    )
    cfg["logging"]["diagnostic_interval"] = 2
    spec = get_algorithm_spec(cfg["algorithm"]["name"])
    algorithm = spec.create(env, cfg, device="cpu", seed=42)
    observations, info = env.reset(seed=42)
    state = info["global_state"]
    for _ in range(5):
        actions = {mg: torch.zeros(env.action_dim).numpy() for mg in env.agent_ids}
        next_obs, rewards, done, step_info = env.step(actions)
        algorithm.replay.add(
            observations, state, actions, float(rewards[env.agent_ids[0]]),
            next_obs, step_info["global_state_after"], done)
        observations, state = next_obs, step_info["global_state_after"]
    first = algorithm.update(batch_size=2)
    assert first == {}
    second = algorithm.update(batch_size=2)
    assert algorithm.peak_critic_updates == 2
    assert second["peak"]["actor_weight"] > 0
    assert second["peak"]["actor_penalty"] is not None
    assert second["peak"]["actor_peak_gradient_norm"] > 0
    assert all(param.grad is None for param in algorithm.peak_critic.parameters())
    assert all(param.grad is None for param in algorithm.target_peak_critic.parameters())
    directory = tmp_path / "recovery"
    algorithm.save_training_state(directory, {"completed_episode": 1})
    restored = spec.create(env, cfg, device="cpu", seed=42)
    progress = restored.load_training_state(directory)
    assert progress["completed_episode"] == 1
    assert restored.peak_critic_updates == 2
    assert len(restored.replay) == len(algorithm.replay)
    for name, tensor in algorithm.peak_critic.state_dict().items():
        assert torch.equal(tensor, restored.peak_critic.state_dict()[name]), name
    for name, tensor in algorithm.actors.state_dict().items():
        assert torch.equal(tensor, restored.actors.state_dict()[name]), name


def _fill_identical_replay(env, algorithms, steps=8):
    observations, info = env.reset(seed=4242, deterministic=True)
    state = info["global_state"]
    for _ in range(steps):
        actions = {
            mg: torch.tensor([0.0, 0.0], dtype=torch.float32).numpy()
            for mg in env.agent_ids
        }
        next_obs, rewards, done, step_info = env.step(actions)
        for algorithm in algorithms:
            algorithm.replay.add(
                observations, state, actions, float(rewards[env.agent_ids[0]]),
                next_obs, step_info["global_state_after"], done,
            )
        observations, state = next_obs, step_info["global_state_after"]
        if done:
            observations, info = env.reset(seed=4243, deterministic=True)
            state = info["global_state"]


def test_peak_critic_parameter_budget_is_below_forty_percent(env_and_configs):
    env, cfg = env_and_configs
    algorithm = get_algorithm_spec(cfg["algorithm"]["name"]).create(
        env, cfg, device="cpu", seed=42)
    parent_critic = algorithm.critics[env.agent_ids[0]]
    parent_count = sum(p.numel() for p in parent_critic.parameters())
    peak_count = sum(p.numel() for p in algorithm.peak_critic.parameters())
    assert peak_count / parent_count <= 0.40


def test_disabled_peak_full_update_matches_predictive_parent(env_and_configs):
    from algorithms.matd3_hierarchical_transformer_predictive_dr.algorithm import (
        Matd3HierarchicalTransformerPredictiveDr,
    )

    env, child_cfg = env_and_configs
    parent_cfg = yaml.safe_load(Path(
        "configs/algorithms/matd3_hierarchical_transformer_predictive_dr_hvac.yaml"
    ).read_text(encoding="utf-8"))
    child_cfg["algorithm"]["peak_risk"].update(
        enabled=False, token_enabled=False, critic_enabled=False,
        actor_weight_final=0.0,
    )
    parent = Matd3HierarchicalTransformerPredictiveDr.from_env(
        env, parent_cfg, device="cpu", seed=42)
    child = get_algorithm_spec(child_cfg["algorithm"]["name"]).create(
        env, child_cfg, device="cpu", seed=42)

    # Enforce exact network/optimizer starting state before comparing one update.
    child.actors.load_state_dict(parent.actors.state_dict())
    child.target_actors.load_state_dict(parent.target_actors.state_dict())
    child.critics.load_state_dict(parent.critics.state_dict())
    child.target_critics.load_state_dict(parent.target_critics.state_dict())
    for mg in env.agent_ids:
        child.actor_optimizers[mg].load_state_dict(
            parent.actor_optimizers[mg].state_dict())
        child.critic_optimizers[mg].load_state_dict(
            parent.critic_optimizers[mg].state_dict())

    _fill_identical_replay(env, (parent, child), steps=8)
    parent.update_steps = child.update_steps = 1  # next update is actor-due
    parent.diagnostic_schedule.next_boundary = 2
    child.diagnostic_schedule.next_boundary = 2

    torch.manual_seed(20260922)
    torch_state = torch.get_rng_state().clone()
    parent_metrics = parent.update(batch_size=4)
    torch.set_rng_state(torch_state)
    child_metrics = child.update(batch_size=4)

    for left, right in (
        (parent.actors.state_dict(), child.actors.state_dict()),
        (parent.target_actors.state_dict(), child.target_actors.state_dict()),
        (parent.critics.state_dict(), child.critics.state_dict()),
        (parent.target_critics.state_dict(), child.target_critics.state_dict()),
    ):
        assert left.keys() == right.keys()
        for name in left:
            assert torch.equal(left[name], right[name]), name

    assert parent.update_steps == child.update_steps
    assert parent_metrics.keys() == child_metrics.keys()
    for key in ("critic_loss", "actor_loss", "q1_mean", "q2_mean",
                "target_q_mean", "actor_grad_norm", "critic_grad_norm"):
        if key in parent_metrics:
            assert parent_metrics[key] == pytest.approx(child_metrics[key], rel=0, abs=0)


def test_peak_critic_cadence_moves_target_only_when_due(env_and_configs):
    env, cfg = env_and_configs
    cfg["algorithm"]["peak_risk"]["critic_update_interval"] = 2
    cfg["logging"]["diagnostic_interval"] = 100
    algorithm = get_algorithm_spec(cfg["algorithm"]["name"]).create(
        env, cfg, device="cpu", seed=42)
    _fill_identical_replay(env, (algorithm,), steps=8)

    before_online = {
        k: v.detach().clone() for k, v in algorithm.peak_critic.state_dict().items()
    }
    before_target = {
        k: v.detach().clone() for k, v in algorithm.target_peak_critic.state_dict().items()
    }
    algorithm.update(batch_size=4)  # step 1, peak critic not due
    assert algorithm.peak_critic_updates == 0
    for name, value in algorithm.peak_critic.state_dict().items():
        assert torch.equal(value, before_online[name]), name
    for name, value in algorithm.target_peak_critic.state_dict().items():
        assert torch.equal(value, before_target[name]), name

    algorithm.update(batch_size=4)  # step 2, peak critic due
    assert algorithm.peak_critic_updates == 1
    assert any(
        not torch.equal(value, before_online[name])
        for name, value in algorithm.peak_critic.state_dict().items()
    )
    assert any(
        not torch.equal(value, before_target[name])
        for name, value in algorithm.target_peak_critic.state_dict().items()
    )
