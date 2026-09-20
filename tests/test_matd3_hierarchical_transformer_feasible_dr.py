from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from algorithms.matd3_hierarchical_transformer.algorithm import Matd3HierarchicalTransformer
from algorithms.matd3_hierarchical_transformer_feasible_dr.algorithm import (
    Matd3HierarchicalTransformerFeasibleDR,
)
from algorithms.matd3_hierarchical_transformer_feasible_dr.feasible_dr import (
    FeasibleDRTransform,
    latest_feasible_ev_floor,
)
from algorithms.matd3_hierarchical_transformer_feasible_dr.training import map_noisy_latent
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


@pytest.fixture
def env_and_config():
    config_path = Path("configs/cooperative.yaml")
    parent_cfg_path = Path("configs/algorithms/matd3_hierarchical_transformer.yaml")
    feasible_cfg_path = Path("configs/algorithms/matd3_hierarchical_transformer_feasible_dr.yaml")
    env_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parent_cfg = yaml.safe_load(parent_cfg_path.read_text(encoding="utf-8"))
    feasible_cfg = yaml.safe_load(feasible_cfg_path.read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(env_cfg, split="train")
    return env, parent_cfg, feasible_cfg


def test_feasible_dr_plugin_uses_its_own_config():
    spec = get_algorithm_spec("matd3_hierarchical_transformer_feasible_dr")
    config = yaml.safe_load(Path(spec.default_config_path).read_text(encoding="utf-8"))

    assert spec.slug == "matd3_hierarchical_transformer_feasible_dr"
    assert spec.algorithm_class.__name__ == "Matd3HierarchicalTransformerFeasibleDR"
    assert config["algorithm"]["name"] == spec.slug
    assert config["algorithm"]["feasible_dr"]["agents"] == {
        "mg1": False,
        "mg2": True,
        "mg3": True,
    }


def test_latest_feasible_ev_floor_preserves_future_schedule():
    floor = latest_feasible_ev_floor(
        remaining_energy=torch.tensor([[30.0]]),
        hours_to_departure=torch.tensor([[2.0]]),
        max_charge_kw=20.0,
        efficiency=1.0,
        delta_t=1.0,
        upper=torch.tensor([[1.0]]),
    )
    assert torch.allclose(floor, torch.tensor([[0.5]]))


def test_hvac_transform_maps_random_preferences_inside_bounds():
    transform = FeasibleDRTransform.for_test(
        dr_type="commercial_hvac",
        schema=("hvac_feasible_min", "hvac_feasible_max"),
    )
    generator = torch.Generator().manual_seed(42)
    lower = torch.rand(10_000, 1, generator=generator) * 2.0 - 1.0
    width = torch.rand(10_000, 1, generator=generator) * (1.0 - lower)
    upper = lower + width
    observation = torch.cat([lower, upper], dim=1)
    latent = torch.rand(10_000, 1, generator=generator) * 2.0 - 1.0

    raw = transform(observation, latent)

    assert torch.all(raw >= lower - 1.0e-7)
    assert torch.all(raw <= upper + 1.0e-7)


def test_ev_unavailable_maps_to_zero_charge():
    transform = FeasibleDRTransform.for_test(
        dr_type="ev_charging",
        schema=(
            "remaining_energy",
            "hours_to_departure",
            "ev_available",
            "ev_feasible_min",
            "ev_feasible_max",
        ),
        max_charge_kw=100.0,
        efficiency=0.95,
        episode_length=24,
    )
    observation = torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.0]])
    raw = transform(observation, torch.tensor([[1.0]]))
    assert torch.allclose(raw, torch.tensor([[-1.0]]))


def test_degenerate_interval_is_finite_and_exact():
    transform = FeasibleDRTransform.for_test(
        dr_type="commercial_hvac",
        schema=("hvac_feasible_min", "hvac_feasible_max"),
        interval_epsilon=1.0e-6,
    )
    latent = torch.tensor([[0.25]], requires_grad=True)
    raw = transform(torch.tensor([[0.3, 0.3]]), latent)
    raw.sum().backward()
    assert torch.allclose(raw, torch.tensor([[0.3]]))
    assert torch.isfinite(raw).all()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_non_degenerate_transform_has_nonzero_finite_gradient():
    transform = FeasibleDRTransform.for_test(
        dr_type="commercial_hvac",
        schema=("hvac_feasible_min", "hvac_feasible_max"),
    )
    latent = torch.tensor([[0.0]], requires_grad=True)
    raw = transform(torch.tensor([[-0.5, 0.75]]), latent)
    raw.sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert latent.grad.abs().item() > 0.0


def test_mg1_feasible_actor_matches_parent_actor(env_and_config):
    env, parent_config, feasible_config = env_and_config
    parent = Matd3HierarchicalTransformer.from_env(env, parent_config, device="cpu", seed=42)
    feasible = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, feasible_config, device="cpu", seed=42
    )
    feasible.actors["mg1"].load_state_dict(parent.actors["mg1"].state_dict(), strict=False)
    observation, _ = env.reset(seed=42)
    tensor = torch.as_tensor(observation["mg1"], dtype=torch.float32).unsqueeze(0)
    expected = parent.actors["mg1"](tensor)
    actual = feasible.actors["mg1"](tensor)
    assert torch.allclose(actual, expected, atol=1.0e-7)


def test_mg2_and_mg3_actor_requests_are_feasible(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    for mg in ("mg2", "mg3"):
        tensor = torch.as_tensor(observations[mg], dtype=torch.float32).unsqueeze(0)
        action = algorithm.actors[mg](tensor)
        latent = algorithm.actors[mg].forward_latent(tensor)
        remapped = algorithm.actors[mg].map_latent_actions(tensor, latent)
        assert torch.allclose(action, remapped)
        assert action.shape == (1, 2)


def test_noisy_rollout_actions_remain_feasible(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    for _ in range(24):
        actions = algorithm.select_actions(observations, noise_std=2.0)
        observations, _, done, info = env.step(actions)
        for mg in ("mg2", "mg3"):
            assert not info["plans"][mg]["dr_projected"]
        if done:
            break


def test_select_actions_accepts_heterogeneous_observation_shapes(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    actions = algorithm.select_actions(observations, noise_std=0.2)
    assert set(actions) == set(env.agent_ids)
    assert all(action.shape == (2,) for action in actions.values())


def test_replay_stores_environment_actions_not_latent_preferences(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    state = env.global_state()
    actions = algorithm.select_actions(observations, noise_std=0.5)
    next_observations, rewards, done, _ = env.step(actions)
    algorithm.replay.add(
        observations,
        state,
        actions,
        sum(rewards.values()),
        next_observations,
        env.global_state(),
        done,
    )
    batch = algorithm.replay.sample(1, env.agent_ids, torch.device("cpu"))
    for mg in env.agent_ids:
        assert torch.allclose(batch["actions"][mg][0], torch.as_tensor(actions[mg]))


def test_target_noise_is_mapped_before_projection(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    state = torch.as_tensor(env.global_state(), dtype=torch.float32).unsqueeze(0)
    raw_actions = []
    for mg in env.agent_ids:
        observation = torch.as_tensor(
            observations[mg], dtype=torch.float32
        ).unsqueeze(0)
        actor = algorithm.target_actors[mg]
        latent = actor.forward_latent(observation)
        extreme_noise = torch.full_like(latent, 10.0)
        raw_actions.append(
            map_noisy_latent(
                actor,
                observation,
                latent,
                extreme_noise,
                noise_clip=algorithm.target_noise_clip,
            )
        )
    _, plans = algorithm.project_actions(state, torch.stack(raw_actions, dim=1))
    for index, mg in enumerate(env.agent_ids):
        if mg in ("mg2", "mg3"):
            assert not plans[index]["dr_projected"].any()


def test_feasible_dr_update_and_diagnostics(env_and_config):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    obs, _ = env.reset(seed=42)
    state = env.global_state()
    for _ in range(10):
        actions = algorithm.select_actions(obs, noise_std=0.1)
        next_obs, rewards, done, info = env.step(actions)
        next_state = env.global_state()
        team_reward = sum(rewards.values())
        algorithm.replay.add(obs, state, actions, team_reward, next_obs, next_state, done)
        obs = next_obs
        state = next_state
        if done:
            obs, _ = env.reset()
            state = env.global_state()

    assert len(algorithm.replay) >= 10
    # Ordinary step returns empty dict
    assert algorithm.update(batch_size=4) == {}
    # Trigger diagnostic step
    algorithm.diagnostic_schedule.next_boundary = 2
    metrics = algorithm.update(batch_size=4)
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert "actor_diagnostics" in metrics


def test_checkpoint_round_trip_and_incompatibility(env_and_config, tmp_path):
    env, _, config = env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    observations, _ = env.reset(seed=42)
    before = algorithm.select_actions(observations, noise_std=0.0)
    path = tmp_path / "model.pt"
    algorithm.save(path)
    restored = Matd3HierarchicalTransformerFeasibleDR.from_env(
        env, config, device="cpu", seed=42
    )
    restored.load(path)
    after = restored.select_actions(observations, noise_std=0.0)
    for mg in env.agent_ids:
        assert torch.allclose(torch.as_tensor(before[mg]), torch.as_tensor(after[mg]))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["algorithm_slug"] = "matd3_hierarchical_transformer"
    wrong_slug = tmp_path / "wrong_slug.pt"
    torch.save(payload, wrong_slug)
    with pytest.raises(ValueError, match="Incompatible algorithm checkpoint"):
        restored.load(wrong_slug)

    payload["algorithm_slug"] = "matd3_hierarchical_transformer_feasible_dr"
    payload["feasible_dr_config"] = {"enabled": False}
    wrong_config = tmp_path / "wrong_config.pt"
    torch.save(payload, wrong_config)
    with pytest.raises(ValueError, match="feasible DR configuration"):
        restored.load(wrong_config)


def test_global_disable_overrides_per_agent_enable():
    transform = FeasibleDRTransform.from_config(
        "commercial_hvac",
        ("hvac_feasible_min", "hvac_feasible_max"),
        {
            "enabled": False,
            "agents": {"mg2": True},
            "hvac": {"interval_mapping_enabled": True},
        },
    )
    observation = torch.tensor([[-0.5, 0.25]])
    latent = torch.tensor([[0.8]])

    assert torch.allclose(transform(observation, latent), latent)


def test_hvac_interval_mapping_switch_disables_mapping():
    transform = FeasibleDRTransform.from_config(
        "commercial_hvac",
        ("hvac_feasible_min", "hvac_feasible_max"),
        {
            "enabled": True,
            "agents": {"mg2": True},
            "hvac": {"interval_mapping_enabled": False},
        },
    )
    observation = torch.tensor([[-0.5, 0.25]])
    latent = torch.tensor([[0.8]])

    assert torch.allclose(transform(observation, latent), latent)


def test_schema_indices_remain_device_independent_python_integers():
    transform = FeasibleDRTransform.for_test(
        dr_type="commercial_hvac",
        schema=("hvac_feasible_min", "hvac_feasible_max"),
    )
    transform.to("cpu")

    assert isinstance(transform.hvac_min_idx, int)
    assert isinstance(transform.hvac_max_idx, int)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("exploration_in_latent_space",), False, "exploration_in_latent_space"),
        (("target_smoothing_in_latent_space",), False, "target_smoothing_in_latent_space"),
        (("hvac", "recovery_anchor_enabled"), True, "recovery_anchor_enabled"),
    ],
)
def test_unsupported_v1_modes_are_rejected(env_and_config, path, value, message):
    env, _, config = env_and_config
    current = config["algorithm"]["feasible_dr"]
    if len(path) == 1:
        current[path[0]] = value
    else:
        current[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=message):
        Matd3HierarchicalTransformerFeasibleDR.from_env(
            env, config, device="cpu", seed=42
        )


