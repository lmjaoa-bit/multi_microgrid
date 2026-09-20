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

