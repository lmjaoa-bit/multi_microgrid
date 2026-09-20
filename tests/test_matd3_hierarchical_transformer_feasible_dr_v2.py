from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.algorithm import (
    Matd3HierarchicalTransformerFeasibleDRV2,
)
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
    residual_map,
    smoothstep,
)

SLUG = "matd3_hierarchical_transformer_feasible_dr_v2"
CONFIG_ROOT = Path("configs/algorithms")


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_v2_plugin_and_ablation_configs_are_isolated():
    spec = get_algorithm_spec(SLUG)
    combined = _load(f"{SLUG}.yaml")
    hvac = _load(f"{SLUG}_hvac.yaml")
    ev = _load(f"{SLUG}_ev.yaml")

    assert spec.slug == SLUG
    assert spec.algorithm_class.__name__ == "Matd3HierarchicalTransformerFeasibleDRV2"
    assert Path(spec.default_config_path).name == f"{SLUG}.yaml"
    for config in (combined, hvac, ev):
        assert config["algorithm"]["name"] == SLUG

    assert combined["algorithm"]["adaptive_dr"]["hvac"]["recovery_anchor_enabled"] is True
    assert combined["algorithm"]["adaptive_dr"]["ev"]["economic_reference_enabled"] is True
    assert hvac["algorithm"]["adaptive_dr"]["hvac"]["recovery_anchor_enabled"] is True
    assert hvac["algorithm"]["adaptive_dr"]["ev"]["economic_reference_enabled"] is False
    assert ev["algorithm"]["adaptive_dr"]["hvac"]["recovery_anchor_enabled"] is False
    assert ev["algorithm"]["adaptive_dr"]["ev"]["economic_reference_enabled"] is True


@pytest.fixture
def v2_env_and_config():
    env_config = yaml.safe_load(Path("configs/cooperative.yaml").read_text(encoding="utf-8"))
    algorithm_config = _load(f"{SLUG}.yaml")
    env = MultiMicrogridEnv(env_config, split="train")
    return env, algorithm_config


def test_residual_map_endpoints_and_reference():
    lower = torch.tensor([[-0.4], [0.1], [-0.8]])
    reference = torch.tensor([[0.2], [0.3], [-0.2]])
    upper = torch.tensor([[0.9], [0.7], [0.5]])
    latent = torch.tensor([[-1.0], [0.0], [1.0]])
    actual = residual_map(latent, lower, reference, upper, interval_epsilon=1e-6)
    assert torch.allclose(actual, torch.tensor([[-0.4], [0.3], [0.5]]))


def test_residual_map_keeps_ten_thousand_random_actions_in_interval():
    generator = torch.Generator().manual_seed(42)
    lower = torch.rand(10_000, 1, generator=generator) * 2 - 1
    upper = lower + torch.rand(10_000, 1, generator=generator) * (1 - lower)
    reference = lower + torch.rand(10_000, 1, generator=generator) * (upper - lower)
    latent = torch.rand(10_000, 1, generator=generator) * 2 - 1
    actual = residual_map(latent, lower, reference, upper, interval_epsilon=1e-6)
    assert torch.all(actual >= lower - 1e-7)
    assert torch.all(actual <= upper + 1e-7)


def test_smoothstep_limits_and_midpoint():
    values = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
    actual = smoothstep(values, 0.0, 1.0)
    assert torch.allclose(actual, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_residual_map_has_finite_gradients_on_both_sides():
    latent = torch.tensor([[-0.25], [0.25]], requires_grad=True)
    actual = residual_map(
        latent,
        torch.tensor([[-0.5], [-0.5]]),
        torch.tensor([[0.0], [0.0]]),
        torch.tensor([[0.75], [0.75]]),
        interval_epsilon=1e-6,
    )
    actual.sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert torch.all(latent.grad.abs() > 0)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("hvac", "recovery_pressure_start", 0.9, "recovery_pressure"),
        ("hvac", "recovery_start_hours", 1.0, "recovery.*hours"),
        ("hvac", "recovery_start_hours", 25.0, "episode horizon"),
        ("ev", "price_weight", -0.1, "price_weight"),
        ("ev", "net_load_weight", -0.1, "net_load_weight"),
        ("ev", "softmax_temperature", 0.0, "softmax_temperature"),
    ],
)
def test_invalid_adaptive_config_is_rejected(v2_env_and_config, section, field, value, message):
    env, config = v2_env_and_config
    config["algorithm"]["adaptive_dr"][section][field] = value
    with pytest.raises(ValueError, match=message):
        Matd3HierarchicalTransformerFeasibleDRV2.from_env(env, config, device="cpu", seed=42)


HVAC_PHYSICS = {
    "c_air_kwh_per_c": 15.0,
    "c_mass_kwh_per_c": 250.0,
    "cop_heat": 3.3,
    "cop_cool": 3.2,
    "heating_max_kw": 35.0,
    "cooling_max_kw": 35.0,
    "delta_t": 1.0,
}


def test_hvac_recovery_anchor_uses_gap_energy_and_direction():
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
        hvac_recovery_anchor,
    )

    anchor = hvac_recovery_anchor(
        air_gap_c=torch.tensor([[1.0], [-1.0], [0.0]]),
        mass_gap_c=torch.tensor([[0.2], [-0.2], [0.0]]),
        remaining_hours=torch.tensor([[5.0], [5.0], [5.0]]),
        lower=torch.full((3, 1), -1.0),
        upper=torch.full((3, 1), 1.0),
        **HVAC_PHYSICS,
    )
    heating_kw = (1.0 * 15.0 + 0.2 * 250.0) / 5.0 / 3.3
    cooling_kw = (-1.0 * 15.0 - 0.2 * 250.0) / 5.0 / 3.2
    assert anchor[0, 0].item() == pytest.approx(heating_kw / 35.0)
    assert anchor[1, 0].item() == pytest.approx(cooling_kw / 35.0)
    assert anchor[2, 0].item() == pytest.approx(0.0)


def test_hvac_recovery_gate_limits_and_monotonicity():
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
        hvac_recovery_gate,
    )

    pressure = torch.tensor([[0.1], [0.5], [1.0]])
    hours = torch.tensor([[8.0], [4.0], [1.0]])
    gate = hvac_recovery_gate(
        pressure,
        hours,
        pressure_start=0.35,
        pressure_full=0.85,
        start_hours=6.0,
        full_hours=2.0,
    )
    assert gate[0, 0].item() == pytest.approx(0.0)
    assert 0.0 < gate[1, 0].item() < 1.0
    assert gate[2, 0].item() == pytest.approx(1.0)
    assert torch.all(gate[1:] >= gate[:-1])


HVAC_SCHEMA = (
    "terminal_air_gap", "terminal_mass_gap", "time_remaining",
    "thermal_recovery_pressure", "hvac_feasible_min", "hvac_feasible_max",
)


def make_hvac_v2_transform():
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
        AdaptiveDRTransform,
    )

    return AdaptiveDRTransform.for_hvac_test(
        HVAC_SCHEMA,
        c_air_kwh_per_c=15.0,
        c_mass_kwh_per_c=250.0,
        cop_heat=3.3,
        cop_cool=3.2,
        heating_max_kw=35.0,
        cooling_max_kw=35.0,
        delta_t=1.0,
        episode_length=24,
        recovery_pressure_start=0.35,
        recovery_pressure_full=0.85,
        recovery_start_hours=6.0,
        recovery_full_hours=2.0,
    )


def test_hvac_zero_gate_matches_v1_and_full_gate_selects_anchor():
    observation = torch.tensor([
        [1.0 / 40.0, 0.2 / 40.0, 8.0 / 24.0, 0.1, -0.5, 0.8],
        [1.0 / 40.0, 0.2 / 40.0, 1.0 / 24.0, 1.0, -0.5, 0.8],
    ])
    latent = torch.tensor([[0.2], [0.2]])
    transform = make_hvac_v2_transform()
    details = transform.map_with_details(observation, latent)

    v1_free = -0.5 + ((0.2 + 1.0) / 2.0) * (0.8 - -0.5)
    assert details.action[0, 0].item() == pytest.approx(v1_free)
    assert details.recovery_gate[0, 0].item() == pytest.approx(0.0)
    assert details.recovery_gate[1, 0].item() == pytest.approx(1.0)
    assert details.action[1, 0].item() == pytest.approx(details.recovery_anchor[1, 0].item())


def test_random_hvac_actions_remain_within_feasible_interval():
    generator = torch.Generator().manual_seed(42)
    obs = torch.rand(10_000, len(HVAC_SCHEMA), generator=generator)
    obs[:, 4] = obs[:, 4] * 2 - 1
    obs[:, 5] = obs[:, 4] + torch.rand(10_000, generator=generator) * (1 - obs[:, 4])
    latent = torch.rand(10_000, 1, generator=generator) * 2 - 1
    transform = make_hvac_v2_transform()
    details = transform.map_with_details(obs, latent)
    assert torch.all(details.action >= obs[:, 4:5] - 1e-6)
    assert torch.all(details.action <= obs[:, 5:6] + 1e-6)
    assert torch.isfinite(details.action).all()



