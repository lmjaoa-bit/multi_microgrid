from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_equal_economic_signals_produce_uniform_weights():
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
        economic_slot_weights,
    )

    weights = economic_slot_weights(
        prices=torch.tensor([[0.4, 0.4, 0.4]]),
        system_net_load_kw=torch.tensor([[100.0, 100.0, 100.0]]),
        valid=torch.tensor([[True, True, True]]),
        price_weight=2.0,
        net_load_weight=1.0,
        temperature=1.0,
    )
    assert torch.allclose(weights, torch.full((1, 3), 1.0 / 3.0))


def test_lower_price_and_net_load_receive_more_weight():
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
        economic_slot_weights,
    )

    weights = economic_slot_weights(
        prices=torch.tensor([[0.4, 0.8, 0.8]]),
        system_net_load_kw=torch.tensor([[50.0, 100.0, 150.0]]),
        valid=torch.tensor([[True, True, True]]),
        price_weight=2.0,
        net_load_weight=1.0,
        temperature=1.0,
    )
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]


def test_economic_weights_mask_invalid_and_post_departure_slots():
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
        economic_slot_weights,
    )

    weights = economic_slot_weights(
        prices=torch.tensor([[0.8, 0.4, 0.1, 0.1]]),
        system_net_load_kw=torch.tensor([[100.0, 100.0, 0.0, 0.0]]),
        valid=torch.tensor([[True, True, False, False]]),
        price_weight=2.0,
        net_load_weight=1.0,
        temperature=1.0,
    )
    assert weights[0, 2:].sum().item() == pytest.approx(0.0)
    assert weights[0, :2].sum().item() == pytest.approx(1.0)


def test_ev_reference_respects_floor_upper_and_one_slot_case():
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
        ev_economic_reference,
    )

    reference = ev_economic_reference(
        remaining_battery_kwh=torch.tensor([[30.0]]),
        efficiency=1.0,
        delta_t=1.0,
        max_charge_kw=20.0,
        current_weight=torch.tensor([[1.0]]),
        lower=torch.tensor([[0.5]]),
        upper=torch.tensor([[1.0]]),
    )
    assert torch.allclose(reference, torch.tensor([[1.0]]))


def test_ev_reference_never_weakens_latest_feasible_floor():
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
        ev_economic_reference,
    )

    reference = ev_economic_reference(
        remaining_battery_kwh=torch.tensor([[10.0]]),
        efficiency=1.0,
        delta_t=1.0,
        max_charge_kw=20.0,
        current_weight=torch.tensor([[0.01]]),
        lower=torch.tensor([[0.4]]),
        upper=torch.tensor([[0.8]]),
    )
    assert reference.item() == pytest.approx(0.4)


def test_ev_unavailable_produces_negative_one_action_and_zero_reference():
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
        AdaptiveDRTransform,
    )

    schema = (
        "ev_available", "ev_feasible_min", "ev_feasible_max",
        "remaining_energy", "hours_to_departure",
    )
    transform = AdaptiveDRTransform(
        dr_type="ev_charging",
        schema=schema,
        enabled=True,
    )
    obs = torch.tensor([[0.0, 0.2, 0.8, 0.5, 0.5]])
    latent = torch.tensor([[0.5]])
    details = transform.map_with_details(obs, latent)
    assert details.action.item() == pytest.approx(-1.0)
    assert details.reference.item() == pytest.approx(0.0)
    assert details.available.item() == pytest.approx(0.0)


def test_normalization_reconstruction_computes_exact_physical_signals():
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
        AdaptiveDRTransform,
    )

    normalizers = {
        "mg1": {"load": 100.0, "pv": 200.0, "wind": 1.0, "power": 250.0, "price": 2.0},
        "mg2": {"load": 80.0, "pv": 1.0, "wind": 300.0, "power": 320.0, "price": 2.0},
        "mg3": {"load": 60.0, "pv": 120.0, "wind": 1.0, "power": 150.0, "price": 1.5},
    }
    schema = (
        "ev_available", "ev_feasible_min", "ev_feasible_max", "remaining_energy", "hours_to_departure",
        "grid_buy_price", "load", "pv", "wind", "mg1_net_load", "mg2_net_load",
        "price_t+1", "load_t+1", "pv_t+1", "wind_t+1", "mg1_net_load_t+1", "mg2_net_load_t+1",
        "forecast_valid_t+1",
    )
    obs_map = {
        "ev_available": 1.0,
        "ev_feasible_min": 0.0,
        "ev_feasible_max": 1.0,
        "remaining_energy": 0.5,
        "hours_to_departure": 5.0 / 24.0,
        "grid_buy_price": 0.8,
        "load": 0.5,
        "pv": 0.5,
        "wind": 0.0,
        "mg1_net_load": 0.4,
        "mg2_net_load": -0.2,
        "price_t+1": 0.6,
        "load_t+1": 0.8,
        "pv_t+1": 0.2,
        "wind_t+1": 0.0,
        "mg1_net_load_t+1": 0.2,
        "mg2_net_load_t+1": 0.1,
        "forecast_valid_t+1": 1.0,
    }
    obs_tensor = torch.tensor([[obs_map[k] for k in schema]])
    transform = AdaptiveDRTransform(
        dr_type="ev_charging",
        schema=schema,
        normalizers_by_microgrid=normalizers,
        active_agent="mg3",
        forecast_horizon=1,
    )
    prices, sys_net, valid = transform.reconstruct_economic_signals(obs_tensor)

    assert prices[0, 0].item() == pytest.approx(1.2)
    assert prices[0, 1].item() == pytest.approx(0.9)
    assert sys_net[0, 0].item() == pytest.approx(6.0)
    assert sys_net[0, 1].item() == pytest.approx(106.0)
    assert bool(valid[0, 0].item()) is True
    assert bool(valid[0, 1].item()) is True


def test_shared_interface_and_action_shapes(v2_env_and_config):
    env, config = v2_env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDRV2.from_env(env, config, device="cpu", seed=42)
    required = ("select_actions", "update", "save", "load", "replay")
    assert all(hasattr(algorithm, name) for name in required)
    obs, _ = env.reset(seed=42)
    actions = algorithm.select_actions(obs, noise_std=0.0)
    assert set(actions) == set(env.agent_ids)
    for agent in env.agent_ids:
        assert actions[agent].shape == (2,)

    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.actor import AdaptiveDRActor
    for mg in env.agent_ids:
        assert isinstance(algorithm.actors[mg], AdaptiveDRActor)


def test_exact_mg1_parity_with_v1(v2_env_and_config):
    from algorithms.matd3_hierarchical_transformer_feasible_dr.algorithm import (
        Matd3HierarchicalTransformerFeasibleDR,
    )
    from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.actor import AdaptiveDRActor

    env, config = v2_env_and_config
    v1_config = _load("matd3_hierarchical_transformer_feasible_dr.yaml")
    v1_algo = Matd3HierarchicalTransformerFeasibleDR.from_env(env, v1_config, device="cpu", seed=42)
    v2_algo = Matd3HierarchicalTransformerFeasibleDRV2.from_env(env, config, device="cpu", seed=42)

    v2_algo.actors["mg1"].load_state_dict(v1_algo.actors["mg1"].state_dict())

    obs, _ = env.reset(seed=42)
    obs_tensor = torch.tensor([obs["mg1"]], dtype=torch.float32)

    with torch.no_grad():
        v1_action = v1_algo.actors["mg1"](obs_tensor)
        v2_action = v2_algo.actors["mg1"](obs_tensor)

    assert torch.equal(v1_action, v2_action)
    assert v2_algo.actors["mg1"].adaptive_dr is None


def test_deterministic_feasible_actions_for_mg2_and_mg3(v2_env_and_config):
    env, config = v2_env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDRV2.from_env(env, config, device="cpu", seed=42)

    for mg in ("mg2", "mg3"):
        actor = algorithm.actors[mg]
        schema = algorithm.observation_schemas[mg]
        obs_dim = len(schema)
        generator = torch.Generator().manual_seed(42)
        obs = torch.randn(256, obs_dim, generator=generator)
        latent = torch.rand(256, 2, generator=generator) * 2 - 1
        latent[:64, 1] = -1.0
        latent[64:128, 1] = 1.0
        latent[128:192, 1] = 0.0

        if mg == "mg2":
            min_idx = actor.adaptive_dr.hvac_min_idx
            max_idx = actor.adaptive_dr.hvac_max_idx
            obs[:, min_idx] = obs[:, min_idx].clamp(-1.0, 0.0)
            obs[:, max_idx] = obs[:, max_idx].clamp(0.0, 1.0)
            mapped = actor.map_latent_actions(obs, latent)
            feasible_min = obs[:, min_idx : min_idx + 1]
            feasible_max = obs[:, max_idx : max_idx + 1]
            assert torch.all(mapped[:, 1:2] >= feasible_min - 1e-6)
            assert torch.all(mapped[:, 1:2] <= feasible_max + 1e-6)
        elif mg == "mg3":
            min_idx = actor.adaptive_dr.ev_min_idx
            max_idx = actor.adaptive_dr.ev_max_idx
            avail_idx = actor.adaptive_dr.ev_avail_idx
            obs[:, min_idx] = obs[:, min_idx].clamp(0.0, 0.5)
            obs[:, max_idx] = obs[:, max_idx].clamp(0.5, 1.0)
            obs[:128, avail_idx] = 1.0
            obs[128:, avail_idx] = 0.0
            mapped = actor.map_latent_actions(obs, latent)
            avail_mask = obs[:, avail_idx] > 0.5
            raw_min = obs[:, min_idx : min_idx + 1] * 2.0 - 1.0
            raw_max = obs[:, max_idx : max_idx + 1] * 2.0 - 1.0
            assert torch.all(mapped[avail_mask, 1:2] >= raw_min[avail_mask] - 1e-6)
            assert torch.all(mapped[avail_mask, 1:2] <= raw_max[avail_mask] + 1e-6)
            assert torch.allclose(mapped[~avail_mask, 1:2], torch.tensor([-1.0]))

    obs, _ = env.reset(seed=42)
    done = False
    mg2_projections = 0
    mg3_projections = 0
    while not done:
        actions = algorithm.select_actions(obs, noise_std=0.0)
        obs, rewards, done, info = env.step(actions)
        step_infos = info.get("step_info", [info])
        for sub_info in step_infos:
            constraints = sub_info.get("constraints", {})
            dr_constraints = constraints.get("dr_by_microgrid", {})
            mg2_projections += int(dr_constraints.get("mg2", 0.0) > 0)
            mg3_projections += int(dr_constraints.get("mg3", 0.0) > 0)
    assert mg2_projections == 0
    assert mg3_projections == 0


def test_extreme_noise_feasibility(v2_env_and_config):
    env, config = v2_env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDRV2.from_env(env, config, device="cpu", seed=42)

    observations, info = env.reset(seed=42, deterministic=True)
    state = torch.as_tensor(info["global_state"], dtype=torch.float32).unsqueeze(0)
    raw_actions = []
    for mg in env.agent_ids:
        obs_tensor = torch.as_tensor(observations[mg], dtype=torch.float32).unsqueeze(0)
        actor = algorithm.target_actors[mg]
        latent = actor.forward_latent(obs_tensor)
        extreme_noise = torch.full_like(latent, 10.0)
        from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.training import map_noisy_adaptive_latent
        mapped = map_noisy_adaptive_latent(
            actor,
            obs_tensor,
            latent,
            noise=extreme_noise,
            noise_clip=algorithm.target_noise_clip,
        )
        assert torch.isfinite(mapped).all()
        assert (mapped >= -1.0 - 1e-6).all()
        assert (mapped <= 1.0 + 1e-6).all()
        raw_actions.append(mapped)

    _, plans = algorithm.project_actions(state, torch.stack(raw_actions, dim=1))
    for index, mg in enumerate(env.agent_ids):
        if mg in ("mg2", "mg3"):
            assert not plans[index]["dr_projected"].any()

    # Also test zero-width intervals and unavailable EV rows on batch
    ev_actor = algorithm.actors["mg3"]
    schema = algorithm.observation_schemas["mg3"]
    obs_unavail = torch.zeros(10, len(schema))
    obs_unavail[:, ev_actor.adaptive_dr.ev_avail_idx] = 0.0
    latent = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]).repeat(4, 1)[:10]
    mapped_unavail = map_noisy_adaptive_latent(
        ev_actor,
        obs_unavail,
        latent,
        noise_std=2.0,
        noise_clip=1.5,
    )
    assert torch.isfinite(mapped_unavail).all()
    assert torch.allclose(mapped_unavail[:, 1:2], torch.tensor([-1.0]))

    # Zero-width interval for HVAC
    hvac_actor = algorithm.actors["mg2"]
    schema2 = algorithm.observation_schemas["mg2"]
    obs_zero = torch.zeros(10, len(schema2))
    obs_zero[:, hvac_actor.adaptive_dr.hvac_min_idx] = 0.25
    obs_zero[:, hvac_actor.adaptive_dr.hvac_max_idx] = 0.25
    mapped_zero = map_noisy_adaptive_latent(
        hvac_actor,
        obs_zero,
        latent,
        noise_std=2.0,
        noise_clip=1.5,
    )
    assert torch.isfinite(mapped_zero).all()
    assert torch.allclose(mapped_zero[:, 1:2], torch.tensor([0.25]), atol=1e-5)


def test_ev_available_boundary_statistics_mask_unavailable_rows(v2_env_and_config):
    env, config = v2_env_and_config
    algorithm = Matd3HierarchicalTransformerFeasibleDRV2.from_env(env, config, device="cpu", seed=42)
    ev_actor = algorithm.actors["mg3"]
    schema = algorithm.observation_schemas["mg3"]
    obs_dim = len(schema)

    # Case 1: All rows unavailable
    obs_all_unavail = torch.zeros(10, obs_dim)
    obs_all_unavail[:, ev_actor.adaptive_dr.ev_avail_idx] = 0.0
    latent = torch.full((10, 2), -1.0)
    ev_actor.map_latent_actions(obs_all_unavail, latent, collect_diagnostics=True)
    assert ev_actor.last_diagnostics["ev_available_rate"] == 0.0
    assert ev_actor.last_diagnostics["ev_available_boundary_rate"] == 0.0
    assert not np.isnan(ev_actor.last_diagnostics["ev_available_boundary_rate"])

    # Case 2: Mixed batch: 1 available row at lower boundary (-1.0), 9 unavailable rows
    obs_mixed = torch.zeros(10, obs_dim)
    obs_mixed[0, ev_actor.adaptive_dr.ev_avail_idx] = 1.0
    obs_mixed[0, ev_actor.adaptive_dr.hours_dep_idx] = 0.5
    obs_mixed[0, ev_actor.adaptive_dr.ev_min_idx] = 0.1
    obs_mixed[0, ev_actor.adaptive_dr.ev_max_idx] = 0.8
    ev_actor.map_latent_actions(obs_mixed, latent, collect_diagnostics=True)
    assert ev_actor.last_diagnostics["ev_available_rate"] == pytest.approx(0.1)
    assert ev_actor.last_diagnostics["ev_available_boundary_rate"] == pytest.approx(1.0)


def test_diagnostic_schedule_and_keys(v2_env_and_config):
    env, config = v2_env_and_config
    config["logging"]["diagnostic_interval"] = 2
    config["algorithm"]["policy_delay"] = 2
    algorithm = Matd3HierarchicalTransformerFeasibleDRV2.from_env(env, config, device="cpu", seed=42)

    obs, _ = env.reset(seed=42)
    state = env.global_state()
    for _ in range(50):
        actions = algorithm.select_actions(obs, noise_std=0.1)
        next_obs, rewards, done, info = env.step(actions)
        next_state = env.global_state()
        team_reward = sum(rewards.values())
        algorithm.replay.add(obs, state, actions, team_reward, next_obs, next_state, done)
        obs = next_obs
        state = next_state
        if done:
            obs, _ = env.reset(seed=42)
            state = env.global_state()

    result1 = algorithm.update(batch_size=16)
    assert "actor_diagnostics" not in result1

    result2 = algorithm.update(batch_size=16)
    assert "actor_diagnostics" in result2
    actor_diags = result2["actor_diagnostics"]
    assert "mg2" in actor_diags
    assert "mg3" in actor_diags

    COMMON_KEYS = {
        "actor_dr_latent_mean",
        "actor_dr_latent_std",
        "actor_dr_mapped_mean",
        "actor_dr_feasible_min_mean",
        "actor_dr_feasible_max_mean",
        "actor_dr_interval_width_mean",
        "actor_dr_lower_boundary_rate",
        "actor_dr_upper_boundary_rate",
        "actor_dr_effective_boundary_rate",
        "dr_projection_rate",
        "dr_projection_delta_mean",
    }
    MG2_KEYS = {
        "actor_hvac_recovery_anchor_mean",
        "actor_hvac_recovery_gate_mean",
        "actor_hvac_recovery_gate_active_rate",
        "actor_hvac_recovery_gate_full_rate",
        "actor_hvac_terminal_air_gap_mean_c",
        "actor_hvac_terminal_mass_gap_mean_c",
        "actor_hvac_terminal_recovery_pressure_mean",
    }
    MG3_KEYS = {
        "actor_ev_mandatory_floor_mean",
        "actor_ev_mandatory_floor_active_rate",
        "actor_ev_economic_reference_mean",
        "actor_ev_reference_above_floor_mean",
        "actor_ev_latent_reference_deviation_mean",
        "actor_ev_deadline_pressure_mean",
        "actor_ev_available_rate",
        "actor_ev_available_boundary_rate",
    }

    for k in COMMON_KEYS:
        assert k in actor_diags["mg2"], f"Missing {k} in mg2 diagnostics"
        assert k in actor_diags["mg3"], f"Missing {k} in mg3 diagnostics"
        assert np.isfinite(actor_diags["mg2"][k])
        assert np.isfinite(actor_diags["mg3"][k])

    for k in MG2_KEYS:
        assert k in actor_diags["mg2"], f"Missing {k} in mg2 diagnostics"
        assert np.isfinite(actor_diags["mg2"][k])

    for k in MG3_KEYS:
        assert k in actor_diags["mg3"], f"Missing {k} in mg3 diagnostics"
        assert np.isfinite(actor_diags["mg3"][k])









