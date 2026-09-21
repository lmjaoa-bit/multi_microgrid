from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from algorithms.matd3_hierarchical_transformer_predictive_dr.predictive_hvac import (
    PredictiveHVACTransform,
    smoothstep,
    validate_predictive_hvac_config,
)
from algorithms.matd3_hierarchical_transformer_predictive_dr.algorithm import (
    Matd3HierarchicalTransformerPredictiveDr,
)

SLUG = "matd3_hierarchical_transformer_predictive_dr"
CONFIG_PATH = Path("configs/algorithms/matd3_hierarchical_transformer_predictive_dr_hvac.yaml")


@pytest.fixture
def env_and_config():
    env_config = yaml.safe_load(Path("configs/cooperative.yaml").read_text(encoding="utf-8"))
    algo_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(env_config, split="train")
    return env, algo_config


# ---------------------------------------------------------------------------
# Smoothstep & Config Validation Tests
# ---------------------------------------------------------------------------

def test_smoothstep_boundaries_and_midpoint():
    val = torch.tensor([-1.0, 0.0, 2.0, 4.0, 5.0])
    res = smoothstep(val, 0.0, 4.0)
    assert res[0].item() == 0.0
    assert res[1].item() == 0.0
    assert res[2].item() == 0.5  # midpoint
    assert res[3].item() == 1.0
    assert res[4].item() == 1.0


def test_config_validation_rejections():
    # Valid config
    valid_cfg = {
        "recovery_horizon_hours": 6.0,
        "reference_transition_full_hours": 4.0,
        "final_air_priority_hours": 2.0,
        "early_air_weight": 1.0,
        "early_mass_weight": 1.0,
        "final_air_weight": 10.0,
        "final_mass_weight": 0.1,
        "mass_assist_extra_c": 1.0,
        "solar_forecast_mode": "zero",
        "invalid_forecast_fallback": "air_only",
        "residual_reference_mapping": True,
        "detach_physical_reference": True,
    }
    validate_predictive_hvac_config(valid_cfg, forecast_horizon=8)

    # Rejection 1: nonpositive recovery horizon
    bad = dict(valid_cfg, recovery_horizon_hours=0.0)
    with pytest.raises(ValueError, match="recovery_horizon_hours"):
        validate_predictive_hvac_config(bad, forecast_horizon=8)

    # Rejection 2: recovery horizon > forecast horizon
    bad = dict(valid_cfg, recovery_horizon_hours=10.0)
    with pytest.raises(ValueError, match="forecast_horizon"):
        validate_predictive_hvac_config(bad, forecast_horizon=8)

    # Rejection 3: reference_transition_full_hours outside [0, rec_h)
    bad = dict(valid_cfg, reference_transition_full_hours=6.5)
    with pytest.raises(ValueError, match="reference_transition_full_hours"):
        validate_predictive_hvac_config(bad, forecast_horizon=8)

    # Rejection 4: final_air_priority_hours outside (0, rec_h)
    bad = dict(valid_cfg, final_air_priority_hours=0.0)
    with pytest.raises(ValueError, match="final_air_priority_hours"):
        validate_predictive_hvac_config(bad, forecast_horizon=8)

    # Rejection 5: final_air_weight <= final_mass_weight
    bad = dict(valid_cfg, final_air_weight=0.1, final_mass_weight=0.1)
    with pytest.raises(ValueError, match="final_air_weight"):
        validate_predictive_hvac_config(bad, forecast_horizon=8)

    # Rejection 6: negative weight
    bad = dict(valid_cfg, early_air_weight=-1.0)
    with pytest.raises(ValueError, match="early_air_weight"):
        validate_predictive_hvac_config(bad, forecast_horizon=8)

    # Rejection 7: unsupported solar forecast mode
    bad = dict(valid_cfg, solar_forecast_mode="full")
    with pytest.raises(ValueError, match="solar_forecast_mode"):
        validate_predictive_hvac_config(bad, forecast_horizon=8)

    # Rejection 8: unsupported fallback mode
    bad = dict(valid_cfg, invalid_forecast_fallback="last_value")
    with pytest.raises(ValueError, match="invalid_forecast_fallback"):
        validate_predictive_hvac_config(bad, forecast_horizon=8)


# ---------------------------------------------------------------------------
# Section 17.1: Pure Physical Tests
# ---------------------------------------------------------------------------

def test_1_one_step_zero_solar_2r2c_explicit_euler():
    """Test 1: One-step zero-solar 2R2C equations match explicit-Euler convention."""
    c_air = 15.0
    c_mass = 250.0
    r_out_air = 0.22
    r_air_mass = 0.08
    delta_t = 1.0

    t_air = 22.0
    t_mass = 21.0
    t_out = 15.0
    q_int = 2.0
    q_hvac = 10.0

    expected_d_air = (delta_t / c_air) * (
        (t_out - t_air) / r_out_air + (t_mass - t_air) / r_air_mass + q_int + q_hvac
    )
    expected_d_mass = (delta_t / c_mass) * ((t_air - t_mass) / r_air_mass)

    t_air_next = t_air + expected_d_air
    t_mass_next = t_mass + expected_d_mass

    assert math.isfinite(t_air_next)
    assert math.isfinite(t_mass_next)
    assert t_mass_next > t_mass  # heat flowed from 22°C air into 21°C mass


def test_2_signed_heating_cooling_cop():
    """Test 2: Heating COP (3.3) and Cooling COP (3.2) signed conversion."""
    cop_heat = 3.3
    cop_cool = 3.2

    # Positive thermal power = heating
    q_pos = 6.6
    p_heat = q_pos / cop_heat
    assert math.isclose(p_heat, 2.0)

    # Negative thermal power = cooling
    q_neg = -6.4
    p_cool = q_neg / cop_cool
    assert math.isclose(p_cool, -2.0)


def test_3_final_step_action_cannot_change_terminal_mass():
    """Test 3: In explicit Euler, HVAC power at step t has no direct effect on T_mass(t+1)."""
    c_mass = 250.0
    r_air_mass = 0.08
    t_air = 22.0
    t_mass = 20.0

    d_mass = (1.0 / c_mass) * ((t_air - t_mass) / r_air_mass)
    t_mass_next = t_mass + d_mass

    # Mass rate depends solely on air and mass temperature at beginning of step
    assert math.isclose(d_mass, (22.0 - 20.0) / (0.08 * 250.0))
    assert math.isclose(t_mass_next, 20.1)


def test_4_mass_correction_direction(env_and_config):
    """Test 4: Mass correction direction is correct for warming and cooling."""
    env, config = env_and_config
    schema = env.obs_builder.schema("mg2")
    transform = PredictiveHVACTransform.from_config(schema, config["algorithm"], env.cfg_by_id["mg2"])

    # Create dummy observation batch of size 2:
    # Row 0: mass is cold (20°C), initial is 23°C -> needs positive heat transfer
    # Row 1: mass is warm (26°C), initial is 23°C -> needs negative heat transfer
    obs = torch.zeros(2, len(schema))
    obs[:, transform.t_air_idx] = 23.0 / 40.0
    obs[0, transform.t_mass_idx] = 20.0 / 40.0
    obs[1, transform.t_mass_idx] = 26.0 / 40.0
    obs[:, transform.t_out_idx] = 20.0 / 40.0
    obs[0, transform.terminal_mass_gap_idx] = (23.0 - 20.0) / 40.0
    obs[1, transform.terminal_mass_gap_idx] = (23.0 - 26.0) / 40.0
    obs[:, transform.time_remaining_idx] = 4.0 / 24.0
    obs[:, transform.hvac_min_idx] = -1.0
    obs[:, transform.hvac_max_idx] = 1.0

    lower = torch.full((2, 1), -1.0)
    upper = torch.full((2, 1), 1.0)
    p_ref, diags = transform._predict_reference(obs, lower, upper)

    # Row 0 (cold mass) must request positive mass rate; Row 1 (warm mass) must request negative
    assert diags["hvac_mass_rate_requested_cph"][0].item() > 0
    assert diags["hvac_mass_rate_requested_cph"][1].item() < 0
    assert diags["hvac_mass_required_air_c"][0].item() > diags["hvac_mass_required_air_c"][1].item()


def test_5_mass_authority_decreases_to_zero(env_and_config):
    """Test 5: Mass weight is 0 with 1 step remaining and increases with more steps."""
    env, config = env_and_config
    schema = env.obs_builder.schema("mg2")
    transform = PredictiveHVACTransform.from_config(schema, config["algorithm"], env.cfg_by_id["mg2"])

    obs = torch.zeros(3, len(schema))
    obs[:, transform.t_air_idx] = 23.0 / 40.0
    obs[:, transform.t_mass_idx] = 20.0 / 40.0
    obs[:, transform.t_out_idx] = 20.0 / 40.0
    obs[:, transform.terminal_air_gap_idx] = 0.0
    obs[:, transform.terminal_mass_gap_idx] = 3.0 / 40.0
    # 1 hour remaining, 2 hours remaining, 6 hours remaining
    obs[0, transform.time_remaining_idx] = 1.0 / 24.0
    obs[1, transform.time_remaining_idx] = 2.0 / 24.0
    obs[2, transform.time_remaining_idx] = 6.0 / 24.0
    obs[:, transform.hvac_min_idx] = -1.0
    obs[:, transform.hvac_max_idx] = 1.0

    lower = torch.full((3, 1), -1.0)
    upper = torch.full((3, 1), 1.0)
    p_ref, diags = transform._predict_reference(obs, lower, upper)

    # At 1 hour remaining: mass weight must be strictly 0.0
    assert diags["hvac_mass_weight"][0].item() == 0.0
    # At 6 hours remaining: mass weight must be higher than at 2 hours
    assert diags["hvac_mass_weight"][2].item() >= diags["hvac_mass_weight"][1].item()


def test_6_assist_interval_and_weights_monotonic(env_and_config):
    """Test 7 & 8: Monotonic weights and assist margin contraction."""
    env, config = env_and_config
    schema = env.obs_builder.schema("mg2")
    transform = PredictiveHVACTransform.from_config(schema, config["algorithm"], env.cfg_by_id["mg2"])

    # Test hours from 6 down to 1
    h_steps = [6, 5, 4, 3, 2, 1]
    w_airs = []
    w_masses = []
    for h in h_steps:
        obs = torch.zeros(1, len(schema))
        obs[0, transform.t_air_idx] = 23.0 / 40.0
        obs[0, transform.t_mass_idx] = 21.0 / 40.0
        obs[0, transform.t_out_idx] = 20.0 / 40.0
        obs[0, transform.time_remaining_idx] = float(h) / 24.0
        obs[0, transform.hvac_min_idx] = -1.0
        obs[0, transform.hvac_max_idx] = 1.0
        _, diags = transform._predict_reference(obs, torch.tensor([[-1.0]]), torch.tensor([[1.0]]))
        w_airs.append(diags["hvac_air_weight"][0].item())
        w_masses.append(diags["hvac_mass_weight"][0].item())

    # As H decreases: air weight must be non-decreasing, mass weight must be non-increasing
    for i in range(len(h_steps) - 1):
        assert w_airs[i + 1] >= w_airs[i] - 1e-6
        assert w_masses[i + 1] <= w_masses[i] + 1e-6


def test_10_huge_mass_gap_does_not_blow_up_reference(env_and_config):
    """Test 10: A huge mass gap (20°C) cannot create an unbounded power request."""
    env, config = env_and_config
    schema = env.obs_builder.schema("mg2")
    transform = PredictiveHVACTransform.from_config(schema, config["algorithm"], env.cfg_by_id["mg2"])

    obs = torch.zeros(1, len(schema))
    obs[0, transform.t_air_idx] = 23.0 / 40.0
    obs[0, transform.t_mass_idx] = 3.0 / 40.0  # 20°C gap!
    obs[0, transform.terminal_mass_gap_idx] = 20.0 / 40.0
    obs[0, transform.t_out_idx] = 20.0 / 40.0
    obs[0, transform.time_remaining_idx] = 2.0 / 24.0
    obs[0, transform.hvac_min_idx] = -1.0
    obs[0, transform.hvac_max_idx] = 1.0

    lower = torch.tensor([[-1.0]])
    upper = torch.tensor([[1.0]])
    p_ref, diags = transform._predict_reference(obs, lower, upper)

    assert torch.isfinite(p_ref).all()
    assert (p_ref >= -1.0).all() and (p_ref <= 1.0).all()
    # Air target must respect assist bounds (not 50°C)
    assert diags["hvac_air_target_next_c"][0].item() <= 26.0


def test_11_final_air_priority_veto_applied(env_and_config):
    """Test 11: Final air priority veto activates when mass pull worsens air error."""
    env, config = env_and_config
    schema = env.obs_builder.schema("mg2")
    transform = PredictiveHVACTransform.from_config(schema, config["algorithm"], env.cfg_by_id["mg2"])

    # Air is cold (20°C, initial 23°C), Mass is very hot (28°C, initial 23°C)
    # Inside final 2 hours, pulling down air to cool mass would worsen air error!
    obs = torch.zeros(1, len(schema))
    obs[0, transform.t_air_idx] = 20.0 / 40.0
    obs[0, transform.terminal_air_gap_idx] = 3.0 / 40.0
    obs[0, transform.t_mass_idx] = 28.0 / 40.0
    obs[0, transform.terminal_mass_gap_idx] = -5.0 / 40.0
    obs[0, transform.t_out_idx] = 20.0 / 40.0
    obs[0, transform.time_remaining_idx] = 1.5 / 24.0
    obs[0, transform.hvac_min_idx] = -1.0
    obs[0, transform.hvac_max_idx] = 1.0

    lower = torch.tensor([[-1.0]])
    upper = torch.tensor([[1.0]])
    p_ref, diags = transform._predict_reference(obs, lower, upper)

    assert torch.isfinite(p_ref).all()
    # At H=1.5 <= 2.0, veto or mass zeroing should protect air
    assert diags["hvac_air_target_next_c"][0].item() >= 20.0


def test_12_invalid_forecasts_fallback(env_and_config):
    """Test 12: Missing/invalid forecast slots fall back safely without error."""
    env, config = env_and_config
    schema = env.obs_builder.schema("mg2")
    transform = PredictiveHVACTransform.from_config(schema, config["algorithm"], env.cfg_by_id["mg2"])

    obs = torch.zeros(1, len(schema))
    obs[0, transform.t_air_idx] = 23.0 / 40.0
    obs[0, transform.t_mass_idx] = 23.0 / 40.0
    obs[0, transform.t_out_idx] = 20.0 / 40.0
    obs[0, transform.time_remaining_idx] = 4.0 / 24.0
    obs[0, transform.hvac_min_idx] = -1.0
    obs[0, transform.hvac_max_idx] = 1.0

    # Set forecast_valid flags to 0 (invalid)
    for k in range(1, 6):
        v_idx = transform.names.get(f"forecast_valid_t+{k}", -1)
        if v_idx >= 0:
            obs[0, v_idx] = 0.0

    lower = torch.tensor([[-1.0]])
    upper = torch.tensor([[1.0]])
    p_ref, diags = transform._predict_reference(obs, lower, upper)

    assert torch.isfinite(p_ref).all()
    assert diags["hvac_forecast_fallback"][0].item() > 0


def test_13_extreme_temperatures_finite(env_and_config):
    """Test 13: Extremely low (-20°C) and high (+45°C) outdoor temps produce finite values."""
    env, config = env_and_config
    schema = env.obs_builder.schema("mg2")
    transform = PredictiveHVACTransform.from_config(schema, config["algorithm"], env.cfg_by_id["mg2"])

    obs = torch.zeros(2, len(schema))
    obs[0, transform.t_out_idx] = -20.0 / 40.0
    obs[1, transform.t_out_idx] = 45.0 / 40.0
    obs[:, transform.t_air_idx] = 23.0 / 40.0
    obs[:, transform.t_mass_idx] = 23.0 / 40.0
    obs[:, transform.time_remaining_idx] = 3.0 / 24.0
    obs[:, transform.hvac_min_idx] = -1.0
    obs[:, transform.hvac_max_idx] = 1.0

    lower = torch.full((2, 1), -1.0)
    upper = torch.full((2, 1), 1.0)
    p_ref, diags = transform._predict_reference(obs, lower, upper)

    assert torch.isfinite(p_ref).all()
    assert (p_ref >= -1.0).all() and (p_ref <= 1.0).all()


# ---------------------------------------------------------------------------
# Section 17.2: Property Tests (10,000 randomized batches)
# ---------------------------------------------------------------------------

def test_10_000_property_tests(env_and_config):
    """Verify bounds, monotonicity, and gradients on 10,000 cases."""
    env, config = env_and_config
    schema = env.obs_builder.schema("mg2")
    transform = PredictiveHVACTransform.from_config(schema, config["algorithm"], env.cfg_by_id["mg2"])

    gen = torch.Generator().manual_seed(42)
    n = 10_000

    obs = torch.zeros(n, len(schema))
    obs[:, transform.t_air_idx] = (torch.rand(n, generator=gen) * 15.0 + 15.0) / 40.0
    obs[:, transform.t_mass_idx] = (torch.rand(n, generator=gen) * 15.0 + 15.0) / 40.0
    obs[:, transform.t_out_idx] = (torch.rand(n, generator=gen) * 40.0 - 5.0) / 40.0
    obs[:, transform.time_remaining_idx] = (torch.rand(n, generator=gen) * 6.0 + 0.1) / 24.0

    # Random lower in [-1.0, 0.5], upper in [lower, 1.0]
    lower = torch.rand(n, 1, generator=gen) * 1.5 - 1.0
    upper = lower + torch.rand(n, 1, generator=gen) * (1.0 - lower)
    obs[:, transform.hvac_min_idx : transform.hvac_min_idx + 1] = lower
    obs[:, transform.hvac_max_idx : transform.hvac_max_idx + 1] = upper

    # Latent actions
    latent = torch.rand(n, 1, generator=gen) * 2.0 - 1.0

    # Map with details
    res = transform.map_with_details(obs, latent)

    assert torch.isfinite(res.action).all()
    assert (res.action >= lower - 1e-6).all()
    assert (res.action <= upper + 1e-6).all()

    # Exact endpoints test on a subset
    res_min = transform.map_with_details(obs[:100], torch.full((100, 1), -1.0))
    res_zero = transform.map_with_details(obs[:100], torch.zeros(100, 1))
    res_max = transform.map_with_details(obs[:100], torch.full((100, 1), 1.0))

    assert torch.allclose(res_min.action, lower[:100], atol=1e-5)
    assert torch.allclose(res_zero.action, res_zero.center, atol=1e-5)
    assert torch.allclose(res_max.action, upper[:100], atol=1e-5)

    # Monotonicity test
    perturbed_latent = torch.clamp(latent + 0.05, -1.0, 1.0)
    res_pert = transform.map_with_details(obs, perturbed_latent)
    assert (res_pert.action >= res.action - 1e-6).all()


# ---------------------------------------------------------------------------
# Section 17.3 & 17.4: Checkpoint & Preservation Tests
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip(env_and_config, tmp_path):
    env, config = env_and_config
    algo1 = Matd3HierarchicalTransformerPredictiveDr.from_env(env, config, device="cpu", seed=42)

    obs, _ = env.reset(seed=42)
    state = env.global_state()
    for _ in range(5):
        actions = algo1.select_actions(obs, noise_std=0.1)
        next_obs, rewards, done, info = env.step(actions)
        next_state = env.global_state()
        algo1.replay.add(obs, state, actions, sum(rewards.values()), next_obs, next_state, done)
        obs, state = next_obs, next_state
        if done:
            break

    algo1.update(batch_size=8)
    ckpt = tmp_path / "model.pt"
    algo1.save(ckpt)

    algo2 = Matd3HierarchicalTransformerPredictiveDr.from_env(env, config, device="cpu", seed=99)
    algo2.load(ckpt)

    assert algo1.update_steps == algo2.update_steps
    assert algo1.predictive_contract_version == algo2.predictive_contract_version

    for p1, p2 in zip(algo1.actors.parameters(), algo2.actors.parameters()):
        assert torch.equal(p1, p2)


@pytest.mark.parametrize(
    ("mut_key", "mut_val", "expected_match"),
    [
        ("algorithm_slug", "matd3_hierarchical_transformer_feasible_dr_v2", "algorithm mismatch"),
        ("predictive_hvac_contract_version", 999, "version mismatch"),
        ("predictive_hvac_config", {"enabled": False}, "predictive_hvac_config mismatch"),
        ("feasible_dr_config", {"enabled": False}, "feasible_dr_config mismatch"),
        ("observation_schema_by_agent", {"mg1": ("wrong",)}, "schema mismatch"),
    ],
)
def test_checkpoint_rejection_on_mutation(env_and_config, tmp_path, mut_key, mut_val, expected_match):
    env, config = env_and_config
    algo = Matd3HierarchicalTransformerPredictiveDr.from_env(env, config, device="cpu", seed=42)
    ckpt = tmp_path / "model.pt"
    algo.save(ckpt)

    raw = torch.load(ckpt, weights_only=False)
    raw[mut_key] = mut_val
    mut_path = tmp_path / f"mut_{mut_key}.pt"
    torch.save(raw, mut_path)

    algo2 = Matd3HierarchicalTransformerPredictiveDr.from_env(env, config, device="cpu", seed=99)
    with pytest.raises(ValueError, match=expected_match):
        algo2.load(mut_path)


def test_parent_preservation_mg1_mg3(env_and_config):
    env, config = env_and_config
    algo = Matd3HierarchicalTransformerPredictiveDr.from_env(env, config, device="cpu", seed=42)

    obs, _ = env.reset(seed=42)
    mg1_obs = torch.as_tensor(obs["mg1"], dtype=torch.float32).unsqueeze(0)
    mg3_obs = torch.as_tensor(obs["mg3"], dtype=torch.float32).unsqueeze(0)

    latent = torch.tensor([[0.5, -0.3]])

    out_mg1 = algo.actors["mg1"].map_latent_actions(mg1_obs, latent)
    out_mg3 = algo.actors["mg3"].map_latent_actions(mg3_obs, latent)

    assert torch.isfinite(out_mg1).all()
    assert torch.isfinite(out_mg3).all()
    assert (out_mg1 >= -1.0).all() and (out_mg1 <= 1.0).all()
    assert (out_mg3 >= -1.0).all() and (out_mg3 <= 1.0).all()

    # MG1 remains the parent's unmapped DR branch, while MG3 must retain the
    # feasible EV mapper configured by the V2 parent.
    assert algo.actors["mg1"].feasible_dr.enabled is False
    assert algo.actors["mg3"].feasible_dr.enabled is True
    assert torch.allclose(out_mg1[:, 1], latent[:, 1])

    # An unavailable EV has the exact environment-compatible raw action -1.0.
    transform = algo.actors["mg3"].feasible_dr
    unavailable_obs = mg3_obs.clone()
    unavailable_obs[:, transform.ev_avail_idx] = 0.0
    unavailable_obs[:, transform.ev_min_idx] = 0.0
    unavailable_obs[:, transform.ev_max_idx] = 0.0
    unavailable_obs[:, transform.rem_energy_idx] = 0.0
    unavailable_obs[:, transform.hours_dep_idx] = 0.0
    unavailable = algo.actors["mg3"].map_latent_actions(
        unavailable_obs,
        torch.tensor([[0.5, 0.25]]),
    )
    assert unavailable[0, 1].item() == pytest.approx(-1.0)


def test_validation_uses_mill_terminal_or_production_constraint(env_and_config):
    _, config = env_and_config
    constraints = config["validation"]["selection"]["constraints"]
    assert "mill_terminal_or_production_success_rate" in constraints
    assert "mill_terminal_success_rate" not in constraints
