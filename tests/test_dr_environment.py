from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"


def test_dr_environment_exposes_two_raw_actions_and_private_device_state():
    env = MultiMicrogridEnv(CONFIG, split="test")
    observations, info = env.reset(seed=42, deterministic=True)

    assert env.action_dim == 2
    assert "data_quality_flags" in info
    assert "silo_ratio" in env.obs_builder.schema("mg1")
    assert "t_air" in env.obs_builder.schema("mg2")
    assert "ev_soc" in env.obs_builder.schema("mg3")
    state_schema = env.global_state_builder.schema()
    assert "mg1_dr_min" in state_schema and "mg1_dr_max" in state_schema
    assert "mg2_dr_min" in state_schema and "mg2_dr_max" in state_schema
    assert "mg3_dr_min" in state_schema and "mg3_dr_max" in state_schema
    assert all(observations[mg].shape == (env.observation_dims[mg],) for mg in env.agent_ids)


def test_dr_step_uses_base_plus_flexible_load_before_ess_and_preserves_balance():
    env = MultiMicrogridEnv(CONFIG, split="test")
    _, reset_info = env.reset(seed=43, deterministic=True)
    state_before = torch.tensor(reset_info["global_state"], dtype=torch.float64).unsqueeze(0)
    actions = {
        "mg1": np.asarray([1.0, 1.0], dtype=np.float64),
        "mg2": np.asarray([0.0, 0.0], dtype=np.float64),
        "mg3": np.asarray([0.0, -1.0], dtype=np.float64),
    }
    raw = torch.tensor(
        np.asarray([actions[mg] for mg in env.agent_ids]), dtype=torch.float64
    ).unsqueeze(0)
    with torch.no_grad():
        projected, projected_plans = env.action_projector(state_before, raw)

    _, rewards, _, info = env.step(actions)

    assert len(set(rewards.values())) == 1
    for i, mg in enumerate(env.agent_ids):
        plan = info["plans"][mg]
        assert plan["actual_load_kw"] == pytest.approx(
            plan["base_load_kw"] + plan["flexible_load_kw"]
        )
        assert plan["load_kw"] == pytest.approx(plan["actual_load_kw"])
        assert plan["effective_dr_action"] == pytest.approx(float(projected[0, i, 2]))
        assert plan["flexible_load_kw"] == pytest.approx(
            float(projected_plans[i]["flexible_load_kw"][0])
        )
        assert abs(info["settlements"][mg]["power_balance_residual_kw"]) < 1e-7
        assert "raw_ess_action" in plan and "raw_dr_action" in plan
        assert "ess_projected" in plan and "dr_projected" in plan


def test_feasible_ess_charge_is_not_projection_just_because_p2p_is_masked():
    env = MultiMicrogridEnv(CONFIG, split="test")
    env.reset(seed=45, deterministic=True)
    raw_ess = -0.10
    actions = {
        "mg1": np.asarray([raw_ess, 0.0], dtype=np.float64),
        "mg2": np.asarray([0.0, 0.0], dtype=np.float64),
        "mg3": np.asarray([0.0, 0.0], dtype=np.float64),
    }
    _, _, _, info = env.step(actions)
    plan = info["plans"]["mg1"]
    max_charge_kw = float(env.cfg_by_id["mg1"]["battery"]["max_charge_kw"])

    assert plan["dispatch_requested_power_kw"] == pytest.approx(
        raw_ess * max_charge_kw, abs=1e-7
    )
    assert plan["sharing_action_masked"]
    assert plan["ess_projection_delta"] == pytest.approx(0.0, abs=1e-7)
    assert not plan["ess_projected"]


def test_grid_capacity_reduces_optional_ess_charging_before_actual_load():
    env = MultiMicrogridEnv(CONFIG, split="test")
    env.reset(seed=44, deterministic=True)
    env.grids["mg3"] = type(env.grids["mg3"])(import_limit_kw=5.0, export_limit_kw=200.0)
    actions = {
        "mg1": np.asarray([0.0, 0.0]),
        "mg2": np.asarray([0.0, 0.0]),
        "mg3": np.asarray([-1.0, 1.0]),
    }
    _, _, _, info = env.step(actions)
    plan = info["plans"]["mg3"]
    assert plan["grid_to_load_kw"] <= 5.0 + 1e-9
    assert plan["grid_to_battery_kw"] <= max(0.0, 5.0 - plan["grid_to_load_kw"]) + 1e-9
    assert plan["battery_charge_curtailed_kw"] >= 0.0


def test_matd3_dispatch_dr_replay_and_update_use_projected_three_channel_actions():
    spec = get_algorithm_spec("matd3_dispatch")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(cfg), split="test")
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    settings = dict(settings)
    settings["algorithm"] = dict(settings.get("algorithm", {}))
    settings["algorithm"]["hidden_dims"] = [32, 32]
    settings["algorithm"]["replay_capacity"] = 100
    model = spec.create(env, settings, seed=7)

    observations, reset_info = env.reset(seed=7, deterministic=True)
    state = reset_info["global_state"]
    metrics = {}
    for _ in range(4):
        actions = model.select_actions(observations, noise_std=0.05)
        assert all(value.shape == (2,) for value in actions.values())
        next_obs, rewards, done, info = env.step(actions)
        model.replay.add(
            observations, state, actions, rewards[env.agent_ids[0]],
            next_obs, info["global_state_after"], done,
        )
        metrics = model.update(batch_size=2)
        observations, state = next_obs, info["global_state_after"]

    assert model.raw_action_dim == 2
    assert model.effective_action_dim == 3
    assert np.isfinite(metrics["critic_loss"])
    assert model.update_steps > 0
