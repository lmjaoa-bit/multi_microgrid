from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

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


def test_grid_capacity_reduces_optional_ess_charging_before_actual_load():
    env = MultiMicrogridEnv(CONFIG, split="test")
    env.reset(seed=44, deterministic=True)
    # Force a small grid import cap to exercise load-before-storage settlement.
    env.grids["mg3"] = type(env.grids["mg3"])(import_limit_kw=5.0, export_limit_kw=200.0)
    actions = {
        "mg1": np.asarray([0.0, 0.0]),
        "mg2": np.asarray([0.0, 0.0]),
        "mg3": np.asarray([-1.0, 1.0]),
    }
    _, _, _, info = env.step(actions)
    plan = info["plans"]["mg3"]
    # Any curtailed import-backed storage demand is explicitly accounted for;
    # actual load remains ahead of optional ESS charging in grid allocation.
    assert plan["grid_to_load_kw"] <= 5.0 + 1e-9
    assert plan["grid_to_battery_kw"] <= max(0.0, 5.0 - plan["grid_to_load_kw"]) + 1e-9
    assert plan["battery_charge_curtailed_kw"] >= 0.0
