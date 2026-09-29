from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from core.joint_action_projection import JointActionProjector
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    cfg["reward"] = {
        "semantics_version": "generalized_system_cost_v2",
        "calibration_status": "diagnostic",
    }
    cfg["cost"].update(
        curtailment_price_per_kwh=cfg["cost"]["renewable_curtailment_cost_per_kwh"],
        carbon_price_per_kg=0.1,
        peak_price_per_kw=1.0,
        mill_unserved_price_per_t=10.0,
        comfort_price_per_degree_hour=2.0,
        ev_shortfall_price_per_kwh=cfg["microgrids"][2]["dr"]["departure_penalty_per_kwh"],
    )
    for mg in cfg["microgrids"]:
        mg["grid"]["export_limit_kw"] = 0.0
    cfg["microgrids"][1]["dr"]["control_semantics"] = "soft_comfort"
    cfg["microgrids"][2]["dr"]["control_semantics"] = "no_deadline_minimum"
    cfg["observation"]["noisy_forecast"] = False
    cfg["observation"]["evaluation_forecast_noise"] = False
    return cfg


def _env(seed=42):
    env = MultiMicrogridEnv(_cfg(), split="test")
    env.reset(seed=seed, deterministic=True, forecast_noise=False)
    return env


@pytest.mark.parametrize(
    "actions",
    [
        {"mg1": [0.0, 0.0], "mg2": [0.0, -0.6], "mg3": [0.0, -1.0]},
        {"mg1": [0.4, 0.2], "mg2": [-0.4, 0.0], "mg3": [0.2, 1.0]},
        {"mg1": [-0.5, -0.4], "mg2": [0.5, 0.7], "mg3": [-0.2, 0.3]},
    ],
)
def test_v2_projector_matches_device_execution(actions):
    env = _env(seed=17)
    state = torch.as_tensor(env.global_state(), dtype=torch.float32).unsqueeze(0)
    raw = torch.as_tensor(
        [[actions[mg] for mg in env.agent_ids]], dtype=torch.float32
    )
    effective, projected_plans = env.action_projector(state, raw)

    np_actions = {
        mg: np.asarray(actions[mg], dtype=np.float32) for mg in env.agent_ids
    }
    _, _, _, info = env.step(np_actions)

    for i, mg in enumerate(env.agent_ids):
        env_plan = info["plans"][mg]
        battery = env.cfg_by_id[mg]["battery"]
        scale = max(
            float(battery["max_charge_kw"]),
            float(battery["max_discharge_kw"]),
        )
        expected_local = (
            float(env_plan.get("ess_to_local_kw", 0.0))
            - float(info["devices"][mg].get("charge_kw", 0.0))
        ) / scale
        expected_trade = float(
            info["p2p"]["by_microgrid"][mg].get("ess_p2p_sell_kw", 0.0)
        ) / scale
        assert float(effective[0, i, 0]) == pytest.approx(expected_local, abs=1e-6)
        assert float(effective[0, i, 1]) == pytest.approx(expected_trade, abs=1e-6)
        assert float(effective[0, i, 2]) == pytest.approx(
            float(env_plan["effective_dr_action"]), abs=1e-6
        )
        assert float(projected_plans[i]["flexible_load_kw"]) == pytest.approx(
            float(env_plan["flexible_load_kw"]), abs=2e-6
        )


def test_v2_unsaturated_hvac_action_has_finite_nonzero_gradient():
    env = _env(seed=23)
    state = torch.as_tensor(env.global_state(), dtype=torch.float32).unsqueeze(0)
    raw = torch.zeros((1, len(env.agent_ids), 2), dtype=torch.float32, requires_grad=True)
    raw.data[0, 1, 1] = 0.2
    effective, _ = env.action_projector(state, raw)
    target = effective[0, 1, 2]
    target.backward()
    grad = float(raw.grad[0, 1, 1])
    assert np.isfinite(grad)
    assert abs(grad) > 1e-8


def test_v2_random_legal_actions_remain_finite_and_bounded():
    rng = np.random.default_rng(20260929)
    for episode_seed in range(3):
        env = _env(seed=100 + episode_seed)
        for _ in range(6):
            actions = {
                mg: rng.uniform(-0.8, 0.8, size=2).astype(np.float32)
                for mg in env.agent_ids
            }
            state = torch.as_tensor(env.global_state(), dtype=torch.float32).unsqueeze(0)
            raw = torch.as_tensor(
                [[actions[mg] for mg in env.agent_ids]], dtype=torch.float32
            )
            effective, _ = env.action_projector(state, raw)
            assert torch.isfinite(effective).all()
            _, _, done, info = env.step(actions)
            assert max(
                abs(float(s["power_balance_residual_kw"]))
                for s in info["settlements"].values()
            ) <= 1e-6
            assert max(
                float(s["grid_export_kw"]) for s in info["settlements"].values()
            ) <= 1e-7
            if done:
                break
