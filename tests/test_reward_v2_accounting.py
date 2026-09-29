from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


def _v2_config():
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


def _zeroish_actions(env):
    return {
        "mg1": np.asarray([0.0, -1.0], dtype=np.float32),
        "mg2": np.asarray([0.0, 0.0], dtype=np.float32),
        "mg3": np.asarray([0.0, -1.0], dtype=np.float32),
    }


def test_peak_uses_executed_load_and_resets():
    env = MultiMicrogridEnv(_v2_config(), split="test")
    env.reset(seed=7, deterministic=True, forecast_noise=False)
    _, _, _, info1 = env.step(_zeroish_actions(env))
    assert info1["peak_before_kw"] == 0.0
    assert info1["delta_peak_kw"] == pytest.approx(info1["team_actual_load_kw"])
    assert info1["peak_after_kw"] == pytest.approx(info1["team_actual_load_kw"])

    _, _, _, info2 = env.step(_zeroish_actions(env))
    assert info2["peak_before_kw"] == pytest.approx(info1["peak_after_kw"])
    assert info2["delta_peak_kw"] == pytest.approx(
        max(0.0, info2["team_actual_load_kw"] - info2["peak_before_kw"])
    )
    env.reset(seed=7, deterministic=True, forecast_noise=False)
    assert env.daily_load_peak_kw == 0.0


def test_v2_surplus_is_curtailed_not_exported():
    env = MultiMicrogridEnv(_v2_config(), split="test")
    env.reset(seed=11, deterministic=True, forecast_noise=False)
    for core in env.cores.values():
        core.battery.soc = core.battery.soc_max

    fixed = {
        mg: {
            "load_kw": 1.0,
            "pv_kw": 1000.0,
            "wind_kw": 0.0,
            "renewable_kw": 1000.0,
        }
        for mg in env.agent_ids
    }
    env._exogenous_at = lambda index: {mg: dict(v) for mg, v in fixed.items()}
    _, _, _, info = env.step(_zeroish_actions(env))

    assert all(x["grid_export_kw"] == 0.0 for x in info["settlements"].values())
    total_curtail = sum(
        x["renewable_curtailment_kw"] for x in info["settlements"].values()
    )
    assert total_curtail > 0.0
    assert info["reward_v2_raw"]["curtailment_kwh"] == pytest.approx(total_curtail)


def test_daily_cost_and_shared_reward_conservation():
    env = MultiMicrogridEnv(_v2_config(), split="test")
    env.reset(seed=17, deterministic=True, forecast_noise=False)
    done = False
    reward_sum = 0.0
    generalized_sum = 0.0
    scale = float(env.reward_v2["reward_scale"])
    while not done:
        _, rewards, done, info = env.step(_zeroish_actions(env))
        assert len(set(round(v, 12) for v in rewards.values())) == 1
        reward_sum += next(iter(rewards.values()))
        generalized_sum += float(info["generalized_system_cost"])
        breakdown = info["reward_v2_breakdown"]
        components = [
            breakdown["cost_grid"],
            breakdown["cost_battery"],
            breakdown["cost_curtail"],
            breakdown["cost_carbon"],
            breakdown["cost_peak"],
            breakdown["cost_mill"],
            breakdown["cost_comfort"],
            breakdown["cost_ev"],
            breakdown["cost_unmet"],
        ]
        assert sum(components) == pytest.approx(info["generalized_system_cost"])
    assert reward_sum == pytest.approx(-generalized_sum / scale)


def test_ev_and_curtailment_not_built_from_accounting_profit():
    env = MultiMicrogridEnv(_v2_config(), split="test")
    env.reset(seed=23, deterministic=True, forecast_noise=False)
    _, _, _, info = env.step(_zeroish_actions(env))
    raw = info["reward_v2_raw"]
    breakdown = info["reward_v2_breakdown"]
    assert breakdown["cost_curtail"] == pytest.approx(
        raw["curtailment_kwh"] * env.reward_v2["curtailment_price_per_kwh"]
    )
    assert breakdown["cost_ev"] == pytest.approx(
        raw["ev_departure_shortfall_kwh"]
        * env.reward_v2["ev_shortfall_price_per_kwh"]
    )
    assert info["generalized_system_cost"] >= 0.0
    assert info["team_reward"] <= 0.0
