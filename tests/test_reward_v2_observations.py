from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

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


def test_same_current_load_different_peak_has_different_state():
    env = MultiMicrogridEnv(_cfg(), split="test")
    env.reset(seed=42, deterministic=True, forecast_noise=False)
    current = env._exogenous_at(0)
    dr = env._dr_states(0)

    env.daily_load_peak_kw = 100.0
    obs_100 = env._observations(current, dr)
    state_100 = env._build_global_state(current, dr)

    env.daily_load_peak_kw = 200.0
    obs_200 = env._observations(current, dr)
    state_200 = env._build_global_state(current, dr)

    for mg in env.agent_ids:
        idx = env.obs_builder.schema(mg).index("team_load_peak_so_far")
        assert obs_100[mg][idx] != obs_200[mg][idx]
    sidx = env.global_state_builder.schema().index("team_load_peak_so_far")
    assert state_100[sidx] != state_200[sidx]


def test_normalizers_do_not_depend_on_test_or_episode():
    train = MultiMicrogridEnv(_cfg(), split="train")
    test = MultiMicrogridEnv(_cfg(), split="test")
    assert train.obs_builder.team_power_scale_kw == pytest.approx(
        test.obs_builder.team_power_scale_kw
    )
    assert train.obs_builder.carbon_factor_scale == pytest.approx(
        test.obs_builder.carbon_factor_scale
    )
    assert train.global_state_builder.team_power_scale_kw == pytest.approx(
        test.global_state_builder.team_power_scale_kw
    )
    assert train.global_state_builder.carbon_factor_scale == pytest.approx(
        test.global_state_builder.carbon_factor_scale
    )
    assert train.obs_builder.team_power_scale_kw > 0.0
    assert train.obs_builder.carbon_factor_scale > 0.0


def test_carbon_is_current_only_and_public():
    env = MultiMicrogridEnv(_cfg(), split="test")
    obs, _ = env.reset(seed=9, deterministic=True, forecast_noise=False)
    current_factor = float(env.episode_df.iloc[0]["grid_carbon_factor_kg_per_kwh"])
    for mg in env.agent_ids:
        schema = env.obs_builder.schema(mg)
        idx = schema.index("grid_carbon_factor")
        assert obs[mg][idx] == pytest.approx(
            current_factor / env.obs_builder.carbon_factor_scale
        )
        assert not any(name.startswith("grid_carbon_factor_t+") for name in schema)
    sidx = env.global_state_builder.schema().index("grid_carbon_factor")
    assert env.global_state()[sidx] == pytest.approx(
        current_factor / env.global_state_builder.carbon_factor_scale
    )

    # At the terminal boundary the public carbon feature remains the last
    # in-day current value; there is no hidden next-day carbon observation.
    env.t = env.episode_length
    public = env._public_state(env.t)
    assert public["grid_carbon_factor"] == pytest.approx(
        float(env.episode_df.iloc[-1]["grid_carbon_factor_kg_per_kwh"])
    )
