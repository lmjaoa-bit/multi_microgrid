from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from core.data_manager import DataManager
from core.profiles import HeterogeneousProfileModel
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"


def test_shifted_weather_series_is_cached_and_matches_deterministic_forecast() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["_project_root"] = str(ROOT)
    config["microgrids"][1]["renewables"]["weather_time_shift_hours"] = 24.0

    train = DataManager(config, split="train", seed=42)
    profiles = HeterogeneousProfileModel(config, train.df)
    unshifted_config = deepcopy(config)
    unshifted_config["microgrids"][1]["renewables"].pop("weather_time_shift_hours")
    unshifted_profiles = HeterogeneousProfileModel(unshifted_config, train.df)
    shifted = profiles.generation_series("mg2", train.df)
    unshifted = unshifted_profiles.generation_series("mg2", train.df)
    assert np.allclose(shifted["pv_kw"], np.roll(unshifted["pv_kw"], 24))
    assert np.allclose(shifted["wind_kw"], np.roll(unshifted["wind_kw"], 24))

    env = MultiMicrogridEnv(config, split="test")
    env.reset(seed=13, deterministic=True)
    assert env._episode_exogenous is not None
    forecast = env.forecasts.combined("mg2", 0, 1)
    assert np.isclose(forecast["pv_kw"], env._episode_exogenous["mg2"]["pv_kw"][1])
    assert np.isclose(forecast["wind_kw"], env._episode_exogenous["mg2"]["wind_kw"][1])


def test_global_config_uses_local_weather_generation_tariffs_and_dr_state() -> None:
    env = MultiMicrogridEnv(CONFIG, split="test")
    day = str(env.data.valid_days[0].date())
    env.reset(day=day, seed=42, deterministic=True)
    assert env._episode_exogenous is not None
    assert np.allclose(env._episode_exogenous["mg1"]["wind_kw"], 0.0)
    assert np.allclose(env._episode_exogenous["mg2"]["pv_kw"], 0.0)
    assert np.allclose(env._episode_exogenous["mg3"]["wind_kw"], 0.0)

    # Nearby Yancheng-area NASA POWER points are allowed to resolve to the same
    # source grid cell. Distinct coordinates must never be tested by demanding
    # numerically different weather series.
    for mg in env.agent_ids:
        renewable = env.cfg_by_id[mg]["renewables"]
        assert renewable["ghi_column"] in env.episode_context_df
        assert renewable["wind_speed_column"] in env.episode_context_df

    hour_16 = env.episode_context_df[env.episode_context_df["timestamp"].dt.hour == 16].iloc[0]
    prices = [env.data.prices_at(hour_16, mg) for mg in env.agent_ids]
    assert all(np.allclose(prices[0], item) for item in prices[1:])
    assert np.isclose(prices[0][1], 0.50 * prices[0][0])
    assert env.data.df["buy_price_per_kwh"].nunique() > 10

    forecast = env.forecasts.combined("mg3", 0, 1)
    expected_buy, expected_sell = env.data.prices_at(env.episode_context_df.iloc[1], "mg3")
    assert np.isclose(forecast["grid_price_per_kwh"], expected_buy)
    assert np.isclose(forecast["grid_sell_price_per_kwh"], expected_sell)

    schema = env.global_state_builder.schema()
    assert env.global_state_dim == len(schema)
    for mg in env.agent_ids:
        assert f"{mg}_feasible_charge_max" in schema
    assert "mg1_dr_min" in schema and "mg1_dr_max" in schema
    assert "mg2_dr_min" in schema and "mg2_dr_max" in schema
    assert "mg3_ev_feasible_min" in schema and "mg3_ev_feasible_max" in schema
    assert "mg1_silo_ratio" in schema
    assert "mg2_t_air" in schema
    assert "mg2_t_mass" in schema
    assert "mg3_ev_G1_remaining_energy" in schema


def test_global_config_is_selected_yancheng_dr_setting() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["action_design"] == "single_dispatch_dr"
    assert config["p2p"]["loss_rate"] == 0.0
    assert config["p2p"]["pair_capacity_kw"] is None
    assert config["p2p"]["allow_grid_resale"] is False
    assert [item["type"] for item in config["microgrids"]] == [
        "industrial_pv_raw_mill",
        "commercial_wind_hvac",
        "residential_pv_ev",
    ]
    assert [item["dr"]["type"] for item in config["microgrids"]] == [
        "industrial_mill",
        "commercial_hvac",
        "ev_charging",
    ]
    assert [item["battery"]["initial_soc"] for item in config["microgrids"]] == [
        0.10,
        0.10,
        0.10,
    ]
    assert all(item["grid"]["import_limit_kw"] is None for item in config["microgrids"])
    assert all(item["grid"]["export_limit_kw"] == 100.0 for item in config["microgrids"])
