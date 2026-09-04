from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from core.data_manager import DataManager
from core.profiles import HeterogeneousProfileModel
from envs.multi_microgrid_env import MultiMicrogridEnv
from scripts.run_ess_scenario_study import resolve_scenario


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"


def test_scenario_resolution_preserves_base_and_applies_global_overrides() -> None:
    base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    original = deepcopy(base)
    scenario = {
        "id": "test",
        "global": {
            "grid_import_limit_kw": 200.0,
            "grid_export_limit_kw": 200.0,
            "renewable_capacity_scale": 1.5,
            "battery_capacity_scale": 0.5,
        },
    }
    resolved = resolve_scenario(base, scenario, ROOT)
    assert base == original
    for before, after in zip(base["microgrids"], resolved["microgrids"]):
        assert after["grid"]["import_limit_kw"] == 200.0
        assert after["grid"]["export_limit_kw"] == 200.0
        assert np.isclose(
            after["renewables"]["pv_capacity_kw"],
            before["renewables"]["pv_capacity_kw"] * 1.5,
        )
        assert np.isclose(
            after["renewables"]["wind_capacity_kw"],
            before["renewables"]["wind_capacity_kw"] * 1.5,
        )
        assert np.isclose(
            after["battery"]["capacity_kwh"],
            before["battery"]["capacity_kwh"] * 0.5,
        )


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


def test_global_config_uses_local_weather_generation_and_tariffs() -> None:
    env = MultiMicrogridEnv(CONFIG, split="test")
    day = str(env.data.valid_days[0].date())
    env.reset(day=day, seed=42, deterministic=True)
    assert env._episode_exogenous is not None
    assert np.allclose(env._episode_exogenous["mg1"]["wind_kw"], 0.0)
    assert np.allclose(env._episode_exogenous["mg2"]["pv_kw"], 0.0)
    assert np.allclose(env._episode_exogenous["mg3"]["wind_kw"], 0.0)
    assert not np.allclose(
        env.episode_context_df["mg1_ghi_w_per_m2"],
        env.episode_context_df["mg3_ghi_w_per_m2"],
    )

    hour_16 = env.episode_context_df[env.episode_context_df["timestamp"].dt.hour == 16].iloc[0]
    assert env.data.prices_at(hour_16, "mg1") == (0.60, 0.18)
    assert np.allclose(env.data.prices_at(hour_16, "mg3"), (1.35, 0.16))
    assert env.data.df["mg2_buy_price_per_kwh"].nunique() > 10

    forecast = env.forecasts.combined("mg3", 0, 1)
    expected_buy, expected_sell = env.data.prices_at(env.episode_context_df.iloc[1], "mg3")
    assert np.isclose(forecast["grid_price_per_kwh"], expected_buy)
    assert np.isclose(forecast["grid_sell_price_per_kwh"], expected_sell)
    assert env.global_state_dim == 24


def test_global_config_is_the_selected_lossless_crossgrid_setting() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["p2p"]["loss_rate"] == 0.0
    assert config["p2p"]["pair_capacity_kw"] == 50.0
    assert config["p2p"]["allow_grid_resale"] is False
    assert [item["type"] for item in config["microgrids"]] == [
        "pv_only_day_industrial",
        "wind_only_continuous_industrial",
        "pv_only_evening_industrial",
    ]
