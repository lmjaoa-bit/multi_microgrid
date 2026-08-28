from pathlib import Path

import numpy as np
import pytest

from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def test_three_microgrids_have_distinct_load_shapes_and_target_means():
    env = MultiMicrogridEnv(CONFIG, split="train")
    frame = env.normalization_data.df
    expected = {"mg1": 60.0, "mg2": 78.0, "mg3": 67.0}
    hourly = {}
    for mg in env.agent_ids:
        load = env.profiles.load_series(mg, frame)
        assert float(np.mean(load)) == pytest.approx(expected[mg], abs=1e-8)
        hours = frame["timestamp"].dt.hour.to_numpy()
        hourly[mg] = np.asarray([load[hours == hour].mean() for hour in range(24)])
    assert not np.allclose(hourly["mg1"], hourly["mg2"])
    assert not np.allclose(hourly["mg1"], hourly["mg3"])
    assert not np.allclose(hourly["mg2"], hourly["mg3"])
    assert int(np.argmax(hourly["mg1"])) in range(9, 17)
    assert int(np.argmax(hourly["mg3"])) in range(17, 21)


def test_installed_renewable_capacities_are_structurally_different():
    env = MultiMicrogridEnv(CONFIG)
    cfg = {mg["id"]: mg for mg in env.config["microgrids"]}
    assert cfg["mg1"]["renewables"]["pv_capacity_kw"] > cfg["mg2"]["renewables"]["pv_capacity_kw"]
    assert cfg["mg2"]["renewables"]["wind_capacity_kw"] > cfg["mg1"]["renewables"]["wind_capacity_kw"]
    assert cfg["mg3"]["renewables"]["pv_capacity_kw"] < cfg["mg1"]["renewables"]["pv_capacity_kw"]


def test_public_weather_and_price_forecasts_are_shared_but_power_forecasts_are_local():
    env = MultiMicrogridEnv(CONFIG)
    day = str(env.data.valid_days[0].date())
    obs, _ = env.reset(day=day, seed=123)
    schemas = {mg: env.obs_builder.schema(mg) for mg in env.agent_ids}
    for field in ("grid_price_t+1", "ghi_t+1", "wind_speed_t+1"):
        values = [float(obs[mg][schemas[mg].index(field)]) for mg in env.agent_ids]
        assert values[0] == pytest.approx(values[1])
        assert values[0] == pytest.approx(values[2])
    local_power = [env.forecasts.local(mg, 0, 1) for mg in env.agent_ids]
    # Installed capacities differ, so the same public weather maps to different
    # physical renewable forecasts (wind is nonzero on the baseline midnight).
    wind_values = [float(item["wind_kw"]) for item in local_power]
    assert len(set(round(value, 6) for value in wind_values)) > 1


def test_reward_is_linear_scaled_cost_without_clipping():
    env = MultiMicrogridEnv(CONFIG)
    day = str(env.data.valid_days[0].date())
    env.reset(day=day, deterministic=True)
    actions = {mg: np.asarray([0.0, -1.0, -1.0, 0.0], dtype=np.float32) for mg in env.agent_ids}
    _, rewards, _, info = env.step(actions)
    for mg in env.agent_ids:
        assert rewards[mg] == pytest.approx(-info["costs"][mg]["total_cost"] / 100.0)
        assert info["costs"][mg]["reward_scale"] == 100.0
