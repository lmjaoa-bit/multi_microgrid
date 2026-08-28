from pathlib import Path
import copy
import numpy as np
import pytest
import yaml

from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def _off_actions(env):
    return {
        mg: np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        for mg in env.agent_ids
    }


def test_one_day_and_ctde_interfaces():
    env = MultiMicrogridEnv(CONFIG, split="train")
    obs, info = env.reset(day=str(env.data.valid_days[0].date()))
    assert set(obs) == {"mg1", "mg2", "mg3"}
    assert info["global_state"].shape == (env.global_state_dim,)
    actions = {mg: np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32) for mg in env.agent_ids}
    n = 0
    done = False
    while not done:
        obs, rewards, done, info = env.step(actions)
        assert set(rewards) == set(env.agent_ids)
        n += 1
    assert n == 24


def test_bid_mapping_is_always_inside_grid_prices():
    env = MultiMicrogridEnv(CONFIG, split="train")
    env.reset(day=str(env.data.valid_days[0].date()))
    row = env.episode_df.iloc[0]
    lo, hi = float(row["sell_price_per_kwh"]), float(row["buy_price_per_kwh"])
    for raw in [-1, -0.4, 0, 0.7, 1]:
        bid = env.action_codec.decode([1, -1, -1, raw], lo, hi)["p2p_bid_price"]
        assert lo <= bid <= hi


def test_all_splits_use_train_normalization_statistics():
    envs = [MultiMicrogridEnv(CONFIG, split=split) for split in ("train", "val", "test")]
    for mg in envs[0].agent_ids:
        expected = envs[0].obs_builder._norm[mg]
        assert envs[1].obs_builder._norm[mg] == expected
        assert envs[2].obs_builder._norm[mg] == expected


def test_episode_seed_and_deterministic_evaluation_are_reproducible():
    env = MultiMicrogridEnv(CONFIG, split="train")
    day = str(env.data.valid_days[0].date())
    first, _ = env.reset(day=day, seed=123)
    second, _ = env.reset(day=day, seed=123)
    for mg in env.agent_ids:
        assert np.array_equal(first[mg], second[mg])

    sampled_1, sampled_info_1 = env.reset(seed=456)
    sampled_2, sampled_info_2 = env.reset(seed=456)
    assert sampled_info_1["day"] == sampled_info_2["day"]
    for mg in env.agent_ids:
        assert np.array_equal(sampled_1[mg], sampled_2[mg])

    deterministic_1, _ = env.reset(day=day, deterministic=True)
    deterministic_2, _ = env.reset(day=day, deterministic=True)
    for mg in env.agent_ids:
        assert np.array_equal(deterministic_1[mg], deterministic_2[mg])

    advancing_1, _ = env.reset(day=day)
    advancing_2, _ = env.reset(day=day)
    assert any(
        not np.array_equal(advancing_1[mg], advancing_2[mg])
        for mg in env.agent_ids
    )


def test_returned_observation_does_not_alias_internal_state():
    env = MultiMicrogridEnv(CONFIG)
    day = str(env.data.valid_days[0].date())
    obs, _ = env.reset(day=day, seed=1)
    expected = env.global_state()
    obs["mg1"][:] = 999.0
    actions = _off_actions(env)
    _, _, _, info = env.step(actions)
    assert np.array_equal(info["global_state_before"], expected)
    assert not np.any(info["global_state_before"] == 999.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_action_fails_fast(bad):
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()))
    actions = _off_actions(env)
    actions["mg1"][3] = bad
    with pytest.raises(ValueError, match="finite"):
        env.step(actions)


def test_no_trade_price_is_grid_midpoint_and_market_names_are_explicit():
    env = MultiMicrogridEnv(CONFIG)
    env.p2p_enabled = False
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    row = env.episode_df.iloc[0]
    midpoint = 0.5 * (float(row["buy_price_per_kwh"]) + float(row["sell_price_per_kwh"]))
    actions = _off_actions(env)
    obs, _, _, info = env.step(actions)
    assert info["market"]["weighted_average_price"] == pytest.approx(midpoint)
    assert info["market"]["submitted_buy_kwh"] >= 0.0
    assert info["market"]["submitted_sell_kwh"] >= 0.0
    assert info["market"]["cleared_buy_kwh"] == 0.0
    assert info["market"]["cleared_sell_kwh"] == 0.0
    schema = env.obs_builder.schema("mg1")
    assert "market_prev_submitted_buy" in schema
    assert "market_prev_submitted_sell" in schema
    assert "market_prev_total_buy" not in schema
    assert obs["mg1"][schema.index("market_prev_avg_price")] > 0.0


def test_forecast_context_reads_next_day_and_global_state_is_dedicated():
    env = MultiMicrogridEnv(CONFIG)
    day = str(env.data.valid_days[0].date())
    obs, info = env.reset(day=day, deterministic=True)
    assert len(env.episode_df) == 24
    assert len(env.episode_context_df) >= 28
    assert info["global_state"].shape == (84,)
    assert env.global_state_dim == 84
    assert any("t+1" in name for name in env.global_state_schema)
    assert env.global_state_schema.count("grid_buy_price") == 1
    assert env.global_state_schema.count("hour_sin") == 1

    # Public weather/price forecasts are identical across actors in one episode.
    schemas = {mg: env.obs_builder.schema(mg) for mg in env.agent_ids}
    for field in ("grid_price_t+1", "ghi_t+1", "wind_speed_t+1"):
        values = [obs[mg][schemas[mg].index(field)] for mg in env.agent_ids]
        assert values[0] == pytest.approx(values[1]) == pytest.approx(values[2])

    # Forecast noise is part of the critic's legal current information, so a new
    # episode seed changes both actor observations and centralized state.
    noisy_obs_1, noisy_info_1 = env.reset(day=day, seed=1)
    noisy_obs_2, noisy_info_2 = env.reset(day=day, seed=2)
    assert any(not np.array_equal(noisy_obs_1[mg], noisy_obs_2[mg]) for mg in env.agent_ids)
    assert not np.array_equal(noisy_info_1["global_state"], noisy_info_2["global_state"])

    env.reset(day=day, deterministic=True)
    actions = _off_actions(env)
    for _ in range(23):
        obs, _, done, _ = env.step(actions)
        assert not done
    schema = env.obs_builder.schema("mg1")
    forecast_load = obs["mg1"][schema.index("load_t+1")]
    expected = env.forecasts.local("mg1", 23, 1)["load_kw"] / env.obs_builder._norm["mg1"]["load"]
    assert forecast_load == pytest.approx(expected)


def test_step_after_done_has_clear_error():
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    actions = _off_actions(env)
    for _ in range(24):
        _, _, done, _ = env.step(actions)
    assert done
    with pytest.raises(RuntimeError, match="terminated"):
        env.step(actions)


def test_config_validator_rejects_duplicate_ids_and_unsupported_market_modes():
    with open(CONFIG, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    duplicate = copy.deepcopy(base)
    duplicate["microgrids"][1]["id"] = duplicate["microgrids"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        MultiMicrogridEnv(duplicate)

    unsupported = copy.deepcopy(base)
    unsupported["p2p"]["pricing"] = "pay_as_bid"
    with pytest.raises(ValueError, match="not supported"):
        MultiMicrogridEnv(unsupported)


@pytest.mark.parametrize("delta_t", [0.5, 2.0])
def test_config_validator_requires_exactly_one_hour_delta_t(delta_t):
    with open(CONFIG, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["time"]["delta_t"] = delta_t
    with pytest.raises(ValueError, match="exactly 1.0 hour"):
        MultiMicrogridEnv(config)


@pytest.mark.parametrize("soft_commitment", [-0.2, 0.0, 0.2])
def test_environment_rejects_soft_generator_commitment(soft_commitment):
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    actions = _off_actions(env)
    actions["mg1"] = np.array([0.0, soft_commitment, -1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="hard encoded"):
        env.step(actions)


def test_environment_rejects_noncanonical_off_generator_setpoint():
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    actions = _off_actions(env)
    actions["mg1"] = np.array([0.0, -1.0, 0.25, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="setpoint must be -1"):
        env.step(actions)


def test_environment_accepts_legal_hard_on_and_off_actions():
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    actions = _off_actions(env)
    actions["mg1"] = np.array([0.2, 1.0, 0.35, -0.4], dtype=np.float32)
    _, _, _, info = env.step(actions)
    assert info["plans"]["mg1"]["generator_commitment_action"] == 1.0
    assert info["plans"]["mg1"]["generator_setpoint_action"] == pytest.approx(0.35)
    assert info["plans"]["mg2"]["generator_commitment_action"] == -1.0
    assert info["plans"]["mg2"]["generator_setpoint_action"] == -1.0
    assert info["executed_actions"]["mg1"][1] == 1.0
    assert info["executed_actions"]["mg2"][1] == -1.0
    assert info["executed_actions"]["mg2"][2] == -1.0


def test_environment_canonicalizes_only_within_hard_action_float_tolerance():
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    actions = _off_actions(env)
    actions["mg1"] = np.array(
        [0.0, -1.0 + 5e-7, -1.0 + 5e-7, 0.0],
        dtype=np.float32,
    )
    _, _, _, info = env.step(actions)
    assert np.array_equal(
        info["executed_actions"]["mg1"][1:3],
        np.array([-1.0, -1.0], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "path,value,error",
    [
        (("microgrids", 0, "battery", "capacity_kwh"), 0.0, "positive"),
        (("microgrids", 0, "battery", "eta_charge"), 0.0, "inside"),
        (("microgrids", 0, "battery", "soc_min"), 0.95, "SOC"),
        (("microgrids", 0, "generator", "min_power_kw"), 100.0, "exceeds"),
        (("microgrids", 0, "grid", "export_limit_kw"), 14.0, "remain 15"),
        (("price", "sell_ratio"), 1.2, "inside"),
    ],
)
def test_config_validator_rejects_invalid_physical_and_price_parameters(path, value, error):
    with open(CONFIG, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=error):
        MultiMicrogridEnv(config)
