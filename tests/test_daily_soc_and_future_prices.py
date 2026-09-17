from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.multi_microgrid_env import MultiMicrogridEnv
from optimization.common_constraints import DispatchData, solve_dispatch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative_legacy_pre_dr.yaml"


def test_milp_none_allows_retaining_terminal_energy():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data = DispatchData(
        timestamps=["last-hour"], agent_ids=["mg1"],
        load_kw={"mg1": np.array([0.0])},
        renewable_kw={"mg1": np.array([100.0])},
        buy_price=np.array([1.0]), sell_price=np.array([-1.0]),
    )
    free = solve_dispatch(config, data, ["mg1"], cooperative=False)
    config["terminal_soc"]["mode"] = "hard"
    constrained = solve_dispatch(config, data, ["mg1"], cooperative=False)
    assert free.trajectories.iloc[0]["battery_charge_kw"] > 20
    assert constrained.trajectories.iloc[0]["battery_charge_kw"] < 10


def test_empty_start_no_terminal_target_or_shaping():
    env = MultiMicrogridEnv(CONFIG)
    env.reset(seed=42, deterministic=True)
    for core in env.cores.values():
        assert core.battery.soc == core.battery.soc_min
        assert core.battery.available_discharge_kw() == pytest.approx(0)
    env.t = 23
    for core in env.cores.values():
        core.battery.soc = .6
    env._last_global_state = env._build_global_state()
    observations, rewards, done, info = env.step({mg: [-1.] for mg in env.agent_ids})
    assert done
    assert info["shaping_reward"] == 0
    assert all(value == 0 for value in info["terminal_soc_penalty"].values())
    for mg in env.agent_ids:
        assert "dispatch_forced" not in info["plans"][mg]
        assert info["devices"][mg]["charge_kw"] > 0
        assert env.cores[mg].battery.soc > .6
        assert observations[mg][env.obs_builder.schema(mg).index("terminal_soc_gap")] == 0
    profit = sum(cost["physical_profit"] for cost in info["costs"].values())
    assert rewards[env.agent_ids[0]] == pytest.approx(profit / env.accounting.reward_scale)
    env.reset(seed=43, deterministic=True)
    assert all(core.battery.soc == core.battery.soc_min for core in env.cores.values())


@pytest.mark.parametrize("deterministic", [True, False])
def test_critic_prices_match_actor_forecasts(deterministic):
    env = MultiMicrogridEnv(CONFIG)
    observations, _ = env.reset(seed=42, deterministic=deterministic)
    for _ in range(24):
        state = env.global_state()
        schema = env.global_state_builder.schema()
        for mg in env.agent_ids:
            obs_schema = env.obs_builder.schema(mg)
            for k in range(1, env.obs_builder.future_h + 1):
                assert state[schema.index(f"{mg}_price_t+{k}")] == pytest.approx(
                    observations[mg][obs_schema.index(f"price_t+{k}")])
        observations, _, done, _ = env.step({mg: np.zeros(1) for mg in env.agent_ids})
        if done:
            break


def test_24_hour_forecast_masks_values_beyond_natural_day():
    env = MultiMicrogridEnv(CONFIG, split="test")
    observations, _ = env.reset(
        day=str(env.data.valid_days[0].date()), seed=42, deterministic=True,
    )
    assert env.obs_builder.future_h == 24
    for mg in env.agent_ids:
        schema = env.obs_builder.schema(mg)
        assert observations[mg][schema.index("forecast_valid_t+1")] == 1.0
        assert observations[mg][schema.index("forecast_valid_t+23")] == 1.0
        assert observations[mg][schema.index("forecast_valid_t+24")] == 0.0
        for name in ("load", "pv", "wind", "price"):
            assert observations[mg][schema.index(f"{name}_t+24")] == 0.0

    state = env.global_state()
    state_schema = env.global_state_builder.schema()
    assert state[state_schema.index("forecast_valid_t+23")] == 1.0
    assert state[state_schema.index("forecast_valid_t+24")] == 0.0


def test_fixed_seed_keeps_noisy_forecasts_reproducible_during_deterministic_rollout():
    env = MultiMicrogridEnv(CONFIG, split="test")
    day = str(env.data.valid_days[0].date())
    env.reset(day=day, seed=77, deterministic=True, forecast_noise=True)
    first = np.asarray([
        env.forecasts.local("mg1", 0, k)["load_kw"] for k in range(1, 24)
    ])
    actual = env._episode_exogenous["mg1"]["load_kw"][1:24]
    env.reset(day=day, seed=77, deterministic=True, forecast_noise=True)
    repeated = np.asarray([
        env.forecasts.local("mg1", 0, k)["load_kw"] for k in range(1, 24)
    ])
    assert np.array_equal(first, repeated)
    assert not np.allclose(first, actual)
    relative_error = np.abs(first / actual - 1.0)
    assert np.max(relative_error) <= env.forecasts.error_max["load"] + 1e-12
