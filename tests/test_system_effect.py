from pathlib import Path
import numpy as np
import pytest

from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def run_case(enabled: bool):
    env = MultiMicrogridEnv(CONFIG, split="train")
    env.p2p_enabled = enabled
    env.reset(day=str(env.data.valid_days[0].date()))
    actions = {mg: np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32) for mg in env.agent_ids}
    done = False
    trade = curtail = 0.0
    while not done:
        _, _, done, info = env.step(actions)
        trade += float(info["market"]["total_trade_kwh"])
        curtail += sum(float(x["renewable_curtailment_kw"]) * env.delta_t for x in info["settlements"].values())
    return trade, curtail


def test_p2p_creates_trade_and_does_not_increase_curtailment():
    trade_on, curtail_on = run_case(True)
    _, curtail_off = run_case(False)
    assert trade_on > 0.0
    assert curtail_on <= curtail_off + 1e-9


def test_random_action_stress_preserves_all_system_invariants():
    rng = np.random.default_rng(2026)
    env = MultiMicrogridEnv(CONFIG, split="train")
    for day in env.data.valid_days[:5]:
        env.reset(day=str(day.date()), seed=123)
        done = False
        while not done:
            actions = {}
            for mg in env.agent_ids:
                commitment = float(rng.choice([-1.0, 1.0]))
                generator_setpoint = (
                    float(rng.uniform(-1.0, 1.0)) if commitment > 0.0 else -1.0
                )
                actions[mg] = np.asarray(
                    [
                        rng.uniform(-1.0, 1.0),
                        commitment,
                        generator_setpoint,
                        rng.uniform(-1.0, 1.0),
                    ],
                    dtype=np.float32,
                )
            _, rewards, done, info = env.step(actions)
            assert all(np.isfinite(value) for value in rewards.values())
            assert sum(
                item["p2p_net_cost"] for item in info["market"]["by_microgrid"].values()
            ) == pytest.approx(0.0, abs=1e-7)
            for mg in env.agent_ids:
                plan = info["plans"][mg]
                settlement = info["settlements"][mg]
                assert plan["generator_commitment_action"] in (-1.0, 1.0)
                if plan["generator_commitment_action"] < 0.0:
                    assert plan["generator_setpoint_action"] == -1.0
                assert abs(settlement["power_balance_residual_kw"]) <= 1e-6
                assert settlement["grid_buy_kw"] <= env.grid[mg].import_limit_kw + 1e-9
                assert settlement["grid_sell_kw"] <= env.grid[mg].export_limit_kw + 1e-9
                assert settlement["grid_to_battery_kw"] == 0.0
                assert settlement["p2p_to_battery_kw"] == 0.0
                assert settlement["battery_grid_sell_kw"] == 0.0
                assert plan["controllable_to_battery_kw"] == 0.0
                assert env.cores[mg].battery.soc_min <= env.cores[mg].battery.soc <= env.cores[mg].battery.soc_max
