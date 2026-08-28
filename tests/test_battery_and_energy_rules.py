from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from components.battery import BatteryModel
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def _row() -> pd.Series:
    return pd.Series({
        "timestamp": pd.Timestamp("2018-01-01 00:00:00"),
        "load_kw": 60.0,
        "buy_price_per_kwh": 0.6,
        "sell_price_per_kwh": 0.3,
        "grid_carbon_factor_kg_per_kwh": 0.5,
        "ghi_w_per_m2": 0.0,
        "wind_speed_mps": 0.0,
    })


def test_battery_action_sign_and_zero_semantics_at_all_soc_levels():
    battery = BatteryModel(200.0, 50.0, 50.0)
    battery.reset()
    for soc in (battery.soc_min, 0.5, battery.soc_max):
        battery.soc = soc
        assert battery.map_action(0.0)["power_kw"] == pytest.approx(0.0)
        assert battery.map_action(-0.5)["power_kw"] <= 0.0
        assert battery.map_action(0.5)["power_kw"] >= 0.0


def test_battery_charging_uses_only_local_renewable_surplus():
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    core = env.cores["mg1"]

    core.exogenous = lambda row: {"load_kw": 0.0, "pv_kw": 100.0, "wind_kw": 0.0, "renewable_kw": 100.0}
    renewable_plan = core.plan_local(_row(), {
        "battery": -1.0,
        "generator_commitment": -1.0,
        "generator_setpoint": -1.0,
    })
    assert renewable_plan["renewable_to_battery_kw"] > 0.0
    assert renewable_plan["controllable_to_battery_kw"] == 0.0
    assert renewable_plan["battery_external_charge_demand_kw"] == 0.0

    core.exogenous = lambda row: {"load_kw": 0.0, "pv_kw": 0.0, "wind_kw": 0.0, "renewable_kw": 0.0}
    generator_plan = core.plan_local(_row(), {
        "battery": -1.0,
        "generator_commitment": 1.0,
        "generator_setpoint": 1.0,
    })
    settlement = env.grid["mg1"].settle(generator_plan, {})
    assert generator_plan["controllable_to_battery_kw"] == 0.0
    assert generator_plan["generator_external_surplus_kw"] > 0.0
    assert settlement["grid_to_battery_kw"] == 0.0
    assert settlement["p2p_to_battery_kw"] == 0.0


def test_battery_can_supply_local_load_and_create_p2p_surplus():
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    core = env.cores["mg1"]
    core.battery.soc = 0.5
    core.exogenous = lambda row: {"load_kw": 10.0, "pv_kw": 0.0, "wind_kw": 0.0, "renewable_kw": 0.0}
    plan = core.plan_local(_row(), {
        "battery": 1.0,
        "generator_commitment": -1.0,
        "generator_setpoint": -1.0,
    })
    assert plan["controllable_to_load_kw"] == pytest.approx(plan["load_kw"])
    assert plan["battery_external_surplus_kw"] > 0.0
    assert plan["external_surplus_kw"] > 0.0
    p2p_sell = min(20.0, plan["external_surplus_kw"])
    settlement = env.grid["mg1"].settle(plan, {"p2p_sell_kwh": p2p_sell})
    device = core.commit(plan, settlement)
    assert settlement["battery_p2p_sell_kw"] == pytest.approx(p2p_sell)
    assert settlement["battery_grid_sell_kw"] == 0.0
    assert device["battery"]["discharge_kw"] == pytest.approx(
        plan["battery_to_load_kw"] + settlement["battery_p2p_sell_kw"]
    )


def test_environment_order_threshold_includes_exact_boundary():
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    plans = {mg: {"net_external_kw": 0.0} for mg in env.agent_ids}
    plans["mg1"]["net_external_kw"] = 1.0
    row = env.episode_df.iloc[0]
    decoded = {
        mg: env.action_codec.decode(
            np.array([0.0, -1.0, -1.0, 0.0]),
            float(row["sell_price_per_kwh"]),
            float(row["buy_price_per_kwh"]),
        )
        for mg in env.agent_ids
    }
    orders, roles = env._build_orders(plans, decoded)
    assert len(orders) == 1
    assert orders[0].quantity_kwh == pytest.approx(1.0)
    assert roles["mg1"] == -1.0
