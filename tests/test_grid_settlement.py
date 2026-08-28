from pathlib import Path

import pandas as pd
import pytest

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


def test_import_limit_unmet_load_and_unserved_charge_are_separate():
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    core = env.cores["mg1"]
    core.exogenous = lambda row: {"load_kw": 200.0, "pv_kw": 0.0, "wind_kw": 0.0, "renewable_kw": 0.0}
    plan = core.plan_local(_row(), {
        "battery": -1.0,
        "generator_commitment": -1.0,
        "generator_setpoint": -1.0,
    })
    settlement = env.grid["mg1"].settle(plan, {})
    assert settlement["grid_buy_kw"] == pytest.approx(70.0)
    assert settlement["grid_to_battery_kw"] == 0.0
    assert settlement["unmet_load_kw"] == pytest.approx(plan["load_external_demand_kw"] - 70.0)
    assert settlement["unserved_battery_charge_kw"] == pytest.approx(
        plan["battery_unserved_local_charge_request_kw"]
    )


@pytest.mark.parametrize("surplus_kw,curtailment", [(15.0, 0.0), (16.0, 1.0)])
def test_export_limit_exact_boundary(surplus_kw, curtailment):
    env = MultiMicrogridEnv(CONFIG)
    env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    core = env.cores["mg1"]
    core.exogenous = lambda row: {"load_kw": 0.0, "pv_kw": surplus_kw, "wind_kw": 0.0, "renewable_kw": surplus_kw}
    plan = core.plan_local(_row(), {
        "battery": 0.0,
        "generator_commitment": -1.0,
        "generator_setpoint": -1.0,
    })
    settlement = env.grid["mg1"].settle(plan, {})
    assert settlement["grid_sell_kw"] == pytest.approx(15.0)
    assert settlement["renewable_curtailment_kw"] == pytest.approx(curtailment)
