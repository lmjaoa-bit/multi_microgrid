from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from optimization.common_constraints import DispatchData, solve_dispatch


@pytest.mark.parametrize("cooperative", [True, False])
def test_full_information_oracle_models_all_three_dr_types(cooperative: bool) -> None:
    pytest.importorskip("highspy")
    config = yaml.safe_load(open("configs/cooperative.yaml", encoding="utf-8"))
    times = pd.date_range("2018-04-27", periods=24, freq="h")
    frame = pd.DataFrame({
        "timestamp": times,
        "mg2_temperature_c": np.full(24, 23.0),
        "mg2_ghi_w_per_m2": np.zeros(24),
    })
    ids = ["mg1", "mg2", "mg3"]
    base = {"mg1": 45.0, "mg2": 60.0, "mg3": 35.0}
    data = DispatchData(
        timestamps=[str(x) for x in times],
        agent_ids=ids,
        load_kw={mg: np.full(24, base[mg]) for mg in ids},
        renewable_kw={mg: np.zeros(24) for mg in ids},
        buy_price={mg: np.full(24, 0.6) for mg in ids},
        sell_price={mg: np.full(24, 0.1) for mg in ids},
        frame=frame,
    )
    result = solve_dispatch(
        config, data, ids, cooperative=cooperative,
        include_dr=True, solver_output=False,
    )
    assert result.status == "kOptimal"
    assert len(result.trajectories) == 72
    assert result.mode.endswith("full_information") or result.mode.endswith("relaxed")
    assert result.cost_breakdown is not None
    costs = result.cost_breakdown
    recovered_cost = (sum(value for key,value in costs.items()
                          if key != "grid_export_revenue") - costs["grid_export_revenue"])
    assert recovered_cost == pytest.approx(result.total_cost, abs=1e-4)
    rows = result.trajectories
    assert np.allclose(rows["load_kw"], rows["base_load_kw"] + rows["flexible_load_kw"])
    assert np.allclose(rows["load_kw"] + rows["battery_charge_kw"] +
                       rows["grid_export_kw"] + rows["p2p_export_kw"] +
                       rows["curtailment_kw"],
                       rows["renewable_kw"] + rows["battery_discharge_kw"] +
                       rows["grid_import_kw"] + rows["p2p_import_kw"] +
                       rows["unmet_load_kw"], atol=1e-5)
    mill = rows.loc[rows.microgrid == "mg1"].sort_values("timestamp")
    hvac = rows.loc[rows.microgrid == "mg2"].sort_values("timestamp")
    ev = rows.loc[rows.microgrid == "mg3"].sort_values("timestamp")
    assert mill["silo_after_t"].between(3.51-1e-7, 15.80+1e-7).all()
    assert (hvac["hvac_heating_kw"] * hvac["hvac_cooling_kw"] < 1e-7).all()
    assert (ev.iloc[7:]["flexible_load_kw"].abs() < 1e-7).all()
    assert ev["ev_soc_after"].between(0.5-1e-7, 0.8+1e-7).all()


def test_full_information_requires_24h_frame() -> None:
    pytest.importorskip("highspy")
    config = yaml.safe_load(open("configs/cooperative.yaml", encoding="utf-8"))
    data = DispatchData(
        timestamps=["2018-04-27 00:00:00"],
        agent_ids=["mg1", "mg2", "mg3"],
        load_kw={mg: np.asarray([50.0]) for mg in ["mg1","mg2","mg3"]},
        renewable_kw={mg: np.asarray([0.0]) for mg in ["mg1","mg2","mg3"]},
        buy_price={mg: np.asarray([0.6]) for mg in ["mg1","mg2","mg3"]},
        sell_price={mg: np.asarray([0.1]) for mg in ["mg1","mg2","mg3"]},
        frame=None,
    )
    with pytest.raises(ValueError, match="complete 24-hour episode frame"):
        solve_dispatch(config, data, data.agent_ids, cooperative=True, include_dr=True)
