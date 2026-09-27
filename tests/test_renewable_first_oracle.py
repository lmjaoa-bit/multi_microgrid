from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimization.common_constraints import DispatchData, solve_dispatch


def _cfg():
    return {
        "dispatch": {"priority_mode": "renewable_load_first"},
        "time": {"delta_t": 1.0},
        "cost": {
            "unmet_load_cost_per_kwh": 1000.0,
            "renewable_curtailment_cost_per_kwh": 0.01,
        },
        "terminal_soc": {"mode": "none", "penalty_per_kwh": 0.0, "tolerance": 0.0},
        "p2p": {
            "pair_capacity_kw": None,
            "allow_grid_resale": False,
            "allow_ess_to_ess": False,
        },
        "microgrids": [
            {
                "id": mg,
                "battery": {
                    "capacity_kwh": 100.0, "soc_min": 0.1, "soc_max": 0.9,
                    "initial_soc": 0.5, "max_charge_kw": 30.0,
                    "max_discharge_kw": 30.0, "eta_charge": 1.0,
                    "eta_discharge": 1.0, "degradation_cost_per_kwh": 0.0,
                },
                "grid": {"import_limit_kw": None, "export_limit_kw": 100.0},
            }
            for mg in ("mg1","mg2","mg3")
        ],
    }


def _data(load, renewable):
    ids=["mg1","mg2","mg3"]
    return DispatchData(
        timestamps=["2026-01-01T00:00:00"],
        agent_ids=ids,
        load_kw={mg:np.asarray([float(load[i])]) for i,mg in enumerate(ids)},
        renewable_kw={mg:np.asarray([float(renewable[i])]) for i,mg in enumerate(ids)},
        buy_price={mg:np.asarray([1.0]) for mg in ids},
        sell_price={mg:np.asarray([0.1]) for mg in ids},
        frame=pd.DataFrame({"timestamp":["2026-01-01T00:00:00"]}),
    )


def test_coalition_renewable_covers_load_before_grid_or_ess():
    cfg=_cfg()
    result=solve_dispatch(
        cfg,_data([30,0,0],[0,30,0]),["mg1","mg2","mg3"],
        cooperative=True,p2p_export_source="unrestricted",
    )
    rows=result.trajectories
    assert rows["grid_to_load_kw"].sum()==pytest.approx(0.0,abs=1e-7)
    assert rows["ess_to_local_load_kw"].sum()==pytest.approx(0.0,abs=1e-7)
    assert rows["renewable_peer_load_export_kw"].sum()==pytest.approx(30.0)
    assert result.metadata["allocation"]=="source_priority_free_allocation"


def test_received_renewable_does_not_forbid_ess_export_to_other_load():
    cfg=_cfg()
    cfg["microgrids"][0]["battery"]["initial_soc"] = 0.5
    cfg["microgrids"][1]["battery"]["initial_soc"] = 0.1
    cfg["microgrids"][2]["battery"]["initial_soc"] = 0.1
    result=solve_dispatch(
        cfg,_data([10,0,20],[0,10,0]),["mg1","mg2","mg3"],
        cooperative=True,p2p_export_source="unrestricted",
    )
    rows=result.trajectories.set_index("microgrid")
    assert rows.loc["mg2","renewable_peer_load_export_kw"]==pytest.approx(10.0)
    assert rows.loc["mg1","ess_peer_load_export_kw"] > 0.0
    assert rows.loc["mg1","grid_to_load_kw"] == pytest.approx(0.0, abs=1e-7)


def test_independent_mode_does_not_virtualize_peer_renewable():
    cfg=_cfg()
    result=solve_dispatch(
        cfg,_data([20,0,0],[0,20,0]),["mg1","mg2","mg3"],
        cooperative=False,p2p_export_source="unrestricted",
    )
    rows=result.trajectories.set_index("microgrid")
    assert rows.loc["mg1","renewable_peer_load_export_kw"]==pytest.approx(0.0)
    assert rows.loc["mg1","grid_to_load_kw"] > 0.0


def test_legacy_mode_keeps_legacy_allocation_metadata():
    cfg=_cfg()
    cfg.pop("dispatch")
    result=solve_dispatch(
        cfg,_data([20,0,0],[0,20,0]),["mg1","mg2","mg3"],
        cooperative=True,p2p_export_source="unrestricted",
    )
    assert result.metadata["source_priority"]=="legacy_local_first"
    assert result.metadata["allocation"]=="legacy_oracle_allocation"


def test_two_member_coalition_cannot_use_outside_member_renewable():
    cfg = _cfg()
    data = _data([20, 0, 0], [0, 0, 50])
    result = solve_dispatch(
        cfg,
        data,
        ["mg1", "mg2"],
        cooperative=True,
        p2p_export_source="unrestricted",
    )
    rows = result.trajectories.set_index("microgrid")
    assert set(rows.index) == {"mg1", "mg2"}
    assert rows.loc["mg1", "grid_to_load_kw"] > 0.0
    assert rows["renewable_peer_load_export_kw"].sum() == pytest.approx(0.0)


def test_d3_peer_renewable_load_priority_allows_grid_charging_at_seller():
    cfg = _cfg()
    # Make grid charging economically attractive after A's 30 kW renewable
    # has first served B's 30 kW load. This is legal: the exported power is
    # renewable; the imported power terminates at A's own battery.
    data = _data([0, 30, 0], [30, 0, 0])
    # A negative import price makes grid charging strictly attractive, so the
    # test actually exercises simultaneous renewable P2P export and grid->ESS.
    data.buy_price["mg1"][0] = -1.0
    result = solve_dispatch(
        cfg,
        data,
        ["mg1", "mg2", "mg3"],
        cooperative=True,
        p2p_export_source="unrestricted",
    )
    rows = result.trajectories.set_index("microgrid")
    assert rows.loc["mg1", "renewable_peer_load_export_kw"] == pytest.approx(30.0)
    assert rows.loc["mg2", "grid_to_load_kw"] == pytest.approx(0.0, abs=1e-7)
    # The source/use model must not prohibit simultaneous renewable export
    # and grid import merely because both happen at the same microgrid.
    assert rows.loc["mg1", "grid_to_charge_kw"] > 0.0
    assert rows.loc["mg1", "grid_import_kw"] == pytest.approx(
        rows.loc["mg1", "grid_to_charge_kw"], abs=1e-7
    )
    assert result.metadata["grid_resale_allowed"] is False


def test_p2p_disabled_cooperative_request_degrades_to_independent_no_flows():
    cfg = _cfg()
    cfg["p2p"]["enabled"] = False
    result = solve_dispatch(
        cfg,
        _data([20, 0, 0], [0, 20, 0]),
        ["mg1", "mg2", "mg3"],
        cooperative=True,
        p2p_export_source="unrestricted",
    )
    assert result.pair_flows_kwh == {}
    assert result.trajectories["p2p_import_kw"].sum() == pytest.approx(0.0)
    assert result.trajectories["p2p_export_kw"].sum() == pytest.approx(0.0)
    assert result.metadata["cooperative_requested"] is True
    assert result.metadata["p2p_enabled"] is False
    assert result.metadata["cooperative_effective"] is False
