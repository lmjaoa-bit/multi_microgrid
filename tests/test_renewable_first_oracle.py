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
