from __future__ import annotations

import pandas as pd
import pytest

from optimization.common_constraints import DispatchResult
from optimization.oracle_replay import compare_oracle_market_replay


def test_market_replay_reports_exact_match():
    oracle = DispatchResult(
        mode="cooperative_full_information_relaxed",
        status="kOptimal",
        total_cost=-10.0,
        total_profit=10.0,
        profit_by_microgrid={"mg1": 10.0},
        trajectories=pd.DataFrame([{
            "timestamp":"2026-01-01T00:00:00",
            "microgrid":"mg1",
            "renewable_to_local_load_kw":5.0,
            "renewable_to_local_charge_kw":0.0,
            "ess_to_local_load_kw":0.0,
            "grid_to_load_kw":0.0,
            "grid_to_charge_kw":0.0,
            "renewable_peer_load_export_kw":0.0,
            "renewable_peer_charge_export_kw":0.0,
            "ess_peer_load_export_kw":0.0,
            "grid_import_kw":0.0,
            "grid_export_kw":0.0,
        }]),
        pair_flows_kwh={},
        metadata={"allocation":"source_priority_free_allocation"},
    )
    replay=[{
        "timestamp":"2026-01-01T00:00:00",
        "mg1_renewable_to_local_load_kw":5.0,
        "mg1_renewable_to_battery_kw":0.0,
        "mg1_ess_to_local_load_kw":0.0,
        "mg1_grid_to_load_kw":0.0,
        "mg1_grid_to_battery_kw":0.0,
        "mg1_renewable_peer_load_export_kw":0.0,
        "mg1_renewable_peer_charge_export_kw":0.0,
        "mg1_ess_p2p_sell_kw":0.0,
        "mg1_grid_import_kw":0.0,
        "mg1_grid_export_kw":0.0,
        "mg1_hourly_profit":10.0,
    }]
    result=compare_oracle_market_replay(oracle,replay)
    assert result["exactly_replayable"] is True
    assert result["profit_gap"] == pytest.approx(0.0)


def test_market_replay_exposes_free_allocation_gap():
    oracle = DispatchResult(
        mode="cooperative_full_information_relaxed",
        status="kOptimal",
        total_cost=-10.0,
        total_profit=10.0,
        profit_by_microgrid={"mg1": 10.0},
        trajectories=pd.DataFrame([{
            "timestamp":"2026-01-01T00:00:00",
            "microgrid":"mg1",
            "renewable_to_local_load_kw":5.0,
            "renewable_to_local_charge_kw":0.0,
            "ess_to_local_load_kw":0.0,
            "grid_to_load_kw":0.0,
            "grid_to_charge_kw":0.0,
            "renewable_peer_load_export_kw":0.0,
            "renewable_peer_charge_export_kw":0.0,
            "ess_peer_load_export_kw":0.0,
            "grid_import_kw":0.0,
            "grid_export_kw":0.0,
        }]),
        pair_flows_kwh={},
        metadata={"allocation":"source_priority_free_allocation"},
    )
    replay=[{
        "timestamp":"2026-01-01T00:00:00",
        "mg1_renewable_to_local_load_kw":4.0,
        "mg1_renewable_to_battery_kw":0.0,
        "mg1_ess_to_local_load_kw":0.0,
        "mg1_grid_to_load_kw":1.0,
        "mg1_grid_to_battery_kw":0.0,
        "mg1_renewable_peer_load_export_kw":0.0,
        "mg1_renewable_peer_charge_export_kw":0.0,
        "mg1_ess_p2p_sell_kw":0.0,
        "mg1_grid_import_kw":1.0,
        "mg1_grid_export_kw":0.0,
        "mg1_hourly_profit":9.0,
    }]
    result=compare_oracle_market_replay(oracle,replay)
    assert result["exactly_replayable"] is False
    assert result["status"]=="oracle_not_exactly_replayable"
    assert result["allocation_replay_gap_max_abs_kw"] == pytest.approx(1.0)
    assert result["profit_gap"] == pytest.approx(-1.0)
