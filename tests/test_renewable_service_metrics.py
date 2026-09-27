from types import SimpleNamespace

import pytest

from evaluation.service_metrics import aggregate_service_days, summarize_day_services
from utils.metrics_collector import EpisodeMetricsCollector, json_ready


def test_non_curtailment_includes_exports_and_reports_end_inventory():
    battery = SimpleNamespace(soc=0.1, initial_soc=0.1, capacity_kwh=100)
    env = SimpleNamespace(
        agent_ids=["mg1"],
        action_design="single_dispatch",
        cores={"mg1": SimpleNamespace(battery=battery)},
        energy_pool=SimpleNamespace(pair_capacity_kw=None),
    )
    collector = EpisodeMetricsCollector(env)
    collector.energy["mg1"].update(
        renewable_available_kwh=100,
        renewable_used_kwh=80,
        renewable_curtailment_kwh=20,
        grid_export_kwh=70,
    )
    collector.soc["mg1"].append(0.3)
    result = collector.finalize_episode()
    assert result["curtailment_kwh"] == 20
    assert result["curtailment_rate"] == pytest.approx(0.2)
    assert result["renewable_non_curtailment_rate"] == pytest.approx(0.8)
    assert result["renewable_utilization_rate"] == result["renewable_non_curtailment_rate"]
    assert result["soc_statistics_by_microgrid"]["mg1"]["final_energy_kwh"] == 30
    assert result["soc_statistics_by_microgrid"]["mg1"]["energy_delta_kwh"] == pytest.approx(20)
    assert result["p2p_statistics"]["pair_capacity_kw"] is None
    assert json_ready(result)[1] == []


def test_departure_service_uses_once_only_cumulative_event_totals():
    config = {
        "time": {"delta_t": 1.0},
        "microgrids": [
            {"id": "mg3", "dr": {"type": "ev_charging"}}
        ],
    }
    rows = [
        {
            "mg3_unmet_load_kw": 0.0,
            "mg3_dr_supply_infeasible": 0,
            "mg3_ev_departure_count": 1,
            "mg3_ev_completed_group_count": 1,
            "mg3_ev_departure_shortfall_kwh": 0.0,
        },
        {
            "mg3_unmet_load_kw": 0.0,
            "mg3_dr_supply_infeasible": 0,
            "mg3_ev_departure_count": 3,
            "mg3_ev_completed_group_count": 2,
            "mg3_ev_departure_shortfall_kwh": 4.0,
        },
    ]
    result = summarize_day_services(
        rows, config, True, "service_metrics_v2"
    )
    assert result["ev"]["departure_count"] == 3
    assert result["ev"]["completed_group_count"] == 2
    assert result["ev"]["departure_shortfall_kwh"] == pytest.approx(4.0)
    assert result["ev"]["departure_success_rate"] == pytest.approx(2 / 3)
    assert result["strict_feasible_day"] is False


def test_disabled_terminal_restoration_is_not_counted_as_success_or_failure():
    day = {
        "data_complete": True,
        "strict_feasible_day": True,
        "unmet_energy_kwh": 0.0,
        "unmet_load_day": False,
        "dr_supply_infeasible_day": False,
        "hvac": {
            "occupied_success_count": 8,
            "occupied_boundary_count": 8,
            "boundary_success_count": 24,
            "boundary_count": 24,
            "comfort_violation_degree_hours": 0.0,
            "terminal_applicable": False,
            "terminal_success": None,
        },
        "mill": {
            "production_t": 16.51,
            "consumption_required_t": 23.40,
            "actual_consumption_t": 23.40,
            "downstream_unserved_t": 0.0,
            "terminal_applicable": False,
            "terminal_success": None,
        },
        "ev": {
            "departure_count": 3,
            "completed_group_count": 3,
            "departure_shortfall_kwh": 0.0,
        },
    }
    result = aggregate_service_days([day])
    assert result["hvac_terminal_joint_success_rate"] is None
    assert result["mill_terminal_success_rate"] is None
    assert result["hvac_comfort_rate"] == pytest.approx(1.0)
    assert result["mill_production_coverage_ratio"] == pytest.approx(16.51 / 23.40)
    assert result["mill_downstream_service_completion_rate"] == pytest.approx(1.0)
    assert result["ev_departure_success_rate"] == pytest.approx(1.0)
