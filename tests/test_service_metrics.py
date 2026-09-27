from __future__ import annotations

import pytest

from evaluation.service_metrics import (
    ServiceMetricDataError,
    aggregate_cooperative_value,
    aggregate_service_days,
    summarize_day_services,
)


def _config():
    return {
        "time": {"delta_t": 1.0},
        "terminal_soc": {"mode": "none"},
        "microgrids": [
            {"id": "mg1", "dr": {"type": "industrial_mill", "terminal_mode": "none", "silo_min_t": 3.0, "silo_max_t": 16.0, "silo_initial_t": 10.0, "terminal_tolerance_t": 0.2}},
            {"id": "mg2", "dr": {"type": "commercial_hvac", "terminal_mode": "none", "terminal_air_tolerance_c": 1.0, "terminal_mass_tolerance_c": 0.3}},
            {"id": "mg3", "dr": {"type": "ev_charging"}},
        ],
    }


def _row(start, result, *, occupied, low, high, t_air=23.0):
    row = {
        "timestamp": start,
        "mg2_result_timestamp": result,
        "mg2_t_air_c": t_air,
        "mg2_result_occupied": int(occupied),
        "mg2_result_comfort_low_c": low,
        "mg2_result_comfort_high_c": high,
        "mg1_mill_actual_production_t": 0.8,
        "mg1_mill_downstream_required_t": 1.0,
        "mg1_mill_actual_consumption_t": 1.0,
        "mg1_mill_downstream_unserved_t": 0.0,
        "mg1_mill_silo_before_t": 10.0,
        "mg1_mill_silo_after_unclipped_t": 9.8,
        "mg1_silo_level_t": 9.8,
        "mg3_ev_departure_count": 0,
        "mg3_ev_completed_group_count": 0,
        "mg3_ev_departure_shortfall_kwh": 0.0,
    }
    for mg in ("mg1", "mg2", "mg3"):
        row[f"{mg}_unmet_load_kw"] = 0.0
        row[f"{mg}_dr_supply_infeasible"] = 0
    return row


def test_hvac_06_to_07_uses_result_occupied_band():
    row = _row("2026-09-01T06:00:00", "2026-09-01T07:00:00", occupied=True, low=21.0, high=25.0, t_air=26.0)
    result = summarize_day_services([row], _config(), True, "service_metrics_v2")
    assert result["hvac"]["comfort_rate"] == 0.0
    assert result["strict_feasible_day"] is False


def test_hvac_18_to_19_uses_result_unoccupied_band():
    row = _row("2026-09-01T18:00:00", "2026-09-01T19:00:00", occupied=False, low=16.0, high=30.0, t_air=27.0)
    result = summarize_day_services([row], _config(), True, "service_metrics_v2")
    assert result["hvac"]["all_boundary_comfort_rate"] == 1.0
    assert result["strict_feasible_day"] is True


def test_hvac_aggregation_uses_success_over_opportunities():
    first = {"data_complete": True, "strict_feasible_day": True, "unmet_energy_kwh": 0.0, "hvac": {"occupied_success_count": 1, "occupied_boundary_count": 1, "boundary_success_count": 1, "boundary_count": 1, "comfort_violation_degree_hours": 0.0, "terminal_success": None}, "mill": {}, "ev": {}}
    second = {"data_complete": True, "strict_feasible_day": True, "unmet_energy_kwh": 0.0, "hvac": {"occupied_success_count": 8, "occupied_boundary_count": 9, "boundary_success_count": 8, "boundary_count": 9, "comfort_violation_degree_hours": 0.0, "terminal_success": None}, "mill": {}, "ev": {}}
    assert aggregate_service_days([first, second])["hvac_comfort_rate"] == pytest.approx(0.9)


def test_current_schema_rejects_missing_required_field():
    row = _row("2026-09-01T18:00:00", "2026-09-01T19:00:00", occupied=False, low=16.0, high=30.0)
    row.pop("mg2_result_timestamp")
    with pytest.raises(ServiceMetricDataError):
        summarize_day_services([row], _config(), True, "service_metrics_v2")


def test_cooperative_value_capture_is_ratio_of_sums():
    rows = [
        {"date": "d1", "algorithm_joint_profit": 9.0, "independent_total_profit": 0.0, "cooperative_milp_profit": 10.0, "coalition_solve_failure_count": 0},
        {"date": "d2", "algorithm_joint_profit": 1.0, "independent_total_profit": 0.0, "cooperative_milp_profit": 2.0, "coalition_solve_failure_count": 0},
    ]
    result = aggregate_cooperative_value(rows)
    assert result["cooperative_value_capture_ratio_total"] == pytest.approx(10.0 / 12.0)
    assert result["cooperative_value_capture_ratio_daily_mean"] == pytest.approx(0.7)
    assert result["aggregation_method"] == "ratio_of_sums"
