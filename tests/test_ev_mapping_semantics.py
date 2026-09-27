from __future__ import annotations

import pytest

from components.ev_charging import EVChargingDR


def _device():
    cfg = {
        "target_soc": 0.8,
        "max_charge_kw": 30.0,
        "eta_charge": 1.0,
        "cohorts": [{
            "group_id": "G",
            "count": 1,
            "capacity_per_vehicle_kwh": 100.0,
            "max_charge_per_vehicle_kw": 30.0,
            "arrival_hours": [0],
            "arrival_probabilities": [1.0],
            "departure_hours": [1],
            "departure_probabilities": [1.0],
            "need_per_vehicle_kwh": [20.0, 20.0],
        }],
    }
    device = EVChargingDR(cfg, delta_t=1.0, episode_length=24)
    device.sessions = [{
        "group_id": "G",
        "arrival_hour": 0,
        "departure_hour": 1,
        "max_charge_kw": 30.0,
        "capacity_kwh": 100.0,
        "demand_kwh": 20.0,
    }]
    device.remaining = {"G": 20.0}
    device.delivered_kwh = {"G": 0.0}
    device._departed = set()
    device.departure_events = []
    return device


def test_affine_mapping_is_not_safety_projection():
    device = _device()
    device._group_bounds = lambda t: {"G": (10.0, 30.0)}
    plan = device._cohort_plan(0.0, 0)
    assert plan["mapped_power_kw"] == pytest.approx(20.0)
    assert plan["effective_power_kw"] == pytest.approx(20.0)
    assert plan["mapping_unit_intent"] == pytest.approx(0.5)
    assert plan["safety_correction_kw"] == pytest.approx(0.0)
    assert plan["safety_projected"] is False
    assert plan["projected"] is False


@pytest.mark.parametrize("bounds,expected", [
    ((10.0, 10.0), 10.0),
    ((0.0, 0.0), 0.0),
])
def test_forced_mapping_is_explicit_without_projection(bounds, expected):
    device = _device()
    device._group_bounds = lambda t: {"G": bounds}
    plan = device._cohort_plan(1.0, 0)
    assert plan["mapped_power_kw"] == pytest.approx(expected)
    assert plan["mapping_forced"] is True
    assert plan["safety_projected"] is False


def test_delivery_shortfall_is_separate_from_mapping():
    device = _device()
    device._group_bounds = lambda t: {"G": (20.0, 20.0)}
    plan = device._cohort_plan(0.0, 0)
    result = device._cohort_commit(plan, 0.75)
    assert plan["safety_correction_kw"] == pytest.approx(0.0)
    assert result["planned_charge_kw"] == pytest.approx(20.0)
    assert result["served_charge_kw"] == pytest.approx(15.0)
    assert result["delivery_shortfall_kw"] == pytest.approx(5.0)
    assert result["ev_departure_shortfall_kwh"] == pytest.approx(5.0)
    assert result["ev_departure_count"] == pytest.approx(1.0)
