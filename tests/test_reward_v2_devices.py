from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from components.commercial_hvac import CommercialHVACDR
from components.ev_charging import EVChargingDR


def _hvac_config():
    return {
        "control_semantics": "soft_comfort",
        "forecast_viability": True,
        "r_out_air_c_per_kw": 0.22,
        "r_air_mass_c_per_kw": 0.08,
        "c_air_kwh_per_c": 15.0,
        "c_mass_kwh_per_c": 250.0,
        "heating_max_kw": 35.0,
        "cooling_max_kw": 35.0,
        "cop_heat": 3.3,
        "cop_cool": 3.2,
        "temperature_column": "temperature_c",
        "ghi_column": "ghi_w_per_m2",
        "occupied_hours": list(range(7, 19)),
        "occupied_comfort_c": [21.0, 25.0],
        "unoccupied_comfort_c": [16.0, 30.0],
        "terminal_air_tolerance_c": 1.0,
        "terminal_mass_tolerance_c": 0.3,
        "terminal_mode": "none",
        "solar_gain_kw_per_wm2": 0.025,
        "solar_air_fraction": 0.20,
        "internal_gain_occupied_fraction": 0.50,
        "internal_gain_unoccupied_fraction": 0.20,
    }


def _hvac():
    d = CommercialHVACDR(_hvac_config(), delta_t=1.0, episode_length=24)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=24, freq="h"),
        "temperature_c": np.linspace(-5, 10, 24),
        "ghi_w_per_m2": np.zeros(24),
    })
    d.reset(frame, np.full(24, 20.0))
    return d


def _ev_config():
    return dict(
        control_semantics="no_deadline_minimum",
        target_soc=.8,
        eta_charge=.93,
        max_charge_kw=93.6,
        scenario_seed=42,
        cohorts=[
            dict(
                group_id=f"G{i}",
                count=n,
                capacity_per_vehicle_kwh=60,
                max_charge_per_vehicle_kw=7.2,
                arrival_hours=[a],
                arrival_probabilities=[1],
                departure_hours=[d],
                departure_probabilities=[1],
                need_per_vehicle_kwh=[14, 22],
            )
            for i, n, a, d in [(1, 6, 0, 24), (2, 4, 0, 10), (3, 3, 10, 15)]
        ],
    )


def _ev():
    d = EVChargingDR(_ev_config(), delta_t=1.0, episode_length=24)
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=24, freq="h")})
    d.reset(frame, np.zeros(24))
    return d


def test_soft_hvac_never_calls_comfort_projection(monkeypatch):
    d = _hvac()

    def boom(*args, **kwargs):
        raise AssertionError("comfort projection must not be used")

    monkeypatch.setattr(d, "_one_step_bounds_q", boom)
    monkeypatch.setattr(d, "prepare_viability", boom)
    assert d.feasible_bounds(0) == (-35.0, 35.0)
    plan = d.plan(0.5, 0)
    assert plan["effective_signed_power_kw"] == pytest.approx(17.5)
    state = d.state(0)
    assert state["hvac_feasible_min_kw"] == -35.0
    assert state["hvac_feasible_max_kw"] == 35.0


def test_soft_hvac_allows_actual_comfort_violation():
    d = _hvac()
    d.t_air = 10.0
    plan = d.plan(0.0, 7)
    result = d.commit(plan)
    assert np.isfinite(result["t_air_c"])
    assert result["t_air_c"] < 21.0


def test_ev_zero_request_is_zero_for_all_groups():
    d = _ev()
    plan = d.plan(-1.0, 9)
    assert plan["feasible_min_kw"] == 0.0
    assert plan["effective_power_kw"] == 0.0
    assert sum(plan["group_power_kw"].values()) == 0.0
    result = d.commit(plan)
    assert result["served_charge_kw"] == 0.0


def test_departure_at_24_is_charged_once():
    d = _ev()
    g1 = next(s for s in d.sessions if s["group_id"] == "G1")
    assert g1["departure_hour"] == 24
    seen = []
    for t in range(24):
        result = d.commit(d.plan(-1.0, t))
        seen.append(result["ev_departure_shortfall_kwh"])
    assert seen[22] == 0.0
    assert seen[23] == pytest.approx(g1["demand_kwh"])
    total_events = len(d.departure_events)
    d.state(24)
    assert len(d.departure_events) == total_events
