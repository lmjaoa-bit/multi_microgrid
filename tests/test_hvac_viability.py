from __future__ import annotations

import numpy as np
import pytest

from components.hvac_viability import (
    HVACForecastSnapshot,
    solve_hvac_viability,
    thermal_linear_coefficients,
)


def _cfg():
    return {
        "r_out_air_c_per_kw": 0.22,
        "r_air_mass_c_per_kw": 0.08,
        "c_air_kwh_per_c": 15.0,
        "c_mass_kwh_per_c": 250.0,
        "heating_max_kw": 35.0,
        "cooling_max_kw": 35.0,
        "cop_heat": 3.3,
        "cop_cool": 3.2,
        "solar_gain_kw_per_wm2": 0.025,
        "solar_air_fraction": 0.20,
        "internal_gain_occupied_fraction": 0.50,
        "internal_gain_unoccupied_fraction": 0.20,
    }


def _snapshot(h=3, low=16.0, high=30.0):
    return HVACForecastSnapshot(
        snapshot_id="fixture",
        boundary_timestamps=tuple(f"2026-01-01T0{k}:00:00" for k in range(h+1)),
        outside_temperature_c=np.full(h, 5.0),
        ghi_w_per_m2=np.zeros(h),
        base_load_kw=np.full(h, 10.0),
        interval_occupied=np.zeros(h, dtype=bool),
        comfort_low_c=np.full(h, low),
        comfort_high_c=np.full(h, high),
    )


def test_linear_coefficients_match_hand_derived_zero_gain_step():
    cfg=_cfg()
    a,b,dt=thermal_linear_coefficients(cfg,0.5)
    x=np.asarray([22.0,23.0])
    q=15.0
    tout=5.0
    expected_air=22.0+0.5*((tout-22.0)/cfg["r_out_air_c_per_kw"]+(23.0-22.0)/cfg["r_air_mass_c_per_kw"]+q)/cfg["c_air_kwh_per_c"]
    expected_mass=23.0+0.5*((22.0-23.0)/cfg["r_air_mass_c_per_kw"])/cfg["c_mass_kwh_per_c"]
    linear=a@x+b*q+np.asarray([0.5/cfg["c_air_kwh_per_c"]*tout/cfg["r_out_air_c_per_kw"],0.0])
    assert linear[0]==pytest.approx(expected_air)
    assert linear[1]==pytest.approx(expected_mass)


def test_feasible_snapshot_returns_ordered_first_action_interval():
    result=solve_hvac_viability(
        23.0,23.0,_snapshot(),config=_cfg(),delta_t=1.0
    )
    assert result.status=="optimal"
    assert result.feasible
    assert result.min_signed_power_kw <= result.max_signed_power_kw
    assert result.min_signed_power_kw >= -35.0-1e-7
    assert result.max_signed_power_kw <= 35.0+1e-7


def test_impossible_comfort_is_reported_as_degraded_with_slack():
    snap=_snapshot(h=2,low=40.0,high=41.0)
    result=solve_hvac_viability(
        20.0,20.0,snap,config=_cfg(),delta_t=1.0
    )
    assert not result.feasible
    assert result.status=="degraded_infeasible"
    assert result.current_min_slack_c > 0.0
