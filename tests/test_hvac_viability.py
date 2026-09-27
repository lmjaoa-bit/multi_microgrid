from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import components.hvac_viability as hvac_viability
from components.commercial_hvac import CommercialHVACDR
from components.hvac_viability import (
    HVACForecastSnapshot,
    HVACViabilitySolverError,
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


def test_solver_failure_is_not_reported_as_feasible(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(hvac_viability, "_build_model", boom)
    with pytest.raises(HVACViabilitySolverError, match="solver failure"):
        solve_hvac_viability(
            23.0, 23.0, _snapshot(), config=_cfg(), delta_t=1.0
        )


def test_same_snapshot_is_cached_and_hidden_truth_cannot_change_bounds():
    cfg = {
        **_cfg(),
        "terminal_air_tolerance_c": 1.0,
        "terminal_mass_tolerance_c": 0.3,
        "temperature_column": "temperature_c",
        "ghi_column": "ghi_w_per_m2",
        "occupied_hours": list(range(7, 19)),
        "occupied_comfort_c": [21.0, 25.0],
        "unoccupied_comfort_c": [16.0, 30.0],
        "terminal_mode": "none",
        "forecast_viability": True,
    }
    device = CommercialHVACDR(cfg, delta_t=1.0, episode_length=24)
    device.t_air = 23.0
    device.t_mass = 23.0

    snap = _snapshot()
    first = device.prepare_viability(0, snap)

    # Mutating data that is not part of the frozen forecast snapshot must not
    # change the already-prepared interval.  This models hidden future truth.
    device._frame = object()
    second = device.prepare_viability(0, snap)
    assert second is first
    assert second.min_signed_power_kw == pytest.approx(first.min_signed_power_kw)
    assert second.max_signed_power_kw == pytest.approx(first.max_signed_power_kw)


def test_changed_snapshot_invalidates_hvac_viability_cache():
    cfg = {
        **_cfg(),
        "terminal_air_tolerance_c": 1.0,
        "terminal_mass_tolerance_c": 0.3,
        "temperature_column": "temperature_c",
        "ghi_column": "ghi_w_per_m2",
        "occupied_hours": list(range(7, 19)),
        "occupied_comfort_c": [21.0, 25.0],
        "unoccupied_comfort_c": [16.0, 30.0],
        "terminal_mode": "none",
        "forecast_viability": True,
    }
    device = CommercialHVACDR(cfg, delta_t=1.0, episode_length=24)
    device.t_air = 23.0
    device.t_mass = 23.0
    first = device.prepare_viability(0, _snapshot())

    changed = HVACForecastSnapshot(
        snapshot_id="changed",
        boundary_timestamps=_snapshot().boundary_timestamps,
        outside_temperature_c=np.full(3, -5.0),
        ghi_w_per_m2=np.zeros(3),
        base_load_kw=np.full(3, 10.0),
        interval_occupied=np.zeros(3, dtype=bool),
        comfort_low_c=np.full(3, 16.0),
        comfort_high_c=np.full(3, 30.0),
    )
    second = device.prepare_viability(0, changed)
    assert second is not first
    assert second.snapshot_id == "changed"
    assert device._viability_result is second
    assert device._viability_t == 0


def _device_cfg():
    return {
        **_cfg(),
        "temperature_column": "temperature_c",
        "ghi_column": "ghi_w_per_m2",
        "occupied_hours": [7],
        "occupied_comfort_c": [21.0, 25.0],
        "unoccupied_comfort_c": [16.0, 30.0],
        "terminal_air_tolerance_c": 1.0,
        "terminal_mass_tolerance_c": 0.3,
        "terminal_mode": "none",
        "forecast_viability": False,
    }


@pytest.mark.parametrize("q_hvac_kw", [-48.0, 0.0, 72.0])
def test_linearized_viability_step_matches_device_next_state_with_gains(q_hvac_kw):
    cfg = _device_cfg()
    device = CommercialHVACDR(cfg, delta_t=0.5, episode_length=2)
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01T06:00:00",
            "2026-01-01T06:30:00",
        ]),
        "temperature_c": [4.0, 3.0],
        "ghi_w_per_m2": [240.0, 180.0],
    })
    base_load = np.asarray([18.0, 18.0])
    device.reset(frame, base_load)
    air0, mass0 = 20.3, 21.1
    actual_air, actual_mass = device._next_state(
        air0, mass0, q_hvac_kw, 0
    )

    a, b, dt = thermal_linear_coefficients(cfg, 0.5)
    solar = cfg["solar_gain_kw_per_wm2"] * 240.0
    q_air = cfg["solar_air_fraction"] * solar
    q_mass = (1.0 - cfg["solar_air_fraction"]) * solar
    q_internal = (
        base_load[0] * cfg["internal_gain_unoccupied_fraction"]
    )
    affine = a @ np.asarray([air0, mass0]) + b * q_hvac_kw
    affine += np.asarray([
        dt / cfg["c_air_kwh_per_c"] * (
            4.0 / cfg["r_out_air_c_per_kw"] + q_internal + q_air
        ),
        dt / cfg["c_mass_kwh_per_c"] * q_mass,
    ])
    assert actual_air == pytest.approx(affine[0], abs=1e-10)
    assert actual_mass == pytest.approx(affine[1], abs=1e-10)


def test_future_occupied_boundary_requires_preheating_before_one_step_violation():
    cfg = _device_cfg()
    snapshot = HVACForecastSnapshot(
        snapshot_id="preheat-two-step",
        boundary_timestamps=(
            "2026-01-01T06:00:00",
            "2026-01-01T06:30:00",
            "2026-01-01T07:00:00",
        ),
        outside_temperature_c=np.asarray([0.0, 0.0]),
        ghi_w_per_m2=np.zeros(2),
        base_load_kw=np.zeros(2),
        interval_occupied=np.asarray([False, False]),
        # 06:30 is still unoccupied; 07:00 is occupied.
        comfort_low_c=np.asarray([16.0, 21.0]),
        comfort_high_c=np.asarray([30.0, 25.0]),
    )
    result = solve_hvac_viability(
        20.0, 20.0, snapshot, config=cfg, delta_t=0.5
    )
    assert result.feasible
    # With q0=0, the 06:30 state is still inside the instantaneous
    # unoccupied band, so a one-step-only guard would accept no heating.
    a, _, dt = thermal_linear_coefficients(cfg, 0.5)
    air_no_heat = (
        a @ np.asarray([20.0, 20.0])
        + np.asarray([
            dt / cfg["c_air_kwh_per_c"] * (
                0.0 / cfg["r_out_air_c_per_kw"]
            ),
            0.0,
        ])
    )[0]
    assert 16.0 <= air_no_heat <= 30.0
    # The full-horizon viability interval excludes that myopic action because
    # maximum heating in the next half-hour would otherwise miss 07:00.
    assert result.min_signed_power_kw > 0.0
