import numpy as np
import pandas as pd
import pytest

from components.commercial_hvac import CommercialHVACDR
from components.controllable_load import build_controllable_load
from components.ev_charging import EVChargingDR
from components.industrial_mill import IndustrialMillDR


def day_frame(*, tout=20.0, ghi=0.0):
    timestamps = pd.date_range("2018-01-02", periods=24, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "mg2_temperature_c": np.full(24, float(tout)),
        "mg2_ghi_w_per_m2": np.full(24, float(ghi)),
    })


MG1 = {
    "type": "industrial_mill",
    "rated_power_kw": 28.5,
    "rated_production_tph": 1.186,
    "downstream_consumption_tph": 0.975,
    "silo_capacity_t": 17.55,
    "silo_min_t": 3.51,
    "silo_max_t": 15.80,
    "silo_initial_t": 10.53,
    "terminal_tolerance_t": 0.20,
}

MG2 = {
    "type": "commercial_hvac",
    "r_out_air_c_per_kw": 0.22,
    "r_air_mass_c_per_kw": 0.08,
    "c_air_kwh_per_c": 15.0,
    "c_mass_kwh_per_c": 250.0,
    "heating_max_kw": 35.0,
    "cooling_max_kw": 35.0,
    "cop_heat": 3.3,
    "cop_cool": 3.2,
    "temperature_column": "mg2_temperature_c",
    "ghi_column": "mg2_ghi_w_per_m2",
    "occupied_hours": list(range(7, 19)),
    "occupied_comfort_c": [21.0, 25.0],
    "unoccupied_comfort_c": [16.0, 30.0],
    "terminal_air_tolerance_c": 1.0,
    "terminal_mass_tolerance_c": 0.3,
    "solar_gain_kw_per_wm2": 0.025,
    "solar_air_fraction": 0.20,
    "internal_gain_occupied_fraction": 0.50,
    "internal_gain_unoccupied_fraction": 0.20,
}

MG3 = {
    "type": "ev_charging",
    "fleet_capacity_kwh": 780.0,
    "initial_soc": 0.50,
    "target_soc": 0.80,
    "max_soc": 0.80,
    "max_charge_kw": 93.6,
    "eta_charge": 0.93,
    "available_start_hour": 0,
    "departure_hour": 7,
}


def test_controllable_load_factory_builds_heterogeneous_devices():
    assert isinstance(build_controllable_load(MG1, delta_t=1.0, episode_length=24), IndustrialMillDR)
    assert isinstance(build_controllable_load(MG2, delta_t=1.0, episode_length=24), CommercialHVACDR)
    assert isinstance(build_controllable_load(MG3, delta_t=1.0, episode_length=24), EVChargingDR)
    with pytest.raises(ValueError, match="Unsupported controllable load"):
        build_controllable_load({"type": "unknown"}, delta_t=1.0, episode_length=24)


def test_mill_full_duty_updates_inventory_and_power():
    mill = IndustrialMillDR(MG1, delta_t=1.0, episode_length=24)
    mill.reset(day_frame(), np.full(24, 48.6))
    assert mill.silo_t == pytest.approx(10.53)
    lo, hi = mill.feasible_bounds(0)
    assert 0.0 <= lo <= hi <= 1.0
    plan = mill.plan(1.0, 0)
    assert plan["requested_duty"] == pytest.approx(1.0)
    assert plan["effective_duty"] <= hi + 1e-12
    assert plan["flexible_load_kw"] == pytest.approx(28.5 * plan["effective_duty"])
    old = mill.silo_t
    mill.commit(plan)
    assert mill.silo_t == pytest.approx(old + 1.186 * plan["effective_duty"] - 0.975)


def test_mill_last_hour_projection_preserves_terminal_silo_band():
    mill = IndustrialMillDR(MG1, delta_t=1.0, episode_length=24)
    mill.reset(day_frame(), np.full(24, 48.6))
    mill.silo_t = 10.53
    lo, hi = mill.feasible_bounds(23)
    for raw in (-1.0, 1.0):
        plan = mill.plan(raw, 23)
        assert lo - 1e-12 <= plan["effective_duty"] <= hi + 1e-12
        terminal = 10.53 + 1.186 * plan["effective_duty"] - 0.975
        assert abs(terminal - 10.53) <= 0.20 + 1e-9


def test_ev_forces_charging_early_enough_to_hit_departure_target():
    ev = EVChargingDR(MG3, delta_t=1.0, episode_length=24)
    ev.reset(day_frame(), np.full(24, 57.52))
    assert ev.soc == pytest.approx(0.50)
    for t in range(7):
        lo, hi = ev.feasible_bounds(t)
        assert 0.0 <= lo <= hi <= 93.6 + 1e-9
        ev.commit(ev.plan(-1.0, t))
    assert ev.soc == pytest.approx(0.80, abs=1e-9)
    assert ev.feasible_bounds(7) == pytest.approx((0.0, 0.0))
    assert ev.state(7)["deadline_pressure"] == pytest.approx(0.0)


def test_ev_never_charges_above_target_or_after_departure():
    ev = EVChargingDR(MG3, delta_t=1.0, episode_length=24)
    ev.reset(day_frame(), np.full(24, 57.52))
    for t in range(7):
        ev.commit(ev.plan(1.0, t))
    assert ev.soc <= 0.80 + 1e-9
    after = ev.plan(1.0, 8)
    assert after["flexible_load_kw"] == pytest.approx(0.0)


def test_hvac_initial_state_is_weather_dependent_but_buffered():
    hvac = CommercialHVACDR(MG2, delta_t=1.0, episode_length=24)
    hvac.reset(day_frame(tout=5.0), np.full(24, 75.0))
    assert hvac.t_air == pytest.approx(19.4)
    assert hvac.t_mass == pytest.approx(21.2)
    assert hvac.initial_t_air == pytest.approx(19.4)
    assert hvac.initial_t_mass == pytest.approx(21.2)


def test_hvac_next_state_uses_next_clock_comfort_band():
    hvac = CommercialHVACDR(MG2, delta_t=1.0, episode_length=24)
    hvac.reset(day_frame(tout=20.0), np.full(24, 75.0))
    # The state reached after interval 06:00-07:00 must already satisfy the
    # occupied band at 07:00; the state reached at 19:00 is unoccupied.
    assert hvac._state_comfort_bounds(7) == pytest.approx((21.0, 25.0))
    assert hvac._state_comfort_bounds(19) == pytest.approx((16.0, 30.0))
    assert hvac._state_comfort_bounds(24) == pytest.approx((16.0, 30.0))


def test_hvac_signed_control_stays_within_physical_cap_and_comfort_projection():
    hvac = CommercialHVACDR(MG2, delta_t=1.0, episode_length=24)
    hvac.reset(day_frame(tout=5.0), np.full(24, 75.0))
    lo, hi = hvac.feasible_bounds(0)
    assert -35.0 <= lo <= hi <= 35.0
    heating = hvac.plan(1.0, 0)
    assert 0.0 <= heating["effective_signed_power_kw"] <= 35.0
    assert heating["flexible_load_kw"] == pytest.approx(abs(heating["effective_signed_power_kw"]))
    old_air = hvac.t_air
    hvac.commit(heating)
    assert hvac.t_air >= old_air - 1e-9


def test_hvac_feasible_bound_cache_avoids_duplicate_highs_solves():
    hvac = CommercialHVACDR(MG2, delta_t=1.0, episode_length=24)
    hvac.reset(day_frame(tout=18.0), np.full(24, 75.0))
    calls = 0
    original = hvac._solve_terminal_q

    def counted(t, *, maximize):
        nonlocal calls
        calls += 1
        return original(t, maximize=maximize)

    hvac._solve_terminal_q = counted
    first = hvac.feasible_bounds(0)
    second = hvac.feasible_bounds(0)
    assert first == pytest.approx(second)
    assert calls == 2
    hvac.commit(hvac.plan(0.0, 0))
    hvac.feasible_bounds(1)
    assert calls == 4


def test_hvac_terminal_reachability_keeps_end_state_in_daily_band():
    hvac = CommercialHVACDR(MG2, delta_t=1.0, episode_length=24)
    frame = day_frame(tout=18.0, ghi=0.0)
    hvac.reset(frame, np.full(24, 75.0))
    for t in range(24):
        plan = hvac.plan(-1.0 if t % 2 == 0 else 1.0, t)
        hvac.commit(plan)
    if not hvac.terminal_thermal_infeasible:
        assert abs(hvac.t_air - hvac.initial_t_air) <= 1.0 + 1e-6
        assert abs(hvac.t_mass - hvac.initial_t_mass) <= 0.3 + 1e-6
