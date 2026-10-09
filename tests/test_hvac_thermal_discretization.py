from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from components.commercial_hvac import CommercialHVACDR
from components.hvac_viability import HVACForecastSnapshot, solve_hvac_viability


def config(mode="exact_zoh"):
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    dr = deepcopy(cfg["microgrids"][1]["dr"])
    dr["thermal_discretization"] = mode
    return dr


def device(*, dt=1.0, outside=23.0, ghi=0.0, base=0.0, mode="exact_zoh"):
    cfg = config(mode)
    hvac = CommercialHVACDR(cfg, delta_t=dt, episode_length=24)
    frame = pd.DataFrame({"timestamp": pd.date_range("2018-01-02", periods=24, freq="h"),
                          cfg["temperature_column"]: outside, cfg["ghi_column"]: ghi})
    hvac.reset(frame, np.full(24, base))
    return hvac


@pytest.mark.parametrize("thermal_power", [33.0, -32.0])
def test_heat_and_cool_pulses_decay_without_crossing_ambient(thermal_power):
    hvac = device()
    state = (23.0, 23.0)
    for hour in range(24):
        state = hvac._next_state(*state, thermal_power if hour == 0 else 0.0, hour)
        deviation = np.asarray(state) - 23.0
        assert (deviation * np.sign(thermal_power) >= -1e-12).all()


@pytest.mark.parametrize("dt,thermal_power", [(0.25, -80.0), (1.0, 65.0), (3.0, 0.0)])
def test_exact_step_matches_independent_continuous_ode_rk4(dt, thermal_power):
    hvac = device(dt=dt, outside=5.0, ghi=650.0, base=80.0)
    state = np.array([22.0, 24.0])
    # Integrate the continuous heat balance independently, without the discrete helper.
    solar = 0.025 * 650.0
    internal = 80.0 * 0.20

    def derivative(x):
        air, mass = x
        return np.array([((5.0-air)/0.22 + (mass-air)/0.08 + internal + 0.20*solar + thermal_power)/15.0,
                         ((air-mass)/0.08 + 0.80*solar)/250.0])

    expected = state.copy()
    step = dt / 2000
    for _ in range(2000):
        k1 = derivative(expected)
        k2 = derivative(expected + step*k1/2)
        k3 = derivative(expected + step*k2/2)
        k4 = derivative(expected + step*k3)
        expected += step*(k1 + 2*k2 + 2*k3 + k4)/6
    actual = hvac._next_state(*state, thermal_power, 0)
    assert actual == pytest.approx(expected, abs=1e-9)


def test_exact_step_does_not_depend_on_internal_step_partition():
    full = device(outside=10.0, ghi=400.0, base=60.0)
    half = device(dt=0.5, outside=10.0, ghi=400.0, base=60.0)
    initial = (24.0, 22.0)
    first = half._next_state(*initial, -60.0, 0)
    second = half._next_state(*first, -60.0, 0)
    assert full._next_state(*initial, -60.0, 0) == pytest.approx(second, abs=1e-11)


def test_missing_mode_preserves_historical_euler_snapshots():
    cfg = config()
    cfg.pop("thermal_discretization")
    hvac = CommercialHVACDR(cfg, delta_t=1.0, episode_length=24)
    frame = pd.DataFrame({"timestamp": pd.date_range("2018-01-02", periods=24, freq="h"),
                          cfg["temperature_column"]: 23.0, cfg["ghi_column"]: 0.0})
    hvac.reset(frame, np.zeros(24))
    first = hvac._next_state(23.0, 23.0, 33.0, 0)
    second = hvac._next_state(*first, 0.0, 1)
    assert first == pytest.approx((25.2, 23.0))
    assert second == pytest.approx((22.7, 23.11))


def test_unknown_discretization_is_rejected():
    with pytest.raises(ValueError, match="thermal_discretization"):
        device(mode="misspelled_exact")


def test_exact_one_step_bounds_and_viability_use_the_device_response():
    hvac = device(outside=20.0, base=20.0)
    hvac.control_semantics = "legacy_comfort_projection"
    hvac.t_air, hvac.t_mass = 20.0, 22.0
    lower_q, upper_q = hvac._one_step_bounds_q(6)
    assert hvac._next_state(20.0, 22.0, lower_q, 6)[0] == pytest.approx(21.0, abs=1e-8)
    assert hvac._next_state(20.0, 22.0, upper_q, 6)[0] == pytest.approx(25.0, abs=1e-8)
    snapshot = HVACForecastSnapshot("exact-single", ("2018-01-02T06:00", "2018-01-02T07:00"),
                                    np.array([20.0]), np.zeros(1), np.array([20.0]), np.array([False]),
                                    np.array([21.0]), np.array([25.0]))
    result = solve_hvac_viability(20.0, 22.0, snapshot, config=hvac.config, delta_t=1.0)
    assert result.feasible
    assert result.min_signed_power_kw == pytest.approx(hvac._q_to_signed_electric(lower_q), abs=1e-8)
    assert result.max_signed_power_kw == pytest.approx(hvac._q_to_signed_electric(upper_q), abs=1e-8)


def test_exact_forecast_constraints_propagate_heat_into_both_nodes():
    import components.hvac_viability as viability

    hvac = device(outside=10.0, ghi=500.0, base=40.0)
    snapshot = HVACForecastSnapshot("exact-multi", tuple(f"2018-01-02T0{k}:00" for k in range(4)),
                                    np.full(3, 10.0), np.full(3, 500.0), np.full(3, 40.0),
                                    np.zeros(3, dtype=bool), np.full(3, -100.0), np.full(3, 100.0))
    model, q, air, mass, _ = viability._build_model(23.0, 22.0, snapshot, config=hvac.config,
                                                   delta_t=1.0, relaxed=False)
    powers = [33.0, -32.0, 0.0]
    for var, power in zip(q, powers):
        model.addConstr(var == power)
    model.minimize(q[0])
    assert "optimal" in model.modelStatusToString(model.getModelStatus()).lower()
    state = (23.0, 22.0)
    for t, power in enumerate(powers):
        state = hvac._next_state(*state, power, t)
        assert [model.val(air[t+1]), model.val(mass[t+1])] == pytest.approx(state, abs=1e-9)


@pytest.mark.parametrize("mode", ["legacy_euler", "exact_zoh"])
@pytest.mark.parametrize("day", ["2018-01-28", "2018-07-28"])
def test_milp_thermal_trajectory_replays_in_the_same_physics(mode, day):
    from optimization.common_constraints import build_dispatch_data
    from optimization.cooperative_dispatch import solve_generalized_dispatch_reference

    root = Path(__file__).parents[1]
    cfg = yaml.safe_load((root / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(root)
    cfg["microgrids"][1]["dr"]["thermal_discretization"] = mode
    _, data = build_dispatch_data(cfg, split="test", day=day)
    result = solve_generalized_dispatch_reference(cfg, split="test", day=day, solver_output=False)
    assert result.metadata["hvac_thermal_discretization"] == {"mg2": mode}
    hvac = CommercialHVACDR(cfg["microgrids"][1]["dr"], delta_t=1.0, episode_length=24)
    hvac.reset(data.frame, np.asarray(data.load_kw["mg2"]))
    state = (hvac.initial_t_air, hvac.initial_t_mass)
    rows = result.trajectories[result.trajectories.microgrid == "mg2"].reset_index(drop=True)
    for t, row in rows.iterrows():
        heat = float(row.hvac_heating_kw)
        cool = float(row.hvac_cooling_kw)
        state = hvac._next_state(*state, hvac.cop_heat*heat - hvac.cop_cool*cool, t)
        assert state == pytest.approx((row.hvac_air_after_c, row.hvac_mass_after_c), abs=1e-7)
