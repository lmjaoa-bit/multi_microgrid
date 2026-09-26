import numpy as np
import pandas as pd
import pytest

from components.ev_charging import EVChargingDR


def config():
    return dict(target_soc=.8, eta_charge=.93, max_charge_kw=93.6, scenario_seed=42,
                cohorts=[dict(group_id=f'G{i}', count=n, capacity_per_vehicle_kwh=60,
                              max_charge_per_vehicle_kw=7.2, arrival_hours=[a],
                              arrival_probabilities=[1], departure_hours=[d],
                              departure_probabilities=[1], need_per_vehicle_kwh=[14, 22])
                         for i, n, a, d in [(1, 6, 0, 7), (2, 4, 0, 10), (3, 3, 10, 15)]])


def device(date='2026-01-01'):
    ev = EVChargingDR(config(), delta_t=1, episode_length=24)
    ev.reset(pd.DataFrame({'timestamp': pd.date_range(date, periods=24, freq='h')}), np.zeros(24))
    return ev


def test_date_stable_sessions_and_hidden_future():
    ev = device()
    assert ev.sessions == device().sessions
    assert ev.sessions != device('2026-01-02').sessions
    assert all(value == 0 for key, value in ev.state(0).items() if key.startswith('ev_G3_'))
    assert ev.state(10)['ev_G3_connected'] == 1


@pytest.mark.parametrize('action', [-1., 0., 1.])
def test_feasible_full_day_energy_and_last_interval(action):
    ev = device()
    energy = 0
    for t in range(24):
        plan = ev.plan(action, t)
        assert plan['feasible_min_kw'] - 1e-9 <= plan['effective_power_kw'] <= plan['feasible_max_kw'] + 1e-9
        result = ev.commit(plan)
        energy += result['served_charge_kw'] * ev.eta
    assert energy == pytest.approx(sum(s['demand_kwh'] for s in ev.sessions))
    assert len(ev.departure_events) == 3
    assert sum(e['shortfall_kwh'] for e in ev.departure_events) < 1e-8
    assert ev.feasible_bounds(24) == (0, 0)


def test_partial_supply_shortfall_once_and_infeasible_floor_clamped():
    ev = device()
    total = 0
    for t in range(24):
        plan = ev.plan(-1, t)
        assert plan['feasible_min_kw'] <= plan['feasible_max_kw']
        result = ev.commit(plan, delivered_fraction=0)
        total += result['ev_departure_shortfall_kwh']
        if t == 6:
            assert result['ev_departure_shortfall_kwh'] == pytest.approx(ev.sessions[0]['demand_kwh'])
        if t == 7:
            assert result['ev_departure_shortfall_kwh'] == 0
    assert total == pytest.approx(sum(s['demand_kwh'] for s in ev.sessions))
    assert ev.first_infeasible_hour is not None
