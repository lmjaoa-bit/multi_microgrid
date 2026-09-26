"""Date-stable equivalent EV sessions and their revealed observation contract."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

GROUP_FIELDS = ('revealed', 'connected', 'departed', 'remaining_energy',
                'hours_to_departure', 'rated_power', 'minimum_power', 'maximum_power')


def cohort_schema(config: Mapping[str, Any]) -> list[str]:
    return [f"ev_{g['group_id']}_{field}" for g in config['cohorts'] for field in GROUP_FIELDS] + [
        'ev_connected_remaining', 'ev_feasible_min', 'ev_feasible_max', 'ev_station_limit']


def cohort_values(config: Mapping[str, Any], state: Mapping[str, float]) -> list[float]:
    station = float(config['max_charge_kw'])
    values = []
    for group in config['cohorts']:
        capacity = float(group['count']) * float(group['capacity_per_vehicle_kwh'])
        for field, scale in zip(GROUP_FIELDS, [1, 1, 1, capacity, 24, station, station, station]):
            values.append(float(state[f"ev_{group['group_id']}_{field}"]) / scale)
    total_capacity = sum(float(g['count']) * float(g['capacity_per_vehicle_kwh']) for g in config['cohorts'])
    values.extend([state['ev_remaining_battery_kwh'] / total_capacity,
                   state['ev_feasible_min_kw'] / station, state['ev_feasible_max_kw'] / station, 1.0])
    return values


def generate_sessions(config: Mapping[str, Any], date: str) -> list[dict[str, Any]]:
    version = 'ev-cohorts-v1'
    key = f"{version}|{date}|{int(config.get('scenario_seed', 42))}"
    digest = hashlib.sha256(key.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:16], 'little'))
    sessions = []
    for group in config['cohorts']:
        def sample(name: str) -> int:
            return int(rng.choice(group[f'{name}_hours'], p=group.get(f'{name}_probabilities')))
        arrival = sample('arrival')
        departure = sample('departure') if 'departure_hours' in group else arrival + sample('dwell')
        need = float(rng.uniform(*group['need_per_vehicle_kwh']))
        count = int(group['count'])
        capacity = float(group['capacity_per_vehicle_kwh']) * count
        power = float(group['max_charge_per_vehicle_kw']) * count
        demand = need * count
        eta = float(config['eta_charge'])
        target = float(config['target_soc'])
        initial = target - demand / capacity
        if not 0 <= arrival < departure <= 24 or initial < 0 or demand > eta * power * (departure - arrival) + 1e-8:
            raise ValueError('Infeasible EV cohort session')
        sessions.append(dict(group_id=str(group['group_id']), count=count, date=date,
                             scenario_id=digest.hex()[:16], generator_version=version,
                             arrival_hour=arrival, departure_hour=departure, demand_kwh=demand,
                             capacity_kwh=capacity, max_charge_kw=power, initial_soc=initial,
                             target_soc=target, eta_charge=eta))
    return sessions
