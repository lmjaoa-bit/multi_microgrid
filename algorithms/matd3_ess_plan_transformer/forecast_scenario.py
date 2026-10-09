"""Causal, materialized forecast scenarios using the unchanged environment physics."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from components.controllable_load import build_controllable_load
from core.forecast import SharedForecastProvider
from core.global_state import GlobalStateBuilder
from core.observation import ObservationBuilder
from core.joint_action_projection import JointActionProjector
from core.tariffs import buy_price_column, sell_price_column, prices_from_row


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json,
                                    allow_nan=False).encode()).hexdigest()


class ForecastContextRejected(ValueError):
    pass


@dataclass(frozen=True)
class ForecastContext:
    t: int
    agent_ids: tuple[str, ...]
    observations: dict[str, np.ndarray]
    global_state: np.ndarray
    plan_mask: np.ndarray
    context_hash: str
    forecast_hash: str
    contract_hash: str
    metadata: dict
    template: object = field(repr=False, compare=False)


def _median(hours, probabilities=None):
    pairs = sorted(zip(hours, probabilities or [1 / len(hours)] * len(hours)))
    total = sum(float(p) for _, p in pairs)
    accum = 0.0
    for hour, p in pairs:
        accum += float(p)
        if accum >= total / 2 - 1e-12:
            return int(hour)
    raise ForecastContextRejected('Empty EV conditional forecast')


def _visible_ev(device, t):
    public = device.state(t)
    sessions, remaining, delivered = [], {}, {}
    for group in device.config['cohorts']:
        key = str(group['group_id'])
        if public[f'ev_{key}_revealed']:
            source = next(s for s in device.sessions if s['group_id'] == key)
            session = copy.deepcopy(source)
            remaining[key] = float(device.remaining[key])
            delivered[key] = float(device.delivered_kwh[key])
        else:
            choices = list(zip(group['arrival_hours'], group.get('arrival_probabilities') or
                               [1 / len(group['arrival_hours'])] * len(group['arrival_hours'])))
            choices = [(h, p) for h, p in choices if int(h) > t and p > 0]
            if not choices:
                raise ForecastContextRejected('Empty EV conditional forecast')
            arrival = _median([h for h, _ in choices], [p for _, p in choices])
            departure = (_median(group['departure_hours'], group.get('departure_probabilities'))
                         if 'departure_hours' in group else
                         arrival + _median(group['dwell_hours'], group.get('dwell_probabilities')))
            count = int(group['count'])
            capacity = count * float(group['capacity_per_vehicle_kwh'])
            power = count * float(group['max_charge_per_vehicle_kw'])
            need = count * float(np.mean(group['need_per_vehicle_kwh']))
            eta, target = device.eta, device.target_soc
            if not (t < arrival < departure <= 24) or need > min(
                    capacity * target, eta * power * (departure - arrival)) + 1e-8:
                raise ForecastContextRejected('Invalid nominal EV forecast')
            session = dict(group_id=key, count=count, arrival_hour=arrival,
                           departure_hour=departure, demand_kwh=need, capacity_kwh=capacity,
                           max_charge_kw=power, initial_soc=target - need / capacity,
                           target_soc=target, eta_charge=eta, date='forecast',
                           scenario_id='nominal', generator_version='nominal-conditional-v1')
            remaining[key], delivered[key] = need, 0.0
        sessions.append(session)
    return sessions, remaining, delivered


def capture_forecast_context(env, observations, global_state, metadata) -> ForecastContext:
    if env.split != 'train' or not 0 <= env.t < 24:
        raise ForecastContextRejected('Planning context requires an active train episode')
    t, ids = int(env.t), tuple(env.agent_ids)
    obs = {mg: np.asarray(observations[mg], dtype=np.float32).copy() for mg in ids}
    state = np.asarray(global_state, dtype=np.float32).copy()
    if any(not np.isfinite(x).all() for x in [state, *obs.values()]):
        raise ForecastContextRejected('Nonfinite planning observation')
    # Only this current row is read from the realized scenario. All other rows
    # are constructed from the snapshot, including device-specific frame caches.
    current = env.episode_df.iloc[t].copy()
    anchor = pd.Timestamp(current['timestamp']) - pd.Timedelta(hours=t)
    frame = pd.DataFrame([current.to_dict() for _ in range(49)])
    frame['timestamp'] = pd.date_range(anchor, periods=len(frame), freq='h')
    actual = env._exogenous_at(t)
    exogenous = {mg: {key: np.full(len(frame), float(value)) for key, value in actual[mg].items()}
                 for mg in ids}
    for k in range(1, 24 - t):
        if env.forecasts.valid(t, k) < .5:
            raise ForecastContextRejected('Missing valid forecast before episode end')
        for mg in ids:
            public = env.forecasts.public(t, k)
            local = env.forecasts.local(mg, t, k)
            forecast = {**public, **local}
            if not all(np.isfinite(v) for v in [*local.values(), *public.values()]):
                raise ForecastContextRejected('Nonfinite forecast')
            pos = t + k
            for name in ('load_kw', 'pv_kw', 'wind_kw'):
                exogenous[mg][name][pos] = forecast[name]
            exogenous[mg]['renewable_kw'][pos] = forecast['pv_kw'] + forecast['wind_kw']
            frame.loc[pos, buy_price_column(mg)] = forecast['grid_price_per_kwh']
            frame.loc[pos, sell_price_column(mg)] = forecast['grid_sell_price_per_kwh']
            if 'grid_price_per_kwh' in public:
                frame.loc[pos, 'buy_price_per_kwh'] = public['grid_price_per_kwh']
                frame.loc[pos, 'sell_price_per_kwh'] = forecast['grid_sell_price_per_kwh']
            dr_cfg = env.cfg_by_id[mg]['dr']
            if dr_cfg['type'] == 'commercial_hvac':
                frame.loc[pos, dr_cfg['temperature_column']] = forecast['temperature_c']
                frame.loc[pos, dr_cfg['ghi_column']] = forecast['ghi_w_per_m2']
            frame.loc[pos, 'ghi_w_per_m2'] = public['ghi_w_per_m2']
            frame.loc[pos, 'wind_speed_mps'] = public['wind_speed_mps']
    # Current heterogeneous tariffs remain exact; padding never reads tomorrow.
    for mg in ids:
        buy, sell = prices_from_row(current, mg)
        frame.loc[:t, buy_price_column(mg)] = buy
        frame.loc[:t, sell_price_column(mg)] = sell
        for values in exogenous[mg].values():
            values[24:] = values[23]
    frame.loc[24:, :] = frame.loc[23].to_numpy()
    frame['timestamp'] = pd.date_range(anchor, periods=len(frame), freq='h')
    # carbon is persistence because no public future carbon channel exists.
    frame['grid_carbon_factor_kg_per_kwh'] = float(current['grid_carbon_factor_kg_per_kwh'])

    worker = copy.copy(env)
    worker.config = copy.deepcopy(env.config)
    worker.cfg_by_id = {str(c['id']): c for c in worker.config['microgrids']}
    worker.agent_ids = list(ids)
    worker.data = SimpleNamespace(data_quality_flags=[], rng=np.random.default_rng(0))
    profiles = copy.copy(env.profiles)
    profiles.config = worker.config
    profiles.mg_cfg = worker.cfg_by_id
    profiles.train_frame = pd.DataFrame()
    profiles._exogenous_cache = {mg: {int(timestamp.value): tuple(float(exogenous[mg][key][i])
        for key in ('load_kw','pv_kw','wind_kw','renewable_kw'))
        for i,timestamp in enumerate(frame['timestamp'])} for mg in ids}
    worker.profiles = profiles
    worker.forecasts = SharedForecastProvider(worker.config, profiles, seed=0)
    worker.forecasts.begin_episode(frame, seed=0, deterministic=True, forecast_noise=False,
                                  exogenous_by_microgrid=exogenous)
    worker.obs_builder = ObservationBuilder(worker.config, profiles, worker.forecasts, env.delta_t)
    worker.global_state_builder = GlobalStateBuilder(worker.config, profiles, worker.forecasts, env.delta_t)
    worker.cores = copy.deepcopy(env.cores)
    for mg in ids:
        worker.cores[mg].profiles = profiles
    worker.dr_devices = {}
    devices_payload = {}
    for mg in ids:
        source = env.dr_devices[mg]
        target = build_controllable_load(worker.cfg_by_id[mg]['dr'], delta_t=env.delta_t,
                                         episode_length=24)
        target._frame = frame.iloc[:24].copy()
        target._base_load_kw = exogenous[mg]['load_kw'][:24].copy()
        kind = worker.cfg_by_id[mg]['dr']['type']
        if kind == 'ev_charging' and target.cohort_mode:
            target.sessions, target.remaining, target.delivered_kwh = _visible_ev(source, t)
            target._departed = copy.deepcopy(source._departed)
            target.departure_events = copy.deepcopy(source.departure_events)
            target.first_infeasible_hour = source.first_infeasible_hour
            target.target_missed = source.target_missed
            target._update_soc()
            payload = dict(sessions=target.sessions, remaining=target.remaining,
                           delivered=target.delivered_kwh, departed=sorted(target._departed),
                           events=target.departure_events)
        else:
            values = {name: copy.deepcopy(value) for name, value in vars(source).items()
                      if isinstance(value, (float, int, bool, str, type(None)))
                      and not name.startswith('_')}
            for name, value in values.items():
                setattr(target, name, value)
            payload = values
        worker.dr_devices[mg] = target
        devices_payload[mg] = payload
    worker.grids = copy.deepcopy(env.grids)
    worker.energy_pool = copy.deepcopy(env.energy_pool)
    worker.accounting = copy.deepcopy(env.accounting)
    worker.action_codec = copy.deepcopy(env.action_codec)
    worker.action_projector = JointActionProjector(worker.config, worker.global_state_builder.schema(),
        worker.global_state_builder.norm, worker.global_state_builder.price_norm)
    worker.dispatch_projector = worker.action_projector
    worker.episode_context_df = frame
    worker.episode_df = frame.iloc[:24].copy()
    worker._episode_exogenous = exogenous
    worker.history = copy.deepcopy(env.history)
    worker._last_observations = copy.deepcopy(obs)
    worker._last_global_state = state.copy()
    worker._terminated = False
    forecast_hash = digest(dict(frame=frame.to_dict('list'), exogenous=exogenous))
    contract_hash = digest(dict(config=worker.config, normalizers=env.obs_builder.norm,
                                rule='causal-forecast-local-weather-conditional-ev-v2'))
    context_hash = digest(dict(t=t, state=state, obs=obs, forecast=forecast_hash,
        contract=contract_hash, devices=devices_payload, history=worker.history,
        peak=worker.daily_load_peak_kw))
    mask = np.arange(24) < 24 - t
    for value in [state, mask, *obs.values()]:
        value.setflags(write=False)
    return ForecastContext(t, ids, obs, state, mask, context_hash, forecast_hash,
                           contract_hash, copy.deepcopy(dict(metadata)), worker)


def build_forecast_environment(context: ForecastContext):
    return copy.deepcopy(context.template)
