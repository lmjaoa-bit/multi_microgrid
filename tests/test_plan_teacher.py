from copy import deepcopy
from pathlib import Path
import pickle
import hashlib

import numpy as np
import pytest
import yaml

from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case():
    cfg = yaml.safe_load((ROOT / 'configs/cooperative.yaml').read_text(encoding='utf-8'))
    cfg['_project_root'] = str(ROOT)
    env = MultiMicrogridEnv(cfg, split='train')
    obs, info = env.reset(seed=17, deterministic=True, forecast_noise=False)
    return env, obs, info['global_state']


def fingerprint(env):
    return hashlib.sha256(pickle.dumps(env.__dict__, protocol=5)).hexdigest()


def test_context_is_pure_and_forecast_environment_restores_current_state(case):
    from algorithms.matd3_plan_transformer.forecast_scenario import capture_planning_context, build_forecast_environment
    env, obs, state = case
    before = fingerprint(env)
    context = capture_planning_context(env, obs, state, {'day': 'train'})
    worker = build_forecast_environment(context)
    assert fingerprint(env) == before
    assert worker.t == env.t
    assert worker.daily_load_peak_kw == env.daily_load_peak_kw
    assert context.plan_mask.sum() == 24
    np.testing.assert_array_equal(worker.global_state(), state)
    actions = {mg: np.zeros(2, dtype=np.float32) for mg in env.agent_ids}
    worker.step(actions)
    assert env.t == 0
    assert fingerprint(env) == before


def test_context_does_not_read_real_future_or_unrevealed_ev(case):
    from algorithms.matd3_plan_transformer.forecast_scenario import capture_planning_context, build_forecast_environment
    env, obs, state = case
    original = capture_planning_context(env, obs, state, {})
    env.episode_df.loc[1:, 'ghi_w_per_m2'] = 777
    env.episode_context_df.loc[1:, 'grid_carbon_factor_kg_per_kwh'] = 777
    for values in env._episode_exogenous.values():
        for arr in values.values():
            arr[1:] = 999
    env.forecasts._public_arrays['ghi_w_per_m2'][1:] = 999
    for session in env.dr_devices['mg3'].sessions:
        if session['group_id'] == 'G3':
            session['demand_kwh'] = 700
            env.dr_devices['mg3'].remaining['G3'] = 700
    changed = capture_planning_context(env, obs, state, {})
    assert original.context_hash == changed.context_hash
    assert original.forecast_hash == changed.forecast_hash
    a, b = build_forecast_environment(original), build_forecast_environment(changed)
    for _ in range(24):
        actions = {mg: np.zeros(2, dtype=np.float32) for mg in env.agent_ids}
        _, ra, _, ia = a.step(actions)
        _, rb, _, ib = b.step(actions)
        assert ra == rb
        np.testing.assert_allclose(ia['global_state_after'], ib['global_state_after'])


def test_context_day_end_and_midday_state(case):
    from algorithms.matd3_plan_transformer.forecast_scenario import capture_planning_context, build_forecast_environment
    env, obs, state = case
    for _ in range(23):
        obs, _, _, info = env.step({mg: np.zeros(2) for mg in env.agent_ids})
        state = info['global_state_after']
    context = capture_planning_context(env, obs, state, {})
    assert context.plan_mask.sum() == 1
    worker = build_forecast_environment(context)
    assert worker.t == 23
    assert worker.dr_devices['mg1'].silo_t == env.dr_devices['mg1'].silo_t
    assert worker.dr_devices['mg2'].t_air == env.dr_devices['mg2'].t_air
    assert worker.dr_devices['mg2'].t_mass == env.dr_devices['mg2'].t_mass
    assert worker.daily_load_peak_kw == env.daily_load_peak_kw
    assert worker.step({mg: np.zeros(2) for mg in env.agent_ids})[2]


def test_context_rejects_missing_valid_forecast(case):
    from algorithms.matd3_plan_transformer.forecast_scenario import capture_planning_context, PlanningContextRejected
    env, obs, state = case
    env.forecasts._valid_array[0, 0] = 0
    with pytest.raises(PlanningContextRejected, match='forecast'):
        capture_planning_context(env, obs, state, {})


def test_teacher_preserves_local_hvac_weather_separately_from_public_forecast(case):
    from algorithms.matd3_plan_transformer.forecast_scenario import capture_planning_context, build_forecast_environment
    env, obs, state = case
    worker = build_forecast_environment(capture_planning_context(env, obs, state, {}))
    dr = env.cfg_by_id['mg2']['dr']
    for k in (6,12,18):
        local, public = env.forecasts.local('mg2',0,k), env.forecasts.public(0,k)
        row = worker.episode_df.iloc[k]
        assert row[dr['ghi_column']] == pytest.approx(local['ghi_w_per_m2'])
        assert row[dr['temperature_column']] == pytest.approx(local['temperature_c'])
        assert row['ghi_w_per_m2'] == pytest.approx(public['ghi_w_per_m2'])


def test_teacher_profile_queries_match_frozen_exogenous_arrays(case):
    from algorithms.matd3_plan_transformer.forecast_scenario import capture_planning_context, build_forecast_environment
    env, obs, state = case
    worker = build_forecast_environment(capture_planning_context(env, obs, state, {}))
    for k in (6,12,18):
        for mg in env.agent_ids:
            query = worker.profiles.exogenous(mg,worker.episode_df.iloc[k])
            for field in ('load_kw','pv_kw','wind_kw','renewable_kw'):
                assert query[field] == pytest.approx(worker._episode_exogenous[mg][field][k])


def test_teacher_is_reproducible_cached_and_physically_attainable(case):
    from algorithms.matd3_plan_transformer.forecast_scenario import capture_planning_context, build_forecast_environment
    from algorithms.matd3_plan_transformer.plan_teacher import PlanTeacher
    env, obs, state = case
    before = fingerprint(env)
    ctx = capture_planning_context(env, obs, state, {})
    settings = dict(population=4, elite=2, iterations=1, max_labels=2)
    teacher = PlanTeacher(settings, seed=19)
    label = teacher.solve(ctx)
    other = PlanTeacher(settings, seed=19).solve(ctx)
    assert label.valid
    assert label.best_cost <= min(label.baseline_costs) + 1e-7
    np.testing.assert_array_equal(label.actions, other.actions)
    assert teacher.solve(ctx) is label
    assert teacher.solve_calls == 1
    worker = build_forecast_environment(ctx)
    cost = 0.0
    for j in range(24):
        _, _, _, info = worker.step(dict(zip(env.agent_ids, label.actions[j])))
        cost += info['generalized_system_cost']
        for mg in env.agent_ids:
            assert label.soc_after[mg][j] == pytest.approx(worker.cores[mg].battery.soc)
    assert cost == pytest.approx(label.best_cost)
    assert fingerprint(env) == before


@pytest.mark.parametrize('settings', [{'elite': 33}, {'population': 0}, {'iterations': 0}, {'initial_std': float('nan')}, {'update_rate': 0}])
def test_teacher_rejects_invalid_settings(settings):
    from algorithms.matd3_plan_transformer.plan_teacher import PlanTeacher
    with pytest.raises(ValueError):
        PlanTeacher(settings, seed=1)


def test_auxiliary_pool_is_separate_and_context_consumed_once(case):
    from algorithms.matd3_plan_transformer.forecast_scenario import capture_planning_context
    from algorithms.matd3_plan_transformer.plan_teacher import PlanTeacher
    from algorithms.matd3_plan_transformer.replay_buffer import PlanReplayBuffer
    import torch
    env, obs, state = case
    ctx = capture_planning_context(env, obs, state, {})
    label = PlanTeacher(dict(population=2, elite=1, iterations=1), 3).solve(ctx)
    replay = PlanReplayBuffer(10, 2, 1, 2)
    replay.bind_context(ctx, label)
    actions = {mg: np.zeros(2) for mg in env.agent_ids}
    nxt, rewards, done, info = env.step(actions)
    replay.add(obs, state, actions, rewards['mg1'], nxt, info['global_state_after'], done)
    assert len(replay) == 1
    aux = replay.sample_auxiliary(32, env.agent_ids, torch.device('cpu'))
    assert aux.mask.shape == (1, 24)
    assert not aux.soc_after['mg1'].requires_grad
    assert replay.pending_context is None
    replay.bind_context(ctx, label)
    with pytest.raises(ValueError, match='context'):
        replay.add(nxt, info['global_state_after'], actions, 0, nxt, info['global_state_after'], False)
