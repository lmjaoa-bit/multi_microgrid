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
