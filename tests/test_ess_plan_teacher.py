import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from algorithms.matd3_ess_plan_transformer.forecast_scenario import capture_forecast_context
from algorithms.matd3_ess_plan_transformer.plan_teacher import PlanTeacher, solve_storage_model
import highspy
import algorithms.matd3_ess_plan_transformer.plan_teacher as teacher_module
from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]


def case():
    torch.set_num_threads(1)
    spec = get_algorithm_spec('matd3_ess_plan_transformer')
    env = MultiMicrogridEnv(spec.build_environment_config(yaml.safe_load(
        (ROOT/'configs/cooperative.yaml').read_text(encoding='utf-8'))), split='train')
    cfg = yaml.safe_load(spec.default_config_path.read_text(encoding='utf-8'))
    cfg['algorithm']['network'].update(embed_dim=8, num_heads=2, num_layers=1, hidden_dim=16)
    model = spec.create(env, cfg, device='cpu', seed=42)
    obs, _ = env.reset(seed=42, deterministic=True)
    return env, model, obs


def test_context_never_reads_realized_future_and_preserves_environment():
    env, _, obs = case()
    before_rng = copy.deepcopy(env.rng.bit_generator.state) if hasattr(env, 'rng') else None
    a = capture_forecast_context(env, obs, env.global_state(), {})
    env.episode_df.iloc[1:, env.episode_df.columns.get_loc('buy_price_per_kwh')] = 999.
    for device in env.dr_devices.values():
        if hasattr(device, 'sessions'):
            public = device.state(env.t)
            for session in device.sessions:
                if not public[f"ev_{session['group_id']}_revealed"]:
                    session['demand_kwh'] = 999.
    b = capture_forecast_context(env, obs, env.global_state(), {})
    assert a.context_hash == b.context_hash
    assert before_rng is None or before_rng == env.rng.bit_generator.state
    assert env.t == 0
    env.split = 'test'
    with pytest.raises(ValueError):
        capture_forecast_context(env, obs, env.global_state(), {})


def test_storage_model_arbitrage_and_no_useless_discharge():
    b = dict(capacity_kwh=100., max_charge_kw=20., max_discharge_kw=20.,
             soc_min=.1, soc_max=.9, eta_charge=1., eta_discharge=1., degradation_cost_per_kwh=.01)
    result = solve_storage_model(np.array([[0.], [20.]]), np.zeros((2,1)),
        np.array([[.1], [1.]]), np.zeros(2), [b], [.1], 1., .05, True, 1.)
    assert result.valid and result.net_kw[0,0] < -19.9 and result.net_kw[1,0] > 19.9
    result = solve_storage_model(np.zeros((2,1)), np.zeros((2,1)), np.ones((2,1)),
        np.zeros(2), [b], [.9], 1., .05, True, 1.)
    assert result.valid and np.abs(result.net_kw).max() < 1e-6


def test_teacher_label_is_physically_checked_and_does_not_touch_training_env():
    env, model, obs = case()
    context = capture_forecast_context(env, obs, env.global_state(), {})
    before = {mg: env.cores[mg].battery.soc for mg in env.agent_ids}
    teacher = PlanTeacher(time_limit_seconds=1.)
    label = teacher.solve(context, model.actors)
    assert teacher.solve_calls == 1 and label.elapsed_seconds > 0
    assert before == {mg: env.cores[mg].battery.soc for mg in env.agent_ids} and env.t == 0
    if label.valid:
        assert label.verified_cost <= label.baseline_cost+1e-5
        assert label.max_residual <= 1e-6
        assert np.isfinite(label.intents).all()
    else:
        assert label.rejection_reason


def test_feasible_timeout_is_rejected(monkeypatch):
    monkeypatch.setattr(highspy.Highs,'getModelStatus',lambda self: highspy.HighsModelStatus.kTimeLimit)
    b = dict(capacity_kwh=100.,max_charge_kw=20.,max_discharge_kw=20.,soc_min=.1,soc_max=.9,
             eta_charge=1.,eta_discharge=1.,degradation_cost_per_kwh=.01)
    candidate=solve_storage_model(np.array([[0.],[20.]]),np.zeros((2,1)),np.array([[.1],[1.]]),
        np.zeros(2),[b],[.1],1.,.05,True,1.)
    assert not candidate.valid and 'TimeLimit' in candidate.status


def test_available_forecast_changes_context_hash(monkeypatch):
    env,model,obs=case()
    before=capture_forecast_context(env,obs,env.global_state(),{})
    local=env.forecasts.local
    def changed(mg,t,k):
        result=dict(local(mg,t,k))
        result['load_kw']+=10.
        return result
    monkeypatch.setattr(env.forecasts,'local',changed)
    after=capture_forecast_context(env,obs,env.global_state(),{})
    assert before.forecast_hash!=after.forecast_hash
    assert before.context_hash!=after.context_hash


def test_existing_highs_thread_pool_is_reused_without_global_reset():
    code='''
import numpy as np
import highspy
from algorithms.matd3_ess_plan_transformer.plan_teacher import solve_storage_model
prior=highspy.Highs()
prior.setOptionValue('output_flag',False)
prior.setOptionValue('threads',2)
x=prior.addVariable(lb=0.,ub=1.)
assert prior.minimize(x)==highspy.HighsStatus.kOk
b=dict(capacity_kwh=100.,max_charge_kw=20.,max_discharge_kw=20.,soc_min=.1,soc_max=.9,
       eta_charge=1.,eta_discharge=1.,degradation_cost_per_kwh=.01)
result=solve_storage_model(np.array([[0.],[20.]]),np.zeros((2,1)),np.array([[.1],[1.]]),
    np.zeros(2),[b],[.1],1.,.05,True,1.)
assert result.valid, result.status
assert prior.minimize(x)==highspy.HighsStatus.kOk
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,capture_output=True,text=True)


def test_capture_and_solve_preserve_actual_rngs():
    env,model,obs=case()
    generators=[env.data.rng,env.forecasts.rng,model.rng,model.replay.rng]
    states=[copy.deepcopy(g.bit_generator.state) for g in generators]
    torch_state=torch.get_rng_state().clone()
    numpy_state=np.random.get_state()
    context=capture_forecast_context(env,obs,env.global_state(),{})
    PlanTeacher().solve(context,model.actors)
    assert all(s==g.bit_generator.state for s,g in zip(states,generators))
    assert torch.equal(torch_state,torch.get_rng_state())
    after=np.random.get_state()
    assert numpy_state[0]==after[0] and np.array_equal(numpy_state[1],after[1]) and numpy_state[2:]==after[2:]


def test_future_weather_and_carbon_do_not_enter_snapshot():
    env,model,obs=case()
    state=env.global_state()
    before=capture_forecast_context(env,obs,state,{})
    for frame in [env.episode_df,env.episode_context_df]:
        for column in frame.columns:
            if 'temperature' in column or 'carbon_factor' in column or 'ghi' in column:
                frame.loc[frame.index[1:],column]=999.
    after=capture_forecast_context(env,obs,state,{})
    assert before.forecast_hash==after.forecast_hash


def test_solver_failure_and_cost_worse_do_not_become_labels(monkeypatch):
    env,model,obs=case()
    context=capture_forecast_context(env,obs,env.global_state(),{})
    monkeypatch.setattr(teacher_module,'solve_storage_model',lambda *a:teacher_module.CandidateResult(
        False,np.zeros((24,3)),'failure'))
    rejected=PlanTeacher().solve(context,model.actors)
    assert not rejected.valid and rejected.rejection_reason=='solver_no_feasible_solution'
    baseline=teacher_module.rollout_frozen_baseline(context,model.actors)
    baseline.cost=-1e6
    candidate=teacher_module.CandidateResult(True,np.zeros((24,3)),'test')
    rejected=teacher_module.verify_candidate(context,baseline,candidate)
    assert not rejected.valid and rejected.rejection_reason=='cost_worse'
