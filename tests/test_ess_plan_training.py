import copy

import numpy as np
import pytest
import torch

from test_ess_plan_teacher import case
from algorithms.matd3_ess_plan_transformer.plan_buffer import PlanBuffer
from algorithms.matd3_ess_plan_transformer.plan_supervision import TeacherBudget, plan_supervision_loss
from algorithms.matd3_ess_plan_transformer.plan_teacher import PlanTeacher
from algorithms.matd3_ess_plan_transformer.forecast_scenario import capture_forecast_context


def test_all_valid_slots_receive_gradient_and_dr_is_excluded():
    env, model, obs = case()
    actor = model.actors[env.agent_ids[0]]
    output = actor.forward_with_plan(torch.tensor(obs[env.agent_ids[0]])[None])
    output.intents.retain_grad()
    target = dict(intents=torch.zeros_like(output.intents), request_kw=torch.zeros_like(output.request_kw),
                  nominal_soc_after=torch.ones_like(output.nominal_soc_after)*.5, mask=output.mask)
    loss,_ = plan_supervision_loss(output, target, actor.battery)
    loss.backward()
    assert output.intents.grad[:, -1].abs().sum() > 0
    assert all(p.grad is None for p in actor.dr_actor.parameters())


def test_budget_tracks_total_generation_and_stops_when_exhausted():
    budget = TeacherBudget(.15, 1.)
    assert not budget.admit(0.)
    assert budget.admit(10.)
    budget.record(2.)
    assert not budget.admit(10.) and budget.seconds == 2.


def test_label_pool_reuses_labels_without_solves():
    env, model, obs = case()
    teacher = PlanTeacher()
    label = teacher.solve(capture_forecast_context(env,obs,env.global_state(),{}), model.actors)
    assert label.valid
    pool = PlanBuffer(4, 12)
    pool.add(label)
    for _ in range(3):
        batch = pool.sample(2, 'cpu')
        assert len(batch[env.agent_ids[0]]['observations']) == 2
    assert teacher.solve_calls == 1


def test_inference_checkpoint_preflight_and_no_teacher_at_inference(tmp_path):
    env, model, obs = case()
    model.select_actions(obs)
    assert model.teacher.solve_calls == 0
    path = tmp_path/'model.pt'
    model.save(path)
    model.load(path)
    saved = torch.load(path, weights_only=True)
    saved['actor_network_config']['architecture'] = 'wrong'
    torch.save(saved,path)
    before = copy.deepcopy(model.actors.state_dict())
    with pytest.raises(ValueError):
        model.load(path)
    assert all(torch.equal(v,model.actors.state_dict()[k]) for k,v in before.items())
    with pytest.raises(RuntimeError,match='resume'):
        model.load_training_state(tmp_path)


def test_disabled_supervision_has_no_capture_and_real_rl_update_is_finite():
    env, model, obs = case()
    model.plan_enabled = False
    model.capture_training_context(env,obs,env.global_state(),{'episode_index':50})
    assert model.teacher.solve_calls == 0 and len(model.plan_buffer) == 0
    model.learning_starts = 8
    state = env.global_state()
    for _ in range(8):
        model.replay.add(obs,state,model.select_actions(obs),-1.,obs,state,False)
    model.update(4)
    result = model.update(4)
    assert model.update_steps == 2
    assert all(torch.isfinite(p).all() for p in model.actors.parameters())


def test_captured_label_trains_plan_without_extra_critic_updates():
    env, model, obs = case()
    label = PlanTeacher().solve(capture_forecast_context(env,obs,env.global_state(),{}),model.actors)
    assert label.valid
    model.plan_buffer.add(label)
    model.plan_update_interval = 1
    model.learning_starts = 8
    state = env.global_state()
    for _ in range(8):
        model.replay.add(obs,state,model.select_actions(obs),-1.,obs,state,False)
    before = model.actors[env.agent_ids[0]].intent_head.weight.detach().clone()
    model.update(4)
    model.update(4)
    assert model.plan_supervision_updates == 1 and model.update_steps == 2
    assert not torch.equal(before,model.actors[env.agent_ids[0]].intent_head.weight)


def test_formal_policy_plan_is_recorded_without_teacher(tmp_path):
    import json
    env,model,obs=case()
    model.configure_diagnostics(tmp_path,{})
    model.set_diagnostic_phase('formal_test',{'days':['2018-01-01']})
    actions=model.select_actions(obs)
    records=list((tmp_path/'policy_plans').glob('*.json'))
    assert len(records)==1
    record=json.loads(records[0].read_text(encoding='utf-8'))
    assert record['date']=='2018-01-01' and record['hour']==0
    for mg in env.agent_ids:
        plan=record['agents'][mg]
        assert len(plan['request_kw'])==len(plan['nominal_soc_after'])==24
        assert plan['action'][0]==pytest.approx(actions[mg][0])
        assert plan['intents'][0]==pytest.approx(actions[mg][0])
    assert model.teacher.solve_calls==0


def test_nonfinite_checkpoint_rejected_before_any_write(tmp_path):
    env,model,obs=case()
    path=tmp_path/'model.pt'
    model.save(path)
    saved=torch.load(path,weights_only=True)
    key=next(k for k in saved['actors'] if k.endswith('weight'))
    saved['actors'][key].flatten()[0]=float('nan')
    torch.save(saved,path)
    before=copy.deepcopy(model.actors.state_dict())
    with pytest.raises(ValueError,match='nonfinite'):
        model.load(path)
    assert all(torch.equal(v,model.actors.state_dict()[k]) for k,v in before.items())
