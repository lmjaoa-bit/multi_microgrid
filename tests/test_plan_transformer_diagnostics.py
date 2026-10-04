import json
from pathlib import Path
import numpy as np
import pytest
from test_plan_transformer_pipeline import case


def test_recording_one_forward_per_step_preserves_actions_and_no_teacher(case, tmp_path, monkeypatch):
    _, env, _, model, obs, _ = case
    expected = model.select_actions(obs)
    model.configure_diagnostics(tmp_path / 'auxiliary_plan', {'checkpoint': 'best_model.pt'})
    calls = {mg: 0 for mg in env.agent_ids}
    for mg in env.agent_ids:
        original = model.actors[mg].forward_with_plan
        def counted(*args, _mg=mg, _original=original, **kwargs):
            calls[_mg] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(model.actors[mg], 'forward_with_plan', counted)
    model.set_diagnostic_phase('formal_test', {'days':['2020-01-01'],'seeds':[17]})
    for t in range(24):
        actions = model.select_actions(obs)
        if t == 0:
            for mg in env.agent_ids:
                np.testing.assert_array_equal(actions[mg], expected[mg])
        obs, _, _, _ = env.step(actions)
    model.set_diagnostic_phase(None, {'formal_test_complete':True})
    assert list(calls.values()) == [24,24,24]
    assert model.teacher.solve_calls == 0
    rows = [json.loads(x) for x in (tmp_path/'auxiliary_plan/formal_test_plan_attention.jsonl').read_text().splitlines()]
    assert len(rows) == 72
    assert rows[-1]['t'] == 23
    assert sum(rows[-1]['plan_mask']) == 1
    assert sum(rows[-1]['forecast_mask']) == 0
    assert rows[-1]['plan_result_times'][0] == '2020-01-02T00:00:00'
    assert all(x is None for x in rows[-1]['forecast_token_times'])
    assert np.asarray(rows[0]['attention']).shape == (2,4,26)
    assert rows[0]['observations']
    with pytest.raises(ValueError, match='exists'):
        model.set_diagnostic_phase('formal_test', {'days':['2020-01-01'],'seeds':[17]})


def test_training_completion_rejects_empty_supervision(case, tmp_path):
    _, _, _, model, _, _ = case
    model.configure_diagnostics(tmp_path/'auxiliary_plan', {})
    with pytest.raises(RuntimeError, match='supervision'):
        model.set_diagnostic_phase(None, {'training_complete':True})
    summary = json.loads((tmp_path/'auxiliary_plan/training_supervision_summary.json').read_text())
    assert summary['valid_labels'] == 0


def test_saved_plotter_replots_without_policy_or_environment(case, tmp_path, monkeypatch):
    from evaluation.plotting import generate_plan_diagnostic_figures
    _, env, _, model, obs, _ = case
    directory = tmp_path/'auxiliary_plan'
    model.configure_diagnostics(directory, {'checkpoint':'best_model.pt'})
    model.set_diagnostic_phase('formal_test', {'days':['2020-01-01'],'seeds':[17]})
    model.select_actions(obs)
    model.set_diagnostic_phase(None, {})
    (tmp_path/'test_summary.json').write_text(json.dumps({'checkpoint_path':'best_model.pt','test_days':['2020-01-01']}))
    trajectory = tmp_path/'test_hourly_trajectories'
    trajectory.mkdir()
    (trajectory/'2020-01-01.csv').write_text('timestep,mg1_soc,mg1_battery_charge_kw,mg1_battery_discharge_kw,mg2_soc,mg2_battery_charge_kw,mg2_battery_discharge_kw,mg3_soc,mg3_battery_charge_kw,mg3_battery_discharge_kw\n0,.5,0,0,.5,0,0,.5,0,0\n')
    monkeypatch.setattr(env, 'step', lambda *a: pytest.fail('plotter ran env'))
    monkeypatch.setattr(model, 'inspect_policy', lambda *a: pytest.fail('plotter ran policy'))
    before = {p:p.read_bytes() for p in directory.glob('*')}
    first = generate_plan_diagnostic_figures(tmp_path, formats=('png',))
    second = generate_plan_diagnostic_figures(tmp_path, formats=('png',))
    assert first == second
    assert len(first) == 6
    assert all(p.stat().st_size > 1000 for p in first)
    assert all(p.parent == tmp_path/'figures/plan_diagnostics' for p in first)
    assert before == {p:p.read_bytes() for p in directory.glob('*')}
