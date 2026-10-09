"""Frozen-policy paired daily-value diagnosis. No training or source-result writes."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('value_locator', ROOT/'reports/critic_error_location_20261008/audit.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
np, pd, torch, base = a.np, a.pd, a.torch, a.base
RUN = ROOT/'outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/runs/DUELINGCRITICT-QH24_2026-10-07_22-11-08_seed42/artifacts'
KEYS = ['cost_grid', 'cost_battery', 'cost_curtail', 'cost_carbon', 'cost_peak', 'cost_mill', 'cost_comfort', 'cost_ev', 'cost_unmet']


def paired(snapshot, actions_list, model):
    trials = [copy.deepcopy(snapshot) for _ in actions_list]
    totals = np.zeros(len(trials))
    immediate = np.zeros(len(trials))
    components = np.zeros((len(trials), len(KEYS)))
    powers = []
    observations = []
    for i, (trial, actions) in enumerate(zip(trials, actions_list)):
        obs, reward, done, info = trial.step(actions)
        value = info['generalized_system_cost']
        assert abs(float(reward['mg1'])*100+value) < 1e-8
        totals[i] = immediate[i] = value
        components[i] += [info['reward_v2_breakdown'][k] for k in KEYS]
        observations.append(obs)
        powers.append({mg: info['plans'][mg]['battery_discharge_kw']-info['plans'][mg]['battery_charge_kw'] for mg in model.agent_ids})
    while not trials[0]._terminated:
        next_obs = []
        for i, trial in enumerate(trials):
            actions = model.select_actions(observations[i], noise_std=0.)
            obs, reward, done, info = trial.step(actions)
            value = info['generalized_system_cost']
            assert abs(float(reward['mg1'])*100+value) < 1e-8
            totals[i] += value
            components[i] += [info['reward_v2_breakdown'][k] for k in KEYS]
            next_obs.append(obs)
        observations = next_obs
    assert np.max(abs(components.sum(axis=1)-totals)) < 1e-7
    return totals, immediate, components, powers


def main():
    a.other.verify_source()
    manifest = json.loads((ROOT/'reports/milp_gap_audit_20261008/source_manifest.json').read_text())
    for relative, expected in manifest['source_hashes'].items():
        assert base.digest(ROOT/relative) == expected, relative
    latest = ROOT/'reports/critic_error_location_20261008/latest_snapshot/best_model.pt'
    assert base.digest(latest) == base.digest(RUN/'checkpoints/best_model.pt')
    assert base.digest(latest) == 'dbdcb3edcfc43745fd7869211209c5c54e90bce6aa86ade7573511d6df629cdc'
    env, model, config = base.load()
    model.load(latest)
    raw_replay_env = base.MultiMicrogridEnv(config['environment'],split='test')
    assert model.gamma == 1
    daily = pd.read_csv(RUN/'test_daily_metrics.csv')
    assert len(daily) == 55 and daily.date.nunique() == 55
    # One central test date per month, selected before action-value/cost outcomes.
    dated = daily[['date']].sort_values('date').copy()
    dated['month'] = pd.to_datetime(dated.date).dt.month
    chosen = [str(g.iloc[len(g)//2].date) for _, g in dated.groupby('month')]
    old = pd.read_csv(ROOT/'reports/critic_error_location_20261008/latest/pair_decomposition.csv')
    # Reproduce all three device examples, plus a terminal HVAC example.
    cases = []
    for agent, date, hour, kw in [('mg1','2018-01-29',5,None), ('mg2','2018-11-26',8,None),
                                  ('mg3','2018-01-30',9,None), ('mg3','2018-03-31',1,54.),
                                  ('mg2','2018-09-27',23,None)]:
        g = old[(old.agent==agent)&(old.date==date)&(old.hour==hour)&(old.gain>.001)&(old.dq1<0)]
        if kw is not None: g = g[np.isclose(g.candidate_kw,kw)]
        assert len(g), (agent,date,hour)
        cases.append(g.sort_values('gain',ascending=False).iloc[0].to_dict())
    days = [str(x.date()) for x in env.data.valid_days]
    records, replay, reproduced = [], [], []
    updates_before = model.update_steps
    for idx, day in enumerate(days):
        obs, _ = env.reset(day=day,seed=42+idx,deterministic=True,forecast_noise=True)
        raw_replay_env.reset(day=day,seed=42+idx,deterministic=True,forecast_noise=True)
        saved_actions = pd.read_csv(RUN/'test_hourly_trajectories'/f'{day}.csv')
        total = 0.
        raw_total = 0.
        action_error = 0.
        while True:
            hour = env.t
            actions = model.select_actions(obs,noise_std=0.)
            state = env.global_state().copy()
            if day in chosen and hour in [0,6,12,18]:
                alternatives = [actions]
                labels = []
                for agent in model.agent_ids:
                    current = float(actions[agent][0])
                    seen = {round(current,6)}
                    for raw in [np.clip(current-.25,-1,1),np.clip(current+.25,-1,1),0.]:
                        raw = float(np.float32(raw))
                        if round(raw,6) in seen: continue
                        seen.add(round(raw,6))
                        change = {mg:x.copy() for mg,x in actions.items()}
                        change[agent][0] = raw
                        alternatives.append(change)
                        labels.append((agent,current,raw))
                costs, imm, comp, powers = paired(env,alternatives,model)
                ss = torch.tensor(np.repeat(state[None],len(alternatives),axis=0))
                raws = torch.tensor(np.stack([np.stack([x[mg] for mg in model.agent_ids]) for x in alternatives]))
                with torch.no_grad():
                    eff,_ = model.project_actions(ss,raws)
                    qs = {mg:model.critics[mg](ss,eff.flatten(1)) for mg in model.agent_ids}
                for j,(mg,current,raw) in enumerate(labels,1):
                    if abs(powers[j][mg]-powers[0][mg]) < 1e-5: continue
                    gain = costs[0]-costs[j]
                    records.append(dict(date=day,hour=hour,agent=mg,reference_raw=current,candidate_raw=raw,
                        reference_kw=powers[0][mg],candidate_kw=powers[j][mg],cost_reduction=gain,
                        immediate_cost_reduction=imm[0]-imm[j],later_cost_reduction=gain-(imm[0]-imm[j]),
                        predicted_cost_reduction_q1=100*float(qs[mg][0][j]-qs[mg][0][0]),
                        predicted_cost_reduction_q2=100*float(qs[mg][1][j]-qs[mg][1][0]),
                        **{f'delta_{KEYS[k]}':comp[j,k]-comp[0,k] for k in range(len(KEYS))}))
            for case in [x for x in cases if x['date']==day and x['hour']==hour]:
                changed = {mg:x.copy() for mg,x in actions.items()}
                changed[case['agent']][1] = case['candidate_raw']
                costs, imm, comp, _ = paired(env,[actions,changed],model)
                ss = torch.tensor(np.repeat(state[None],2,axis=0))
                raw = torch.tensor(np.stack([np.stack([x[mg] for mg in model.agent_ids]) for x in [actions,changed]]))
                with torch.no_grad():
                    effective,_ = model.project_actions(ss,raw)
                    q1,q2 = model.critics[case['agent']](ss,effective.flatten(1))
                gain, dq = float(costs[0]-costs[1]), float(q1[1]-q1[0])
                assert abs(gain-100*case['gain']) < 1e-5, (day,hour,case['agent'],gain,100*case['gain'])
                assert abs(dq-case['dq1']) < 1e-5
                reproduced.append(dict(agent=case['agent'],date=day,hour=hour,reference_kw=case['reference_kw'],
                    candidate_kw=case['candidate_kw'],cost_reduction=gain,predicted_cost_reduction_q1=100*dq,
                    immediate_cost_reduction=float(imm[0]-imm[1]),later_cost_reduction=float(gain-imm[0]+imm[1])))
            obs, reward, done, info = env.step(actions)
            total += info['generalized_system_cost']
            saved = {mg:np.asarray([saved_actions[f'{mg}_raw_action_0'].iloc[hour],
                                   saved_actions[f'{mg}_raw_action_1'].iloc[hour]]) for mg in model.agent_ids}
            action_error = max(action_error,max(float(np.max(abs(saved[mg]-actions[mg]))) for mg in model.agent_ids))
            _,_,raw_done,raw_info = raw_replay_env.step(saved)
            raw_total += raw_info['generalized_system_cost']
            if done: break
        expected = float(daily.loc[daily.date==day,'algorithm_generalized_system_cost'].iloc[0])
        assert raw_done and abs(raw_total-expected)<1e-7, (day,raw_total,expected)
        # Neural floating-point execution can differ slightly across CPU thread/batch settings.
        # This bound is 100x smaller than the 0.1 cost threshold for meaningful comparisons.
        assert abs(total-expected)<1e-3, (day,total,expected)
        replay.append(dict(date=day,replayed_cost=total,formal_cost=expected,absolute_error=abs(total-expected),
                           saved_action_replay_error=abs(raw_total-expected),regenerated_raw_action_max_error=action_error))
        if (idx+1)%5==0: print('verified days',idx+1,'/55; ESS pairs',len(records),flush=True)
    assert model.update_steps == updates_before
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'ess_action_pairs.csv',index=False)
    pd.DataFrame(replay).to_csv(OUT/'policy_replay.csv',index=False)
    pd.DataFrame(reproduced).to_csv(OUT/'reproduced_dr_cases.csv',index=False)
    meaningful = frame[frame.cost_reduction.abs()>.1]
    beneficial_missed = meaningful[(meaningful.cost_reduction>0)&(meaningful.predicted_cost_reduction_q1<0)]
    # At most ONE missed beneficial correction per date, never sum interacting candidates.
    best = beneficial_missed.groupby('date').cost_reduction.max().reindex(chosen,fill_value=0.)
    service_keys = ['cost_mill','cost_ev','cost_unmet']
    safe = beneficial_missed[(beneficial_missed[[f'delta_{k}' for k in service_keys]]<=1e-7).all(axis=1)]
    safe_best = safe.groupby('date').cost_reduction.max().reindex(chosen,fill_value=0.)
    reference = pd.read_csv(ROOT/'reports/milp_gap_audit_20261008/environment_replay_results.csv')
    assert not reference.duplicated(['date','mode']).any() and len(reference)==440
    mode_stats = reference.groupby('mode').agg(days=('date','nunique'),mean_cost=('cost','mean'),mean_saving=('gain','mean')).reset_index().to_dict('records')
    stats = dict(checkpoint_episode=4600,checkpoint_sha256=base.digest(latest),training_episodes=5000,formal_days=55,
        baseline_replay_max_cost_error=max(x['absolute_error'] for x in replay),learning_updates=0,
        saved_action_replay_max_cost_error=max(x['saved_action_replay_error'] for x in replay),
        ess_sample_dates=chosen,ess_hours=[0,6,12,18],ess_candidates='raw +/-0.25 and 0, clipped and physical duplicates excluded',
        ess_pairs=len(frame),ess_meaningful_pairs=len(meaningful),
        ess_q1_wrong=int((meaningful.cost_reduction*meaningful.predicted_cost_reduction_q1<0).sum()),
        ess_q2_wrong=int((meaningful.cost_reduction*meaningful.predicted_cost_reduction_q2<0).sum()),
        ess_missed_beneficial_pairs=len(beneficial_missed),ess_days_with_missed_benefit=len(best[best>0]),
        ess_one_correction_per_sampled_day_mean=float(best.mean()),ess_one_correction_per_day=list(best.items()),
        ess_one_correction_no_extra_mill_ev_unmet_penalty_mean=float(safe_best.mean()),
        reproduced_dr_cases=reproduced,
        formal_cost_mean=float(daily.algorithm_generalized_system_cost.mean()),
        generalized_reference_mean=float(daily.generalized_reference_cost.mean()),
        frozen_curve_reoptimization=mode_stats,
        caveats=['ESS sample is 12 preselected monthly dates, four hours per day, not all states or a random population estimate.',
                 'One-correction saving uses hindsight to pick the best of tested Q1-misranked candidates; it is measured diagnostic opportunity, not retrained-policy performance.',
                 'Critic ranking errors do not identify optimization, replay coverage, representation or long-horizon planning as a unique cause.',
                 'Full-information fixed-curve reoptimization gaps must not be summed or attributed entirely to critic error.',
                 'Ordinary MATD3 is still running; this diagnosis establishes the phenomenon in the completed Dueling run.'])
    (OUT/'verification_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    inputs = [latest,RUN/'checkpoints/best_model.pt',RUN/'test_daily_metrics.csv',ROOT/'reports/critic_error_location_20261008/latest/pair_decomposition.csv',
              ROOT/'reports/milp_gap_audit_20261008/environment_replay_results.csv',Path(__file__)]
    hashes = {str(p.relative_to(ROOT)):base.digest(p) for p in inputs}
    (OUT/'sources.json').write_text(json.dumps(dict(source_hashes_verified=True,input_hashes=hashes,
        unchanged_original_source_manifest=str(ROOT/'reports/milp_gap_audit_20261008/source_manifest.json')),indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
