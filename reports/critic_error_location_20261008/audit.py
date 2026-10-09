"""Read-only temporal/Bellman localization, without learning updates."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('other_dr', ROOT/'reports/dr_critic_audit_20261007/audit_other_dr.py')
other = importlib.util.module_from_spec(spec); spec.loader.exec_module(other)
base = other.base; np, pd, torch = base.np, base.pd, base.torch


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def freeze_latest(source):
    dest = OUT/'latest_snapshot'; dest.mkdir(exist_ok=True)
    run = Path(source['source_run'])/'artifacts'
    original = run/'checkpoints/best_model.pt'
    content = original.read_bytes(); sha = base.hashlib.sha256(content).hexdigest()
    assert sha == base.digest(original)
    (dest/'best_model.pt').write_bytes(content)
    for name in ['validation_history.json','run_status.json','config_snapshot.yaml']:
        (dest/name).write_bytes((run/name).read_bytes())
    h = json.loads((dest/'validation_history.json').read_text())
    return dict(checkpoint_sha256=sha, episode=next(x['episode'] for x in reversed(h) if x.get('is_best')))


def policy_batch(model, observations, *, target=False):
    modules = model.target_actors if target else model.actors
    with torch.no_grad():
        return torch.stack([modules[mg](torch.tensor(np.stack([obs[mg] for obs in observations])))
                            for mg in model.agent_ids], dim=1)


def first_candidates(env, actions, agent):
    if agent == 'mg2':
        current = float(actions[agent][1]); power = current*35.
        probes = [(float(np.float32(np.clip(current+d/35.,-1,1))),d) for d in [-1.,1.]]
        return [(raw, raw*35.) for raw,d in probes if abs(abs(raw*35.-power)-1.) < 1e-5]
    low,high=env.dr_devices[agent].feasible_bounds(env.t)
    if high-low < 1e-8: return []
    original_power = env.dr_devices[agent].plan(float(actions[agent][1]), env.t)['flexible_load_kw']
    return [(raw,power) for raw,power,kind in other.candidate_actions(env,agent,actions)
            if abs(abs(power-original_power)-1.) < 1e-5]


def rollout(snapshot, first, model, agent, trajectory_id):
    trial = copy.deepcopy(snapshot); actions = {mg:a.copy() for mg,a in first.items()}; rows=[]
    while True:
        hour=trial.t; state=trial.global_state().copy()
        raw=np.stack([actions[mg] for mg in model.agent_ids])
        obs,reward,done,info=trial.step(actions)
        rows.append(dict(trajectory_id=trajectory_id, agent=agent, hour=hour, state=state, raw=raw,
                         next_state=trial.global_state().copy(), next_obs=obs,
                         reward=float(reward[agent]), done=done,
                         costs={k:float(v) for k,v in info['reward_v2_breakdown'].items() if k.startswith('cost_')}))
        if done: break
        actions=model.select_actions(obs,noise_std=0.)
    return rows


def main(checkpoint, draws):
    source=other.verify_source(); env,model,config=base.load()
    assert model.gamma==1.
    metadata=dict(checkpoint_sha256=source['checkpoint_sha256'],episode=1900)
    if checkpoint=='latest':
        metadata=freeze_latest(source); model.load(OUT/'latest_snapshot/best_model.pt')
    path=OUT/checkpoint; path.mkdir(exist_ok=True)
    originals=pd.concat([pd.read_csv(base.OUT/'counterfactual_probes.csv').assign(agent='mg2'),
                         pd.read_csv(other.OUT/'counterfactual_probes.csv')],ignore_index=True)
    selected={(mg,day,int(hour)) for mg,day,hour in originals[['agent','date','hour']].itertuples(index=False,name=None)}
    days=[str(x.date()) for x in env.data.valid_days]
    trajectories=[]; comparisons=[]; first_records={}; cache={}; next_id=0
    baseline_rewards=[]
    for day_idx,day in enumerate(days):
        obs,_=env.reset(day=day,seed=base.SEED+day_idx,deterministic=True,forecast_noise=True)
        while True:
            hour=env.t; actions=model.select_actions(obs,noise_std=0.)
            for agent in model.agent_ids:
                chosen=(agent,day,hour) in selected
                if not chosen and hour!=23: continue
                probes=first_candidates(env,actions,agent)
                # Retain the two large wrong-sign examples from the earlier audit.
                if (agent,day,hour)==('mg3','2018-03-31',1):
                    low,high=env.dr_devices[agent].feasible_bounds(hour)
                    if high>=54 and high-low>0: probes.append((float(np.float32(2*(54-low)/(high-low)-1)),54.))
                if (agent,day,hour)==('mg1','2018-11-26',15):
                    probes.append((-1.,env.dr_devices[agent].plan(-1.,hour)['flexible_load_kw']))
                if not probes: continue
                reference_power=(float(actions[agent][1])*35. if agent=='mg2' else
                                 env.dr_devices[agent].plan(float(actions[agent][1]),hour)['flexible_load_kw'])
                key=(agent,day,hour)
                if key not in cache:
                    reference=rollout(env,actions,model,agent,next_id); trajectories.extend(reference)
                    first_records[next_id]=reference[0]; cache[key]=(next_id,reference); next_id+=1
                ref_id,reference=cache[key]
                seen=set()
                for raw,power in probes:
                    if abs(power-reference_power)<1e-5 or round(power,5) in seen: continue
                    seen.add(round(power,5))
                    changed={mg:a.copy() for mg,a in actions.items()}; changed[agent][1]=raw
                    alternative=rollout(env,changed,model,agent,next_id); trajectories.extend(alternative)
                    first_records[next_id]=alternative[0]
                    cost_keys=reference[0]['costs']
                    comparisons.append(dict(agent=agent,date=day,hour=hour,reference_id=ref_id,candidate_id=next_id,
                        candidate_raw=raw,reference_raw=float(actions[agent][1]),reference_kw=reference_power,
                        candidate_kw=power,power_delta_kw=power-reference_power,
                        sample='stratified' if chosen else 'terminal_extra',
                        gain=sum(x['reward'] for x in alternative)-sum(x['reward'] for x in reference),
                        immediate_gain=alternative[0]['reward']-reference[0]['reward'],
                        **{f'delta_{k}':sum(x['costs'][k] for x in alternative)-sum(x['costs'][k] for x in reference) for k in cost_keys}))
                    next_id+=1
            obs,rewards,done,_=env.step(actions)
            baseline_rewards.append(dict(date=day,hour=hour,reward=float(rewards['mg1'])))
            if done: break
        if (day_idx+1)%10==0: print(checkpoint,'replayed',day_idx+1,'days; pairs',len(comparisons),flush=True)
    # Critic outputs along actual deterministic branches, batch evaluated.
    outputs={}; traces=[]; gradient_records=[]
    for agent in model.agent_ids:
        records=[r for r in trajectories if r['agent']==agent]
        for start in range(0,len(records),128):
            batch=records[start:start+128]; ss=torch.tensor(np.stack([r['state'] for r in batch]))
            raw=torch.tensor(np.stack([r['raw'] for r in batch]))
            with torch.no_grad():
                effective,_=model.project_actions(ss,raw); parts=model.critics[agent].decompose(ss,effective.flatten(1))
            for i,r in enumerate(batch):
                values={k:float(v[i,0]) for k,v in parts.items()}; outputs[(r['trajectory_id'],r['hour'])]=values
                traces.append(dict(trajectory_id=r['trajectory_id'],agent=agent,hour=r['hour'],reward=r['reward'],done=r['done'],**values))
    trace=pd.DataFrame(traces).sort_values(['trajectory_id','hour'])
    trace['return_to_go']=trace.groupby('trajectory_id').reward.transform(lambda s:s.iloc[::-1].cumsum().iloc[::-1])
    trace['next_q1']=trace.groupby('trajectory_id').q1.shift(-1).fillna(0.)
    trace['bellman_residual_q1']=trace.q1-trace.reward-trace.next_q1
    # Reconstruct each one-step TD3 target with 128 common-random-number draws.
    targets={}; rng=np.random.default_rng(810008)
    common_noise=torch.tensor(rng.normal(0,model.target_policy_noise,size=(draws,len(model.agent_ids),2)),dtype=torch.float32).clamp(-model.target_noise_clip,model.target_noise_clip)
    for agent in model.agent_ids:
        ids=[idx for idx,r in first_records.items() if r['agent']==agent and not r['done']]
        for start in range(0,len(ids),32):
            batch_ids=ids[start:start+32]; records=[first_records[idx] for idx in batch_ids]
            next_states=torch.tensor(np.stack([r['next_state'] for r in records]))
            next_obs=[r['next_obs'] for r in records]
            raw_online=policy_batch(model,next_obs); raw_target=policy_batch(model,next_obs,target=True)
            with torch.no_grad():
                eff_online,_=model.project_actions(next_states,raw_online)
                online_q=model.critics[agent](next_states,eff_online.flatten(1))
                lag_q=model.target_critics[agent](next_states,eff_online.flatten(1))
                eff_target,_=model.project_actions(next_states,raw_target)
                det_q=model.target_critics[agent](next_states,eff_target.flatten(1))
                noisy_raw=(raw_target[:,None]+common_noise[None]).clamp(-1,1)
                repeated=next_states[:,None].expand(-1,draws,-1).reshape(len(records)*draws,-1)
                eff_noise,_=model.project_actions(repeated,noisy_raw.flatten(0,1))
                qn1,qn2=model.target_critics[agent](repeated,eff_noise.flatten(1))
                noise_values=torch.minimum(qn1,qn2).reshape(len(records),draws).numpy()
            for i,idx in enumerate(batch_ids):
                targets[idx]=dict(online_q1=float(online_q[0][i]),online_min=float(torch.minimum(*online_q)[i]),
                    lag_online=float(torch.minimum(*lag_q)[i]),target_det=float(torch.minimum(*det_q)[i]),
                    noise_values=noise_values[i])
        print(checkpoint,'targets complete',agent,len(ids),flush=True)
    for idx,r in first_records.items():
        if r['done']: targets[idx]=dict(online_q1=0.,online_min=0.,lag_online=0.,target_det=0.,noise_values=np.zeros(draws))
    rows=[]; contribution_rows=[]; identity_error=0.; original_match=0.
    for pair_idx,c in enumerate(comparisons):
        ref,cand=c['reference_id'],c['candidate_id']; r0,a0=first_records[ref],first_records[cand]
        dq=outputs[(cand,c['hour'])]['q1']-outputs[(ref,c['hour'])]['q1']
        ys={k:c['immediate_gain']+targets[cand][k]-targets[ref][k] for k in ['online_q1','online_min','lag_online','target_det']}
        noise_pair=c['immediate_gain']+targets[cand]['noise_values']-targets[ref]['noise_values']
        ys['smooth']=float(noise_pair.mean())
        smoothing=noise_pair-ys['target_det']
        row=dict(pair_id=pair_idx,**c,dq1=dq,error=dq-c['gain'],
                 **{f'delta_y_{k}':value for k,value in ys.items()},
                 smooth_target_se=float(noise_pair.std(ddof=1)/np.sqrt(draws)),
                 smoothing_effect_se=float(smoothing.std(ddof=1)/np.sqrt(draws)),
                 component_fit=dq-ys['smooth'],component_smoothing=ys['smooth']-ys['target_det'],
                 component_actor_lag=ys['target_det']-ys['lag_online'],
                 component_critic_lag=ys['lag_online']-ys['online_min'],
                 component_min=ys['online_min']-ys['online_q1'],
                 component_future=ys['online_q1']-c['gain'])
        parts=[row[k] for k in row if k.startswith('component_')]
        identity_error=max(identity_error,abs(sum(parts)-row['error']))
        rt=trace[trace.trajectory_id==ref].set_index('hour'); at=trace[trace.trajectory_id==cand].set_index('hour')
        deltas=at.bellman_residual_q1-rt.bellman_residual_q1
        assert abs(deltas.sum()-row['error'])<1e-10
        row['direct_bellman_error']=float(deltas.iloc[0]);row['later_bellman_error']=float(deltas.iloc[1:].sum())
        row['future_v_delta']=0. if r0['done'] else outputs[(cand,c['hour']+1)]['v1']-outputs[(ref,c['hour']+1)]['v1']
        row['future_adv_delta']=0. if r0['done'] else outputs[(cand,c['hour']+1)]['adv1']-outputs[(ref,c['hour']+1)]['adv1']
        for hour,value in deltas.items():
            contribution_rows.append(dict(pair_id=pair_idx,agent=c['agent'],date=c['date'],start_hour=c['hour'],hour=hour,
                differential_bellman_residual=float(value),reference_q=float(rt.loc[hour,'q1']),candidate_q=float(at.loc[hour,'q1']),
                reference_reward=float(rt.loc[hour,'reward']),candidate_reward=float(at.loc[hour,'reward'])))
        if checkpoint=='old' and c['sample']=='stratified':
            g=originals[(originals.agent==c['agent'])&(originals.date==c['date'])&(originals.hour==c['hour'])&np.isclose(originals.candidate_kw,c['candidate_kw'],atol=1e-5,rtol=0)]
            if len(g):
                assert abs(c['gain']-g.gain.iloc[0])<1e-10
                original_match=max(original_match,abs(dq-g.dq1.iloc[0]))
        rows.append(row)
    assert identity_error<1e-10 and original_match<2e-5
    frame=pd.DataFrame(rows); frame.to_csv(path/'pair_decomposition.csv',index=False)
    trace.to_csv(path/'trajectory_values.csv',index=False)
    pd.DataFrame(contribution_rows).to_csv(path/'hourly_error_contributions.csv',index=False)
    pd.DataFrame(baseline_rewards).to_csv(path/'baseline_rewards.csv',index=False)
    stats=[]
    for sample_type in ['local_stratified','terminal']:
        subset=frame[(frame['sample']=='stratified')&np.isclose(abs(frame.power_delta_kw),1.,atol=1e-5,rtol=0)] if sample_type=='local_stratified' else frame[frame.hour==23]
        for agent,g in subset.groupby('agent'):
            meaningful=g[g.gain.abs()>.001];wrong=meaningful[meaningful.dq1*meaningful.gain<0]
            item=dict(sample=sample_type,agent=agent,pairs=len(g),meaningful=len(meaningful),wrong=len(wrong),
                q_error_median=float(g.error.abs().median()),
                target_det_wrong=int(sum(meaningful.delta_y_target_det*meaningful.gain<0)),
                target_smooth_wrong=int(sum(meaningful.delta_y_smooth*meaningful.gain<0)),
                matched_online_target_wrong=int(sum(meaningful.delta_y_online_q1*meaningful.gain<0)),
                direct_larger_than_later=int(sum(wrong.direct_bellman_error.abs()>wrong.later_bellman_error.abs())))
            if len(wrong):
                item['wrong_pair_component_median_abs']={k:float(wrong[k].abs().median()) for k in frame if k.startswith('component_')}
                item['wrong_pair_median_abs_total_error']=float(wrong.error.abs().median())
                item['wrong_pair_median_abs_direct']=float(wrong.direct_bellman_error.abs().median())
                item['wrong_pair_median_abs_later']=float(wrong.later_bellman_error.abs().median())
                item['smoothing_small_vs_error_count']=int(sum(wrong.component_smoothing.abs()<wrong.error.abs()*.25))
            stats.append(item)
    other.verify_source()
    summary=dict(**metadata,checkpoint_label=checkpoint,days=55,noise_draws=draws,pair_records=len(frame),trajectory_rows=len(trace),
        stats=stats,checks=dict(error_decomposition_identity_max=identity_error,old_q_reproduction_max_error=original_match,
                                source_hashes_unchanged=True,learning_updates=0),
        state_contract=dict(peak_observed='team_load_peak_so_far' in model.state_schema,
            ev_cohort_remaining_features=[s for s in model.state_schema if s.startswith('mg3_ev_') and 'remaining' in s],
            ev_cohort_deadline_features=[s for s in model.state_schema if s.startswith('mg3_ev_') and 'departure' in s]),
        caveats=['All target residuals are reconstructed on audit states, not actual saved training replay losses.',
                 'No saved replay or optimizer trajectory exists; action coverage and optimization causes cannot be causally separated.',
                 'Smoothing is a one-step expected backup estimated using common random numbers; recurring future smoothed-policy returns are not yet measured.',
                 'Component decomposition is an algebraic attribution, not causal attribution to training hyperparameters.',
                 'Latest and old checkpoints encounter different states under their own policies on shared dates; rates are not paired causal effects.'])
    write(path/'audit_summary.json',summary)
    write(path/'source_manifest.json',dict(**metadata,script_sha256=base.digest(Path(__file__)),parent_checkpoint_sha256=source['checkpoint_sha256'],
        input_hashes={str(p.relative_to(ROOT)):base.digest(p) for p in [base.OUT/'counterfactual_probes.csv',other.OUT/'counterfactual_probes.csv']},
        extra_source_hashes={str(p.relative_to(ROOT)):base.digest(p) for p in [ROOT/'core/global_state.py',ROOT/'components/ev_sessions.py']},
        learning_updates=0))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',choices=['old','latest'],required=True)
    parser.add_argument('--noise-draws',type=int,default=128);args=parser.parse_args()
    main(args.checkpoint,args.noise_draws)
