"""Independent training-date terminal checks and smoothed-policy objective checks."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('locator',OUT/'audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
base,np,pd,torch=a.base,a.np,a.pd,a.torch


def load():
    source=a.other.verify_source();env,model,config=base.load()
    model.load(OUT/'latest_snapshot/best_model.pt')
    return source,env,model,config


def terminal_train():
    source,_,model,config=load()
    env=base.MultiMicrogridEnv(config['environment'],split='train')
    # Metadata load explicitly checks that the training and test feature contracts match.
    train_model=base.get_algorithm_spec(base.SLUG).create(env,config['algorithm'],device='cpu',seed=42)
    train_model.load(OUT/'latest_snapshot/best_model.pt')
    for module in [train_model.actors,train_model.target_actors,train_model.critics,train_model.target_critics]:module.eval()
    frame=pd.DataFrame(dict(date=[str(x.date()) for x in env.data.valid_days]))
    frame['month']=pd.to_datetime(frame.date).dt.month
    chosen=[]
    for month,g in frame.groupby('month'):
        for idx in np.linspace(0,len(g)-1,min(5,len(g))).round().astype(int): chosen.append(str(g.iloc[idx].date))
    rows=[]
    for day_idx,day in enumerate(chosen):
        obs,_=env.reset(day=day,seed=42+day_idx,deterministic=True,forecast_noise=True)
        while env.t<23:
            obs,_,done,_=env.step(train_model.select_actions(obs,noise_std=0.))
            assert not done
        actions=train_model.select_actions(obs,noise_std=0.);state=env.global_state().copy()
        raw=torch.tensor(np.stack([actions[mg] for mg in model.agent_ids])[None])
        with torch.no_grad():
            eff,_=model.project_actions(torch.tensor(state[None]),raw)
        reference=copy.deepcopy(env);_,rr,done,_=reference.step(actions);assert done
        for agent in model.agent_ids:
            for candidate,power in a.first_candidates(env,actions,agent):
                changed={mg:x.copy() for mg,x in actions.items()};changed[agent][1]=candidate
                branch=copy.deepcopy(env);_,cr,done,info=branch.step(changed);assert done
                probes=raw.repeat(2,1,1);probes[1,model.agent_ids.index(agent),1]=candidate
                with torch.no_grad():
                    es,_=model.project_actions(torch.tensor(np.repeat(state[None],2,axis=0)),probes)
                    q1,q2=model.critics[agent](torch.tensor(np.repeat(state[None],2,axis=0)),es.flatten(1))
                rows.append(dict(date=day,agent=agent,hour=23,gain=float(cr[agent]-rr[agent]),
                    dq1=float(q1[1]-q1[0]),dq2=float(q2[1]-q2[0]),candidate_raw=candidate,candidate_kw=power,
                    reference_raw=float(actions[agent][1])))
        if (day_idx+1)%10==0:print('training-date terminal',day_idx+1,'/',len(chosen),flush=True)
    result=pd.DataFrame(rows);result.to_csv(OUT/'training_date_terminal_pairs.csv',index=False)
    stats=[]
    for agent,g in result.groupby('agent'):
        significant=g[g.gain.abs()>.001]
        stats.append(dict(agent=agent,pairs=len(g),meaningful=len(significant),
            q1_wrong=int(sum(significant.dq1*significant.gain<0)),q2_wrong=int(sum(significant.dq2*significant.gain<0))))
    history_path=Path(source['source_run'])/'artifacts/training_history.json'
    history=json.loads(history_path.read_text());coverage=[]
    for low,high in [(43,1900),(1600,1900),(4300,4600)]:
        records=[]
        for h in history:
            if not low<=h['episode']<=high:continue
            d=h.get('action_statistics',{}).get('by_microgrid',{}).get('mg3',{}).get('dimensions',{}).get('dr_raw',{})
            if d:records.append(d)
        coverage.append(dict(episodes=[low,high],days=len(records),all_raw_positive_days=sum(d['min']>0 for d in records),
            all_raw_above_half_days=sum(d['min']>.5 for d in records),median_daily_raw_mean=float(np.median([d['mean'] for d in records])),
            median_daily_raw_min=float(np.median([d['min'] for d in records]))))
    summary=dict(days=len(chosen),dates=chosen,checkpoint_episode=4600,stats=stats,ev_training_action_summary=coverage,
        training_history_sha256=base.digest(history_path),
        caveats=['Training-date states were regenerated with the final best policy, not restored from actual training replay.',
                 'Daily action extrema cover all hours including forced EV hours, and include exploration; they are coarse coverage evidence.',
                 'The checkpoint recreates its training feature contract successfully; this check does not prove full state coverage.'])
    a.write(OUT/'training_date_checks.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def objective(draws):
    source,env,model,config=load()
    frame=pd.read_csv(OUT/'latest/pair_decomposition.csv')
    chosen=[]
    for agent,g in frame[(frame['sample']=='stratified')&(frame.hour<23)&np.isclose(frame.power_delta_kw.abs(),1.,atol=1e-5,rtol=0)].groupby('agent'):
        wrong=g[(g.gain.abs()>.001)&(g.dq1*g.gain<0)]
        chosen.append(wrong.loc[wrong.error.abs().idxmax()].to_dict())
    chosen.append(frame[(frame.agent=='mg3')&(frame.date=='2018-03-31')&(frame.hour==1)&np.isclose(frame.candidate_kw,54.)].iloc[0].to_dict())
    all_days=[str(x.date()) for x in env.data.valid_days];rows=[]
    for case_idx,case in enumerate(chosen):
        day=case['date'];obs,_=env.reset(day=day,seed=42+all_days.index(day),deterministic=True,forecast_noise=True)
        while env.t<int(case['hour']):obs,_,_,_=env.step(model.select_actions(obs,noise_std=0.))
        first=model.select_actions(obs,noise_std=0.)
        candidate={mg:x.copy() for mg,x in first.items()};candidate[case['agent']][1]=case['candidate_raw']
        for mode in ['target_det','target_smoothed']:
            count=1 if mode=='target_det' else draws
            rng=np.random.default_rng(910008+case_idx)
            noises=np.clip(rng.normal(0,model.target_policy_noise,size=(draws,24,len(model.agent_ids),2)),
                           -model.target_noise_clip,model.target_noise_clip).astype(np.float32)
            gains=[]
            for start in range(0,count,8):
                indices=list(range(start,min(start+8,count)));trials=[];totals=[];observations=[]
                for idx in indices:
                    for first_action in [first,candidate]:
                        trial=copy.deepcopy(env);new_obs,rewards,done,_=trial.step(first_action)
                        trials.append(trial);totals.append(float(rewards[case['agent']]));observations.append(new_obs)
                while not trials[0]._terminated:
                    hour=trials[0].t
                    raws=a.policy_batch(model,observations,target=True).numpy()
                    if mode=='target_smoothed':
                        shared=np.stack([noises[idx,hour] for idx in indices]).repeat(2,axis=0)
                        raws=np.clip(raws+shared,-1,1)
                    next_obs=[]
                    for j,trial in enumerate(trials):
                        action={mg:raws[j,k] for k,mg in enumerate(model.agent_ids)}
                        o,reward,done,_=trial.step(action);totals[j]+=float(reward[case['agent']]);next_obs.append(o)
                    observations=next_obs
                gains.extend(totals[j+1]-totals[j] for j in range(0,len(totals),2))
            for draw,gain in enumerate(gains):rows.append(dict(case_id=case_idx,agent=case['agent'],date=day,hour=case['hour'],
                mode=mode,draw=draw,gain=gain,online_gain=case['gain'],dq1=case['dq1'],candidate_kw=case['candidate_kw'],
                reference_kw=case['reference_kw']))
            print('objective',case_idx,case['agent'],mode,'gain',np.mean(gains),flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'policy_objective_rollouts.csv',index=False)
    summaries=[]
    for (case,mode),g in f.groupby(['case_id','mode']):
        avg=float(g.gain.mean());se=float(g.gain.std(ddof=1)/np.sqrt(len(g))) if len(g)>1 else 0.
        summaries.append(dict(case_id=int(case),agent=g.agent.iloc[0],date=g.date.iloc[0],hour=int(g.hour.iloc[0]),mode=mode,
            draws=len(g),gain_mean=avg,gain_se=se,monte_carlo_interval_95=[avg-1.96*se,avg+1.96*se],
            online_gain=float(g.online_gain.iloc[0]),dq1=float(g.dq1.iloc[0]),
            candidate_kw=float(g.candidate_kw.iloc[0]),reference_kw=float(g.reference_kw.iloc[0])))
    a.write(OUT/'policy_objective_summary.json',dict(checkpoint_episode=4600,summaries=summaries,
        caveats=['Four deliberately chosen examples, not a population estimate.',
                 'This tests physical returns of the frozen target policy with recurring smoothing; clipped-min bootstrap is not itself a physical policy.',
                 'Intervals quantify only Monte Carlo sampling error under this fixed environment scenario, not model or generalization uncertainty.']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['terminal_train','objective']);p.add_argument('--draws',type=int,default=64)
    args=p.parse_args();{'terminal_train':terminal_train,'objective':lambda:objective(args.draws)}[args.stage]()
