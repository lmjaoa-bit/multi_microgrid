"""Generate immutable capacity-probe labels from protected full best-model sources."""
from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import torch
from diagnostics.critic_probe import (SLUGS,FrozenPolicy,collect_cases,evaluate_candidates,axis_candidates,
    sha256,write_json,validate_source_run,fingerprint,Candidate,environment_signature)
from diagnostics.learning_chain_probe import load_phase_one,_tensor_fingerprint
from diagnostics.capacity_probe import training_dates,confirmation_dates,MaskedTeacher
from envs.multi_microgrid_env import MultiMicrogridEnv


def sources_and_config(root,probe,device='cpu'):
    phase,cases,scores,baselines=load_phase_one(probe)
    entries={slug:[s for s in phase['sources'] if s['algorithm']==slug] for slug in SLUGS}
    if any(len(v)!=1 for v in entries.values()):raise ValueError('Exactly one source per supported policy is required')
    receipts=[entries[slug][0] for slug in SLUGS]
    sources=[validate_source_run(s['run_directory'],s['algorithm']) for s in receipts]
    for source,receipt in zip(sources,receipts):
        if source.files!=receipt['sha256']:raise ValueError('Original best model evidence changed')
    configs=[dict(s.environment) for s in sources]
    for c in configs:c.pop('_project_root',None)
    if fingerprint(configs[0])!=fingerprint(configs[1]) or fingerprint(configs[0])!=phase['environment_hash']:
        raise ValueError('Source environments do not match')
    cfg=dict(configs[0],_project_root=str(root))
    if sha256(root/cfg['data']['val_path'])!=phase['validation_data_sha256']:
        raise ValueError('Validation CSV changed since original audit')
    torch.set_num_threads(1)
    env=MultiMicrogridEnv(cfg,split='train')
    policies=[FrozenPolicy(s,env,device) for s in sources]
    return phase,cases,sources,cfg,policies


class RecordingPolicy:
    def __init__(self,policy,env,records):
        self.policy,self.env,self.records=policy,env,records
        self.source,self.gamma=policy.source,policy.gamma
        self.day,self.seed='unknown',0
    def actions(self,observations):
        self.records.append(dict(day=self.day,seed=self.seed,origin=self.source.slug,state=self.env.global_state(),
                                 observations={k:v.copy() for k,v in observations.items()}))
        return self.policy.actions(observations)


def capacity_candidates(raw):
    result=axis_candidates(raw)
    for agent in range(3):
        for control in range(2):
            for sign in (-1,1):
                changed=raw.copy()
                changed[agent,control]=np.clip(raw[agent,control]+sign*.1,-1,1)
                result.append(Candidate(changed,agent,control))
    return result


def append_case(block,case,candidates,effective,returns,predicted):
    for cid,candidate in enumerate(candidates):
        control='current' if cid==0 else f'mg{candidate.agent_index+1}_'+('ESS' if candidate.control_index==0 else 'DR')
        block.append(dict(x=np.r_[case.state,effective[cid].flatten()],y=returns[cid],base_x=np.r_[case.state,effective[0].flatten()],
            base_y=returns[0],case_id=case.case_id,day=case.day,control=control,old_q1=predicted[cid,:,0],old_q2=predicted[cid,:,1]))


def save_block(path,rows):
    if not rows:raise ValueError('Empty value block')
    np.savez_compressed(path,**{key:np.asarray([r[key] for r in rows]) for key in rows[0]})


def load_block(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def audit_block(probe,phase_cases,slug):
    lookup={c.case_id:c for c in phase_cases}
    bycase={}
    with (Path(probe)/'candidate_results.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['algorithm']==slug:bycase.setdefault(r['case_id'],[]).append(r)
    rows=[]
    for case_id,records in bycase.items():
        records.sort(key=lambda r:int(r['candidate_id']))
        c=lookup[case_id]
        candidates=[];effective=[];returns=[];pred=[]
        for r in records:
            aid=None if int(r['candidate_id'])==0 else int(r['changed_agent'].replace('mg',''))-1
            control=None if aid is None else (0 if r['changed_control']=='ESS' else 1)
            candidates.append(Candidate(np.asarray(json.loads(r['raw_actions']),dtype=np.float32),aid,control))
            effective.append(json.loads(r['effective_actions']))
            returns.append(float(r['realized_return']))
            pred.append([[float(r[f'mg{i}_{q}']) for q in ('q1','q2')] for i in (1,2,3)])
        append_case(rows,c,candidates,np.asarray(effective),np.asarray(returns),np.asarray(pred))
    return rows


def prepare_capacity_data(root,probe,output,*,smoke=False):
    root,probe,output=Path(root).resolve(),Path(probe).resolve(),Path(output).resolve()
    if output.exists() or output.is_relative_to(probe) or any(output.is_relative_to(root/d) for d in ('outputs/working_runs','outputs/releases','algorithms')):
        raise ValueError('Choose a new separate diagnostic output')
    phase,audit_cases,sources,cfg,policies=sources_and_config(root,probe)
    if any(output.is_relative_to(source.path) for source in sources):
        raise ValueError('Choose a new separate diagnostic output outside source runs')
    env=MultiMicrogridEnv(cfg,split='train')
    val_env=MultiMicrogridEnv(cfg,split='val')
    days=[str(d.date()) for d in env.data.valid_days]
    fit_days,dev_days=training_dates(days)
    confirm=confirmation_dates([str(d.date()) for d in val_env.data.valid_days],phase['days'])
    if smoke:fit_days,dev_days,confirm=fit_days[:1],dev_days[:1],confirm[:1]
    partitions=[set(d) for d in (fit_days,dev_days,confirm,phase['days'])]
    if any(partitions[i]&partitions[j] for i in range(4) for j in range(i)):
        raise ValueError('Calendar splits overlap')
    protected={}
    for s in sources:protected.update({s.path/name:sha256(s.path/name) for name in s.files})
    for split in ('train','val'):
        p=root/cfg['data'][f'{split}_path'];protected[p]=sha256(p)
    for name in ('manifest.json','summary.json','state_pool.json','candidate_results.csv','state_scores.csv'):
        protected[probe/name]=sha256(probe/name)
    for name in ('actor.py','critic.py','matd3.py','replay_buffer.py','plugin.py'):
        p=root/'algorithms/matd3'/name;protected[p]=sha256(p)
    original=[_tensor_fingerprint(p.model) for p in policies]
    manifest=dict(status='data_running',purpose='capacity_diagnostic',created_at=datetime.now().astimezone().isoformat(),
        smoke=smoke,sources=[s.receipt() for s in sources],phase_one_directory=str(probe),
        fit_days=fit_days,dev_days=dev_days,confirmation_days=confirm,audit_days=phase['days'],
        critic_forecast_horizon=24,actor_forecast_horizon=8,critic_data_seed=814121,actor_extra_seed=915131,
        initialization_seeds=[17,29,43],learning_rates=[.0002,.001],max_updates=2000,batch_size=256,
        candidate_design='baseline; each coordinate at -.8,+.8 and current±.1',continuation=SLUGS[1],
        forecast_noise=phase['forecast_noise'],reward_scale=float(cfg['cost']['reward_scale']),
        state_schema=list(policies[1].model.state_schema),observation_schemas={k:list(v) for k,v in policies[1].model.observation_schemas.items()},
        protected_hashes={str(p):s for p,s in protected.items()},source_tensors=original,
        code_sha256={n:sha256(root/'diagnostics'/n) for n in ('capacity_data.py','capacity_probe.py')})
    output.mkdir(parents=True)
    write_json(output/'manifest.json',manifest)
    records=[];state_pool=[];blocks={k:[] for k in ('fit','dev','confirmation')}
    start=time.perf_counter();rollouts=steps=0
    teacher=MaskedTeacher(policies[0])
    try:
        all_days=sorted(fit_days+dev_days)+confirm
        for day_index,day in enumerate(all_days):
            split='confirmation' if day in confirm else 'fit' if day in fit_days else 'dev'
            active_env=val_env if split=='confirmation' else env
            actual_seed=814121+day_index
            recorders=[RecordingPolicy(p,active_env,records) for p in policies]
            for r in recorders:r.day,r.seed=day,actual_seed
            selected=collect_cases(active_env,recorders,[day],actual_seed,6,phase['forecast_noise'])
            if smoke:selected=selected[:1]
            for case in selected:
                case.forecast_seed_base=814121
                state_pool.append(dict(split=split,**vars(case)))
                raw=np.stack([policies[1].actions(case.observations)[mg] for mg in active_env.agent_ids])
                candidates=capacity_candidates(raw)
                pred=policies[1].q_values(case.state,candidates)
                values=evaluate_candidates(active_env,policies[1],case,candidates,forecast_noise=phase['forecast_noise'])
                append_case(blocks[split],case,candidates,values['effective'],values['returns'],pred)
                rollouts+=values['rollouts'];steps+=values['steps']
            # Extra predictor-noise paths expand Actor inputs without extra Q branches.
            for noise_seed in (actual_seed,915131+day_index):
                for name,policy in (('h8_teacher',teacher),('tq',policies[1])):
                    obs,_=active_env.reset(day=day,seed=noise_seed,deterministic=True,forecast_noise=phase['forecast_noise'])
                    for hour in range(active_env.episode_length):
                        records.append(dict(day=day,seed=noise_seed,origin=name,state=active_env.global_state(),
                            observations={k:v.copy() for k,v in obs.items()}))
                        obs,_,done,_=active_env.step(policy.actions(obs))
                        if done:break
            for name,rows in blocks.items():
                if rows:save_block(output/f'value_{name}.npz',rows)
            print(f'Data {day_index+1}/{len(all_days)} {day} {split}: Q cases={len(selected)}, branches={rollouts}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
        # Reuse the completed original validation audit only after fixed splits.
        save_block(output/'value_audit.npz',audit_block(probe,audit_cases,SLUGS[1]))
        write_json(output/'state_pool.json',state_pool)
        for split,dates in (('fit',fit_days),('dev',dev_days),('confirmation',confirm)):
            rows=[r for r in records if r['day'] in dates]
            arrays=dict(day=np.asarray([r['day'] for r in rows]),seed=np.asarray([r['seed'] for r in rows]),
                        origin=np.asarray([r['origin'] for r in rows]),state=np.stack([r['state'] for r in rows]))
            for mg in active_env.agent_ids:arrays[mg]=np.stack([r['observations'][mg] for r in rows])
            np.savez_compressed(output/f'actor_{split}.npz',**arrays)
        if any(sha256(p)!=digest for p,digest in protected.items()):raise ValueError('Protected source changed')
        if original!=[_tensor_fingerprint(p.model) for p in policies]:raise ValueError('Original model tensors changed')
        manifest.update(status='data_complete',source_integrity_verified=True,model_tensors_unchanged=True,
            data_elapsed_seconds=time.perf_counter()-start,actual_q_branches=rollouts,environment_q_steps=steps,
            dataset_hashes={p.name:sha256(p) for p in output.glob('*.npz')})
        write_json(output/'manifest.json',manifest)
        return manifest
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(output/'manifest.json',manifest)
        raise
