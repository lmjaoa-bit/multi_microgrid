"""Add a fresh January date without changing the original generator or evidence."""
from pathlib import Path
import time
import numpy as np
import torch

from diagnostics.capacity_data import (sources_and_config,RecordingPolicy,capacity_candidates,
    append_case,save_block,load_block)
from diagnostics.capacity_probe import MaskedTeacher
from diagnostics.critic_probe import collect_cases,evaluate_candidates,sha256,read_json,write_json
from diagnostics.learning_chain_probe import _tensor_fingerprint
from envs.multi_microgrid_env import MultiMicrogridEnv


def prepare_confirmation_supplement(root,output):
    from diagnostics.capacity_fit import validated_data,validate_state_pool
    root,output=Path(root).resolve(),Path(output).resolve()
    target=output/'confirmation_supplement'
    if target.exists():raise ValueError('Confirmation supplement already exists')
    original,_,_=validated_data(root,output)
    if original['smoke']:raise ValueError('Fresh formal confirmation is for the complete diagnostic only')
    _,_,sources,cfg,policies=sources_and_config(root,Path(original['phase_one_directory']))
    env=MultiMicrogridEnv(cfg,split='val')
    already=set(original['fit_days']+original['dev_days']+original['confirmation_days']+original['audit_days'])
    available=sorted(str(d.date()) for d in env.data.valid_days if str(d.date()).startswith('2018-01') and str(d.date()) not in already)
    if not available:raise ValueError('No unused January validation date')
    day=available[-1]
    # This calendar rule was fixed before inspecting the additional date.
    primary=[day if d=='2018-01-25' else d for d in original['confirmation_days']]
    if len(primary)!=12 or '2018-01-25' in primary:raise ValueError('Unexpected primary confirmation calendar')
    original_hashes=dict(original['protected_hashes'])
    original_hashes[str(output/'manifest.json')]=sha256(output/'manifest.json')
    original_hashes.update({str(output/n):h for n,h in original['dataset_hashes'].items()})
    source_tensors=[_tensor_fingerprint(p.model) for p in policies]
    target.mkdir()
    manifest=dict(status='running',excluded_from_primary=['2018-01-25'],new_day=day,
        primary_confirmation_days=primary,calendar_rule='last unused January validation date',
        reason='January 25 was inspected in smoke and cannot remain independent of design',
        parent_manifest_sha256=sha256(output/'manifest.json'),data_seed=814169,extra_seed=915179,
        original_protected_hashes=original_hashes,code_sha256=sha256(Path(__file__)))
    write_json(target/'manifest.json',manifest)
    records=[]
    recorders=[RecordingPolicy(p,env,records) for p in policies]
    for r in recorders:r.day,r.seed=day,manifest['data_seed']
    cases=collect_cases(env,recorders,[day],manifest['data_seed'],6,original['forecast_noise'])
    start=time.perf_counter();rows=[];branches=steps=0
    for case in cases:
        actions=policies[1].actions(case.observations)
        candidates=capacity_candidates(np.stack([actions[mg] for mg in env.agent_ids]))
        prediction=policies[1].q_values(case.state,candidates)
        actual=evaluate_candidates(env,policies[1],case,candidates,forecast_noise=original['forecast_noise'])
        append_case(rows,case,candidates,actual['effective'],actual['returns'],prediction)
        branches+=actual['rollouts'];steps+=actual['steps']
    teacher=MaskedTeacher(policies[0])
    for seed in (manifest['data_seed'],manifest['extra_seed']):
        for name,policy in (('h8_teacher',teacher),('tq',policies[1])):
            obs,_=env.reset(day=day,seed=seed,deterministic=True,forecast_noise=original['forecast_noise'])
            for hour in range(env.episode_length):
                records.append(dict(day=day,seed=seed,origin=name,state=env.global_state(),
                    observations={k:v.copy() for k,v in obs.items()}))
                obs,_,done,_=env.step(policy.actions(obs))
                if done:break
    save_block(target/'value_confirmation.npz',rows)
    arrays=dict(day=np.asarray([r['day'] for r in records]),seed=np.asarray([r['seed'] for r in records]),
        origin=np.asarray([r['origin'] for r in records]),state=np.stack([r['state'] for r in records]))
    for mg in env.agent_ids:arrays[mg]=np.stack([r['observations'][mg] for r in records])
    np.savez_compressed(target/'actor_confirmation.npz',**arrays)
    write_json(target/'state_pool.json',[dict(split='confirmation',**vars(c)) for c in cases])
    if any(sha256(Path(p))!=h for p,h in original_hashes.items()):raise ValueError('Original evidence changed during supplement')
    if source_tensors!=[_tensor_fingerprint(p.model) for p in policies]:raise ValueError('Source tensors changed during supplement')
    manifest.update(status='complete',source_integrity_verified=True,source_tensors_unchanged=True,
        actual_q_branches=branches,environment_q_steps=steps,elapsed_seconds=time.perf_counter()-start,
        dataset_hashes={p.name:sha256(p) for p in target.iterdir() if p.suffix=='.npz' or p.name=='state_pool.json'})
    write_json(target/'manifest.json',manifest)
    check=dict(original,confirmation_days=[day],dataset_hashes=manifest['dataset_hashes'])
    validate_state_pool(target,check,{'confirmation':load_block(target/'value_confirmation.npz')},
        {'confirmation':load_block(target/'actor_confirmation.npz')})
    print(f'Fresh confirmation {day}: {branches} actual branches; original evidence preserved',flush=True)
    return manifest


def primary_data(root,output,manifest,values,actors):
    from diagnostics.capacity_fit import validate_state_pool,validate_partition_membership
    target=output/'confirmation_supplement';extra=read_json(target/'manifest.json')
    if extra['status']!='complete' or extra.get('source_integrity_verified') is not True:raise ValueError('Fresh supplement is incomplete')
    if extra['parent_manifest_sha256']!=sha256(output/'manifest.json'):raise ValueError('Supplement parent changed')
    for name,h in extra['dataset_hashes'].items():
        if sha256(target/name)!=h:raise ValueError('Fresh confirmation evidence changed')
    for path,h in extra['original_protected_hashes'].items():
        if sha256(Path(path))!=h:raise ValueError('Original evidence changed')
    new_value=load_block(target/'value_confirmation.npz');new_actor=load_block(target/'actor_confirmation.npz')
    check=dict(manifest,confirmation_days=[extra['new_day']],dataset_hashes=extra['dataset_hashes'])
    validate_state_pool(target,check,{'confirmation':new_value},{'confirmation':new_actor})
    values=dict(values);actors=dict(actors)
    for collection,new in ((values,new_value),(actors,new_actor)):
        old=collection['confirmation'];keep=~np.isin(old['day'],extra['excluded_from_primary'])
        collection['confirmation']={k:np.concatenate([v[keep],new[k]],axis=0) for k,v in old.items()}
    manifest=dict(manifest,confirmation_days=extra['primary_confirmation_days'],
        original_confirmation_days=manifest['confirmation_days'],excluded_smoke_dates=extra['excluded_from_primary'],
        confirmation_supplement=extra)
    validate_partition_membership(manifest,values);validate_partition_membership(manifest,actors)
    return manifest,values,actors
