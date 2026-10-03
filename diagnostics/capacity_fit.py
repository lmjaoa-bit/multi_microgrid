"""Train and verify isolated diagnostic networks; never update a source policy."""
from __future__ import annotations

import copy
import csv
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from diagnostics.capacity_data import load_block, sources_and_config, capacity_candidates
from diagnostics.capacity_probe import DenseValue, SequenceValue, TrainScaler, value_metrics, FlatStudent, MaskedTeacher
from diagnostics.critic_probe import read_json, write_json as _write_json, sha256, score_choices
from diagnostics.learning_chain_probe import _tensor_fingerprint

EXECUTION_SOURCE_SHA256=sha256(Path(__file__))


def write_json(path,value):
    # Windows indexers can briefly keep a just-written destination open.
    for attempt in range(6):
        try:return _write_json(path,value)
        except PermissionError:
            if attempt==5:raise
            time.sleep(.05*(attempt+1))


def validate_partition_membership(manifest, blocks):
    registered={name:set(manifest[f'{name}_days']) for name in ('fit','dev','confirmation','audit')}
    names=list(registered)
    if any(registered[names[i]] & registered[names[j]] for i in range(4) for j in range(i)):
        raise ValueError('Registered date partitions overlap')
    for name, block in blocks.items():
        actual=set(str(d) for d in block['day'])
        if actual!=registered[name]:
            raise ValueError(f'{name} row date membership disagrees with manifest')


def validated_data(root, output):
    output=Path(output).resolve()
    manifest=read_json(output/'manifest.json')
    if manifest['status']!='data_complete' or manifest.get('source_integrity_verified') is not True:
        raise ValueError('Data generation did not complete with protected sources')
    for path,digest in manifest['protected_hashes'].items():
        if sha256(Path(path))!=digest:raise ValueError(f'Protected evidence changed: {path}')
    for name,digest in manifest['code_sha256'].items():
        if sha256(Path(root)/'diagnostics'/name)!=digest:raise ValueError('Data generator changed since preparation')
    for name,digest in manifest['dataset_hashes'].items():
        if sha256(output/name)!=digest:raise ValueError(f'Diagnostic dataset changed: {name}')
    values={name:load_block(output/f'value_{name}.npz') for name in ('fit','dev','confirmation','audit')}
    actors={name:load_block(output/f'actor_{name}.npz') for name in ('fit','dev','confirmation')}
    validate_partition_membership(manifest,values)
    validate_partition_membership(manifest,actors)
    for name,block in values.items():
        for field in ('x','base_x','y','base_y','old_q1','old_q2'):
            if not np.isfinite(block[field]).all():raise ValueError(f'Nonfinite {name} {field}')
        if block['x'].shape!=block['base_x'].shape:raise ValueError('Baseline input dimension differs')
        base=baseline_indices(block)
        if not np.allclose(block['base_x'],block['x'][base],rtol=0,atol=1e-7):raise ValueError('Wrong paired baseline input')
        if not np.allclose(block['base_y'],block['y'][base],rtol=0,atol=1e-7):raise ValueError('Wrong paired baseline return')
    for name,block in actors.items():
        if any(not np.isfinite(block[k]).all() for k in ('state','mg1','mg2','mg3')):
            raise ValueError(f'Nonfinite Actor {name} rows')
    validate_state_pool(output,manifest,values,actors)
    supplement=output/'confirmation_supplement'/'manifest.json'
    if supplement.exists():
        from diagnostics.capacity_confirmation import primary_data
        manifest,values,actors=primary_data(root,output,manifest,values,actors)
    return manifest,values,actors


def validate_state_pool(output,manifest,values,actors,*,pool_path=None,receipt_path=None):
    pool_path=Path(pool_path or output/'state_pool.json')
    receipt_path=Path(receipt_path or output/'state_pool_integrity.json')
    digest=sha256(pool_path)
    if receipt_path.exists() and read_json(receipt_path)['state_pool_sha256']!=digest:
        raise ValueError('Protected diagnostic state pool changed')
    pool=read_json(pool_path)
    ids=[r['case_id'] for r in pool]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate state-pool case ID')
    expected={c for name in ('fit','dev','confirmation') if name in values for c in values[name]['case_id']}
    if set(ids)!=expected:raise ValueError('State pool case membership disagrees')
    dimension=len(manifest['state_schema'])
    for case in pool:
        name=case['split']
        if name not in values or name not in actors:raise ValueError('Unexpected state-pool split')
        v=values[name];a=actors[name]
        qrows=np.flatnonzero(v['case_id']==case['case_id'])
        if not len(qrows) or any(str(d)!=case['day'] for d in v['day'][qrows]):raise ValueError('State pool date disagrees with value labels')
        state=np.asarray(case['state'])
        if not np.allclose(v['x'][qrows,:dimension],state[None],rtol=0,atol=1e-7):raise ValueError('State pool state disagrees with value labels')
        candidates=np.flatnonzero((a['day']==case['day'])&(a['seed']==case['seed'])&(a['origin']==case['origin']))
        matching=[i for i in candidates if np.allclose(a['state'][i],state,rtol=0,atol=1e-7)]
        if not matching:raise ValueError('State pool has no matching protected Actor trajectory')
        if not any(all(np.allclose(a[mg][i],case['observations'][mg],rtol=0,atol=1e-7) for mg in ('mg1','mg2','mg3')) for i in matching):
            raise ValueError('State pool observations disagree with protected trajectory')
    validate_partition_membership(manifest,actors)
    if not receipt_path.exists():
        write_json(receipt_path,dict(state_pool_sha256=digest,case_count=len(pool),
            state_day_observation_membership_verified=True,verified_against_dataset_hashes=manifest.get('dataset_hashes',{}),
            verifier_sha256=sha256(Path(__file__)),note='Supplemental check; original generator identity is preserved'))
    return pool


def baseline_indices(block):
    lookup={}
    for i,(case,control) in enumerate(zip(block['case_id'],block['control'])):
        if control=='current':
            if case in lookup:raise ValueError('Duplicate baseline')
            lookup[case]=i
    try:return np.array([lookup[c] for c in block['case_id']],dtype=np.int64)
    except KeyError as exc:raise ValueError('Missing baseline') from exc


def gain_mae(block, predicted, *, reward_scale):
    pred=np.asarray(predicted)
    bases=baseline_indices(block)
    rows=np.flatnonzero(block['control']!='current')
    truth=block['y'][rows]-block['y'][bases[rows]]
    if pred.ndim==2:
        columns=np.array([int(str(block['control'][i]).split('_')[0][2:])-1 for i in rows])
        delta=pred[rows,columns]-pred[bases[rows],columns]
    else:delta=pred[rows]-pred[bases[rows]]
    return float(np.mean(np.abs(delta-truth))*reward_scale)


def paired_day_difference(candidate, baseline, *, field):
    def reduce(rows):
        groups={}
        for r in rows:groups.setdefault(str(r['date']),[]).append(float(r[field]))
        return {d:float(np.mean(v)) for d,v in groups.items()}
    a,b=reduce(candidate),reduce(baseline)
    if set(a)!=set(b) or not a:raise ValueError('Paired dates disagree')
    days=sorted(a);delta=np.array([a[d]-b[d] for d in days])
    rng=np.random.default_rng(961)
    bootstrap=delta[rng.integers(len(days),size=(10000,len(days)))].mean(1)
    return dict(date_count=len(days),mean_difference=float(delta.mean()),
        descriptive_95_interval=np.quantile(bootstrap,[.025,.975]).tolist(),
        improved_dates=int(np.sum(delta<0)),worse_dates=int(np.sum(delta>0)),
        by_date=[dict(date=d,difference=float(v)) for d,v in zip(days,delta)])


class NormalizedValue(nn.Module):
    def __init__(self, network, train):
        super().__init__()
        self.network=network
        scaler=TrainScaler.fit(train['x'])
        self.register_buffer('x_mean',torch.as_tensor(scaler.mean))
        self.register_buffer('x_std',torch.as_tensor(scaler.std))
        self.register_buffer('y_mean',torch.tensor(float(np.mean(train['y'])),dtype=torch.float32))
        self.register_buffer('y_std',torch.tensor(max(float(np.std(train['y'])),.01),dtype=torch.float32))
        self.register_buffer('gain_std',torch.tensor(max(float(np.std(train['y']-train['base_y'])),.001),dtype=torch.float32))

    def forward(self,x):
        return self.network((x-self.x_mean)/self.x_std)*self.y_std+self.y_mean


@torch.no_grad()
def predict_value(model, x, batch_size=512):
    device=next(model.parameters()).device
    model.eval()
    return np.concatenate([model(torch.as_tensor(x[i:i+batch_size],dtype=torch.float32,device=device)).cpu().numpy()
                           for i in range(0,len(x),batch_size)])


def value_loss(model, x, base_x, y, base_y):
    prediction=model(x);baseline=model(base_x)
    absolute=F.smooth_l1_loss((prediction-model.y_mean)/model.y_std,(y-model.y_mean)/model.y_std)
    gains=F.smooth_l1_loss((prediction-baseline)/model.gain_std,(y-base_y)/model.gain_std)
    return absolute+gains


def fit_value_supervised(network, train, dev, *, seed, learning_rate, updates, batch_size,
                         device, evaluate_every=100, reward_scale=100.):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=NormalizedValue(network,train).to(device)
    arrays={k:torch.as_tensor(train[k],dtype=torch.float32,device=device) for k in ('x','base_x','y','base_y')}
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    history=[];best_score=None;best=None
    for step in range(updates+1):
        if step>0:
            model.train()
            idx=torch.as_tensor(rng.integers(len(train['x']),size=batch_size),device=device)
            loss=value_loss(model,*(arrays[k][idx] for k in ('x','base_x','y','base_y')))
            if not torch.isfinite(loss):raise ValueError('Nonfinite Critic fitting loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
        if step % evaluate_every==0 or step==updates:
            prediction=predict_value(model,dev['x']);base=predict_value(model,dev['base_x'])
            gain_error=float(np.mean(np.abs(prediction-base-dev['y']+dev['base_y']))*reward_scale)
            mae=float(np.mean(np.abs(prediction-dev['y']))*reward_scale)
            if 'case_id' in dev:
                metrics=value_metrics(dev['y'],prediction,dev['case_id'],dev['control'],reward_scale=reward_scale)
                score=(metrics['mean_selection_regret_cost'],gain_error,mae)
            else:metrics={};score=(gain_error,mae)
            with torch.no_grad():
                sample=slice(0,min(1024,len(train['x'])))
                loss=float(value_loss(model,*(arrays[k][sample] for k in ('x','base_x','y','base_y'))))
            history.append(dict(step=step,training_loss=loss,dev_gain_mae_cost=gain_error,dev_value_mae_cost=mae,**metrics))
            if best_score is None or score<best_score:
                best_score=score;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.selected_step=step
    model.load_state_dict(best);model.eval()
    return model,history


def choice_details(block,predicted,reward_scale):
    result=[]
    for case in sorted(set(block['case_id'])):
        members=np.flatnonzero(block['case_id']==case)
        base=members[block['control'][members]=='current']
        for control in sorted(set(block['control'][members])-{'current'}):
            rows=np.r_[base,members[block['control'][members]==control]]
            p=predicted[rows]
            if p.ndim==2:p=p[:,int(str(control).split('_')[0][2:])-1]
            scored=score_choices(block['y'][rows],p,reward_scale=reward_scale,tie_cost=.1)
            result.append(dict(case_id=str(case),control=str(control),date=str(block['day'][rows[0]]),**scored))
    return result


def per_day_choices(details):
    result=[]
    for day in sorted({d['date'] for d in details}):
        rows=[r for r in details if r['date']==day]
        pairs=sum(r['rank_pair_count'] for r in rows);correct=sum(r['rank_correct_count'] for r in rows)
        result.append(dict(date=day,regret=float(np.mean([r['selection_regret_cost'] for r in rows])),
            oracle_gain=float(np.mean([r['actor_improvement_cost'] for r in rows])),
            rank_accuracy=correct/pairs if pairs else None,rank_correct_count=correct,rank_pair_count=pairs))
    return result


def stage_critic(root,output,manifest,values):
    target=output/'critic'
    if target.exists():raise ValueError('Critic fitting output already exists')
    target.mkdir()
    torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    schema=manifest['state_schema'];dimension=values['fit']['x'].shape[1]
    updates=80 if manifest['smoke'] else manifest['max_updates']
    seeds=manifest['initialization_seeds'][:1] if manifest['smoke'] else manifest['initialization_seeds']
    rates=manifest['learning_rates'][:1] if manifest['smoke'] else manifest['learning_rates']
    summary=dict(status='running',diagnostic_only=True,device=device,models=[],learning_rate_pilots=[],references=[],
        architecture_parameters={},fit_dates=manifest['fit_days'],dev_dates=manifest['dev_days'],confirmation_dates=manifest['confirmation_days'])
    for name in ('mlp','transformer'):
        def construct(seed):
            torch.manual_seed(seed)
            return DenseValue(dimension) if name=='mlp' else SequenceValue(schema,dimension-len(schema))
        summary['architecture_parameters'][name]=sum(p.numel() for p in construct(seeds[0]).parameters())
        pilots=[]
        for lr in rates:
            model,history=fit_value_supervised(construct(seeds[0]),values['fit'],values['dev'],seed=seeds[0],
                learning_rate=lr,updates=updates,batch_size=manifest['batch_size'],device=device,reward_scale=manifest['reward_scale'])
            pred=predict_value(model,values['dev']['x'])
            metrics=value_metrics(values['dev']['y'],pred,values['dev']['case_id'],values['dev']['control'],reward_scale=manifest['reward_scale'])
            pilots.append((metrics['mean_selection_regret_cost'],gain_mae(values['dev'],pred,reward_scale=manifest['reward_scale']),lr,model,history))
            write_json(target/f'{name}_lr{lr}_pilot_curve.json',history)
            summary['learning_rate_pilots'].append(dict(architecture=name,learning_rate=lr,selected_step=model.selected_step,**metrics))
            print(f'Critic dev pilot {name} lr={lr}: regret={metrics["mean_selection_regret_cost"]:.4f}, rank={metrics["rank_accuracy"]:.4f}',flush=True)
        chosen=min(pilots,key=lambda v:v[:2]);lr=chosen[2]
        for seed in seeds:
            if seed==seeds[0]:model,history=chosen[3:]
            else:model,history=fit_value_supervised(construct(seed),values['fit'],values['dev'],seed=seed,
                learning_rate=lr,updates=updates,batch_size=manifest['batch_size'],device=device,reward_scale=manifest['reward_scale'])
            ident=f'{name}_seed{seed}'
            torch.save(dict(diagnostic_only=True,architecture=name,seed=seed,learning_rate=lr,
                selected_step=model.selected_step,state_dict=model.cpu().state_dict()),target/f'{ident}.pt')
            model.to(device)
            write_json(target/f'{ident}_curve.json',history)
            for split in ('fit','dev','confirmation','audit'):
                block=values[split];pred=predict_value(model,block['x'])
                np.savez_compressed(target/f'{ident}_{split}_predictions.npz',prediction=pred)
                metrics=value_metrics(block['y'],pred,block['case_id'],block['control'],reward_scale=manifest['reward_scale'])
                details=choice_details(block,pred,manifest['reward_scale'])
                write_json(target/f'{ident}_{split}_choices.json',details)
                summary['models'].append(dict(model=ident,architecture=name,seed=seed,learning_rate=lr,
                    selected_step=model.selected_step,split=split,gain_mae_cost=gain_mae(block,pred,reward_scale=manifest['reward_scale']),
                    daily=per_day_choices(details),**metrics))
            print(f'Critic finished {ident}',flush=True)
            write_json(target/'summary.json',summary)
    for label in ('old_q1','old_q2','old_min'):
        for split in ('confirmation','audit'):
            block=values[split]
            pred=np.minimum(block['old_q1'],block['old_q2']) if label=='old_min' else block[label]
            metrics=value_metrics(block['y'],pred,block['case_id'],block['control'],reward_scale=manifest['reward_scale'])
            summary['references'].append(dict(model=label,split=split,gain_mae_cost=gain_mae(block,pred,reward_scale=manifest['reward_scale']),
                daily=per_day_choices(choice_details(block,pred,manifest['reward_scale'])),**metrics))
    summary['status']='complete';summary['code_sha256']=EXECUTION_SOURCE_SHA256
    write_json(target/'summary.json',summary)
    return summary


def construct_actors(kind,policy,env,seed,*,warm=False):
    from algorithms.matd3_hierarchical_transformer_q_readout.actor import DeviceForecastReadoutActor
    torch.manual_seed(seed)
    if warm:
        result=copy.deepcopy(policy.model.actors)
    elif kind in ('raw_mlp','semantic_mlp'):
        result=nn.ModuleDict({mg:FlatStudent(policy.model.observation_schemas[mg],policy.model.actors[mg],kind=='semantic_mlp')
                              for mg in policy.ids})
    elif kind=='transformer':
        net=policy.source.config['algorithm']['network']
        result=nn.ModuleDict({mg:DeviceForecastReadoutActor(
            policy.model.observation_schemas[mg],policy.ids,mg,env.cfg_by_id[mg],
            power_normalizers=policy.model.actor_feature_normalizers[mg],embed_dim=int(net['embed_dim']),
            num_heads=int(net['num_heads']),num_layers=int(net['num_layers']),hidden_dim=int(net['hidden_dim']),
            forecast_horizon=8,include_peer_forecast=True,peer_forecast_features=tuple(net['peer_forecast_features']))
            for mg in policy.ids})
    else:raise ValueError(f'Unknown Actor architecture {kind}')
    return result.requires_grad_(True)


@torch.no_grad()
def predict_actors(actors,block,*,logits=False,batch_size=512):
    device=next(actors.parameters()).device
    actors.eval()
    all_agents=[]
    for mg,actor in actors.items():
        values=[]
        for i in range(0,len(block[mg]),batch_size):
            obs=torch.as_tensor(block[mg][i:i+batch_size],dtype=torch.float32,device=device)
            values.append((actor.action_logits(obs) if logits else actor(obs)).cpu().numpy())
        all_agents.append(np.concatenate(values))
    return np.stack(all_agents,axis=1)


def teacher_logits(teacher,block):
    if isinstance(teacher,MaskedTeacher):
        return np.stack([teacher.logits(mg,block[mg]) for mg in teacher.ids],axis=1)
    return predict_actors(teacher.model.actors,block,logits=True)


def actor_objective(prediction,target,scale):
    return F.smooth_l1_loss(prediction/scale,target/scale)+4*F.mse_loss(prediction.tanh(),target.tanh())


def fit_actor_supervised(actors,train,dev,train_targets,dev_targets,*,seed,learning_rate,updates,batch_size,
                         device,evaluate_every=100):
    torch.set_num_threads(1);torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    actors.to(device).train()
    observations={mg:torch.as_tensor(train[mg],dtype=torch.float32,device=device) for mg in actors}
    targets=torch.as_tensor(train_targets,dtype=torch.float32,device=device)
    scale=torch.as_tensor(np.maximum(np.std(train_targets,axis=0),1.),dtype=torch.float32,device=device)
    optimizer=torch.optim.Adam(actors.parameters(),lr=learning_rate)
    best=None;best_score=None;history=[];selected=0
    for step in range(updates+1):
        if step:
            actors.train()
            idx=torch.as_tensor(rng.integers(len(train_targets),size=batch_size),device=device)
            prediction=torch.stack([actor.action_logits(observations[mg][idx]) for mg,actor in actors.items()],1)
            loss=actor_objective(prediction,targets[idx],scale)
            if not torch.isfinite(loss):raise ValueError('Nonfinite Actor fitting loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(actors.parameters(),10.);optimizer.step()
        if step % evaluate_every==0 or step==updates:
            dev_prediction=predict_actors(actors,dev,logits=True)
            dev_loss=float(actor_objective(torch.as_tensor(dev_prediction,device=device),torch.as_tensor(dev_targets,device=device),scale))
            with torch.no_grad():
                sample=slice(0,min(512,len(train_targets)))
                train_prediction=torch.stack([a.action_logits(observations[mg][sample]) for mg,a in actors.items()],1)
                train_loss=float(actor_objective(train_prediction,targets[sample],scale))
            raw_rmse=float(np.sqrt(np.mean((np.tanh(dev_prediction)-np.tanh(dev_targets))**2)))
            history.append(dict(step=step,training_loss=train_loss,dev_loss=dev_loss,dev_action_rmse=raw_rmse))
            if best_score is None or dev_loss<best_score:
                best_score=dev_loss;best={k:v.detach().cpu().clone() for k,v in actors.state_dict().items()};selected=step
    actors.load_state_dict(best);actors.eval();actors.selected_step=selected
    return actors,history


def actor_errors(actors,block,target,policy):
    pred=predict_actors(actors,block,logits=True)
    raw=np.tanh(pred);teacher=np.tanh(target)
    with torch.no_grad():
        states=torch.as_tensor(block['state'],dtype=torch.float32,device=policy.model.device)
        a,_=policy.model.project_actions(states,torch.as_tensor(raw,dtype=torch.float32,device=policy.model.device))
        b,_=policy.model.project_actions(states,torch.as_tensor(teacher,dtype=torch.float32,device=policy.model.device))
        effective_delta=(a-b).cpu().numpy()
    daily=[]
    for day in sorted(set(block['day'])):
        rows=block['day']==day
        daily.append(dict(date=str(day),action_rmse=float(np.sqrt(np.mean((raw[rows]-teacher[rows])**2))),
            effective_rmse=float(np.sqrt(np.mean(effective_delta[rows]**2)))))
    return dict(action_rmse=float(np.sqrt(np.mean((raw-teacher)**2))),logit_mae=float(np.mean(np.abs(pred-target))),
        action_rmse_by_agent_control=np.sqrt(np.mean((raw-teacher)**2,axis=0)).tolist(),
        effective_action_rmse=float(np.sqrt(np.mean(effective_delta**2))),
        effective_action_rmse_by_agent=np.sqrt(np.mean(effective_delta**2,axis=0)).tolist(),daily=daily),pred


def monte_carlo_actor_labels(output,values,policy,*,reward_scale):
    """Ex-post one-step labels are training supervision, never inference inputs."""
    records=read_json(output/'state_pool.json')
    supplement=output/'confirmation_supplement'/'state_pool.json'
    if supplement.exists():records+=read_json(supplement)
    if len(records)!=len({r['case_id'] for r in records}):raise ValueError('Duplicate MC case ID')
    pool={r['case_id']:r for r in records}
    blocks={};targets={};receipts=[]
    for split in ('fit','dev','confirmation'):
        value=values[split];cases=sorted(set(value['case_id']));rows=[];labels=[]
        for case_id in cases:
            case=pool[case_id];obs={mg:np.asarray(case['observations'][mg],dtype=np.float32) for mg in policy.ids}
            if case['split']!=split:raise ValueError('MC label split mismatch')
            action=policy.actions(obs);raw=np.stack([action[mg] for mg in policy.ids])
            candidates=capacity_candidates(raw)
            indices=np.flatnonzero(value['case_id']==case_id)
            if len(indices)!=25:raise ValueError('MC label requires all twenty-five candidates')
            if any(str(d)!=case['day'] for d in value['day'][indices]):raise ValueError('MC label date mismatch')
            dimension=len(policy.model.state_schema)
            if not np.allclose(value['x'][indices,:dimension],np.asarray(case['state'])[None],rtol=0,atol=1e-7):
                raise ValueError('MC label state mismatch')
            with torch.no_grad():
                states=torch.as_tensor(case['state'],dtype=torch.float32)[None].expand(25,-1)
                effective,_=policy.model.project_actions(states,torch.as_tensor(np.stack([c.raw for c in candidates])))
            if not np.allclose(effective.flatten(1).numpy(),value['x'][indices,dimension:],rtol=0,atol=1e-6):
                raise ValueError('MC candidate order disagrees with protected labels')
            selected=int(np.argmax(value['y'][indices]));best=candidates[selected].raw
            block={mg:obs[mg][None] for mg in policy.ids}
            current_logits=predict_actors(policy.model.actors,block,logits=True)[0]
            changed=best!=raw
            label=current_logits.copy()
            label[changed]=np.arctanh(np.clip(best[changed],-1+1e-6,1-1e-6))
            labels.append(label)
            rows.append(dict(day=case['day'],state=case['state'],**obs))
            receipts.append(dict(case_id=case_id,split=split,selected_candidate=selected,
                remaining_cost_gain=float((value['y'][indices[selected]]-value['y'][indices[0]])*reward_scale),
                raw_labels=best.tolist()))
        blocks[split]={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        targets[split]=np.asarray(labels,dtype=np.float32)
    manifest=read_json(output/'manifest.json')
    supplement_manifest=output/'confirmation_supplement'/'manifest.json'
    if supplement_manifest.exists():manifest['confirmation_days']=read_json(supplement_manifest)['primary_confirmation_days']
    validate_partition_membership(manifest,blocks)
    return blocks,targets,receipts


def stage_actor(root,output,manifest,blocks,values):
    from envs.multi_microgrid_env import MultiMicrogridEnv
    target=output/'actor'
    if target.exists():raise ValueError('Actor fitting output already exists')
    target.mkdir();torch.set_num_threads(1)
    _,_,_,cfg,policies=sources_and_config(root,Path(manifest['phase_one_directory']))
    source_tensors=[_tensor_fingerprint(p.model) for p in policies]
    env=MultiMicrogridEnv(cfg,split='train')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    updates=80 if manifest['smoke'] else manifest['max_updates']
    seeds=manifest['initialization_seeds'][:1] if manifest['smoke'] else manifest['initialization_seeds']
    rates=manifest['learning_rates'][:1] if manifest['smoke'] else manifest['learning_rates']
    summary=dict(status='running',diagnostic_only=True,device=device,models=[],references=[],learning_rate_pilots=[],parameter_counts={})
    for teacher_name,teacher in (('mlp_h8',MaskedTeacher(policies[0])),('current_t',policies[1])):
        targets={split:teacher_logits(teacher,block).astype(np.float32) for split,block in blocks.items()}
        if any(not np.isfinite(v).all() for v in targets.values()):raise ValueError('Nonfinite teacher logits')
        for split,value in targets.items():np.savez_compressed(target/f'target_{teacher_name}_{split}.npz',logits=value)
        for split,block in blocks.items():
            errors,_=actor_errors(policies[1].model.actors,block,targets[split],policies[1])
            summary['references'].append(dict(model='original_t_q',teacher=teacher_name,split=split,**errors))
        designs=[('raw_mlp',False),('semantic_mlp',False),('transformer',False)]
        if teacher_name=='mlp_h8':designs.append(('transformer',True))
        for kind,warm in designs:
            label='t_warm' if warm else kind
            sample=construct_actors(kind,policies[1],env,seeds[0],warm=warm)
            summary['parameter_counts'][label]={mg:sum(p.numel() for p in a.parameters()) for mg,a in sample.items()}
            pilots=[]
            for lr in rates:
                actors,history=fit_actor_supervised(construct_actors(kind,policies[1],env,seeds[0],warm=warm),
                    blocks['fit'],blocks['dev'],targets['fit'],targets['dev'],seed=seeds[0],learning_rate=lr,
                    updates=updates,batch_size=manifest['batch_size'],device=device)
                selected=min(history,key=lambda r:r['dev_loss'])
                pilots.append((selected['dev_loss'],lr,actors,history))
                write_json(target/f'{teacher_name}_{label}_lr{lr}_pilot_curve.json',history)
                summary['learning_rate_pilots'].append(dict(teacher=teacher_name,architecture=label,learning_rate=lr,**selected))
                print(f'Actor dev pilot teacher={teacher_name} {label} lr={lr}: loss={selected["dev_loss"]:.5f}, raw_rmse={selected["dev_action_rmse"]:.5f}',flush=True)
            chosen=min(pilots,key=lambda v:v[0]);lr=chosen[1]
            for seed in seeds:
                if seed==seeds[0]:actors,history=chosen[2:]
                else:actors,history=fit_actor_supervised(construct_actors(kind,policies[1],env,seed,warm=warm),
                    blocks['fit'],blocks['dev'],targets['fit'],targets['dev'],seed=seed,learning_rate=lr,
                    updates=updates,batch_size=manifest['batch_size'],device=device)
                ident=f'{teacher_name}_{label}_seed{seed}'
                torch.save(dict(diagnostic_only=True,architecture=kind,warm=warm,teacher=teacher_name,seed=seed,
                    learning_rate=lr,selected_step=actors.selected_step,state_dict=actors.cpu().state_dict()),target/f'{ident}.pt')
                actors.to(device)
                write_json(target/f'{ident}_curve.json',history)
                for split,block in blocks.items():
                    errors,pred=actor_errors(actors,block,targets[split],policies[1])
                    if split=='confirmation':np.savez_compressed(target/f'{ident}_{split}_predictions.npz',logits=pred)
                    summary['models'].append(dict(model=ident,teacher=teacher_name,architecture=label,
                        random_initialization=not warm,seed=seed,split=split,learning_rate=lr,selected_step=actors.selected_step,**errors))
                print(f'Actor finished {ident}',flush=True)
                write_json(target/'summary.json',summary)
    # Recovery of the actual current Actor with realized-return labels removes
    # the MLP teacher as the only alternative signal. No rollout oracle deploys.
    mc_blocks,mc_targets,receipts=monte_carlo_actor_labels(output,values,policies[1],reward_scale=manifest['reward_scale'])
    write_json(target/'mc_label_receipts.json',receipts)
    for split,block in mc_blocks.items():
        errors,_=actor_errors(policies[1].model.actors,block,mc_targets[split],policies[1])
        summary['references'].append(dict(model='original_t_q',teacher='mc_candidates',split=split,**errors))
    pilots=[]
    for lr in rates:
        actors,history=fit_actor_supervised(construct_actors('transformer',policies[1],env,seeds[0],warm=True),
            mc_blocks['fit'],mc_blocks['dev'],mc_targets['fit'],mc_targets['dev'],seed=seeds[0],learning_rate=lr,
            updates=updates,batch_size=manifest['batch_size'],device=device)
        selected=min(history,key=lambda r:r['dev_loss'])
        pilots.append((selected['dev_loss'],lr,actors,history))
        write_json(target/f'mc_candidates_t_warm_lr{lr}_pilot_curve.json',history)
        summary['learning_rate_pilots'].append(dict(teacher='mc_candidates',architecture='t_warm',learning_rate=lr,**selected))
        print(f'Actor MC recovery dev lr={lr}: loss={selected["dev_loss"]:.5f}',flush=True)
    chosen=min(pilots,key=lambda v:v[0]);lr=chosen[1]
    for seed in seeds:
        if seed==seeds[0]:actors,history=chosen[2:]
        else:actors,history=fit_actor_supervised(construct_actors('transformer',policies[1],env,seed,warm=True),
            mc_blocks['fit'],mc_blocks['dev'],mc_targets['fit'],mc_targets['dev'],seed=seed,learning_rate=lr,
            updates=updates,batch_size=manifest['batch_size'],device=device)
        ident=f'mc_candidates_t_warm_seed{seed}'
        torch.save(dict(diagnostic_only=True,architecture='transformer',warm=True,teacher='mc_candidates',seed=seed,
            learning_rate=lr,selected_step=actors.selected_step,state_dict=actors.cpu().state_dict()),target/f'{ident}.pt')
        actors.to(device);write_json(target/f'{ident}_curve.json',history)
        for split,block in mc_blocks.items():
            errors,pred=actor_errors(actors,block,mc_targets[split],policies[1])
            np.savez_compressed(target/f'{ident}_{split}_predictions.npz',logits=pred)
            summary['models'].append(dict(model=ident,teacher='mc_candidates',architecture='t_warm',
                random_initialization=False,seed=seed,split=split,learning_rate=lr,selected_step=actors.selected_step,**errors))
        print(f'Actor MC recovery finished {ident}',flush=True)
        write_json(target/'summary.json',summary)
    if source_tensors!=[_tensor_fingerprint(p.model) for p in policies]:raise ValueError('Original Actor/Critic tensors changed during fitting')
    summary['source_tensors_unchanged']=True
    summary['status']='complete';summary['code_sha256']=EXECUTION_SOURCE_SHA256
    write_json(target/'summary.json',summary)
    return summary


def actions_from_modules(actors,observations):
    with torch.no_grad():
        return {mg:actor(torch.as_tensor(observations[mg],dtype=torch.float32)[None])[0].numpy()
                for mg,actor in actors.items()}


def evaluate_modules(env,actors,days,seed,forecast_noise):
    from evaluation.evaluate_algorithm import evaluate_policy_on_days
    actors.cpu().eval()
    return evaluate_policy_on_days(env,days,lambda obs:actions_from_modules(actors,obs),seed=seed,
                                   forecast_noise=forecast_noise,capture_hourly=True)


def flatten_policy_days(env,daily,hourly):
    from evaluation.service_metrics import summarize_day_services,SERVICE_METRICS_SCHEMA_VERSION
    rows=[]
    for d in daily:
        service=summarize_day_services(hourly[d['date']],env.config,
            d['constraint_statistics']['physical_feasible_episode'],SERVICE_METRICS_SCHEMA_VERSION)
        if not service['data_complete']:raise ValueError('Incomplete policy service evidence')
        hvac,mill,ev=service['hvac'],service['mill'],service['ev']
        raw=d['reward_v2_raw_components']
        rows.append(dict(date=d['date'],seed=d['test_seed'],cost=d['generalized_system_cost'],
            physical_profit=d['team_profit'],carbon_kg=raw['carbon_kg'],peak_kw=d['team_actual_load_peak_kw'],
            p2p_kwh=d['p2p_total_energy_kwh'],curtailment_kwh=d['curtailment_kwh'],
            unmet_kwh=service['unmet_energy_kwh'],comfort_degree_hours=hvac['comfort_violation_degree_hours'],
            occupied_comfort_success=hvac['occupied_success_count'],occupied_comfort_count=hvac['occupied_boundary_count'],
            mill_unserved_t=mill['downstream_unserved_t'],ev_shortfall_kwh=ev['departure_shortfall_kwh'],
            ev_success_count=ev['completed_group_count'],ev_departure_count=ev['departure_count'],
            strict_feasible_day=service['strict_feasible_day'],dr_infeasible_hours=service['dr_supply_infeasible_hour_count'],
            hvac_terminal_failure=bool(hvac['terminal_applicable'] and not hvac['terminal_success']),
            mill_terminal_failure=bool(mill['terminal_applicable'] and not mill['terminal_success']),
            physical_feasible_day=service['physical_feasible_day'],service_data_complete=service['data_complete']))
    return rows


def service_no_worse(candidate,baseline):
    a={(r['date'],r['seed']):r for r in candidate};b={(r['date'],r['seed']):r for r in baseline}
    if set(a)!=set(b):raise ValueError('Service guard day/seed pairs disagree')
    for key in a:
        x,y=a[key],b[key]
        if not x['service_data_complete'] or not x['physical_feasible_day']:return False
        for field in ('unmet_kwh','ev_shortfall_kwh','mill_unserved_t','comfort_degree_hours','dr_infeasible_hours',
                      'hvac_terminal_failure','mill_terminal_failure'):
            if float(x[field])>float(y[field])+1e-6:return False
        if x['occupied_comfort_success']<y['occupied_comfort_success']:return False
        if x['ev_success_count']<y['ev_success_count']:return False
    return True


def load_value_checkpoint(path,manifest,values,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('diagnostic_only') is not True:raise ValueError('Not a diagnostic Critic checkpoint')
    dimension=values['fit']['x'].shape[1]
    network=DenseValue(dimension) if saved['architecture']=='mlp' else SequenceValue(manifest['state_schema'],dimension-len(manifest['state_schema']))
    result=NormalizedValue(network,values['fit'])
    result.load_state_dict(saved['state_dict']);result.to(device).eval().requires_grad_(False)
    return result


def short_actor_transfer(value,policy,env,train,dev_days,*,seed,forecast_noise,smoke=False):
    """Identical frozen-Q Actor updates; dev service guard always allows step zero."""
    device=next(value.parameters()).device
    actors=copy.deepcopy(policy.model.actors).to(device).requires_grad_(True).train()
    observations={mg:torch.as_tensor(train[mg],dtype=torch.float32,device=device) for mg in actors}
    states=torch.as_tensor(train['state'],dtype=torch.float32,device=device)
    with torch.no_grad():original=torch.as_tensor(predict_actors(policy.model.actors,train),device=device)
    optimizer=torch.optim.Adam(actors.parameters(),lr=2e-4)
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    snapshots=[0,10] if smoke else [0,100,250,500]
    selected=None;selected_step=0;selected_cost=None;history=[];baseline=None
    for step in range(snapshots[-1]+1):
        if step:
            actors.to(device).train()
            idx=torch.as_tensor(rng.integers(len(states),size=256),device=device)
            raw=torch.stack([a(observations[mg][idx]) for mg,a in actors.items()],1)
            effective,_=policy.model.project_actions(states[idx].to('cpu'),raw.to('cpu'))
            effective=effective.to(device)
            q=value(torch.cat([states[idx],effective.flatten(1)],-1))
            loss=-q.mean()/value.gain_std+20*F.mse_loss(raw,original[idx])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(actors.parameters(),10.);optimizer.step()
        if step in snapshots:
            _,daily,hourly=evaluate_modules(env,actors,dev_days,427777,forecast_noise)
            rows=flatten_policy_days(env,daily,hourly)
            if step==0:baseline=rows
            cost=float(np.mean([r['cost'] for r in rows]));guard=service_no_worse(rows,baseline)
            history.append(dict(step=step,dev_cost_mean=cost,service_no_worse=guard,daily=rows))
            if step==0 or (guard and cost<selected_cost):
                selected={k:v.detach().cpu().clone() for k,v in actors.state_dict().items()}
                selected_cost=cost;selected_step=step
    actors.load_state_dict(selected);actors.cpu().eval().requires_grad_(False)
    return actors,selected_step,history


def stage_policy(root,output,manifest,values,blocks):
    from envs.multi_microgrid_env import MultiMicrogridEnv
    from evaluation.evaluate_algorithm import evaluate_policy_on_days
    target=output/'policy'
    if target.exists():raise ValueError('Policy verification output already exists')
    for stage in ('critic','actor'):
        if read_json(output/stage/'summary.json')['status']!='complete':raise ValueError(f'{stage} stage incomplete')
    target.mkdir();torch.set_num_threads(1)
    _,_,_,cfg,policies=sources_and_config(root,Path(manifest['phase_one_directory']))
    source_tensors=[_tensor_fingerprint(p.model) for p in policies]
    val_env=MultiMicrogridEnv(cfg,split='val');dev_env=MultiMicrogridEnv(cfg,split='train')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    seeds=manifest['initialization_seeds'][:1] if manifest['smoke'] else manifest['initialization_seeds']
    summary=dict(status='running',diagnostic_only=True,models=[],transfers=[],forecast_seed_bases=[814157,915167],
        critic_transfer_learning_rate=2e-4,critic_transfer_raw_trust_weight=20,critic_transfer_snapshots=[0,100,250,500],
        confirmation_days=manifest['confirmation_days'])
    def verify(name,policy_fn,category,attributes=None):
        rows=[]
        for seed in summary['forecast_seed_bases']:
            aggregate,daily,hourly=evaluate_policy_on_days(val_env,manifest['confirmation_days'],policy_fn,seed=seed,
                forecast_noise=manifest['forecast_noise'],capture_hourly=True)
            rows.extend(flatten_policy_days(val_env,daily,hourly))
            write_json(target/f'{name}_seed{seed}_daily.json',dict(aggregate=aggregate,daily=daily))
            with gzip.open(target/f'{name}_seed{seed}_hourly.json.gz','wt',encoding='utf-8') as f:
                json.dump(hourly,f,ensure_ascii=False,allow_nan=False)
        fields=('cost','physical_profit','carbon_kg','peak_kw','p2p_kwh','curtailment_kwh','unmet_kwh',
                'comfort_degree_hours','mill_unserved_t','ev_shortfall_kwh')
        averaged={f+'_mean':float(np.mean([r[f] for r in rows])) for f in fields}
        c=sum(r['occupied_comfort_count'] for r in rows);e=sum(r['ev_departure_count'] for r in rows)
        item=dict(model=name,category=category,daily=rows,**averaged,
            occupied_comfort_rate=sum(r['occupied_comfort_success'] for r in rows)/c if c else None,
            ev_departure_success_rate=sum(r['ev_success_count'] for r in rows)/e if e else None,
            physical_feasible_day_rate=float(np.mean([r['physical_feasible_day'] for r in rows])),
            strict_feasible_day_rate=float(np.mean([r['strict_feasible_day'] for r in rows])),**(attributes or {}))
        summary['models'].append(item)
        write_json(target/'summary.json',summary)
        print(f'Policy {name}: mean cost={item["cost_mean"]:.3f}, comfort={item["occupied_comfort_rate"]:.3f}',flush=True)
        return item
    for name,policy in (('original_matd3_q',policies[0]),('mlp_h8_teacher',MaskedTeacher(policies[0])),('original_t_q',policies[1])):
        verify(name,policy.actions,'source_reference')
    for path in sorted((output/'actor').glob('*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('diagnostic_only') is not True:raise ValueError('Not a diagnostic Actor checkpoint')
        actors=construct_actors(saved['architecture'],policies[1],dev_env,saved['seed'],warm=saved['warm']).cpu()
        actors.load_state_dict(saved['state_dict']);actors.eval().requires_grad_(False)
        verify(path.stem,lambda obs,modules=actors:actions_from_modules(modules,obs),'supervised_actor',
            dict(teacher=saved['teacher'],architecture='t_warm' if saved['warm'] else saved['architecture'],seed=saved['seed']))
    # Same Actor and update budget, changing only the frozen fitted value function.
    for kind in ('mlp','transformer'):
        for seed in seeds:
            name=f'critic_transfer_{kind}_seed{seed}'
            value=load_value_checkpoint(output/'critic'/f'{kind}_seed{seed}.pt',manifest,values,device)
            actors,step,history=short_actor_transfer(value,policies[1],dev_env,blocks['fit'],manifest['dev_days'],
                seed=seed,forecast_noise=manifest['forecast_noise'],smoke=manifest['smoke'])
            write_json(target/f'{name}_dev_selection.json',history)
            torch.save(dict(diagnostic_only=True,source_actor='original_t_q',fitted_critic=kind,seed=seed,
                selected_step=step,state_dict=actors.state_dict()),target/f'{name}.pt')
            summary['transfers'].append(dict(model=name,critic=kind,seed=seed,selected_step=step,dev_history=history))
            verify(name,lambda obs,modules=actors:actions_from_modules(modules,obs),'critic_transfer',dict(critic=kind,seed=seed,selected_step=step))
    baseline=next(r for r in summary['models'] if r['model']=='original_t_q')
    for r in summary['models']:
        r['paired_cost_vs_current_t']=paired_day_difference(r['daily'],baseline['daily'],field='cost')
        r['confirmation_service_no_worse_vs_current_t']=service_no_worse(r['daily'],baseline['daily'])
    if source_tensors!=[_tensor_fingerprint(p.model) for p in policies]:raise ValueError('Original Actor/Critic tensors changed during policy verification')
    summary['source_tensors_unchanged']=True
    summary['status']='complete';summary['code_sha256']=EXECUTION_SOURCE_SHA256
    write_json(target/'summary.json',summary)
    return summary


def aggregate_models(rows,fields):
    return {f:dict(mean=float(np.mean([r[f] for r in rows])),std=float(np.std([r[f] for r in rows])),
        values=[float(r[f]) for r in rows]) for f in fields}


def stage_report(root,output,manifest,values,blocks):
    stages={n:read_json(output/n/'summary.json') for n in ('critic','actor','policy')}
    if any(r['status']!='complete' for r in stages.values()):raise ValueError('All fitting and policy stages must complete')
    critic,actor,policy=(stages[n] for n in ('critic','actor','policy'))
    result=dict(status='complete',diagnostic_only=True,smoke=manifest['smoke'],
        fit_days=manifest['fit_days'],dev_days=manifest['dev_days'],primary_confirmation_days=manifest['confirmation_days'],
        excluded_smoke_dates=manifest.get('excluded_smoke_dates',[]),audit_days=manifest['audit_days'],
        sources=manifest['sources'],critic={},actor={},policy={},comparisons={},
        critic_parameter_counts=critic['architecture_parameters'],actor_parameter_counts=actor['parameter_counts'],
        counts=dict(value_fit_rows=len(values['fit']['y']),value_dev_rows=len(values['dev']['y']),
            value_confirmation_rows=len(values['confirmation']['y']),actor_fit_rows=len(blocks['fit']['day']),
            actor_dev_rows=len(blocks['dev']['day']),actor_confirmation_rows=len(blocks['confirmation']['day'])),
        scope='Frozen-policy return fitting and supervised/short-adaptation diagnostics; not full RL placement superiority',
        source_files_unchanged=True,actor_source_tensors_unchanged=actor.get('source_tensors_unchanged',False),
        policy_source_tensors_unchanged=policy.get('source_tensors_unchanged',False))
    for split in ('fit','dev','confirmation','audit'):
        for kind in ('mlp','transformer'):
            rows=[r for r in critic['models'] if r['split']==split and r['architecture']==kind]
            result['critic'][f'{kind}_{split}']=aggregate_models(rows,('mean_selection_regret_cost','rank_accuracy','gain_mae_cost','value_mae_cost'))
        if split in ('confirmation','audit'):
            a=[d for r in critic['models'] if r['split']==split and r['architecture']=='transformer' for d in r['daily']]
            b=[d for r in critic['models'] if r['split']==split and r['architecture']=='mlp' for d in r['daily']]
            result['comparisons'][f'critic_t_minus_mlp_{split}']=paired_day_difference(a,b,field='regret')
    result['critic_references']=[r for r in critic['references'] if r['split']=='confirmation']
    for teacher in ('mlp_h8','current_t','mc_candidates'):
        for kind in ('raw_mlp','semantic_mlp','transformer','t_warm'):
            rows=[r for r in actor['models'] if r['teacher']==teacher and r['architecture']==kind and r['split']=='confirmation']
            if rows:result['actor'][f'{teacher}_{kind}']=aggregate_models(rows,('action_rmse','logit_mae','effective_action_rmse'))
    result['actor_references']=[r for r in actor['references'] if r['split']=='confirmation']
    for r in policy['models']:
        result['policy'][r['model']]={k:v for k,v in r.items() if k!='daily'}
    baseline=next(r for r in policy['models'] if r['model']=='original_t_q')
    families={}
    for r in policy['models']:
        if r['category']=='source_reference':continue
        label=f"{r.get('teacher','transfer')}_{r.get('architecture',r.get('critic'))}"
        families.setdefault(label,[]).append(r)
    result['policy_families']={}
    for label,rows in families.items():
        daily=[d for r in rows for d in r['daily']]
        result['policy_families'][label]=dict(model_count=len(rows),
            cost=aggregate_models(rows,('cost_mean','comfort_degree_hours_mean','carbon_kg_mean','peak_kw_mean','p2p_kwh_mean')),
            paired_cost_vs_original_t=paired_day_difference(daily,baseline['daily'],field='cost'),
            service_no_worse_seed_count=sum(r['confirmation_service_no_worse_vs_current_t'] for r in rows))
    transfer_effect=[]
    for kind in ('mlp','transformer'):
        rows=[r for r in policy['models'] if r['category']=='critic_transfer' and r['critic']==kind]
        transfer_effect.append(dict(critic=kind,selected_updates=[r['selected_step'] for r in rows],
            mean_cost_difference=float(np.mean([r['cost_mean']-baseline['cost_mean'] for r in rows])),
            safe_confirmation_seed_count=sum(r['confirmation_service_no_worse_vs_current_t'] for r in rows)))
    result['critic_transfer']=transfer_effect
    contrast=result['comparisons']['critic_t_minus_mlp_confirmation']
    stable=True
    for seed in manifest['initialization_seeds'][:1] if manifest['smoke'] else manifest['initialization_seeds']:
        a=next(r for r in critic['models'] if r['split']=='confirmation' and r['architecture']=='transformer' and r['seed']==seed)
        b=next(r for r in critic['models'] if r['split']=='confirmation' and r['architecture']=='mlp' and r['seed']==seed)
        stable &= a['mean_selection_regret_cost']<b['mean_selection_regret_cost']
    mean_t=result['critic']['transformer_confirmation'];mean_m=result['critic']['mlp_confirmation']
    supported=bool(stable and contrast['descriptive_95_interval'][1]<0 and
        mean_t['rank_accuracy']['mean']>=mean_m['rank_accuracy']['mean'] and
        mean_t['gain_mae_cost']['mean']<mean_m['gain_mae_cost']['mean'])
    result['bounded_verdict']=dict(transformer_critic_advantage_supported=supported,
        explanation='Strictly a statement about these fixed-data structures, dates and fitting budget; no causal blame percentage or universal proof')
    controls=[]
    for kind in ('mlp','transformer'):
        details=[d for r in critic['models'] if r['split']=='confirmation' and r['architecture']==kind
                 for d in read_json(output/'critic'/f"{r['model']}_confirmation_choices.json")]
        for control in sorted({d['control'] for d in details}):
            rows=[d for d in details if d['control']==control]
            pairs=sum(d['rank_pair_count'] for d in rows)
            controls.append(dict(model=kind,control=control,
                regret=float(np.mean([d['selection_regret_cost'] for d in rows])),
                rank=sum(d['rank_correct_count'] for d in rows)/pairs if pairs else None))
    result['critic_by_control']=controls
    by_control={(r['model'],r['control']):r['regret'] for r in controls}
    total_gain=sum(by_control['mlp',c]-by_control['transformer',c] for c in sorted({r['control'] for r in controls}))
    mg2_gain=by_control['mlp','mg2_DR']-by_control['transformer','mg2_DR']
    result['mg2_dr_net_regret_gain_share']=mg2_gain/total_gain if total_gain>0 else None
    result['mc_recovery_selected_steps']=[r['selected_step'] for r in actor['models']
        if r['teacher']=='mc_candidates' and r['split']=='confirmation']
    result['report_code_sha256']=EXECUTION_SOURCE_SHA256
    result['fitting_code_sha256']={name:stages[name]['code_sha256'] for name in stages}
    write_json(output/'analysis_verified.json',result)
    lines=['# Actor / Critic 结构能力诊断','',
        '这是独立诊断，读取两个完整5000回合训练的最佳模型。原算法、原模型及已有结果保持不变。',
        'Critic使用原H24信息；Actor保持H8。学习率与检查点只用开发日期选择，主要确认日期不用于调参。','',
        f"拟合{len(manifest['fit_days'])}天，开发{len(manifest['dev_days'])}天，主要确认{len(manifest['confirmation_days'])}天，旧审计{len(manifest['audit_days'])}天。",
        f"主要确认日期：{', '.join(manifest['confirmation_days'])}。试跑已看过日期{manifest.get('excluded_smoke_dates',[])}只保留作复用证据。",'',
        '## 本轮支持的决策','',
        ('下一轮结构试验优先选择同H8信息的简单MLP Actor + Transformer Critic，重点检验MG2需求响应评价。'
         if supported else '本轮固定数据判据没有支持Transformer Critic优势，应先检查学习信号和拟合。'),
        '本轮没有获得通过逐日服务筛选的闭环收益，不能宣布新算法优于MATD3-Q，也不能给出Actor/Critic的因果责任比例。','',
        '## 相同回报监督下的 Critic 比较','',
        '下面的误选代价是每个状态/控制组的候选损失，不能加总为日节省。标准差来自三个初始化，整日区间保留日期相关性。','',
        '| 结构 | 参数量 | 主要确认平均误选代价 | 排序正确率 | 动作收益差MAE |','|---|---:|---:|---:|---:|']
    for kind in ('mlp','transformer'):
        r=result['critic'][f'{kind}_confirmation']
        lines.append(f"| {kind} | {critic['architecture_parameters'][kind]} | {r['mean_selection_regret_cost']['mean']:.4f} ± {r['mean_selection_regret_cost']['std']:.4f} | {r['rank_accuracy']['mean']:.2%} | {r['gain_mae_cost']['mean']:.4f} |")
    lines += ['',f"Transformer减MLP的整日配对误选代价差为{contrast['mean_difference']:.4f}，描述性95%区间{contrast['descriptive_95_interval']}。",
        f"预设固定数据优势判据通过：{supported}。判据要求三个初始化代价均降低、描述性日期bootstrap区间低于0、排序与收益差误差不恶化。",
        f"相对同预算新拟合MLP，Transformer平均误选代价降低{1-mean_t['mean_selection_regret_cost']['mean']/mean_m['mean_selection_regret_cost']['mean']:.2%}，{contrast['improved_dates']}/{contrast['date_count']}个确认日期改善。该比例不是相对原训练Q的改善。",'',
        '## 优势集中在哪个控制量','',
        '| 控制量 | 新MLP误选代价 | 新Transformer误选代价 |','|---|---:|---:|']
    for control in sorted({r['control'] for r in controls}):
        lines.append(f"| {control} | {by_control['mlp',control]:.4f} | {by_control['transformer',control]:.4f} |")
    share=result['mg2_dr_net_regret_gain_share']
    if share is not None:
        lines += ['',f"MG2 DR贡献全部控制量净误选改善的{share:.1%}。这是各控制组均值的分解，不是每日成本贡献。"]
    worse=[c for c in sorted({r['control'] for r in controls}) if by_control['transformer',c]>by_control['mlp',c]]
    if worse:
        lines.append(f"误选代价增加的控制量：{', '.join(worse)}。因此优势不是所有通道全面改善，也不支持直接移除既有ESS辅助机制。")
    lines += ['',
        '原训练Q与新监督模型的训练方式不同，以下只作已有信号参考，不能当作相同训练预算的架构对比：']
    for r in result['critic_references']:
        lines.append(f"- {r['model']}：误选代价{r['mean_selection_regret_cost']:.4f}，排序{r['rank_accuracy']:.2%}，收益差MAE{r['gain_mae_cost']:.4f}。")
    lines += ['','## Actor 函数表达与恢复','',
        '两个可执行H8教师均非最优策略。MLP教师屏蔽其训练时的远期预测，必须结合下方闭环质量解释。',
        '原始输入MLP和共同特征MLP用来分开检查信息压缩与结构；共同特征与T的非学习输入一致。MC事后标签只监督原T副本，单独报告。','',
        '| 标签来源 | 学生 | 原动作RMSE | 投影有效动作RMSE |','|---|---|---:|---:|']
    for name,r in result['actor'].items():
        teacher=next(t for t in ('mlp_h8','current_t','mc_candidates') if name.startswith(t+'_'))
        lines.append(f"| {teacher} | {name} | {r['action_rmse']['mean']:.4f} | {r['effective_action_rmse']['mean']:.4f} |")
    actor_mlp_lower=all(result['actor'][t+'_semantic_mlp']['action_rmse']['mean']<
        result['actor'][t+'_transformer']['action_rmse']['mean'] for t in ('mlp_h8','current_t'))
    lines += ['',
        f"两个教师下共同特征MLP的平均动作RMSE都低于T：{actor_mlp_lower}。这是本轮预算下的拟合比较，不能据此判断Actor没有问题或容量充分。",
        f"本轮MC恢复所选更新步数为{result['mc_recovery_selected_steps']}。"+
        ('开发集未认可非零更新，未证实恢复。' if not any(result['mc_recovery_selected_steps'])
         else '恢复性判断须结合下方真实闭环成本与服务，不能仅由事后标签误差推断。')]
    lines += ['','## 真实闭环成本与服务','',
        '每日综合成本来自共享评估器，包含购电、碳、削峰、弃电、设备服务等原奖励项。舒适度按人数/时刻计数汇总，不能靠供电物理可行性代替服务。','',
        '原MATD3-Q参考Actor保留H24预测，原T-Q Actor使用H8；这两个原策略的闭环差异不能用于同信息结构优劣归因。','',
        '| 策略 | 日均综合成本 | 占用时段舒适率 | EV离场成功率 | 日均碳排放kg |','|---|---:|---:|---:|---:|']
    for r in policy['models']:
        lines.append(f"| {r['model']} | {r['cost_mean']:.3f} | {r['occupied_comfort_rate']:.2%} | {r['ev_departure_success_rate']:.2%} | {r['carbon_kg_mean']:.3f} |")
    lines += ['','## Critic向同一Actor的传递','',
        '这里只做冻结拟合Q、动作信任惩罚和500步离线联合适配，没有正式TD3重训或原ESS辅助项。开发服务守卫失败则保留0步原策略。']
    for r in transfer_effect:
        lines.append(f"- {r['critic']}：开发选中更新步数{r['selected_updates']}；确认日均成本相对原T差{r['mean_cost_difference']:.4f}；服务不恶化种子{r['safe_confirmation_seed_count']}。")
    lines += ['','## 结论边界','',
        '- 新MLP能学好的差异支持检查原训练信号/数据，并不意味着原MATD3 Critic结构本身失败。',
        '- 即使T Actor能学习另一映射或在可靠回报监督下恢复，也只能支持表达和优化可达性；本轮是否恢复以所选更新步数和真实闭环表现为准。无法给出Actor/Critic各占多少责任。',
        '- 两端分别使用H8/H24输入，结果只支持当前信息方案下的结构试验优先级，不能证明Transformer天然只适合Critic端。',
        '- 若排序优势不能转化为独立闭环收益，不能据此承诺加Transformer后完整训练会更好。',
        '- 主确认日期没有参与本轮诊断设计，但源最佳模型曾用整个验证集选择；它们并非全新独立正式测试。',
        '- 正式“仅Actor/仅Critic加Transformer谁更强”仍需同信息、同训练预算的端到端训练。当前结论应明确为下一步结构优先级。','',
        '所有原来源哈希在报告前重新核对。拟合模型位于诊断目录，不能作为正式最佳模型。','']
    (output/'diagnosis.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Report saved: {output / "diagnosis.md"}',flush=True)
    return result


def run_stage(root,output,stage):
    root,output=Path(root).resolve(),Path(output).resolve()
    manifest,values,actors=validated_data(root,output)
    if stage=='critic':return stage_critic(root,output,manifest,values)
    if stage=='actor':return stage_actor(root,output,manifest,actors,values)
    if stage=='policy':return stage_policy(root,output,manifest,values,actors)
    if stage=='report':return stage_report(root,output,manifest,values,actors)
    raise ValueError(f'Stage {stage} is not yet implemented')
