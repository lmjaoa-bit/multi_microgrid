"""Read saved A3 runs; reconstruct recorded observations without changing policy or training.

Run from the repository root. Reconstruction replays SAVED actions, verifies
saved dispatch/SOC/profit, and never runs a new policy test or coalition MILP.
All figures/report data consume the derived evidence.json, not the environment.
"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv

torch.set_num_threads(1)
SLUGS = ['matd3_plain_single_dispatch', 'crossgate_single_dispatch']
IDS = ['mg1', 'mg2', 'mg3']
RUNS = {
    SLUGS[0]: ROOT / 'outputs/matd3_plain_single_dispatch/runs/MATD3PLAINA3_2026-09-04_16-19-49_seed42',
    SLUGS[1]: ROOT / 'outputs/crossgate_single_dispatch/runs/CROSSGATESINGLEDISPATCH_2026-09-04_14-03-59_seed42',
}


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


data = {}
result = {'runs': {}, 'method': 'Recorded 55 day pairs; CrossGate minus plain; no retraining or gate removal.'}
for slug, path in RUNS.items():
    cfg = yaml.safe_load((path / 'config_snapshot.yaml').read_text(encoding='utf-8'))
    daily = pd.read_csv(path / 'test_daily_metrics.csv').set_index('date').sort_index()
    hourly = pd.concat([pd.read_csv(p) for p in sorted((path / 'test_hourly_trajectories').glob('*.csv'))]).set_index(['date', 'hour']).sort_index()
    hist = read_json(path / 'training_history.json')
    train = read_json(path / 'training_summary.json')
    test = read_json(path / 'test_summary.json')
    diag = pd.read_csv(path / 'actor_diagnostics.csv')
    assert len(daily) == 55 and len(hourly) == 55 * 24
    assert daily.index.is_unique and hourly.index.is_unique
    assert abs(daily.algorithm_team_profit.mean() - test['joint_team_profit_mean']) < 1e-9
    for day, frame in hourly.groupby(level='date'):
        assert abs(frame[[f'{mg}_hourly_profit' for mg in IDS]].to_numpy().sum() - daily.loc[day, 'algorithm_team_profit']) < 1e-7
    spec = get_algorithm_spec(slug)
    env_cfg = cfg['environment'].copy()
    env_cfg['_project_root'] = str(ROOT)
    env = MultiMicrogridEnv(env_cfg, split='test')
    model = spec.create(env, cfg['algorithm'], seed=42)
    model.load(path / 'checkpoints/best_model.pt')
    data[slug] = dict(path=path, cfg=cfg, daily=daily, hourly=hourly, history=hist, train=train, test=test, diag=diag, env=env, model=model)
    best = train['best_episode']
    window = diag[(diag.episode > best - 100) & (diag.episode <= best)]
    metrics = ['total_grad_norm', 'gate_grad_norm', 'local_feature_grad_norm', 'market_feature_grad_norm', 'peer_feature_grad_norm', 'dispatch_action_grad_rms', 'dispatch_saturation_rate', 'forced_dispatch_rate', 'offer_eligible_rate', 'clip_applied']
    window_stats = {}
    for mg, part in window.groupby('microgrid'):
        window_stats[mg] = {m: float(np.average(part[m+'_mean'], weights=part[m+'_count'])) for m in metrics if m+'_mean' in part}
    result['runs'][slug] = dict(path=str(path), best_episode=best,
        parameter_counts=model.actor_parameter_counts(),
        best100_training_diagnostics=window_stats,
        validation=[dict(episode=r['episode'],profit=r['team_profit_mean']) for r in read_json(path/'validation_history.json')],
        physical_config_hash=cfg['experiment_contract']['physical_config_hash'],
        checkpoint_sha256=hashlib.sha256((path/'checkpoints/best_model.pt').read_bytes()).hexdigest())

plain, cross = (data[s] for s in SLUGS)
assert plain['daily'].index.equals(cross['daily'].index)
assert plain['daily'].test_seed.equals(cross['daily'].test_seed)
assert plain['cfg']['environment'] == cross['cfg']['environment']
delta = cross['daily'].algorithm_team_profit - plain['daily'].algorithm_team_profit
result['paired'] = dict(mean=float(delta.mean()), std=float(delta.std()), median=float(delta.median()), wins=int((delta>0).sum()), losses=int((delta<0).sum()), total=float(delta.sum()))
result['daily'] = [dict(date=d,month=d[:7],plain=float(plain['daily'].loc[d,'algorithm_team_profit']),crossgate=float(cross['daily'].loc[d,'algorithm_team_profit']),gain=float(delta[d])) for d in delta.index]
result['monthly'] = [dict(month=month,count=len(part),gain_mean=float(part.mean()),gain_total=float(part.sum())) for month,part in delta.groupby(delta.index.str[:7])]

# Exact accounting bridge. Residual costs include curtailment, terminal and unmet load.
economic = {}
for slug in SLUGS:
    h = data[slug]['hourly']
    economic[slug] = {k: float(h[[f'{mg}_{k}' for mg in IDS]].to_numpy().sum()/55) for k in ['grid_import_cost','grid_export_revenue','p2p_purchase_cost','p2p_sale_revenue','battery_degradation_cost','hourly_profit']}
    e=economic[slug]
    e['other_cost'] = e['grid_export_revenue'] - e['grid_import_cost'] + e['p2p_sale_revenue'] - e['p2p_purchase_cost'] - e['battery_degradation_cost'] - e['hourly_profit']
    assert abs(e['p2p_sale_revenue'] - e['p2p_purchase_cost']) < 1e-8
result['economics'] = economic
result['bridge'] = [dict(component=k,plain=economic[SLUGS[0]][k],crossgate=economic[SLUGS[1]][k],gain=sign*(economic[SLUGS[1]][k]-economic[SLUGS[0]][k])) for k,sign in [('grid_import_cost',-1),('grid_export_revenue',1),('battery_degradation_cost',-1),('other_cost',-1)]]
assert abs(sum(r['gain'] for r in result['bridge'])-delta.mean()) < 1e-7
result['microgrids'] = []
result['microgrid_external'] = []
for mg in IDS:
    row = dict(microgrid=mg)
    for field in ['physical_profit','ess_offer','ess_accepted','battery_charge','battery_discharge']:
        key = f'{field}_{mg}' + ('_kwh' if field != 'physical_profit' else '')
        row[field+'_plain']=float(plain['daily'][key].mean())
        row[field+'_crossgate']=float(cross['daily'][key].mean())
    row['profit_gain']=row['physical_profit_crossgate']-row['physical_profit_plain']
    result['microgrids'].append(row)
    transfers=[]
    for slug in SLUGS:
        h=data[slug]['hourly']
        transfers.append(float((h[f'{mg}_p2p_sale_revenue'].sum()-h[f'{mg}_p2p_purchase_cost'].sum())/55))
    result['microgrid_external'].append(dict(mg=mg,physical_profit_gain=row['profit_gain'],
        internal_transfer_gain=transfers[1]-transfers[0],
        external_profit_gain=row['profit_gain']-(transfers[1]-transfers[0])))

result['mg3_terminal']=[]
for slug in SLUGS:
    h=data[slug]['hourly']
    r23=h.xs(23,level='hour'); r22=h.xs(22,level='hour')
    result['mg3_terminal'].append(dict(algorithm=slug,
        hour22_soc_mean=float(r22.mg3_soc.mean()),
        hour23_charge_mean=float(r23.mg3_battery_charge_kw.mean()),
        hour23_grid_cost_mean=float(r23.mg3_grid_import_cost.mean()),
        hour23_forced_rate=float(r23.mg3_dispatch_forced.mean())))
    result['runs'][slug]['aligned_training_windows']={}
    for end in [2400,5000]:
        df=data[slug]['diag'];window=df[(df.episode>end-100)&(df.episode<=end)]
        result['runs'][slug]['aligned_training_windows'][str(end)]={mg:{
            m:float(np.average(g[m+'_mean'],weights=g[m+'_count']))
            for m in ['clip_applied','dispatch_saturation_rate']}
            for mg,g in window.groupby('microgrid')}

# Dispatch evidence from saved trajectories; these are test statistics, not replay batches.
for slug in SLUGS:
    h=data[slug]['hourly']
    result['runs'][slug]['test_dispatch']={mg:{
        'saturation_rate':float((h[f'{mg}_raw_action_0'].abs()>=.95).mean()),
        'forced_rate':float(h[f'{mg}_dispatch_forced'].mean()),
        'offer_cap_positive_rate':float((h[f'{mg}_ess_p2p_offer_cap_kw']>1e-6).mean()),
        'positive_offer_rate':float((h[f'{mg}_ess_p2p_offer_kw']>1e-6).mean()),
        'zero_physical_action_rate':float(((h[f'{mg}_effective_local_action'].abs()<1e-6)&(h[f'{mg}_effective_trade_action'].abs()<1e-6)).mean()),
        'offer_acceptance_rate':float(h[f'{mg}_ess_p2p_sell_kw'].sum()/max(h[f'{mg}_ess_p2p_offer_kw'].sum(),1e-12)),
    } for mg in IDS}
    result['runs'][slug]['test_energy']={k:float(data[slug]['daily'][k].mean()) for k in ['p2p_total_energy_kwh','grid_import_kwh','grid_export_kwh','battery_charge_kwh','battery_discharge_kwh','ess_offer_energy_kwh','ess_accepted_energy_kwh','renewable_curtailment_kwh']}

print('Saved-result accounting complete',result['paired'],flush=True)

# Reconstruct each model's already-recorded test observations with its saved raw actions.
# Verify both saved action inference and hourly physical outcomes before using sensitivities.
observation_sets={}
for slug in SLUGS:
    d=data[slug]; env=d['env']; model=d['model']
    obs_buffer={mg:[] for mg in IDS}; states=[]; max_error=0.
    for day, row in d['daily'].iterrows():
        obs,info=env.reset(day=day,seed=int(row.test_seed),deterministic=True)
        for hour,r in d['hourly'].loc[day].iterrows():
            for mg in IDS: obs_buffer[mg].append(obs[mg].copy())
            states.append(info['global_state'] if hour==0 else info['global_state_after'])
            actions={mg:np.array([r[f'{mg}_raw_action_0']],dtype=np.float32) for mg in IDS}
            obs,_,done,info=env.step(actions)
            for mg in IDS:
                max_error=max(max_error,abs(env.cores[mg].battery.soc-r[f'{mg}_soc']))
                max_error=max(max_error,abs(info['devices'][mg]['discharge_kw']-r[f'{mg}_battery_discharge_kw']))
            max_error=max(max_error,abs(info['team_profit']-sum(r[f'{mg}_hourly_profit'] for mg in IDS)))
        assert done
    assert max_error<1e-5, (slug,max_error)
    observation_sets[slug]={mg:torch.tensor(np.asarray(x)) for mg,x in obs_buffer.items()}
    result['runs'][slug]['reconstruction_max_error']=max_error
    for mg in IDS:
        with torch.no_grad(): predicted=model.actors[mg](observation_sets[slug][mg]).numpy().ravel()
        error=float(np.max(np.abs(predicted-d['hourly'][f'{mg}_raw_action_0'].to_numpy())))
        assert error<1e-5, (slug,mg,error)
    print('Verified recorded observations:',slug,flush=True)

# Both networks are inspected on a common union of recorded states (no new rollout).
# Standardized local Jacobian: mean |du/dx_j| times observed SD(x_j).
# This measures local action sensitivity, NOT causal feature importance or profit contribution.
result['sensitivity']=[]
result['gates']=[]
result['projection_gradient']=[]
for mg in IDS:
    inputs=torch.cat([observation_sets[s][mg] for s in SLUGS],dim=0)
    scale=inputs.std(dim=0,unbiased=False)
    schema=cross['model'].observation_schemas[mg]
    grouped=cross['model'].actors[mg]
    group_indices={g:getattr(grouped,'_'+g+'_indices').tolist() for g in ['local','market','peer']}
    for slug in SLUGS:
        actor=data[slug]['model'].actors[mg]
        x=inputs.clone().requires_grad_(True)
        action=actor(x)
        grad=torch.autograd.grad(action.sum(),x)[0]
        score=(grad.abs().mean(dim=0)*scale).detach().numpy()
        total=float(score.sum())
        result['sensitivity'].append(dict(algorithm=slug,microgrid=mg,
            groups={g:float(score[idx].sum()/max(total,1e-15)) for g,idx in group_indices.items()},
            top_features=[dict(feature=schema[i],score=float(score[i])) for i in np.argsort(score)[-8:][::-1]]))
    with torch.no_grad():
        local=grouped.local_encoder(inputs.index_select(-1,grouped._local_indices))
        market=grouped.market_encoder(inputs.index_select(-1,grouped._market_indices))
        peer=grouped.peer_encoder(inputs.index_select(-1,grouped._peer_indices))
        for name,z in [('local_gate',torch.cat([peer,market],-1)),('peer_gate',torch.cat([local,market],-1))]:
            gate=torch.sigmoid(getattr(grouped,name)(z))
            multiplier=1+grouped.gate_strength*(2*gate-1)
            result['gates'].append(dict(microgrid=mg,gate=name,mean=float(gate.mean()),
                across_states_std_mean=float(gate.std(dim=0,unbiased=False).mean()),
                low_rate=float((gate<.1).float().mean()),high_rate=float((gate>.9).float().mean()),
                multiplier_abs_deviation_mean=float((multiplier-1).abs().mean()),
                multiplier_min=float(multiplier.min()),multiplier_max=float(multiplier.max())))

result['source_files']=[str(p.relative_to(ROOT)) for slug in SLUGS for p in [RUNS[slug]/'test_daily_metrics.csv',RUNS[slug]/'training_history.json',RUNS[slug]/'validation_history.json',RUNS[slug]/'actor_diagnostics.csv',RUNS[slug]/'config_snapshot.yaml']]
result['limitations']=['One training seed; 55 adjacent/seasonal held-out days are not 55 independent training replications.', 'Parameter gradients depend on parameterization and are not feature importance.', 'Input sensitivity is a local derivative on recorded-state union, not a causal ablation.', 'Recorded-action reconstruction verified current data/code against saved profit, SOC, discharge and actor outputs. No training or formal evaluation artifacts were changed.']
(OUT/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(result['daily']).to_csv(OUT/'paired_daily.csv',index=False)
print(json.dumps({k:result[k] for k in ['bridge','microgrids','gates','sensitivity','monthly']},ensure_ascii=False,indent=2))
