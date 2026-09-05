"""Read-only comparison of saved experiments; no policy/environment execution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = {
    'MATD3': ROOT/'outputs/matd3_plain_single_dispatch/runs/MATD3PLAINA3_2026-09-04_16-19-49_seed42',
    'PNN': ROOT/'outputs/pnn/runs/PNN_2026-09-04_20-34-39_seed42',
}
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
daily, summaries, training, validation, hourly, configs = {}, {}, {}, [], {}, {}
metrics, diagnostics, history = [], [], []
for label, path in RUNS.items():
    d = pd.read_csv(path/'test_daily_metrics.csv').sort_values('date').set_index('date')
    assert len(d) == 55 and d.index.is_unique and not d.algorithm_team_profit.isna().any()
    daily[label] = d
    s = summaries[label] = read(path/'test_summary.json')
    t = training[label] = read(path/'training_summary.json')
    configs[label] = yaml.safe_load((path/'config_snapshot.yaml').read_text(encoding='utf-8'))
    assert abs(d.algorithm_team_profit.mean()-s['team_profit_statistics']['mean']) < 1e-8
    assert s['checkpoint_used'].endswith('best_model.pt')
    v = pd.DataFrame(read(path/'validation_history.json'))
    assert len(v) == 50 and v.episode.is_unique
    assert v.loc[v.team_profit_mean.idxmax(), 'episode'] == t['best_episode']
    validation.extend(dict(algorithm=label, episode=int(r.episode), profit=float(r.team_profit_mean),
                           std=float(r.team_profit_std), best=bool(r.is_best), seed=42) for r in v.itertuples())
    h = pd.concat([pd.read_csv(path/'test_hourly_trajectories'/f'{date}.csv') for date in d.index], ignore_index=True)
    assert len(h) == 1320 and not h.duplicated(['date','hour']).any()
    hourly[label] = h
    hourly_profit = h.groupby('date')[[f'mg{i}_hourly_profit' for i in [1,2,3]]].sum().sum(axis=1)
    assert np.allclose(hourly_profit, d.algorithm_team_profit, atol=1e-7)
    a = pd.read_csv(path/'actor_diagnostics.csv')
    assert a.filter(like='nonfinite_count').sum().sum() == 0
    tail = a[a.episode > 4500]
    for mg, g in tail.groupby('microgrid'):
        diagnostics.append(dict(algorithm=label, microgrid=mg, episodes=500,
            grad=float(g.total_grad_norm_mean.mean()), clip_rate=float(g.clip_applied_mean.mean()),
            saturation=float(g.dispatch_saturation_rate_mean.mean())))
    for line in (path/'training_history.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        if row['episode'] % 100 == 0:
            history.append({k:row.get(k) for k in ['episode','team_profit_ma_100','critic_loss','actor_loss','actor_parameter_count']} | {'algorithm':label})
    params = row['actor_parameter_count']
    metrics.append(dict(algorithm=label, profit=s['team_profit_statistics']['mean'],
        test_std=s['team_profit_statistics']['std'], test_days=55,
        training_minutes=t['training_seconds']/60, best_episode=t['best_episode'],
        validation_best=t['best_validation_profit_mean'], actor_parameters=params,
        p2p=s['p2p_total_energy_kwh_mean'], ess_offer=d.ess_offer_energy_kwh.mean(),
        ess_accepted=d.ess_accepted_energy_kwh.mean(), ess_unmatched=d.ess_unmatched_energy_kwh.mean(),
        ess_acceptance=d.ess_accepted_energy_kwh.sum()/d.ess_offer_energy_kwh.sum(),
        renewable=s['renewable_utilization_rate'], feasible=s['feasible_day_rate'],
        ir=s['shapley_all_ir_satisfaction_rate'], val_last10_mean=v.tail(10).team_profit_mean.mean(),
        val_last10_std=v.tail(10).team_profit_mean.std(), idle_gain=s['comparison']['algorithm_vs_idle_gain_mean']))
m, p = daily['MATD3'], daily['PNN']
assert m.index.equals(p.index)
for col in ['test_seed','renewable_available_kwh','idle_team_profit','cooperative_milp_profit']:
    assert np.allclose(m[col],p[col],atol=1e-8)
hm,hp=hourly['MATD3'],hourly['PNN']
for i in [1,2,3]:
    for f in ['load_kw','renewable_available_kw']:
        assert np.allclose(hm[f'mg{i}_{f}'],hp[f'mg{i}_{f}'])
for k in ['resolved_environment_hash','physical_config_hash']:
    assert summaries['PNN']['experiment_contract'][k] == summaries['MATD3']['experiment_contract'][k]
diffs=[]
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(a.keys()|b.keys()): diff(a.get(k),b.get(k),path+'.'+k)
    elif a!=b: diffs.append(dict(field=path,matd3=a,pnn=b))
diff(configs['MATD3'],configs['PNN'])
paired=pd.DataFrame({'matd3':m.algorithm_team_profit,'pnn':p.algorithm_team_profit,
    'delta':p.algorithm_team_profit-m.algorithm_team_profit,'idle':p.idle_team_profit,
    'renewable_kwh':p.renewable_available_kwh,'cooperative_milp':p.cooperative_milp_profit})
paired['month']=pd.to_datetime(paired.index).month
paired['quarter']=pd.to_datetime(paired.index).quarter
paired.to_csv(OUT/'paired_daily.csv')
d=paired.delta.to_numpy()
rng=np.random.default_rng(20260905)
ci=np.quantile(rng.choice(d,size=(20000,len(d)),replace=True).mean(axis=1),[.025,.975])
# Sensitivity: preserve within-month date dependence by resampling 12 month clusters.
months=[g.delta.to_numpy() for _,g in paired.groupby('month')]
draw=rng.integers(0,len(months),size=(20000,len(months)))
sums=np.array([x.sum() for x in months]); counts=np.array([len(x) for x in months])
cluster_ci=np.quantile(sums[draw].sum(axis=1)/counts[draw].sum(axis=1),[.025,.975])
microgrids=[]
for i in [1,2,3]:
    def avg(h,k): return float(h[f'mg{i}_{k}'].sum()/55)
    physical=paired.delta.mean()*0+(p[f'physical_profit_mg{i}']-m[f'physical_profit_mg{i}']).mean()
    transfer=(avg(hp,'p2p_sale_revenue')-avg(hp,'p2p_purchase_cost'))-(avg(hm,'p2p_sale_revenue')-avg(hm,'p2p_purchase_cost'))
    microgrids.append(dict(microgrid=f'MG{i}',physical_delta=physical,external_delta=physical-transfer,
        transfer_delta=transfer,matd3_profit=m[f'physical_profit_mg{i}'].mean(),pnn_profit=p[f'physical_profit_mg{i}'].mean()))
econ=[]
for field,sign in [('grid_import_cost',-1),('grid_export_revenue',1),('battery_degradation_cost',-1)]:
    mm=sum(hm[f'mg{i}_{field}'].sum() for i in [1,2,3])/55
    pp=sum(hp[f'mg{i}_{field}'].sum() for i in [1,2,3])/55
    econ.append(dict(component=field,matd3=mm,pnn=pp,gain=sign*(pp-mm)))
econ.append(dict(component='other_costs',gain=float(d.mean()-sum(x['gain'] for x in econ))))
assert abs(sum(x['external_delta'] for x in microgrids)-d.mean())<1e-7
settlement=[]
for label,s in summaries.items():
    for mg in ['mg1','mg2','mg3']:
        settlement.append(dict(algorithm=label,microgrid=mg,
            physical=s['physical_profit_mean_by_microgrid'][mg],
            equal=s['equal_settled_profit_mean_by_microgrid'][mg],
            shapley=s['shapley_settled_profit_mean_by_microgrid'][mg]))
        assert abs(sum(s['shapley_settled_profit_mean_by_microgrid'].values())-s['team_profit_statistics']['mean'])<1e-7
evidence={'metrics':metrics,'paired':paired.reset_index().to_dict('records'),
 'paired_stats':dict(mean=d.mean(),median=float(np.median(d)),std=float(d.std(ddof=1)),
    wins=int(sum(d>1e-8)),losses=int(sum(d< -1e-8)),total=float(d.sum()),
    bootstrap_iid_95=ci.tolist(),bootstrap_month_cluster_95=cluster_ci.tolist(),
    min=float(d.min()),max=float(d.max())),
 'quarter':paired.groupby('quarter').delta.agg(['count','mean','sum']).reset_index().to_dict('records'),
 'validation':validation,'training_sample':history,'diagnostics':diagnostics,
 'microgrids':microgrids,'economics':econ,'settlement':settlement,'config_differences':diffs,
 'sources':{k:str(v.relative_to(ROOT)).replace('\\','/') for k,v in RUNS.items()},
 'qa':{'daily_rows_per_run':55,'hourly_rows_per_run':1320,'duplicate_days':0,
    'duplicate_hours':0,'core_missing':0,'summary_hourly_daily_reconcile':True,
    'same_dates_loads_renewables_baselines_environment_hashes':True,
    'checkpoint_selection_verified':True,'diagnostic_nonfinite_count':0},
 'limitations':['One training seed; 55 dates are not 55 independent training replications.',
    'Saved git commits differ and the old baseline package is absent from its recorded commit: uncommitted run-time source was not archived. Config/physics/data equivalence was verified, exact run-time code equivalence cannot be certified.',
    'Training seconds include implementation and logging overhead; no inference benchmark.',
    'PNN combines product interactions, normalization, grouping, initialization, and size changes; no causal isolation of product branch.']}
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    'cells':[{'cell_type':'markdown','metadata':{},'source':['# PNN 与普通 MATD3：保存结果复核\n','运行下方单元可重新生成 evidence.json 和 paired_daily.csv。仅读取保存的结果。'],'id':'intro'},
    {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':['from pathlib import Path\n',"script = Path('analyze.py').resolve()\n","exec(compile(script.read_text(encoding='utf-8'), str(script), 'exec'), {'__file__': str(script), '__name__': '__main__'})\n"],'id':'analysis'}]}
(OUT/'analysis.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:evidence[k] for k in ['metrics','paired_stats','microgrids','economics','qa']},ensure_ascii=False,indent=2))
