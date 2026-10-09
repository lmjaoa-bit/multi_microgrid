"""Read-only MILP gap accounting and fixed-dispatch reoptimization.

Highs.minimize instrumentation adds equality constraints only to diagnostic models.
Shared solver, algorithms, rewards and formal evidence are never edited.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml
from core.reward_v2 import calculate_reward_v2, resolve_carbon_factors, resolve_reward_v2_config
from optimization.common_constraints import build_dispatch_data, solve_dispatch

OUT = Path(__file__).resolve().parent
RUN = ROOT / 'outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/runs/DUELINGCRITICT-QH24_2026-10-07_22-11-08_seed42/artifacts'
CFG = yaml.safe_load((RUN/'config_snapshot.yaml').read_text(encoding='utf-8'))['environment']
CFG['_project_root'] = str(ROOT)
KEYS = ['cost_grid', 'cost_battery', 'cost_curtail', 'cost_carbon', 'cost_peak',
        'cost_mill', 'cost_comfort', 'cost_ev', 'cost_unmet']
GROUPS = ['ess', 'mill', 'hvac', 'ev']


def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def summarize_trajectory(frame, data):
    ids = data.agent_ids
    prices = {(mg, ts): float(data.buy_price[mg][t]) for mg in ids for t, ts in enumerate(data.timestamps)}
    carbon = dict(zip(data.timestamps, resolve_carbon_factors(CFG, data.frame)))
    batteries = {m['id']: m['battery'] for m in CFG['microgrids']}
    raw = dict(grid_import_cost=0., battery_degradation_cost=0., curtailment_kwh=0.,
               carbon_kg=0., delta_peak_kw=float(frame.groupby('timestamp').load_kw.sum().max()),
               mill_unserved_t=0., comfort_degree_hours=0., ev_departure_shortfall_kwh=0., unmet_kwh=0.)
    for r in frame.to_dict('records'):
        mg, ts = r['microgrid'], r['timestamp']
        raw['grid_import_cost'] += r['grid_import_kw'] * prices[mg, ts]
        raw['battery_degradation_cost'] += (r['battery_charge_kw']+r['battery_discharge_kw'])*batteries[mg]['degradation_cost_per_kwh']
        raw['curtailment_kwh'] += r['curtailment_kw']
        raw['carbon_kg'] += r['grid_import_kw']*carbon[ts]
        raw['unmet_kwh'] += r['unmet_load_kw']
        for k, v in r.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                if k == 'comfort_degree_hours': raw['comfort_degree_hours'] += v
                if k.startswith('ev_') and k.endswith('_departure_shortfall_kwh'): raw['ev_departure_shortfall_kwh'] += v
    return calculate_reward_v2(raw=raw, rates=resolve_reward_v2_config(CFG))


def saved_accounting():
    daily = pd.read_csv(RUN/'test_daily_metrics.csv')
    rows, operations, hours = [], [], []
    sources = [RUN/'config_snapshot.yaml', RUN/'test_summary.json', RUN/'test_daily_metrics.csv', ROOT/'optimization/common_constraints.py']
    for record in daily.to_dict('records'):
        day = record['date']
        policy_path, ref_path = RUN/'test_hourly_trajectories'/f'{day}.csv', RUN/'generalized_reference_trajectories'/f'{day}.csv'
        sources += [policy_path, ref_path]
        policy, ref = pd.read_csv(policy_path), pd.read_csv(ref_path)
        assert len(policy) == 24 and len(ref) == 72
        cfg, data = build_dispatch_data(CFG, 'test', day)
        for mg in data.agent_ids:
            subset = ref[ref.microgrid == mg].sort_values('timestamp')
            assert np.max(abs(subset.base_load_kw.to_numpy()-policy[f'{mg}_base_load_kw'].to_numpy())) < 1e-8
            assert np.max(abs(subset.renewable_kw.to_numpy()-policy[f'{mg}_renewable_available_kw'].to_numpy())) < 1e-8
            assert np.max(abs(subset.base_load_kw.to_numpy()-data.load_kw[mg])) < 1e-8
        reference = summarize_trajectory(ref, data)
        assert abs(reference['generalized_system_cost']-record['generalized_reference_cost']) < 1e-5
        assert abs(policy.generalized_system_cost.sum()-record['algorithm_generalized_system_cost']) < 1e-7
        row = dict(date=day, policy_cost=record['algorithm_generalized_system_cost'], reference_cost=record['generalized_reference_cost'])
        for key in KEYS:
            row[f'policy_{key}'] = float(policy[key].sum())
            row[f'reference_{key}'] = float(reference[key])
            row[f'gap_{key}'] = row[f'policy_{key}']-row[f'reference_{key}']
        row['gap'] = row['policy_cost']-row['reference_cost']
        assert abs(sum(row[f'gap_{k}'] for k in KEYS)-row['gap']) < 1e-5
        rows.append(row)
        for mg in data.agent_ids:
            r = ref[ref.microgrid == mg].sort_values('timestamp').reset_index(drop=True)
            b = next(x['battery'] for x in CFG['microgrids'] if x['id']==mg)
            op = dict(date=day, microgrid=mg)
            for metric, suffix, refcol in [('charge','battery_charge_kw','battery_charge_kw'), ('discharge','battery_discharge_kw','battery_discharge_kw'),
                                          ('grid_import','grid_import_kw','grid_import_kw'), ('curtail','renewable_curtailment_kw','curtailment_kw'),
                                          ('dr_energy','flexible_load_kw','flexible_load_kw')]:
                op[f'policy_{metric}_kwh'] = float(policy[f'{mg}_{suffix}'].sum())
                op[f'reference_{metric}_kwh'] = float(r[refcol].sum())
            op['policy_terminal_energy_kwh'] = float((policy[f'{mg}_soc'].iloc[-1]-b['soc_min'])*b['capacity_kwh'])
            op['reference_terminal_energy_kwh'] = float((r.soc.iloc[-1]-b['soc_min'])*b['capacity_kwh'])
            op['policy_grid_to_battery_kwh'] = float(policy[f'{mg}_grid_to_battery_kw'].sum())
            op['reference_grid_to_battery_kwh'] = float(r.grid_to_charge_kw.sum())
            operations.append(op)
            for t in range(24):
                h = dict(date=day, microgrid=mg, hour=t, buy_price=float(data.buy_price[mg][t]))
                for k, a, c in [('dr','flexible_load_kw','flexible_load_kw'),('charge','battery_charge_kw','battery_charge_kw'),
                                 ('discharge','battery_discharge_kw','battery_discharge_kw'),('grid','grid_import_kw','grid_import_kw')]:
                    h[f'policy_{k}']=float(policy[f'{mg}_{a}'].iloc[t]); h[f'reference_{k}']=float(r[c].iloc[t])
                hours.append(h)
    pd.DataFrame(rows).to_csv(OUT/'cost_decomposition.csv', index=False)
    pd.DataFrame(operations).to_csv(OUT/'operations.csv', index=False)
    pd.DataFrame(hours).to_csv(OUT/'hourly_comparison.csv', index=False)
    d = pd.DataFrame(rows)
    result = dict(days=len(d), policy_cost_mean=float(d.policy_cost.mean()), reference_cost_mean=float(d.reference_cost.mean()),
                  gap_mean=float(d.gap.mean()), components=[dict(component=k, policy=float(d[f'policy_{k}'].mean()),
                  reference=float(d[f'reference_{k}'].mean()), gap=float(d[f'gap_{k}'].mean())) for k in KEYS],
                  operations_by_microgrid=pd.DataFrame(operations).groupby('microgrid').mean(numeric_only=True).reset_index().to_dict('records'),
                  worst_days=d.nlargest(8,'gap')[['date','gap','gap_cost_grid','gap_cost_peak','gap_cost_comfort']].to_dict('records'),
                  verification=dict(component_identity_max=float(abs(d[[f'gap_{k}' for k in KEYS]].sum(axis=1)-d.gap).max()),
                                    matched_exogenous_days=len(d)))
    write('saved_accounting_summary.json',result)
    write('source_manifest.json',dict(source_run=str(RUN.parent), source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    print(json.dumps(result,ensure_ascii=False),flush=True)


def fixed_dispatch(day, released):
    import highspy
    policy=pd.read_csv(RUN/'test_hourly_trajectories'/f'{day}.csv')
    cfg,data=build_dispatch_data(CFG,'test',day)
    fixed={}
    for t in range(24):
        if 'ess' not in released:
            for mg in data.agent_ids:
                for name,col in [('charge','battery_charge_kw'),('discharge','battery_discharge_kw')]:
                    fixed[f'{name}[{mg},{t}]']=float(policy[f'{mg}_{col}'].iloc[t])
                fixed[f'battery_mode[{mg},{t}]']=float(policy[f'{mg}_battery_charge_kw'].iloc[t]>0)
        if 'mill' not in released:
            rated=next(x['dr']['rated_power_kw'] for x in CFG['microgrids'] if x['id']=='mg1')
            fixed[f'dr_duty[mg1,{t}]']=float(policy.mg1_mill_power_kw.iloc[t])/rated
        if 'hvac' not in released:
            fixed[f'dr_heat[mg2,{t}]']=float(policy.mg2_hvac_heating_power_kw.iloc[t])
            fixed[f'dr_cool[mg2,{t}]']=float(policy.mg2_hvac_cooling_power_kw.iloc[t])
            fixed[f'dr_hvac_mode[mg2,{t}]']=float(policy.mg2_hvac_heating_power_kw.iloc[t]>0)
        if 'ev' not in released:
            fixed[f'dr_ev_charge[mg3,{t}]']=float(policy.mg3_ev_effective_power_kw.iloc[t])
    original=highspy.Highs.minimize
    original_val=highspy.Highs.val
    call_checks=[]
    tiny_negative_shortfalls=[]
    nonnegative_names=set()
    def diagnostic_val(model,variable):
        value=original_val(model,variable)
        name=getattr(variable,'name','')
        if name in nonnegative_names and -2e-5<float(value)<0:
            tiny_negative_shortfalls.append(float(value))
            return 0.
        return value
    def instrumented_minimize(model,*args,**kwargs):
        matched=set()
        handles={}
        equalities={}
        model.setOptionValue('threads',1)
        model.setOptionValue('primal_feasibility_tolerance',1e-6)
        model.setOptionValue('mip_feasibility_tolerance',1e-5)
        model.setOptionValue('mip_rel_gap',1e-7)
        model.setOptionValue('mip_abs_gap',1e-5)
        variables=model.getVariables()
        lower_bounds=model.getLp().col_lower_
        nonnegative_names.update(v.name for v in variables if lower_bounds[v.index]>=0.)
        for variable in variables:
            name=variable.name
            if name in fixed:
                equalities[name]=model.addConstr(variable==fixed[name]);matched.add(name);handles[name]=variable
        assert matched==set(fixed),(set(fixed)-matched)
        solved=original(model,*args,**kwargs)
        presolve_retry=False
        if 'kInfeasible' in str(model.getModelStatus()):
            # Exact policy equalities can trigger numerical presolve failures.
            # Retry the identical constraints without presolve; do not relax physics.
            model.setOptionValue('presolve','off')
            solved=original(model,*args,**kwargs)
            presolve_retry=True
        equality_band_retry=False
        if 'kInfeasible' in str(model.getModelStatus()):
            # Imported floating-point curves need at most a 1e-6-unit band.
            # The physical model stays exact. Record and bound this diagnostic approximation.
            for name, constraint in equalities.items():
                if '_mode[' in name:continue
                eps=1e-8 if name.startswith('dr_duty') else 1e-6
                model.changeRowBounds(constraint.index,max(0.,fixed[name]-eps),fixed[name]+eps)
            solved=original(model,*args,**kwargs)
            equality_band_retry=True
        assert 'kOptimal' in str(model.getModelStatus()),str(model.getModelStatus())
        fixed_error=max((abs(float(model.val(v))-fixed[k]) for k,v in handles.items()),default=0.)
        assert fixed_error<2e-5,fixed_error
        call_checks.append(dict(fixed_variables=len(fixed),max_error=fixed_error,presolve_retry=presolve_retry,
                               equality_band_retry=equality_band_retry))
        return solved
    highspy.Highs.minimize=instrumented_minimize
    highspy.Highs.val=diagnostic_val
    try:
        result=solve_dispatch(cfg,data,data.agent_ids,cooperative=True,include_dr=True,objective_kind='generalized_cost')
    finally:
        highspy.Highs.minimize=original
        highspy.Highs.val=original_val
    assert len(call_checks)==1,call_checks
    result.metadata['diagnostic_fixed_checks']=call_checks[0]
    result.metadata['diagnostic_tiny_negative_shortfalls']=tiny_negative_shortfalls
    return result


def optimize(days):
    daily=pd.read_csv(RUN/'test_daily_metrics.csv')
    if days: daily=daily.head(days)
    modes=[('all_fixed',[]),('ess_free',['ess']),('mill_free',['mill']),('hvac_free',['hvac']),
           ('ev_free',['ev']),('dr_free',['mill','hvac','ev']),('all_free',GROUPS)]
    existing=OUT/'fixed_dispatch_results.csv'
    saved=pd.read_csv(existing).to_dict('records') if existing.exists() else []
    requested=set(daily.date)
    saved=[r for r in saved if r['date'] in requested]
    trajectory_dir=OUT/'diagnostic_trajectories';trajectory_dir.mkdir(exist_ok=True)
    for record in daily.to_dict('records'):
        day=record['date']
        for mode,released in modes:
            if any(r['date']==day and r['mode']==mode for r in saved):continue
            result=fixed_dispatch(day,released)
            result.trajectories.to_csv(trajectory_dir/f'{day}_{mode}.csv',index=False)
            saved.append(dict(date=day,mode=mode,cost=result.total_cost,status=result.status,
                              objective_bound=result.metadata['solver_objective_bound'],
                              fixed_variables=result.metadata['diagnostic_fixed_checks']['fixed_variables'],
                              fixed_max_error=result.metadata['diagnostic_fixed_checks']['max_error'],
                              presolve_retry=result.metadata['diagnostic_fixed_checks']['presolve_retry'],
                              equality_band_retry=result.metadata['diagnostic_fixed_checks']['equality_band_retry'],
                              numerical_shortfall_clamps=len(result.metadata['diagnostic_tiny_negative_shortfalls']),
                              **result.cost_breakdown))
            pd.DataFrame(saved).to_csv(OUT/'fixed_dispatch_results.csv',index=False)
        pivot={r['mode']:r['cost'] for r in saved if r['date']==day}
        # Saved formal solves used default MIP tolerances. A tighter diagnostic
        # solution can improve that incumbent but must respect its recorded bound.
        assert pivot['all_free']<=record['generalized_reference_cost']+.1
        assert pivot['all_free']>=record['generalized_reference_objective_bound']-.002
        assert pivot['all_fixed']<=record['algorithm_generalized_system_cost']+1e-4
        assert pivot['all_free']<=min(pivot.values())+1e-4
        for k in pivot:assert pivot[k]<=pivot['all_fixed']+1e-4
        print(json.dumps(dict(date=day, costs=pivot, allocation_gap=record['algorithm_generalized_system_cost']-pivot['all_fixed'])),flush=True)
    table=pd.DataFrame(saved).pivot(index='date',columns='mode',values='cost')
    result={'days':len(table),'mean_costs':table.mean().to_dict(),
            'single_group_release_gains':{g:float((table.all_fixed-table[g+'_free']).mean()) for g in GROUPS},
            'ess_plus_all_dr_interaction':float((table.ess_free+table.dr_free-table.all_fixed-table.all_free).mean()),
            'caveats':['Full-information optimization with free source/use allocation.',
                       'Single-group release gains interact and must not be added.',
                       'EV aggregate power is fixed; cohort assignment remains a solver variable.',
                       'These are diagnostic reoptimizations, not trained-policy gains.']}
    write('fixed_dispatch_summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['saved','optimize']);p.add_argument('--days',type=int,default=0)
    args=p.parse_args()
    if args.phase=='saved':saved_accounting()
    else:optimize(args.days)
