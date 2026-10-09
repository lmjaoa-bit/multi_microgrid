"""Independently reconcile saved ledgers, diagnostic solves and environment replays."""
import hashlib
import json
from pathlib import Path
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def main():
    saved=json.loads((OUT/'saved_accounting_summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'))
    for path,expected in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    cost=pd.read_csv(OUT/'cost_decomposition.csv').set_index('date')
    solves=pd.read_csv(OUT/'fixed_dispatch_results.csv')
    replays=pd.read_csv(OUT/'environment_replay_results.csv')
    assert len(solves)==385 and len(replays)==440
    assert not solves.duplicated(['date','mode']).any()
    assert not replays.duplicated(['date','mode']).any()
    assert set(solves.status)=={'kOptimal'}
    assert solves.fixed_max_error.max()<2e-5
    assert max(abs(solves[[k for k in solves if k.startswith('cost_')]].sum(axis=1)-solves.cost))<1e-6
    assert max(abs(replays[[k for k in replays if k.startswith('cost_')]].sum(axis=1)-replays.cost))<1e-6
    c=solves.pivot(index='date',columns='mode',values='cost')
    r=replays.pivot(index='date',columns='mode',values='cost')
    assert len(c)==len(r)==55 and c.notna().all().all() and r.notna().all().all()
    assert max(abs(cost.policy_cost-r.policy))<1e-7
    fixed_allocation_gap=cost.policy_cost-c.all_fixed
    ranks=[]
    for group,label in [('ess','Storage'),('hvac','HVAC'),('ev','EV'),('mill','Mill'),('dr','All DR jointly')]:
        rr=replays[replays['mode']==group+'_free'].set_index('date')
        gain=float((r.policy-r[group+'_free']).mean())
        ranks.append(dict(module=label, days=55, milp_gain=float((c.all_fixed-c[group+'_free']).mean()),
                          environment_replay_gain=gain, baseline_cost_reduction_pct=100*gain/float(r.policy.mean()),
                          max_power_error_kw=float(rr.max_power_error_kw.max()),
                          curve_mismatch_days=int((rr.max_power_error_kw>.001).sum())))
    result=dict(days=55,baseline_cost=float(r.policy.mean()),reference_cost=saved['reference_cost_mean'],
                saved_gap=saved['gap_mean'],opportunities=ranks,
                fixed_control_allocation_gap_mean=float(fixed_allocation_gap.mean()),
                fixed_control_allocation_gap_max_abs=float(fixed_allocation_gap.abs().max()),
                solver_cost_bound_gap_max=float(abs(solves.cost-solves.objective_bound).max()),
                solver_fixed_variable_error_max=float(solves.fixed_max_error.max()),
                environment_all_fixed_cost_error_max=float(abs(r.policy-r.all_fixed).max()),
                no_policy_retraining=True,source_hashes_unchanged=True,
                caveats=['The optimizer observes all future daily inputs; these are opportunities, not achievable learned-policy forecasts.',
                         'Single-module opportunities overlap; do not add the rows.',
                         'All DR jointly includes three devices, unlike the single-module rows.',
                         'Generalized cost includes noncash service, peak, carbon and curtailment penalties.',
                         'Diagnostic solves use bounded numerical tolerances; immutable formal results are untouched.'])
    timing=pd.read_csv(OUT/'schedule_timing.csv')
    storage=pd.read_csv(OUT/'storage_schedule_metrics.csv')
    timing_rows=[]
    for mode in ['all_fixed','ess_free']:
        t=timing[timing['mode']==mode];s=storage[storage['mode']==mode]
        timing_rows.append(dict(mode=mode,grid_charging_weighted_price=float(t.grid_charge_cost.sum()/t.grid_charge.sum()),
                               discharge_time_weighted_tariff=float(t.discharge_tariff_exposure.sum()/t.discharge.sum()),
                               daily_charge_kwh=float(s.charge.sum()/55),daily_discharge_kwh=float(s.discharge.sum()/55)))
    result['storage_timing']=timing_rows
    (OUT/'review_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    component_rows=[dict(component=k['component'],policy=k['policy'],reference=k['reference'],gap=k['gap']) for k in saved['components']]
    receipt={'schemaVersion':1,'items':[
        {'id':'cost-gap','title':'Electricity purchases and load peaks account for most of the MILP cost gap','queries':[
            {'id':'ledger-components','source':{'label':'55-day formal-test and generalized-reference ledgers',
             'files':[{'label':'test_hourly_trajectories/*.csv'},{'label':'generalized_reference_trajectories/*.csv'},
                      {'label':'cost_decomposition.csv'},{'label':'audit.py'}],
             'metricDefinitions':[{'label':'Daily generalized cost gap','definition':'Mean policy-minus-reference cost across the same 55 dates, with all nine Reward V2 cost components reconciled.'}],
             'filters':['Final best checkpoint selected at episode 4600','Same daily base loads and renewable profiles'],
             'caveats':[result['caveats'][3]]},'rows':component_rows,'columns':['component','policy','reference','gap'],
             'preview':{'kind':'aggregate','note':'Nine mutually exclusive cost components.'},
             'evidenceFlow':[{'kind':'validation','title':'Reconciliation','text':'All 55 exogenous profiles match; daily cost-component gap sums reconcile within 1.9e-12. All source hashes remain unchanged.'}]}]},
        {'id':'module-opportunities','title':'Storage scheduling has the largest individual-module opportunity in this run','queries':[
            {'id':'fixed-curve-release','source':{'label':'Fixed-dispatch reoptimization and unchanged-environment replay',
             'files':[{'label':'fixed_dispatch_results.csv'},{'label':'environment_replay_results.csv'},
                      {'label':'audit.py'},{'label':'replay.py'},{'label':'summarize.py'}],
             'metricDefinitions':[{'label':'Single-module opportunity','definition':'Hold all other executed power curves fixed, optimize one module over the full day, and replay the resulting controls through the unchanged environment.'}],
             'filters':['55 identical test dates','385 diagnostic MILP solves','440 environment day replays'],
             'caveats':result['caveats']},'rows':ranks,'columns':list(ranks[0]),
             'preview':{'kind':'aggregate','note':'Four single-module comparisons and one joint three-device DR comparison.'},
             'evidenceFlow':[{'kind':'validation','title':'Execution checks','text':f"Baseline raw-action replays match formal daily costs within 1e-7. Maximum all-fixed replay cost error is {result['environment_all_fixed_cost_error_max']:.6f}; fixed solver variables deviate by at most {result['solver_fixed_variable_error_max']:.8f}. All diagnostic models report kOptimal."}]}]},
        {'id':'interface-and-clearing','title':'Aggregate EV control limits exact replay, while allocation alone contributes little cost','queries':[
            {'id':'compatibility','source':{'label':'Power-curve compatibility and fixed-control allocation comparison',
             'files':[{'label':'environment_replay_results.csv'},{'label':'fixed_dispatch_results.csv'},
                      {'label':'optimization/common_constraints.py'},{'label':'components/ev_charging.py'},
                      {'label':'evaluation/evaluate_matd3.py'}],
             'metricDefinitions':[{'label':'EV curve mismatch','definition':'MILP chooses cohort-specific charging; the environment maps one aggregate charging request using its urgency-based cohort allocation.'},
                                  {'label':'Allocation-only gap','definition':'Policy cost minus MILP cost with all executed ESS and DR power curves held fixed.'}],
             'caveats':['The existing formal oracle_replay field compares policy and oracle trajectories; it does not replay the same oracle controls.',
                        'Different individual microgrid flows can coexist with nearly identical team generalized costs.']},
             'rows':[dict(metric='Maximum EV-only replay power mismatch (kW)',value=next(x['max_power_error_kw'] for x in ranks if x['module']=='EV')),
                     dict(metric='Maximum allocation-only cost gap magnitude',value=result['fixed_control_allocation_gap_max_abs'])],
             'columns':['metric','value']}]},
        {'id':'storage-timing','title':'Optimized storage charges at lower prices and discharges in higher-price hours','queries':[
            {'id':'price-exposure','source':{'label':'Diagnostic storage schedules and tariffs',
             'files':[{'label':'schedule_timing.csv'},{'label':'storage_schedule_metrics.csv'},
                      {'label':'diagnostic_trajectories/*.csv'},{'label':'hourly_comparison.csv'}],
             'metricDefinitions':[{'label':'Grid-charging weighted price','definition':'Total tariff-weighted grid energy sent to batteries divided by total grid energy sent to batteries, across all 55 dates and three storage units.'},
                                  {'label':'Discharge-time weighted tariff','definition':'Discharged energy weights the contemporaneous purchase tariff; it is a time-exposure measure, not export revenue.'}],
             'caveats':['All non-storage executed power curves are fixed.',result['caveats'][0]]},
             'rows':timing_rows,'columns':list(timing_rows[0])}]}]}
    (OUT/'sources_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
