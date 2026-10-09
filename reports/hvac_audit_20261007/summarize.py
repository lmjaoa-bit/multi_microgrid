"""Recompute audit claims and check conservation/provenance before delivery."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
manifest=json.loads((HERE/'source_manifest.json').read_text())
for path,expected in manifest['source_hashes'].items():
    actual=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    assert actual==expected, f'Source changed during audit: {path}'
for path,expected in manifest['snapshot_hashes'].items():
    assert hashlib.sha256((HERE/'source_snapshot'/path).read_bytes()).hexdigest()==expected

hourly=pd.read_csv(HERE/'hourly_forecast_true.csv')
probes=pd.read_csv(HERE/'counterfactual_probes.csv')
assert len(hourly)==1320 and not hourly.duplicated(['date','hour']).any()
assert len(probes.groupby(['date','hour']))==48
assert not probes.duplicated(['date','hour','candidate_raw']).any()
assert np.isfinite(probes.select_dtypes(include='number').drop(columns=['previous_raw'])).all().all()
errors=[]
for (day,hour),group in probes.groupby(['date','hour']):
    suffix=hourly[(hourly.date==day)&(hourly.hour>=hour)].reward.sum()
    errors.append(abs(suffix-group.reference_return.iloc[0]))
costs=[c for c in probes.columns if c.startswith('delta_cost_')]
cost_error=float(np.max(abs(probes.gain+probes[costs].sum(axis=1)/100)))
assert max(errors)<1e-10 and cost_error<1e-10
local=probes[np.isclose(abs(probes.candidate_kw-probes.reference_kw),1,atol=1e-6,rtol=0)]
meaningful=local[local.gain.abs()>.001]
state=probes.groupby(['date','hour']).first()
base=pd.read_csv(HERE/'daily_forecast_true.csv')
base=base[base.threshold_kw==1].set_index('date')
dead=pd.read_csv(HERE/'deadband_daily.csv')
interventions=[]
for threshold,group in dead.groupby('intervention_threshold_kw'):
    group=group.set_index('date')
    delta_cost=group.generalized_cost-base.generalized_cost
    delta_comfort=group.comfort_degree_hours-base.comfort_degree_hours
    interventions.append(dict(threshold_kw=float(threshold),days=len(group),
        mean_daily_cost_delta=float(delta_cost.mean()),
        mean_comfort_degree_hours_delta=float(delta_comfort.mean()),
        cost_improved_days=int(sum(delta_cost<-.1)),
        comfort_worsened_days=int(sum(delta_comfort>1e-6)),
        adjacent_reversals=int(group.adjacent_reversals.sum()),
        both_modes_days=int(group.both_modes.sum()),off_hours=int(group.off_hours.sum())))
historical=[]
for path in HERE.glob('historical_*.csv'):
    f=pd.read_csv(path)
    assert len(f)==1320 and not f.duplicated(['date','hour']).any()
    both=switches=micro=off=0
    for _,g in f.groupby('date'):
        p=g.sort_values('hour').mg2_hvac_effective_signed_power_kw.to_numpy()
        signs=np.where(p>1,1,np.where(p<-1,-1,0))
        both+=int((signs==1).any() and (signs==-1).any())
        switches+=int(sum(signs[1:]*signs[:-1]==-1))
        micro+=int(sum((abs(p)>1e-6)&(abs(p)<1)))
        off+=int(sum(abs(p)<=1e-6))
    historical.append(dict(source=path.name,days=55,both_modes_days=both,
        adjacent_reversals=switches,micro_hours=micro,off_hours=off))

summary=dict(
    source_run=manifest['source_run'],checkpoint_sha256=manifest['checkpoint_sha256'],
    best_episode=manifest['best_validation_episode'],scope='intermediate checkpoint, test split, deterministic actor, no training updates',
    baseline=json.loads((HERE/'baseline_summary.json').read_text()),historical=historical,
    local_probes=dict(states=48,all_candidate_rows=len(probes),one_kw_comparisons=len(local),
        meaningful_comparisons=len(meaningful),meaningful_reward_threshold=.001,
        q1_wrong_sign=int(sum(meaningful.dq1*meaningful.gain<0)),
        q2_wrong_sign=int(sum(meaningful.dq2*meaningful.gain<0)),
        qmin_wrong_sign=int(sum(meaningful.dqmin*meaningful.gain<0)),
        both_critics_wrong_sign=int(sum((meaningful.dq1*meaningful.gain<0)&(meaningful.dq2*meaningful.gain<0))),
        median_actual_abs_gain=float(local.gain.abs().median()),
        median_q1_abs_difference_error=float((local.dq1-local.gain).abs().median()),
        selection='Four states per month: reversal, smallest action, largest action, late hour. Stratified diagnostic sample; not a population error-rate estimate.'),
    noise=dict(samples_per_state=512,std=.2,clip=.5,std_kw=7.,clip_kw=17.5,
        negative_mean_q_shift_states=int(sum(state.noise_hvac_only_q_shift<0)),
        median_absolute_q_shift=float(state.noise_hvac_only_q_shift.abs().median()),
        nearzero_target_states=int(sum(state.noise_hvac_only_target_raw.abs()<.1)),
        nearzero_sign_cross_rate_median=float(state.loc[state.noise_hvac_only_target_raw.abs()<.1,'noise_hvac_only_sign_cross_rate'].median()),
        limitation='Frozen target-critic sensitivity only; no retraining ablation. Does not establish that target noise caused learned switching.'),
    zero_candidates=dict(states=48,beneficial_without_hourly_hvac_comfort_worsening=int(sum((probes.candidate_raw==0)&(probes.gain>.001)&probes.comfort_no_worse))),
    deadband=interventions,
    checks=dict(rows_unique=True,finite=True,source_hashes_unchanged=True,
        reference_suffix_max_error=max(errors),reward_cost_identity_max_error=cost_error),
    caveats=['The current model was selected at episode 1900 while its 5000-episode training was active.',
        'Historical feasible-actor plot uses legacy Euler physics; the current frozen model uses exact_zoh. They are not a controlled algorithm comparison.',
        'HVAC comfort_no_worse means every remaining hourly comfort deficit is no worse; it does not claim all other device services were checked.',
        'Candidate return follows the unchanged deterministic online policy until day end, whereas TD3 targets include policy smoothing.',
        'Deadband interventions apply an external action filter to a fixed actor; they are not retrained deadband policies.',
        'Observed same-day heating and cooling is not by itself proof of waste. There was no causal retraining experiment to rank all four proposed causes.']
)
(HERE/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
