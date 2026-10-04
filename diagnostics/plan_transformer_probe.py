from __future__ import annotations

from copy import deepcopy
import numpy as np


def perturb_forecasts(observations, contract, case):
    """Change named valid forecast features while retaining all NOW inputs."""
    changed = {mg: np.asarray(value).copy() for mg,value in observations.items()}
    schemas = contract['observation_schemas']
    hours = list(case['hours'])
    if not hours or any(isinstance(k,bool) or not isinstance(k,int) or not 1 <= k <= 24 for k in hours):
        raise ValueError('Forecast hours must be integers in [1,24]')
    indices = {mg:{name:i for i,name in enumerate(schema)} for mg,schema in schemas.items()}
    for mg in changed:
        if any(changed[mg][indices[mg][f'forecast_valid_t+{k}']] < .5 for k in hours):
            raise ValueError('Intervention requires valid future slots')
    kind = case['kind']
    if kind in ('swap_energy_hours','swap_price_hours'):
        if len(hours) != 2 or abs(hours[0]-hours[1]) != 1:
            raise ValueError('Swap requires two adjacent forecast hours')
        a,b = hours
        for mg,schema in schemas.items():
            for name in schema:
                if not name.endswith(f'_t+{a}'):
                    continue
                feature = name.rsplit('_t+',1)[0]
                chosen = feature=='price' if kind=='swap_price_hours' else (feature in ('load','pv','wind','temperature') or
                    any(feature.endswith('_'+suffix) for suffix in ('energy_balance','deficit','surplus')))
                if chosen:
                    i,j = indices[mg][name], indices[mg][f'{feature}_t+{b}']
                    changed[mg][i], changed[mg][j] = observations[mg][j], observations[mg][i]
    elif kind == 'surplus_pv':
        agent = case['agent']
        capacity = float(contract['renewable_capacities'][agent]['pv_capacity_kw'])
        if capacity <= 0:
            raise ValueError('Surplus intervention requires installed PV')
        increment = .1*capacity
        norm = contract['normalizers'][agent]
        for k in hours:
            changed[agent][indices[agent][f'pv_t+{k}']] += increment / float(norm['pv'])
            balance = sum(float(changed[agent][indices[agent][f'{f}_t+{k}']])*float(norm[f])
                          for f in ('pv','wind')) - float(changed[agent][indices[agent][f'load_t+{k}']])*float(norm['load'])
            for mg in changed:
                if mg == agent:
                    continue
                for suffix,value in (('energy_balance',balance),('deficit',max(-balance,0)),('surplus',max(balance,0))):
                    changed[mg][indices[mg][f'{agent}_{suffix}_t+{k}']] = value / float(norm['power'])
    else:
        raise ValueError(f'Unknown intervention: {kind}')
    interventions = []
    for mg in changed:
        for i in np.flatnonzero(changed[mg] != observations[mg]):
            interventions.append(dict(agent=mg, feature=schemas[mg][i],
                before=float(observations[mg][i]), after=float(changed[mg][i])))
    return changed, interventions


def probe_forecast_sensitivity(model, observations, cases):
    baseline = model.inspect_policy(observations)
    reports = []
    for case in cases:
        changed, interventions = perturb_forecasts(observations, model.actor_contract, case)
        output = model.inspect_policy(changed)
        responses = {}
        for mg in output:
            response = {}
            for name in ('actions','soc_after','ess_net_kw','attention'):
                before, after = getattr(baseline[mg],name)[0], getattr(output[mg],name)[0]
                response[name+'_before'] = before.cpu().tolist()
                response[name+'_after'] = after.cpu().tolist()
                key = 'attention_delta' if name=='attention' else name+'_delta'
                response[key] = (after-before).cpu().tolist()
            responses[mg] = response
        reports.append(dict(case=deepcopy(case), diagnostic_only=True,
            interpretation='Controlled observation sensitivity; no realized dispatch return or causal performance proof.',
            interventions=interventions, responses=responses))
    return reports
