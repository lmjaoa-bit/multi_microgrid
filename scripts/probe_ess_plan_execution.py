"""Compare saved neural plans with saved execution, without rerunning physics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def compare_saved_plans(artifacts):
    artifacts=Path(artifacts)
    directory=artifacts/'auxiliary_plan'
    records=sorted((directory/'policy_plans').glob('*.json'))
    if not records:
        raise ValueError('No saved policy plans')
    source_hashes={}
    actual_by_date={}
    rows=[]
    def record_source(path):
        source_hashes[path.relative_to(artifacts).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in records:
        record_source(path)
        record=json.loads(path.read_text(encoding='utf-8'))
        date=record['date']
        if date not in actual_by_date:
            actual_path=artifacts/'test_hourly_trajectories'/f'{date}.csv'
            record_source(actual_path)
            with actual_path.open(encoding='utf-8-sig',newline='') as stream:
                actual_by_date[date]={int(row['hour']):row for row in csv.DictReader(stream)}
        for mg,plan in record['agents'].items():
            for lead,valid in enumerate(plan['mask']):
                if not valid:
                    continue
                hour=record['hour']+lead
                actual=actual_by_date[date][hour]
                power=float(actual[f'{mg}_battery_discharge_kw'])-float(actual[f'{mg}_battery_charge_kw'])
                raw=float(actual[f'{mg}_raw_ess_action'])
                requested=float(actual[f'{mg}_dispatch_requested_power_kw'])
                soc=float(actual[f'{mg}_soc'])  # Shared hourly evidence records post-dispatch SOC.
                rows.append(dict(date=date,agent=mg,origin_hour=record['hour'],lead=lead,target_hour=hour,
                    planned_intent=plan['intents'][lead],actual_intent=raw,
                    planned_request_kw=plan['request_kw'][lead],actual_request_kw=requested,
                    actual_power_kw=power,nominal_soc_after=plan['nominal_soc_after'][lead],actual_soc_after=soc,
                    intent_difference=plan['intents'][lead]-raw,
                    request_difference_kw=plan['request_kw'][lead]-requested,
                    power_difference_kw=plan['request_kw'][lead]-power,
                    soc_difference=plan['nominal_soc_after'][lead]-soc))
    if not rows:
        raise ValueError('Plans contain no valid slots')
    output=directory/'plan_execution_comparison.csv'
    with output.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first=[row for row in rows if row['lead']==0]
    summary=dict(policy_plan_records=len(records),valid_slot_rows=len(rows),first_step_rows=len(first),
        first_action_max_abs_error=max(abs(row['intent_difference']) for row in first),
        first_requested_power_max_abs_error_kw=max(abs(row['request_difference_kw']) for row in first),
        first_actual_power_mean_abs_difference_kw=mean(abs(row['power_difference_kw']) for row in first),
        first_soc_mean_abs_difference=mean(abs(row['soc_difference']) for row in first),
        all_leads_power_mean_abs_difference_kw=mean(abs(row['power_difference_kw']) for row in rows),
        all_leads_soc_mean_abs_difference=mean(abs(row['soc_difference']) for row in rows),
        source_sha256=source_hashes,
        interpretation='First-step action/request errors check coupling. Other differences include clearing, '
            'forecast error and hourly replanning; overlapping future slots are not independent observations.')
    (directory/'plan_execution_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='Working run containing artifacts/')
    args=parser.parse_args()
    summary=compare_saved_plans(args.run/'artifacts')
    print(f"Saved plans: {summary['policy_plan_records']}; valid slots: {summary['valid_slot_rows']}; "
          f"first action max error: {summary['first_action_max_abs_error']:.3g}")


if __name__=='__main__':
    main()
