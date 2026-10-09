import csv
import json

import pytest

from scripts.probe_ess_plan_execution import compare_saved_plans


def test_saved_only_probe_distinguishes_request_and_actual(tmp_path):
    plans=tmp_path/'auxiliary_plan'/'policy_plans'
    plans.mkdir(parents=True)
    trajectories=tmp_path/'test_hourly_trajectories'
    trajectories.mkdir()
    (plans/'formal_test_00000.json').write_text(json.dumps(dict(date='2018-01-01',hour=0,
        agents={'mg1':dict(intents=[.5,.4],request_kw=[20.,10.],nominal_soc_after=[.4,.3],
            mask=[True,True],action=[.5,0.],current_soc=.5)})),encoding='utf-8')
    with (trajectories/'2018-01-01.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['hour','mg1_raw_ess_action','mg1_dispatch_requested_power_kw',
            'mg1_battery_discharge_kw','mg1_battery_charge_kw','mg1_soc']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields,[0,.5,20.,8.,0.,.46])))
        writer.writerow(dict(zip(fields,[1,.1,2.,2.,0.,.44])))
    summary=compare_saved_plans(tmp_path)
    assert summary['valid_slot_rows']==2 and summary['first_action_max_abs_error']==0
    assert summary['first_requested_power_max_abs_error_kw']==0
    assert summary['first_actual_power_mean_abs_difference_kw']==12
    assert summary['first_soc_mean_abs_difference']==pytest.approx(.06)
    assert len(summary['source_sha256'])==2
    assert (tmp_path/'auxiliary_plan'/'plan_execution_comparison.csv').is_file()
