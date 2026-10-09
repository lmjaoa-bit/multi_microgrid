"""Replay diagnostic optimized power curves through the unchanged environment."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import audit
from envs.multi_microgrid_env import MultiMicrogridEnv

OUT=Path(__file__).resolve().parent


def actions_from_curve(env, curve, t):
    actions={}
    desired={}
    for mg in env.agent_ids:
        r=curve[curve.microgrid==mg].sort_values('timestamp').iloc[t]
        battery=env.cores[mg].battery
        net=float(r.battery_discharge_kw-r.battery_charge_kw)
        scale=battery.available_discharge_kw() if net>=0 else battery.available_charge_kw()
        ess=net/scale if scale>1e-10 else 0.
        dr=env.dr_devices[mg]
        if mg=='mg1':raw=2*float(r.flexible_load_kw)/dr.rated_power_kw-1
        elif mg=='mg2':
            power=float(r.hvac_heating_kw-r.hvac_cooling_kw)
            raw=power/(dr.heat_max_kw if power>=0 else dr.cool_max_kw)
        else:
            low,high=dr.feasible_bounds(t)
            raw=2*(float(r.flexible_load_kw)-low)/(high-low)-1 if high-low>1e-10 else -1.
        actions[mg]=np.clip(np.asarray([ess,raw],dtype=np.float64),-1,1)
        desired[mg]=(float(r.battery_charge_kw),float(r.battery_discharge_kw),float(r.flexible_load_kw))
    return actions,desired


def main():
    daily=pd.read_csv(audit.RUN/'test_daily_metrics.csv')
    previous=OUT/'environment_replay_results.csv'
    results=pd.read_csv(previous).to_dict('records') if previous.exists() else []
    env=MultiMicrogridEnv(audit.CFG,split='test')
    modes=['policy','all_fixed','ess_free','mill_free','hvac_free','ev_free','dr_free','all_free']
    hourly_dir=OUT/'environment_replay_hourly';hourly_dir.mkdir(exist_ok=True)
    for idx,record in enumerate(daily.to_dict('records')):
        day=record['date']
        original=pd.read_csv(audit.RUN/'test_hourly_trajectories'/f'{day}.csv')
        for mode in modes:
            if any(r['date']==day and r['mode']==mode for r in results):continue
            curve_path=OUT/'diagnostic_trajectories'/f'{day}_{mode}.csv'
            if mode!='policy' and not curve_path.exists():continue
            curve=pd.read_csv(curve_path) if mode!='policy' else None
            env.reset(day=day,seed=42+idx,deterministic=True,forecast_noise=True)
            hours=[]
            for t in range(24):
                if mode=='policy':
                    actions={mg:np.asarray([original[f'{mg}_raw_action_0'].iloc[t],original[f'{mg}_raw_action_1'].iloc[t]]) for mg in env.agent_ids}
                    desired={mg:tuple(float(original[f'{mg}_{k}'].iloc[t]) for k in ['battery_charge_kw','battery_discharge_kw','flexible_load_kw']) for mg in env.agent_ids}
                else:actions,desired=actions_from_curve(env,curve,t)
                _,_,done,info=env.step(actions)
                h=dict(date=day,mode=mode,hour=t,**info['reward_v2_breakdown'])
                for mg in env.agent_ids:
                    actual=tuple(float(info['plans'][mg][k]) for k in ['battery_charge_kw','battery_discharge_kw','flexible_load_kw'])
                    for k,a,b in zip(['charge','discharge','dr'],actual,desired[mg]):
                        h[f'{mg}_{k}_power_error_kw']=a-b
                hours.append(h)
            assert done
            frame=pd.DataFrame(hours)
            cost=float(frame.generalized_system_cost.sum())
            if mode=='policy':assert abs(cost-record['algorithm_generalized_system_cost'])<1e-7
            row=dict(date=day,mode=mode,cost=cost,gain=record['algorithm_generalized_system_cost']-cost,
                     max_power_error_kw=float(frame.filter(like='power_error_kw').abs().to_numpy().max()))
            row.update({k:float(frame[k].sum()) for k in audit.KEYS})
            results.append(row)
            frame.to_csv(hourly_dir/f'{day}_{mode}.csv',index=False)
            pd.DataFrame(results).to_csv(previous,index=False)
        if (idx+1)%10==0:print(json.dumps(dict(processed_date=day,rows=len(results))),flush=True)
    frame=pd.DataFrame(results)
    summary=dict(rows=len(frame),modes=frame.groupby('mode').agg(days=('date','count'),mean_cost=('cost','mean'),
                 mean_gain=('gain','mean'),max_power_error_kw=('max_power_error_kw','max')).reset_index().to_dict('records'))
    audit.write('environment_replay_summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
