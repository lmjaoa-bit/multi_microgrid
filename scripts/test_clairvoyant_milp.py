"""Small deterministic MILP smoke test independent of external test datasets."""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import numpy as np
import pandas as pd

class Mill:
    def __init__(self,cfg,*,delta_t,episode_length):
        self.rated_power_kw=2.;self.production_tph=1.;self.downstream_tph=.5
        self.initial_silo_t=10.;self.silo_min_t=0.;self.silo_max_t=20.;self.terminal_tolerance_t=.1
    def reset(self,frame,base): pass
class HVAC:
    def __init__(self,cfg,*,delta_t,episode_length):
        self.initial_t_air=23.;self.initial_t_mass=23.;self.heat_max_kw=4.;self.cool_max_kw=4.
        self.cop_heat=3.;self.cop_cool=3.;self.c_air=15.;self.c_mass=250.;self.r_oa=.22;self.r_am=.08;self.solar_air_fraction=.2
        self.terminal_air_tolerance=1.;self.terminal_mass_tolerance=.3
    def reset(self,frame,base): pass
    def _exogenous(self,t):return 23.,0.,0.
    def _state_comfort_bounds(self,t):return 16.,30.
class EV:
    def __init__(self,cfg,*,delta_t,episode_length):
        self.capacity_kwh=20.;self.initial_soc=.5;self.target_soc=.8;self.max_soc=.8
        self.max_charge_kw=3.;self.eta=.93;self.available_start_hour=0;self.departure_hour=7
    def reset(self,frame,base): pass
for name,cls in [('industrial_mill',Mill),('commercial_hvac',HVAC),('ev_charging',EV)]:
    mod=types.ModuleType('components.'+name)
    setattr(mod,{'industrial_mill':'IndustrialMillDR','commercial_hvac':'CommercialHVACDR','ev_charging':'EVChargingDR'}[name],cls)
    sys.modules[mod.__name__]=mod
sys.modules['components']=types.ModuleType('components')
sys.modules['core']=types.ModuleType('core')
sys.modules['core.data_manager']=types.ModuleType('core.data_manager')
sys.modules['core.data_manager'].DataManager=object
sys.modules['optimization']=types.ModuleType('optimization')
sys.modules['optimization.common_constraints']=types.ModuleType('optimization.common_constraints')
for key,value in [('DispatchData',object),('build_dispatch_data',None),('load_config',None)]:
    setattr(sys.modules['optimization.common_constraints'],key,value)
spec=importlib.util.spec_from_file_location('oracle',Path(__file__).with_name('clairvoyant_milp.py'))
mod=importlib.util.module_from_spec(spec);sys.modules['oracle']=mod;spec.loader.exec_module(mod)
mg=[]
for i,kind in enumerate(['industrial_mill','commercial_hvac','ev_charging'],1):
    mg.append({'id':'mg'+str(i),'battery':{'capacity_kwh':20.,'soc_min':.1,'soc_max':.9,'initial_soc':.5,'max_charge_kw':5.,'max_discharge_kw':5.,'eta_charge':.95,'eta_discharge':.95,'degradation_cost_per_kwh':.01},
               'grid':{'import_limit_kw':100.,'export_limit_kw':100.},'dr':{'type':kind}})
cfg={'time':{'delta_t':1.,'episode_length':24},'p2p':{'enabled':True,'mode':'cooperative_energy_pool','loss_rate':0.,'pair_capacity_kw':75.,'allow_grid_resale':False},
     'terminal_soc':{'mode':'penalty','tolerance':.03,'penalty_per_kwh':1.5},
     'cost':{'renewable_curtailment_cost_per_kwh':.02,'unmet_load_cost_per_kwh':20.},'microgrids':mg}
timestamps=[f'2018-01-28 {h:02d}:00:00' for h in range(24)]
data=SimpleNamespace(timestamps=timestamps,agent_ids=['mg1','mg2','mg3'],horizon=24,
  frame=pd.DataFrame({'timestamp':pd.to_datetime(timestamps)}),
  load_kw={x['id']:np.full(24,8.) for x in mg},
  renewable_kw={'mg1':np.r_[np.zeros(8),np.full(8,15.),np.zeros(8)],'mg2':np.full(24,3.),'mg3':np.zeros(24)},
  buy_price={x['id']:np.full(24,0.8) for x in mg},
  sell_price={x['id']:np.full(24,.2) for x in mg})
answer=mod.solve_clairvoyant(cfg,data,allow_unmet=False,time_limit=40)
assert answer.summary['status']=='optimal'
assert abs(answer.summary['primal_residual'])<1e-5
assert answer.summary['unmet_kwh']<1e-8
assert len(answer.hourly)==72
assert abs(sum(x['total_cost'] for x in answer.summary['external_cost_by_microgrid'].values()) - answer.summary['cost']) < 1e-5
for name in ['mg1','mg2','mg3']:
    f=answer.hourly[answer.hourly.microgrid==name]
    residual=f.renewable_kw+f.battery_discharge_kw+f.grid_import_kw+f.p2p_import_kw +f.unmet_kw - f.base_load_kw-f.flexible_load_kw-f.battery_charge_kw-f.grid_export_kw-f.p2p_export_kw-f.curtailment_kw
    assert residual.abs().max()<1e-5
print('PASS: independent 3-MG synthetic 24h MILP; optimality, 72 device-hours, power balance and zero unmet confirmed')
print('synthetic_profit=',round(answer.summary['profit'],4),'peak_kw=',round(answer.summary['team_peak_kw'],3))

# Regression: legacy DispatchData without a frame must still resolve the day,
# and shared price arrays must not override individual microgrid tariff columns.
real_day = pd.DataFrame({'timestamp': pd.to_datetime(timestamps),
                         'buy_price_per_kwh': np.full(24, .8),
                         'sell_price_per_kwh': np.full(24, .4)})
for i, mg_id in enumerate(['mg1', 'mg2', 'mg3']):
    real_day[f'{mg_id}_buy_price_per_kwh'] = np.full(24, .9 + .1 * i)
    real_day[f'{mg_id}_sell_price_per_kwh'] = np.full(24, .2 + .1 * i)
raw = SimpleNamespace(timestamps=timestamps,agent_ids=['mg1','mg2','mg3'],horizon=24,
                      load_kw=data.load_kw,renewable_kw=data.renewable_kw,
                      buy_price=np.full(24,.8),sell_price=np.full(24,.4))
resolved = mod._resolve_dispatch_data(cfg, raw, frame=real_day)
assert resolved.frame is not None
assert resolved.buy_price['mg1'][0] == .9
assert resolved.buy_price['mg3'][0] == 1.1
assert abs(resolved.sell_price['mg2'][0] - .3) < 1e-12
answer2 = mod.solve_clairvoyant(cfg, raw, frame=real_day, allow_unmet=False, time_limit=40)
assert answer2.summary['status'] == 'optimal'
assert len(answer2.hourly) == 72
class InMemoryDataManager:
    def __init__(self, config, split='test'):
        assert split == 'test'
    def get_day(self, day):
        assert day == '2018-01-28'
        return real_day.copy()
mod.DataManager = InMemoryDataManager
answer3 = mod.solve_clairvoyant(cfg, raw, allow_unmet=False, time_limit=40)
assert abs(answer3.summary['profit'] - answer2.summary['profit']) < 1e-7
print('PASS: no-frame DispatchData, per-microgrid tariffs, and automatic day recovery')
