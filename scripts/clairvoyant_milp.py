"""Independent 24h complete-device clairvoyant MILP for multi_microgrid.

Run from the repository root: python scripts/clairvoyant_milp.py --all-test-days
This model is an optimistic, full-realization CENTRAL PLANNER, not a deployable RL
policy. P2P is a source-aware bilateral flow relaxation of sequential pool clearing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import sys

# Allow direct execution from the repository root as scripts/clairvoyant_milp.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from components.commercial_hvac import CommercialHVACDR
from components.ev_charging import EVChargingDR
from components.industrial_mill import IndustrialMillDR
from core.data_manager import DataManager
from optimization.common_constraints import DispatchData, build_dispatch_data, load_config


class Builder:
    """Sparse MILP builder; all physical energy quantities are in kW/kWh."""
    def __init__(self) -> None:
        self.name: list[str] = []
        self.c: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integer: list[int] = []
        self.ri: list[int] = []
        self.ci: list[int] = []
        self.val: list[float] = []
        self.rl: list[float] = []
        self.ru: list[float] = []

    def var(self, name: str, low=0.0, high=np.inf, cost=0.0, binary=False) -> int:
        index = len(self.c)
        self.name.append(name)
        self.c.append(float(cost))
        self.lb.append(float(low))
        self.ub.append(float(high))
        self.integer.append(int(binary))
        return index

    def constrain(self, terms: Mapping[int, float], low=-np.inf, high=np.inf) -> None:
        row = len(self.rl)
        for col, value in terms.items():
            if abs(value) > 1e-14:
                self.ri.append(row)
                self.ci.append(col)
                self.val.append(float(value))
        self.rl.append(float(low))
        self.ru.append(float(high))

    def eq(self, terms: Mapping[int, float], rhs: float) -> None:
        self.constrain(terms, rhs, rhs)

    def le(self, terms: Mapping[int, float], rhs: float) -> None:
        self.constrain(terms, high=rhs)

    def hinge(self, name: str, state: int, target: float, tolerance: float,
              rate: float, positive_only=False) -> tuple[int, ...]:
        plus = self.var(name + '_above', cost=rate)
        self.constrain({plus: 1.0, state: -1.0}, low=-target-tolerance)
        if positive_only:
            return (plus,)
        minus = self.var(name + '_below', cost=rate)
        self.constrain({minus: 1.0, state: 1.0}, low=target-tolerance)
        return plus, minus

    def solve(self, *, time_limit: float = 180.0, mip_gap: float = 1e-8):
        # scipy.optimize.milp delegates mixed-integer optimization to HiGHS.
        matrix = coo_matrix((self.val, (self.ri, self.ci)),
                            shape=(len(self.rl), len(self.c))).tocsr()
        result = milp(np.array(self.c), integrality=np.array(self.integer),
                      bounds=Bounds(self.lb, self.ub),
                      constraints=LinearConstraint(matrix, self.rl, self.ru),
                      options={'time_limit': time_limit, 'mip_rel_gap': mip_gap})
        if result.status != 0 or result.x is None or result.mip_gap > mip_gap + 1e-10:
            raise RuntimeError(f'MILP not proven optimal: status={result.status}, '
                               f'message={result.message}, gap={getattr(result, "mip_gap", None)}')
        residual = matrix @ result.x
        excess = max(float(np.max(np.maximum(np.asarray(self.rl)-residual, 0))),
                     float(np.max(np.maximum(residual-np.asarray(self.ru), 0))))
        if excess > 1e-5:
            raise RuntimeError(f'MILP primal feasibility residual: {excess}')
        return result, excess


def make_devices(config: Mapping[str, Any], data: DispatchData):
    cfgs = {str(item['id']): item for item in config['microgrids']}
    dt = float(config['time']['delta_t'])
    result = {}
    for mg in data.agent_ids:
        cfg = cfgs[mg]['dr']
        cls = {'industrial_mill': IndustrialMillDR,
               'commercial_hvac': CommercialHVACDR,
               'ev_charging': EVChargingDR}[cfg['type']]
        dev = cls(cfg, delta_t=dt, episode_length=data.horizon)
        dev.reset(data.frame, np.asarray(data.load_kw[mg], dtype=float))
        result[mg] = dev
    return result


def _resolve_dispatch_data(config: Mapping[str, Any], data: DispatchData,
                           frame: pd.DataFrame | None = None) -> Any:
    """Attach the actual daily frame and *per-microgrid* tariffs.

    optimization.common_constraints.DispatchData has no ``frame`` attribute and
    historically stores shared prices only.  Do not alter that shared API: wrap
    the oracle's view with the daily frame needed by the physical DR devices.
    """
    if frame is None:
        frame = getattr(data, 'frame', None)
    if frame is None:
        if not data.timestamps:
            raise ValueError('Empty MILP episode')
        manager = DataManager(config, split='test')
        frame = manager.get_day(str(pd.Timestamp(data.timestamps[0]).date()))
    frame = frame.reset_index(drop=True).copy()
    timestamps = pd.to_datetime(frame['timestamp']).tolist()
    source_timestamps = pd.to_datetime(data.timestamps).tolist()
    if len(frame) != data.horizon or timestamps != source_timestamps:
        raise ValueError('Daily device frame and DispatchData timestamps differ')

    buy_prices: dict[str, np.ndarray] = {}
    sell_prices: dict[str, np.ndarray] = {}
    cfg_by_id = {str(item['id']): item for item in config['microgrids']}
    for mg in data.agent_ids:
        buy_col = f'{mg}_buy_price_per_kwh'
        sell_col = f'{mg}_sell_price_per_kwh'
        if buy_col in frame and sell_col in frame:
            buy_prices[mg] = frame[buy_col].to_numpy(dtype=float)
            sell_prices[mg] = frame[sell_col].to_numpy(dtype=float)
        elif 'buy_price_per_kwh' in frame and 'sell_price_per_kwh' in frame:
            # An inherited tariff may use the common price.  A heterogeneous
            # tariff *must* use DataManager's correctly generated MG columns.
            tariff = cfg_by_id[mg].get('tariff', {})
            if tariff and (tariff.get('mode', 'inherited') != 'inherited'
                           or float(tariff.get('sell_ratio',
                               config.get('price', {}).get('sell_ratio', 0.5)))
                           != float(config.get('price', {}).get('sell_ratio', 0.5))
                           or 'sell_price_per_kwh' in tariff):
                raise ValueError(f'{mg}: heterogeneous tariff columns are missing')
            buy_prices[mg] = frame['buy_price_per_kwh'].to_numpy(dtype=float)
            sell_prices[mg] = frame['sell_price_per_kwh'].to_numpy(dtype=float)
        else:
            # For deliberately tiny standalone test fixtures without price
            # columns, preserve prices explicitly supplied by the test input.
            if not hasattr(data, 'frame'):
                raise ValueError(f'{mg}: day frame has no price columns')
            buy = data.buy_price[mg] if isinstance(data.buy_price, Mapping) else data.buy_price
            sell = data.sell_price[mg] if isinstance(data.sell_price, Mapping) else data.sell_price
            buy_prices[mg] = np.asarray(buy, dtype=float)
            sell_prices[mg] = np.asarray(sell, dtype=float)
        if len(buy_prices[mg]) != data.horizon or len(sell_prices[mg]) != data.horizon:
            raise ValueError(f'{mg}: tariff horizon mismatch')
    return SimpleNamespace(
        timestamps=list(data.timestamps), agent_ids=list(data.agent_ids),
        load_kw=data.load_kw, renewable_kw=data.renewable_kw,
        horizon=data.horizon, frame=frame,
        buy_price=buy_prices, sell_price=sell_prices,
    )


def _terms(*pairs) -> dict[int, float]:
    z = defaultdict(float)
    for key, value in pairs:
        z[key] += float(value)
    return dict(z)


@dataclass
class OracleResult:
    summary: dict[str, Any]
    hourly: pd.DataFrame


def solve_clairvoyant(config: Mapping[str, Any], data: DispatchData, *,
                       time_limit: float = 180.0, mip_gap: float = 1e-8,
                       allow_unmet: bool = True,
                       frame: pd.DataFrame | None = None) -> OracleResult:
    """Minimize 24h realized team cost under current device physical parameters."""
    data = _resolve_dispatch_data(config, data, frame=frame)
    cfgs = {str(x['id']): x for x in config['microgrids']}
    ids = list(data.agent_ids)
    h = int(data.horizon)
    dt = float(config['time']['delta_t'])
    if data.frame is None or h != int(config['time']['episode_length']) or h != 24:
        raise ValueError('Expected exact 24h episode with realized weather and prices')
    if len(ids) != 3 or set(ids) != {'mg1', 'mg2', 'mg3'}:
        raise ValueError('Current environment requires mg1, mg2, mg3')
    p2p = config['p2p']
    if (not p2p['enabled'] or p2p['mode'] != 'cooperative_energy_pool'
            or float(p2p['loss_rate']) != 0 or p2p['allow_grid_resale']):
        raise ValueError('Unsupported market setting: requires lossless P2P and no grid resale')
    cap = float(p2p['pair_capacity_kw'])
    rate = float(config['terminal_soc']['penalty_per_kwh'])
    tol_soc = float(config['terminal_soc']['tolerance'])
    terminal_mode = config['terminal_soc']['mode']
    if terminal_mode not in {'none', 'penalty', 'hard'}:
        raise ValueError(f'Unsupported terminal mode {terminal_mode}')
    b = Builder()
    dev = make_devices(config, data)
    v: dict[tuple, int] = {}
    terminal: dict[str, list[int]] = defaultdict(list)

    def new(name: str, mg: str, t: int, **kw) -> int:
        k = (name, mg, t)
        v[k] = b.var(f'{name}_{mg}_{t}', **kw)
        return v[k]

    def at(name: str, mg: str, t: int) -> int:
        return v[name, mg, t]

    for mg in ids:
        c = cfgs[mg]
        battery = c['battery']
        E = float(battery['capacity_kwh'])
        emin, emax = E*float(battery['soc_min']), E*float(battery['soc_max'])
        e0 = E*float(battery['initial_soc'])
        gc = c['grid']
        device = dev[mg]
        new('soc_kwh', mg, 0, low=e0, high=e0)
        for t in range(1, h+1):
            new('soc_kwh', mg, t, low=emin, high=emax)
        for t in range(h):
            ch = new('charge', mg, t, high=float(battery['max_charge_kw']),
                     cost=dt*float(battery['degradation_cost_per_kwh']))
            dis = new('discharge', mg, t, high=float(battery['max_discharge_kw']),
                      cost=dt*float(battery['degradation_cost_per_kwh']))
            mode = new('battery_mode', mg, t, high=1, binary=True)
            b.le({ch:1, mode:-float(battery['max_charge_kw'])}, 0)
            b.le({dis:1, mode:float(battery['max_discharge_kw'])},
                 float(battery['max_discharge_kw']))
            b.eq(_terms((at('soc_kwh',mg,t+1),1),(at('soc_kwh',mg,t),-1),
                        (ch,-dt*float(battery['eta_charge'])),
                        (dis,dt/float(battery['eta_discharge']))), 0)
            buy = float(data.buy_price[mg][t]) if isinstance(data.buy_price, Mapping) else float(data.buy_price[t])
            sell = float(data.sell_price[mg][t]) if isinstance(data.sell_price, Mapping) else float(data.sell_price[t])
            imp = new('grid_import', mg, t, high=float(gc['import_limit_kw']), cost=dt*buy)
            exp = new('grid_export', mg, t, high=float(gc['export_limit_kw']), cost=-dt*sell)
            gmode = new('grid_mode', mg, t, high=1, binary=True)
            b.le({imp:1, gmode:-float(gc['import_limit_kw'])}, 0)
            b.le({exp:1, gmode:float(gc['export_limit_kw'])}, float(gc['export_limit_kw']))
            new('curtail', mg, t, high=float(data.renewable_kw[mg][t]),
                cost=dt*float(config['cost']['renewable_curtailment_cost_per_kwh']))
            new('unmet', mg, t, high=(np.inf if allow_unmet else 0.0),
                cost=dt*float(config['cost']['unmet_load_cost_per_kwh']))

        if terminal_mode == 'penalty':
            terminal['ess'] += list(b.hinge('ess_'+mg, at('soc_kwh',mg,h), e0,
                                             tol_soc*E, rate))
        elif terminal_mode == 'hard':
            b.constrain({at('soc_kwh',mg,h):1}, e0-tol_soc*E, e0+tol_soc*E)

        if isinstance(device, IndustrialMillDR):
            initial = device.initial_silo_t
            new('silo',mg,0,low=initial,high=initial)
            for t in range(h):
                duty = new('mill_duty',mg,t, high=1.0)
                new('silo',mg,t+1,low=device.silo_min_t,high=device.silo_max_t)
                b.eq(_terms((at('silo',mg,t+1),1),(at('silo',mg,t),-1),
                            (duty,-device.production_tph*dt)), -device.downstream_tph*dt)
                # Plan power equals duty * rated electrical power.
                flex = new('flex_load',mg,t,high=device.rated_power_kw)
                b.eq({flex:1, duty:-device.rated_power_kw},0)
            if terminal_mode != 'none':
                terminal['mill'] += list(b.hinge('mill_'+mg,at('silo',mg,h),initial,
                                 device.terminal_tolerance_t,
                                 rate*device.rated_power_kw/device.production_tph))

        elif isinstance(device, CommercialHVACDR):
            new('air',mg,0,low=device.initial_t_air,high=device.initial_t_air)
            new('mass',mg,0,low=device.initial_t_mass,high=device.initial_t_mass)
            for t in range(h):
                a = new('air',mg,t+1,low=-np.inf,high=np.inf)
                z = new('mass',mg,t+1,low=-np.inf,high=np.inf)
                heat = new('heat',mg,t,high=device.heat_max_kw)
                cool = new('cool',mg,t,high=device.cool_max_kw)
                mode = new('hvac_mode',mg,t,high=1,binary=True)
                b.le({heat:1,mode:-device.heat_max_kw},0)
                b.le({cool:1,mode:device.cool_max_kw},device.cool_max_kw)
                t_out,q_solar,q_internal = device._exogenous(t)
                air0,mass0 = at('air',mg,t),at('mass',mg,t)
                r1,r2,ca,cm = device.r_oa,device.r_am,device.c_air,device.c_mass
                b.eq(_terms((a,1),(air0,-1+dt/ca*(1/r1+1/r2)),
                            (mass0,-dt/ca/r2),(heat,-dt/ca*device.cop_heat),
                            (cool,dt/ca*device.cop_cool)),
                     dt/ca*(t_out/r1+q_internal+device.solar_air_fraction*q_solar))
                b.eq(_terms((z,1),(mass0,-1+dt/cm/r2),(air0,-dt/cm/r2)),
                     dt/cm*(1-device.solar_air_fraction)*q_solar)
                lo,hi = device._state_comfort_bounds(t+1)
                b.constrain({a:1},lo,hi)
                flex = new('flex_load',mg,t,high=max(device.heat_max_kw,device.cool_max_kw))
                b.eq({flex:1,heat:-1,cool:-1},0)
            if terminal_mode != 'none':
                for key,initial,tolerance,capacity in (
                    ('air',device.initial_t_air,device.terminal_air_tolerance,device.c_air),
                    ('mass',device.initial_t_mass,device.terminal_mass_tolerance,device.c_mass)):
                    plus,minus = b.hinge('hvac_'+mg+'_'+key,at(key,mg,h),initial,
                                          tolerance,0.0)
                    b.c[plus] = rate*capacity/device.cop_cool
                    b.c[minus] = rate*capacity/device.cop_heat
                    terminal['hvac'].extend([plus,minus])

        elif isinstance(device, EVChargingDR):
            e0_ev = device.capacity_kwh*device.initial_soc
            new('ev_kwh',mg,0,low=e0_ev,high=e0_ev)
            for t in range(h):
                hour = pd.Timestamp(data.timestamps[t]).hour
                available = device.available_start_hour <= hour < device.departure_hour
                power = new('flex_load',mg,t,high=device.max_charge_kw if available else 0.0)
                new('ev_kwh',mg,t+1,high=device.capacity_kwh*device.max_soc)
                b.eq(_terms((at('ev_kwh',mg,t+1),1),(at('ev_kwh',mg,t),-1),
                            (power,-device.eta*dt)),0)
            if terminal_mode != 'none':
                target = device.capacity_kwh*device.target_soc
                short = b.var('ev_shortfall_'+mg,cost=rate)
                b.constrain({short:1,at('ev_kwh',mg,h):1},low=target)
                terminal['ev'].append(short)
        else:
            raise TypeError(f'Unsupported DR device {type(device)}')

    # Bilateral market: one shared physical channel capacity per undirected pair.
    # This intentionally omits the sequential local-first allocation priority.
    for t in range(h):
        for j,first in enumerate(ids):
            for second in ids[j+1:]:
                fs = b.var(f'p2p_{first}_to_{second}_{t}',high=cap,cost=1e-9*dt)
                sf = b.var(f'p2p_{second}_to_{first}_{t}',high=cap,cost=1e-9*dt)
                v['flow',first,second,t] = fs
                v['flow',second,first,t] = sf
                b.le({fs:1,sf:1},cap)

    for t in range(h):
        for mg in ids:
            incoming = [(v['flow',other,mg,t],1) for other in ids if other!=mg]
            outgoing = [(v['flow',mg,other,t],-1) for other in ids if other!=mg]
            load = float(data.load_kw[mg][t])
            renewable = float(data.renewable_kw[mg][t])
            b.eq(_terms((at('grid_import',mg,t),1),(at('grid_export',mg,t),-1),
                        (at('discharge',mg,t),1),(at('charge',mg,t),-1),
                        (at('unmet',mg,t),1),(at('curtail',mg,t),-1),
                        (at('flex_load',mg,t),-1),*incoming,*outgoing),load-renewable)
            # No direct grid energy resale. This is a source-aware P2P RELAXATION:
            # exporters cannot send more than renewables plus ESS discharge.
            b.le(_terms((at('discharge',mg,t),-1),
                        *[(v['flow',mg,other,t],1) for other in ids if other!=mg]),renewable)
            # Import and peer export may co-occur in this generous relaxation.
            # Restricting this without reproducing local-first pool logic could
            # inadvertently exclude an environment-feasible dispatch.

    solved, primal = b.solve(time_limit=time_limit,mip_gap=mip_gap)
    x = solved.x
    get = lambda name,mg,t:float(x[at(name,mg,t)])
    records=[]
    cost_by_mg={mg:defaultdict(float) for mg in ids}
    for t in range(h):
        for mg in ids:
            d=dev[mg]
            cfg=cfgs[mg]
            buy=float(data.buy_price[mg][t]) if isinstance(data.buy_price,Mapping) else float(data.buy_price[t])
            sell=float(data.sell_price[mg][t]) if isinstance(data.sell_price,Mapping) else float(data.sell_price[t])
            imp,exp=get('grid_import',mg,t),get('grid_export',mg,t)
            ch,dis=get('charge',mg,t),get('discharge',mg,t)
            curtail,unmet=get('curtail',mg,t),get('unmet',mg,t)
            price_cost=imp*buy*dt-exp*sell*dt
            degradation=(ch+dis)*dt*float(cfg['battery']['degradation_cost_per_kwh'])
            curtail_cost=curtail*dt*float(config['cost']['renewable_curtailment_cost_per_kwh'])
            unmet_cost=unmet*dt*float(config['cost']['unmet_load_cost_per_kwh'])
            for key,cost in [('grid_net_cost',price_cost),('battery_degradation_cost',degradation),
                             ('renewable_curtailment_cost',curtail_cost),('unmet_load_cost',unmet_cost)]:
                cost_by_mg[mg][key]+=cost
            row={'date':str(data.timestamps[t])[:10],'hour':t,'microgrid':mg,
                 'base_load_kw':float(data.load_kw[mg][t]),'renewable_kw':float(data.renewable_kw[mg][t]),
                 'flexible_load_kw':get('flex_load',mg,t),'grid_import_kw':imp,'grid_export_kw':exp,
                 'battery_charge_kw':ch,'battery_discharge_kw':dis,
                 'soc':get('soc_kwh',mg,t+1)/float(cfg['battery']['capacity_kwh']),
                 'p2p_import_kw':sum(float(x[v['flow',other,mg,t]]) for other in ids if other!=mg),
                 'p2p_export_kw':sum(float(x[v['flow',mg,other,t]]) for other in ids if other!=mg),
                 'curtailment_kw':curtail,'unmet_kw':unmet,'grid_net_cost':price_cost,
                 'degradation_cost':degradation,'curtailment_cost':curtail_cost,'unmet_cost':unmet_cost}
            if isinstance(d,IndustrialMillDR):
                row.update(mill_duty=get('mill_duty',mg,t),silo_t=get('silo',mg,t+1))
            elif isinstance(d,CommercialHVACDR):
                row.update(hvac_heat_kw=get('heat',mg,t),hvac_cool_kw=get('cool',mg,t),
                           hvac_air_c=get('air',mg,t+1),hvac_mass_c=get('mass',mg,t+1))
            elif isinstance(d,EVChargingDR):
                row['ev_soc']=get('ev_kwh',mg,t+1)/d.capacity_kwh
            records.append(row)
    term_cost={}
    for group,idxs in terminal.items():
        term_cost[group]=sum(float(x[idx])*b.c[idx] for idx in idxs)
    aggregate_cost=sum(sum(c.values()) for c in cost_by_mg.values())+sum(term_cost.values())
    terminal_by_mg={mg:sum(float(x[idx])*b.c[idx] for group in terminal.values()
                       for idx in group if '_'+mg in b.name[idx]) for mg in ids}
    for mg in ids:
        cost_by_mg[mg]['terminal_penalty']=terminal_by_mg[mg]
        cost_by_mg[mg]['total_cost']=sum(value for key,value in cost_by_mg[mg].items()
                                      if key!='total_cost')
    if abs(aggregate_cost-float(solved.fun))>max(1e-4,1e-6*abs(float(solved.fun))):
        # Tiny tie-break flow penalty is excluded from physical team cost.
        tie=sum(float(x[v['flow',a,c,t]])*1e-9*dt
                for t in range(h) for a in ids for c in ids if a!=c)
        if abs(aggregate_cost+tie-float(solved.fun))>max(1e-4,1e-6*abs(float(solved.fun))):
            raise RuntimeError('Cost reconciliation failed: model objective != physical cost')
    frame=pd.DataFrame(records)
    team_peak=float(frame.assign(actual_kw=frame['base_load_kw']+frame['flexible_load_kw'])
                    .groupby('hour')['actual_kw'].sum().max())
    summary={'date':str(data.timestamps[0])[:10],'status':'optimal','profit':-aggregate_cost,
             'cost':aggregate_cost,'team_peak_kw':team_peak,'mip_gap':float(solved.mip_gap),
             'primal_residual':primal,'external_cost_by_microgrid':{mg:dict(v) for mg,v in cost_by_mg.items()},
             'terminal_penalties':term_cost,'unmet_kwh':float(frame['unmet_kw'].sum()*dt),
             'p2p_kwh':float(frame['p2p_import_kw'].sum()*dt),
             'reference_type':'24h_clairvoyant_joint_relaxed_p2p',
             'warning':'P2P bilateral flows are not the exact sequential local-first pool clearing.'}
    return OracleResult(summary,frame)


def _read_rl_csv(path: Path) -> pd.Series:
    f=pd.read_csv(path)
    if not {'date','algorithm_team_profit'}.issubset(f.columns):
        raise ValueError(f'Missing date / algorithm_team_profit: {path}')
    f['date']=f['date'].astype(str).str[:10]
    if f['date'].duplicated().any():
        raise ValueError(f'Duplicate test day: {path}')
    return f.set_index('date')['algorithm_team_profit'].astype(float)


def main(argv=None):
    ap=argparse.ArgumentParser(description='Complete-device 24h clairvoyant MILP')
    ap.add_argument('--config',type=Path,default=Path('configs/cooperative.yaml'))
    days=ap.add_mutually_exclusive_group(required=True)
    days.add_argument('--day')
    days.add_argument('--all-test-days',action='store_true')
    ap.add_argument('--matd3-csv',type=Path)
    ap.add_argument('--transformer-csv',type=Path)
    ap.add_argument('--time-limit',type=float,default=180)
    ap.add_argument('--mip-gap',type=float,default=1e-8)
    ap.add_argument('--no-unmet',action='store_true')
    ap.add_argument('--output',type=Path)
    a=ap.parse_args(argv)
    cfg=load_config(a.config)
    dm=DataManager(cfg,split='test')
    dates=[a.day] if a.day else [str(d.date()) for d in dm.valid_days]
    if a.all_test_days and len(dates)!=55:
        raise ValueError(f'Expected 55 complete test days, got {len(dates)}; check test data/config')
    out=a.output or Path('outputs/working_runs/clairvoyant_milp/runs')/('MILP_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    out.mkdir(parents=True,exist_ok=False)
    (out/'hourly').mkdir()
    errors=[]
    daily=[]
    for j,date in enumerate(dates,1):
        try:
            _,data=build_dispatch_data(a.config,'test',date)
            result=solve_clairvoyant(cfg,data,time_limit=a.time_limit,mip_gap=a.mip_gap,
                                     allow_unmet=not a.no_unmet,
                                     frame=dm.get_day(date))
            result.hourly.to_csv(out/'hourly'/f'{date}.csv',index=False)
            daily.append(result.summary)
            print(f'[{j}/{len(dates)}] {date}: optimal, profit={result.summary["profit"]:.3f}, '
                  f'gap={result.summary["mip_gap"]:.3g}',flush=True)
        except (ValueError,RuntimeError) as exc:
            errors.append({'date':date,'error':str(exc)})
            print(f'[{j}/{len(dates)}] {date}: FAILED {exc}',flush=True)
    audit={'status':'complete' if not errors else 'incomplete','days_requested':len(dates),
           'days_completed':len(daily),'errors':errors,
           'reference_type':'24h_clairvoyant_joint_relaxed_p2p',
           'physical_equivalence':'not certified: bilateral P2P does not replicate local-first allocation'}
    if not errors and daily:
        f=pd.DataFrame([{'date':x['date'],'milp_profit':x['profit'],'milp_cost':x['cost'],
                         'milp_peak_kw':x['team_peak_kw'],'milp_p2p_kwh':x['p2p_kwh'],
                         'milp_unmet_kwh':x['unmet_kwh'],
                         **{f'milp_terminal_{k}':v for k,v in x['terminal_penalties'].items()},
                         **{f'milp_{mg}_{k}':v for mg,vals in x['external_cost_by_microgrid'].items()
                            for k,v in vals.items()}} for x in daily])
        for label,path in [('matd3',a.matd3_csv),('transformer',a.transformer_csv)]:
            if path:
                series=_read_rl_csv(path)
                if any(date not in series for date in f['date']):
                    raise ValueError(f'{label}: test dates do not match MILP')
                f[label+'_profit']=f['date'].map(series)
                f[label+'_gap_to_milp']=f['milp_profit']-f[label+'_profit']
                audit[label+'_profit_mean']=float(f[label+'_profit'].mean())
                audit[label+'_gap_to_milp_mean']=float(f[label+'_gap_to_milp'].mean())
        f.to_csv(out/'daily_comparison.csv',index=False)
        audit['milp_profit_mean']=float(f['milp_profit'].mean())
        audit['milp_cost_mean']=float(f['milp_cost'].mean())
        audit['milp_peak_mean_kw']=float(f['milp_peak_kw'].mean())
        audit['milp_unmet_mean_kwh']=float(f['milp_unmet_kwh'].mean())
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),**audit},ensure_ascii=False,indent=2),flush=True)
    return 0 if not errors else 2


if __name__=='__main__':
    raise SystemExit(main())
