"""Small ESS-only proposal model followed by unchanged physical execution."""
from __future__ import annotations

from dataclasses import dataclass
import time

import highspy
import numpy as np
import torch

from core.renewable_first_dispatch import clear_renewable_first
from core.tariffs import prices_from_row
from .forecast_scenario import build_forecast_environment
from .plan_projection import decode_nominal_plan


@dataclass
class CandidateResult:
    valid: bool
    net_kw: np.ndarray
    status: str


def solve_storage_model(load, renewable, prices, carbon_cost, batteries, initial_soc,
                        delta_t, curtail_cost, p2p_enabled, time_limit_seconds):
    load, renewable, prices = [np.asarray(a, dtype=float) for a in (load, renewable, prices)]
    if load.ndim != 2 or renewable.shape != load.shape or prices.shape != load.shape:
        raise ValueError('Expected matching [horizon, agents] profiles')
    if not all(np.isfinite(x).all() for x in (load, renewable, prices, carbon_cost)):
        raise ValueError('Nonfinite optimization inputs')
    if (load < 0).any() or (renewable < 0).any():
        raise ValueError('Negative physical profile')
    horizon, agents = load.shape
    zeros = torch.zeros_like(torch.tensor(load))
    cleared = clear_renewable_first(torch.tensor(renewable), torch.tensor(load), zeros, zeros,
                                    p2p_enabled=p2p_enabled)
    residual = cleared['load_after_renewable_kw'].numpy()
    surplus = cleared['renewable_surplus_for_export_kw'].numpy()
    model = highspy.Highs()
    model.setOptionValue('output_flag', False)
    model.setOptionValue('time_limit', max(.001, float(time_limit_seconds)))
    model.setOptionValue('mip_rel_gap', 1e-6)
    model.setOptionValue('mip_feasibility_tolerance', 1e-7)
    model.setOptionValue('primal_feasibility_tolerance', 1e-7)
    soc, charge, discharge = {}, {}, {}
    objective = 0.
    for i,b in enumerate(batteries):
        for t in range(horizon+1):
            soc[t,i] = model.addVariable(lb=float(b['soc_min']), ub=float(b['soc_max']))
        model.addConstr(soc[0,i] == float(initial_soc[i]))
        for t in range(horizon):
            charge[t,i] = model.addVariable(lb=0., ub=float(b['max_charge_kw']))
            discharge[t,i] = model.addVariable(lb=0., ub=float(b['max_discharge_kw']))
            mode = model.addVariable(lb=0., ub=1., type=highspy.HighsVarType.kInteger)
            model.addConstr(charge[t,i] <= mode*float(b['max_charge_kw']))
            model.addConstr(discharge[t,i] <= (1-mode)*float(b['max_discharge_kw']))
            model.addConstr(soc[t+1,i] == soc[t,i]+delta_t/float(b['capacity_kwh'])*
                (float(b['eta_charge'])*charge[t,i]-discharge[t,i]/float(b['eta_discharge'])))
            objective += delta_t*float(b.get('degradation_cost_per_kwh',0.))*(charge[t,i]+discharge[t,i])
    for t in range(horizon):
        ess_flow, res_flow = {}, {}
        for i in range(agents):
            for j in range(agents):
                ub = highspy.kHighsInf if p2p_enabled or i == j else 0.
                ess_flow[i,j] = model.addVariable(lb=0., ub=ub)
                res_flow[i,j] = model.addVariable(lb=0., ub=ub)
        for i in range(agents):
            grid_load, grid_charge = model.addVariable(lb=0.), model.addVariable(lb=0.)
            curtail = model.addVariable(lb=0.)
            model.addConstr(sum(ess_flow[i,j] for j in range(agents)) == discharge[t,i])
            model.addConstr(sum(ess_flow[j,i] for j in range(agents))+grid_load == float(residual[t,i]))
            model.addConstr(sum(res_flow[i,j] for j in range(agents))+curtail == float(surplus[t,i]))
            model.addConstr(sum(res_flow[j,i] for j in range(agents))+grid_charge == charge[t,i])
            objective += delta_t*((float(prices[t,i])+float(carbon_cost[t]))*(grid_load+grid_charge)
                                  +float(curtail_cost)*curtail)
    model.minimize(objective)
    model_status = model.getModelStatus()
    status = str(model_status)
    solution = model.getSolution()
    # A feasible incumbent at a time limit is deliberately not a training label.
    if (model_status != highspy.HighsModelStatus.kOptimal or not solution.value_valid
            or model.getInfo().max_primal_infeasibility > 1e-6):
        return CandidateResult(False, np.zeros_like(load), status)
    net = np.array([[float(model.val(discharge[t,i]))-float(model.val(charge[t,i]))
                     for i in range(agents)] for t in range(horizon)])
    return CandidateResult(np.isfinite(net).all(), net, status)


@dataclass
class BaselineTrajectory:
    actions: np.ndarray
    load: np.ndarray
    renewable: np.ndarray
    prices: np.ndarray
    initial_soc: np.ndarray
    cost: float


def rollout_frozen_baseline(context, actors):
    env = build_forecast_environment(context)
    rows, load, res, prices, cost = [], [], [], [], 0.
    initial = np.array([env.cores[mg].battery.soc for mg in context.agent_ids])
    with torch.inference_mode():
        while env.t < 24:
            ex = env._exogenous_at(env.t)
            price = [prices_from_row(env.episode_df.iloc[env.t], mg)[0] for mg in context.agent_ids]
            actions = {mg: actors[mg](torch.tensor(env._last_observations[mg], device=next(actors[mg].parameters()).device)[None])[0].cpu().numpy()
                       for mg in context.agent_ids}
            _, _, _, info = env.step(actions)
            rows.append([actions[mg] for mg in context.agent_ids])
            load.append([ex[mg]['load_kw']+info['dr_plans'][mg]['flexible_load_kw'] for mg in context.agent_ids])
            res.append([ex[mg]['pv_kw']+ex[mg]['wind_kw'] for mg in context.agent_ids])
            prices.append(price)
            cost += info['generalized_system_cost']
    return BaselineTrajectory(np.array(rows), np.array(load), np.array(res), np.array(prices), initial, float(cost))


@dataclass
class PlanLabel:
    valid: bool
    rejection_reason: str | None
    context_hash: str
    forecast_hash: str
    observations: dict
    agent_ids: tuple
    intents: np.ndarray
    request_kw: np.ndarray
    nominal_soc_after: np.ndarray
    actual_kw: np.ndarray
    actual_soc_after: np.ndarray
    mask: np.ndarray
    baseline_cost: float
    verified_cost: float | None
    max_residual: float
    solver_status: str
    elapsed_seconds: float
    metadata: dict


def verify_candidate(context, baseline, candidate):
    env = build_forecast_environment(context)
    shape = (24, len(context.agent_ids))
    intents, requests, actual, states = [np.zeros(shape) for _ in range(4)]
    cost, residual = 0., 0.
    for j in range(24-context.t):
        actions = {}
        before = [env.cores[mg].battery.soc for mg in context.agent_ids]
        for i, mg in enumerate(context.agent_ids):
            battery = env.cores[mg].battery
            power = float(candidate.net_kw[j,i])
            limit = battery.available_discharge_kw() if power >= 0 else battery.available_charge_kw()
            raw = np.clip(power/limit, -1., 1.) if limit > 1e-9 else 0.
            actions[mg] = np.array([raw, baseline.actions[j,i,1]], dtype=np.float32)
            intents[j,i] = actions[mg][0]
            requests[j,i] = battery.map_action(actions[mg][0])['power_kw']
        _, _, _, info = env.step(actions)
        cost += float(info['generalized_system_cost'])
        ledger = info['dispatch_ledger']
        for i,mg in enumerate(context.agent_ids):
            b = env.cores[mg].battery
            ch, dis = float(ledger['actual_charge_kw'][i]), float(ledger['actual_discharge_kw'][i])
            actual[j,i], states[j,i] = dis-ch, float(b.soc)
            expected = before[i]+(ch*b.eta_charge-dis/b.eta_discharge)*env.delta_t/b.capacity_kwh
            residual = max(residual, abs(expected-b.soc), max(0., b.soc_min-b.soc,b.soc-b.soc_max),
                           max(0., ch-b.max_charge_kw,dis-b.max_discharge_kw), min(ch,dis))
        for values in info.get('source_balance_residuals', {}).values():
            residual = max(residual, *(abs(float(x)) for x in values.values()))
    nominal_p, nominal_s = [], []
    for i,mg in enumerate(context.agent_ids):
        p,s = decode_nominal_plan(torch.tensor(intents[:,i])[None], torch.tensor([baseline.initial_soc[i]]),
                    torch.tensor(context.plan_mask)[None], env.cfg_by_id[mg]['battery'], env.delta_t)
        nominal_p.append(p.numpy()[0]); nominal_s.append(s.numpy()[0])
    valid = bool(np.isfinite(cost) and residual <= 1e-6 and cost <= baseline.cost+1e-5)
    return PlanLabel(valid, None if valid else ('physical_residual' if residual > 1e-6 else 'cost_worse'),
        context.context_hash, context.forecast_hash, context.observations, context.agent_ids, intents,
        np.stack(nominal_p, 1), np.stack(nominal_s, 1), actual, states, context.plan_mask,
        baseline.cost, cost, residual, candidate.status, 0., dict(context.metadata, t=context.t,
        actual_request_kw=requests.tolist()))


class PlanTeacher:
    def __init__(self, time_limit_seconds=1.):
        self.time_limit_seconds = float(time_limit_seconds)
        self.solve_calls = 0

    def solve(self, context, actors):
        started = time.perf_counter()
        self.solve_calls += 1
        baseline = rollout_frozen_baseline(context, actors)
        cfg = context.template.config
        carbon = float(context.template.episode_df.iloc[context.t]['grid_carbon_factor_kg_per_kwh'])
        if cfg['reward'].get('fixed_carbon_factor_kg_per_kwh') is not None:
            carbon = float(cfg['reward']['fixed_carbon_factor_kg_per_kwh'])
        candidate = solve_storage_model(baseline.load, baseline.renewable, baseline.prices,
            np.full(len(baseline.load), carbon*float(cfg['cost']['carbon_price_per_kg'])),
            [context.template.cfg_by_id[mg]['battery'] for mg in context.agent_ids], baseline.initial_soc,
            context.template.delta_t, cfg['cost']['curtailment_price_per_kwh'], cfg['p2p']['enabled'], self.time_limit_seconds)
        # Solver failure remains a rejection; it never silently becomes a baseline label.
        if not candidate.valid:
            shape = (24, len(context.agent_ids))
            zero = np.zeros(shape)
            label = PlanLabel(False, 'solver_no_feasible_solution', context.context_hash, context.forecast_hash,
                context.observations, context.agent_ids, zero, zero, zero, zero, zero, context.plan_mask,
                baseline.cost, None, 0., candidate.status, 0., dict(context.metadata, t=context.t))
        else:
            label = verify_candidate(context, baseline, candidate)
        label.elapsed_seconds = time.perf_counter()-started
        return label
