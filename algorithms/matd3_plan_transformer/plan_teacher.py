from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from .forecast_scenario import build_forecast_environment, digest


@dataclass(frozen=True)
class PlanLabel:
    valid: bool
    rejection_reason: str | None
    agent_ids: tuple[str, ...]
    context_hash: str
    forecast_hash: str
    contract_hash: str
    observations: dict
    global_state: np.ndarray
    t: int
    mask: np.ndarray
    soc_after: dict
    ess_net_kw: dict
    actions: np.ndarray
    dr_power_kw: dict
    baseline_costs: list
    best_cost: float | None
    seed: int
    budget: dict
    elapsed_seconds: float
    metadata: dict


class PlanTeacher:
    def __init__(self, settings, seed):
        defaults = dict(population=32, elite=8, iterations=3, initial_std=.5,
                        min_std=.05, update_rate=.5, max_labels=2048)
        self.settings = {**defaults, **dict(settings)}
        for key in ('population', 'elite', 'iterations', 'max_labels'):
            value = self.settings[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if self.settings['elite'] > self.settings['population']:
            raise ValueError('elite cannot exceed population')
        for key in ('initial_std', 'min_std', 'update_rate'):
            if not np.isfinite(self.settings[key]) or self.settings[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.settings['update_rate'] > 1:
            raise ValueError('update_rate cannot exceed one')
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        self.solve_calls = 0

    @property
    def cache_size(self):
        return len(self.cache)

    def _rollout(self, context, actions):
        env = build_forecast_environment(context)
        ids = context.agent_ids
        soc = {mg: np.zeros(24) for mg in ids}
        power = {mg: np.zeros(24) for mg in ids}
        dr = {mg: np.zeros(24) for mg in ids}
        cost = 0.0
        for j, raw in enumerate(actions):
            old_soc = {mg: float(env.cores[mg].battery.soc) for mg in ids}
            _, _, _, info = env.step(dict(zip(ids, raw)))
            value = float(info['generalized_system_cost'])
            if not np.isfinite(value):
                return float('inf'), None
            cost += value
            ledger = info['dispatch_ledger']
            for i, mg in enumerate(ids):
                battery = env.cores[mg].battery
                charge = float(ledger['actual_charge_kw'][i])
                discharge = float(ledger['actual_discharge_kw'][i])
                after = float(battery.soc)
                expected = old_soc[mg] + (charge * battery.eta_charge - discharge / battery.eta_discharge) * env.delta_t / battery.capacity_kwh
                if (not np.isfinite(after) or abs(expected - after) > 1e-7 or
                    after < battery.soc_min - 1e-7 or after > battery.soc_max + 1e-7 or
                    charge > battery.max_charge_kw + 1e-7 or discharge > battery.max_discharge_kw + 1e-7 or
                    min(charge, discharge) > 1e-7):
                    return float('inf'), None
                soc[mg][j], power[mg][j] = after, discharge - charge
                dr[mg][j] = float(info['dr_plans'][mg]['flexible_load_kw'])
        return cost, (soc, power, dr)

    def solve(self, context):
        key = digest(dict(context=context.context_hash, version='cem-feasible-v1',
                          settings=self.settings, seed=self.seed))
        if key in self.cache:
            return self.cache[key]
        started = time.perf_counter()
        shape = (24 - context.t, len(context.agent_ids), 2)
        best_raw, best_cost, best_data = np.zeros(shape), float('inf'), None
        baselines = []
        reason = None
        if self.solve_calls >= self.settings['max_labels']:
            reason = 'teacher_budget_exhausted'
        else:
            self.solve_calls += 1
            for dr_intent in (0., -1., 1.):
                candidate = np.zeros(shape)
                candidate[:, :, 1] = dr_intent
                score, data = self._rollout(context, candidate)
                baselines.append(score if np.isfinite(score) else None)
                if score < best_cost:
                    best_raw, best_cost, best_data = candidate, score, data
            mean = np.zeros(shape)
            std = np.full(shape, self.settings['initial_std'])
            for _ in range(self.settings['iterations']):
                population = np.clip(self.rng.normal(mean, std, (self.settings['population'], *shape)), -1, 1)
                scores = []
                for candidate in population:
                    score, data = self._rollout(context, candidate)
                    scores.append(score)
                    if score < best_cost:
                        best_raw, best_cost, best_data = candidate.copy(), score, data
                ordered = np.argsort(scores, kind='stable')
                chosen = [i for i in ordered if np.isfinite(scores[i])][:self.settings['elite']]
                if chosen:
                    elite = population[chosen]
                    rate = self.settings['update_rate']
                    mean = (1 - rate) * mean + rate * elite.mean(axis=0)
                    std = np.maximum(self.settings['min_std'], (1 - rate) * std + rate * elite.std(axis=0))
            if best_data is None:
                reason = 'no_physically_valid_candidate'
        empty = lambda: {mg: np.zeros(24) for mg in context.agent_ids}
        soc, power, dr = best_data or (empty(), empty(), empty())
        padded = np.zeros((24, len(context.agent_ids), 2))
        padded[:shape[0]] = best_raw
        for arr in [padded, *soc.values(), *power.values(), *dr.values()]:
            arr.setflags(write=False)
        label = PlanLabel(best_data is not None, reason, context.agent_ids,
            context.context_hash, context.forecast_hash, context.contract_hash,
            context.observations, context.global_state, context.t, context.plan_mask,
            soc, power, padded, dr, baselines, best_cost if best_data is not None else None,
            self.seed, dict(self.settings), time.perf_counter() - started, context.metadata)
        if self.cache_size < self.settings['max_labels']:
            self.cache[key] = label
        return label
