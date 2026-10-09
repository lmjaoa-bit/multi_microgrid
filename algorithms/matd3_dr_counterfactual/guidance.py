from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Callable

import numpy as np
import torch


def resolve_guidance(settings):
    defaults = dict(capture_probability=0.025, terminal_capture_probability=1.0,
                    capacity=512, batch_size=16, max_age_steps=2400,
                    update_interval=2, critic_weight=0.5, actor_weight=0.01,
                    terminal_anchor_weight=0.1, minimum_gain=0.001)
    if not isinstance(settings, dict) or set(settings) - set(defaults):
        raise ValueError("Unknown dr_counterfactual settings")
    result = {**defaults, **settings}
    for name in ("capacity", "batch_size", "max_age_steps", "update_interval"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"dr_counterfactual.{name} must be a positive integer")
        result[name] = int(value)
    for name in set(defaults) - {"capacity", "batch_size", "max_age_steps", "update_interval"}:
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"dr_counterfactual.{name} must be finite and nonnegative")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"dr_counterfactual.{name} must be finite and nonnegative")
        if name.endswith("probability") and value > 1:
            raise ValueError(f"dr_counterfactual.{name} must be at most one")
        result[name] = float(value)
    return result


def critic_pair_loss(reference, candidate, measured_gain):
    """Fit action differences, retaining both beneficial and harmful probes."""
    return sum(torch.nn.functional.smooth_l1_loss(alternative - original, measured_gain)
               for original, alternative in zip(reference, candidate))


def dr_logit_loss(logits, targets, *, weight):
    """DR-only output supervision; finite endpoint targets can attain float32 +/-1."""
    clipped = targets.clamp(-1 + 1e-6, 1 - 1e-6)
    desired = torch.atanh(clipped)
    desired = torch.where(targets >= 1, torch.full_like(desired, 10.0), desired)
    desired = torch.where(targets <= -1, torch.full_like(desired, -10.0), desired)
    return float(weight) * torch.nn.functional.smooth_l1_loss(logits[:, 1], desired)


@dataclass(frozen=True)
class CounterfactualLabel:
    agent: str
    observation: np.ndarray
    state: np.ndarray
    reference_actions: np.ndarray
    candidate_actions: np.ndarray
    reference_return: float
    candidate_return: float
    candidate_raw: float
    service_safe: bool
    actor_approved: bool
    rollout_steps: int
    step: int

    @property
    def gain(self):
        return self.candidate_return - self.reference_return


@dataclass(frozen=True)
class TrainingContext:
    env: object
    observations: dict
    state: np.ndarray
    agent: str
    step: int


@dataclass(frozen=True)
class RolloutScore:
    total_return: float
    service: np.ndarray
    steps: int


class CounterfactualTeacher:
    """Training-day Monte Carlo comparisons using the unchanged physical environment.

    The first joint action differs in one DR channel. Subsequent actions come
    from the same frozen-in-time deterministic policy callback. Exogenous truth
    is encountered through normal step(), never supplied to the Actor as a plan.
    """

    def __init__(self, agent_ids, *, gamma, minimum_gain):
        self.agent_ids = list(agent_ids)
        self.gamma = float(gamma)
        self.minimum_gain = float(minimum_gain)
        self.compare_calls = 0
        self.rollout_steps = 0

    def _rollout(self, snapshot, first_actions, policy: Callable):
        env = copy.deepcopy(snapshot)
        actions = {mg: a.copy() for mg, a in first_actions.items()}
        total_return, discount, service, steps = 0.0, 1.0, [], 0
        done = False
        while not done:
            obs, rewards, done, info = env.step(actions)
            total_return += discount * float(rewards[self.agent_ids[0]])
            discount *= self.gamma
            for mg in self.agent_ids:
                dr = info["dr_devices"][mg]
                service.extend(float(dr.get(key, 0.0)) for key in (
                    "comfort_degree_hours", "downstream_unserved_t",
                    "inventory_min_violation_t", "inventory_max_violation_t"))
                service.append(float(info["settlements"][mg]["unmet_load_kw"]))
                service.append(float(info["terminal_constraint_penalty"][mg]["total_penalty"]))
            steps += 1
            if not done:
                actions = policy(obs)
        # Preserve each cohort's service, not only the station's aggregate deficit.
        for mg in self.agent_ids:
            device = env.dr_devices[mg]
            events = {item["group_id"]: item["shortfall_kwh"]
                      for item in getattr(device, "departure_events", [])}
            for session in sorted(getattr(device, "sessions", []), key=lambda s: s["group_id"]):
                service.append(float(events[session["group_id"]]))
        service = np.asarray(service, dtype=np.float64)
        if not math.isfinite(total_return) or not np.isfinite(service).all():
            raise ValueError("Non-finite counterfactual rollout")
        self.rollout_steps += steps
        return RolloutScore(total_return, service, steps)

    def compare(self, env, observations, state, actions, agent, policy, *, step):
        if env.split != "train":
            raise ValueError("Counterfactual labels require a training context")
        self.compare_calls += 1
        # Read this context only. Each _rollout already makes its own independent
        # deep copy before stepping; an additional whole-env copy here is unused.
        snapshot = env
        kind = snapshot.cfg_by_id[agent]["dr"]["type"]
        candidates = [-1., -0.5, -0.15, 0., 0.15, 0.5, 1.] if kind == "commercial_hvac" else [-1., 0., 1.]
        original = {mg: np.asarray(actions[mg], dtype=np.float32).copy() for mg in self.agent_ids}
        device = copy.deepcopy(snapshot.dr_devices[agent])
        original_effective = float(device.plan(float(original[agent][1]), snapshot.t)["effective_normalized"])
        unique = []
        for raw in candidates:
            effective = float(device.plan(raw, snapshot.t)["effective_normalized"])
            if abs(effective - original_effective) <= 1e-6:
                continue
            if any(abs(effective - previous) <= 1e-6 for _, previous in unique):
                continue
            unique.append((raw, effective))
        if not unique:
            return []
        reference = self._rollout(snapshot, original, policy)
        reference_raw = np.stack([original[mg] for mg in self.agent_ids])
        labels = []
        for raw, _ in unique:
            changed = {mg: a.copy() for mg, a in original.items()}
            changed[agent][1] = raw
            alternative = self._rollout(snapshot, changed, policy)
            safe = bool(np.all(alternative.service <= reference.service + 1e-6))
            labels.append(CounterfactualLabel(
                agent, np.asarray(observations[agent], dtype=np.float32).copy(),
                np.asarray(state, dtype=np.float32).copy(), reference_raw.copy(),
                np.stack([changed[mg] for mg in self.agent_ids]), reference.total_return,
                alternative.total_return, raw, safe, False, alternative.steps, int(step)))
        approved = [i for i, label in enumerate(labels)
                    if label.service_safe and label.gain > self.minimum_gain]
        if approved:
            best = max(approved, key=lambda i: labels[i].gain)
            labels[best] = replace(labels[best], actor_approved=True)
        return labels
