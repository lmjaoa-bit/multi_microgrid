from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import pytest
import torch

from algorithms.matd3_shapley_q.stability import lower_soc_discharge_gradient_proxy

AGENTS = ("mg1", "mg2", "mg3")
MODULE_PATH = Path(__file__).parents[1] / "algorithms" / "matd3_shapley_q" / "vectorized.py"


def _load_vectorized():
    assert MODULE_PATH.exists(), "vectorized Shapley-Q helper module is missing"
    spec = importlib.util.spec_from_file_location("algorithms.matd3_shapley_q.vectorized", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _coalitions():
    return tuple(
        frozenset(combo)
        for size in range(len(AGENTS) + 1)
        for combo in combinations(AGENTS, size)
    )


class FakeBaseProjector:
    enabled = True
    pair_capacity = 2.5

    def __init__(self):
        self.calls = 0
        self.batteries = {
            mg: {"max_charge_kw": 2.0 + i, "max_discharge_kw": 3.0 + i}
            for i, mg in enumerate(AGENTS)
        }

    def __call__(self, state, raw):
        self.calls += 1
        effective_channels = []
        plans = []
        for i, _ in enumerate(AGENTS):
            u = raw[:, i, 0]
            dis_max = state[:, i].clamp(min=0.0)
            local = torch.where(u >= 0, u * dis_max, -0.4 * u)
            effective_channels.append(torch.stack([local, torch.zeros_like(local)], dim=-1))
            buy = torch.full_like(local, 0.35 + 0.03 * i)
            sell = torch.full_like(local, 0.20 + 0.02 * i)
            demand = torch.full_like(local, 1.0 + 0.25 * i)
            remaining = (1.8 - local).clamp(min=0.0)
            plans.append({
                "local": local,
                "buy": buy,
                "sell": sell,
                "demand": demand,
                "charging": u < 0,
                "deficit": torch.full_like(local, 2.0),
                "remaining": remaining,
                "uncapped_offer": u.clamp(min=0.0) * remaining,
                "dis_max": dis_max,
            })
        return torch.stack(effective_channels, dim=1), plans


def _legacy_project(base, state, raw, coalition):
    effective, plans = base(state, raw)
    membership = torch.tensor(
        [mg in coalition for mg in AGENTS], dtype=torch.bool, device=state.device
    ).unsqueeze(0).expand(state.shape[0], -1)
    channels = []
    for i, mg in enumerate(AGENTS):
        plan = plans[i]
        member = membership[:, i]
        peer_cap = torch.zeros_like(plan["local"])
        for j, other in enumerate(plans):
            if j != i:
                eligible_pair = member & membership[:, j]
                peer_cap = peer_cap + torch.where(
                    eligible_pair & (other["buy"] + 1e-10 >= plan["sell"]),
                    other["demand"].clamp(max=base.pair_capacity),
                    torch.zeros_like(peer_cap),
                )
        cap = torch.where(
            (~member) | plan["charging"] | (plan["deficit"] - plan["local"] > 1e-7),
            torch.zeros_like(peer_cap),
            torch.minimum(plan["remaining"], peer_cap),
        )
        offer = torch.minimum(plan["uncapped_offer"], cap)
        scale = max(base.batteries[mg]["max_charge_kw"], base.batteries[mg]["max_discharge_kw"])
        effective_local, _ = lower_soc_discharge_gradient_proxy(
            effective[:, i, 0], raw[:, i, 0], plan["dis_max"]
        )
        channels.append(torch.stack([effective_local, offer / scale], dim=-1))
    return torch.stack(channels, dim=1)


def test_project_many_matches_legacy_values_and_gradients_with_one_base_projection():
    vectorized = _load_vectorized()
    state = torch.tensor([[0.0, 1.2, 0.8], [1.0, 0.0, 1.4]], dtype=torch.float32)
    raw_fast = torch.tensor(
        [[[0.8], [0.3], [-0.2]], [[0.4], [0.7], [0.5]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    coalitions = _coalitions()

    fast_base = FakeBaseProjector()
    projector = vectorized.CoalitionAwareProjector(fast_base, AGENTS)
    projected, proxy_masks = projector.project_many(state, raw_fast, coalitions)

    assert fast_base.calls == 1
    assert set(projected) == set(coalitions)
    assert proxy_masks["mg1"].tolist() == [True, False]
    assert proxy_masks["mg2"].tolist() == [False, True]

    fast_loss = sum(projected[c].sum() for c in coalitions)
    fast_loss.backward()
    fast_grad = raw_fast.grad.detach().clone()

    raw_legacy = raw_fast.detach().clone().requires_grad_(True)
    legacy_base = FakeBaseProjector()
    legacy = {c: _legacy_project(legacy_base, state, raw_legacy, c) for c in coalitions}
    assert legacy_base.calls == len(coalitions)
    for coalition in coalitions:
        torch.testing.assert_close(projected[coalition].detach(), legacy[coalition].detach())
    legacy_loss = sum(legacy[c].sum() for c in coalitions)
    legacy_loss.backward()
    torch.testing.assert_close(fast_grad, raw_legacy.grad, rtol=1e-5, atol=1e-6)


class CountingCritic:
    def __init__(self):
        self.calls = 0

    def __call__(self, state, before, after, predecessor_mask):
        self.calls += 1
        before_sum = before.flatten(1).sum(dim=1, keepdim=True)
        after_sum = after.flatten(1).sum(dim=1, keepdim=True)
        state_sum = state.sum(dim=1, keepdim=True)
        mask_sum = predecessor_mask.sum(dim=1, keepdim=True)
        q1 = 0.11 * state_sum + 0.07 * before_sum + 0.13 * after_sum + 0.17 * mask_sum
        q2 = -0.05 * state_sum + 0.19 * before_sum + 0.03 * after_sum + 0.23 * mask_sum
        return q1, q2


def _predecessors(active):
    others = [mg for mg in AGENTS if mg != active]
    return (
        (frozenset(), 1.0 / 3.0),
        (frozenset({others[0]}), 1.0 / 6.0),
        (frozenset({others[1]}), 1.0 / 6.0),
        (frozenset(others), 1.0 / 3.0),
    )


def _legacy_agent_values(state, projected, active, critic):
    first, second = [], []
    for predecessor, weight in _predecessors(active):
        joined = predecessor | {active}
        mask = state.new_tensor([mg in predecessor for mg in AGENTS]).expand(state.shape[0], -1)
        q1, q2 = critic(state, projected[predecessor], projected[joined], mask)
        first.append(weight * q1)
        second.append(weight * q2)
    return sum(first), sum(second)


def test_batched_predecessors_match_legacy_and_reduce_critic_calls_four_to_one():
    vectorized = _load_vectorized()
    torch.manual_seed(7)
    state = torch.randn(5, 4)
    coalitions = _coalitions()
    projected_fast = {
        c: torch.randn(5, len(AGENTS), 2, requires_grad=True) for c in coalitions
    }
    active = "mg2"

    fast_critic = CountingCritic()
    fast_q1, fast_q2 = vectorized.batched_agent_shapley_values(
        state, projected_fast, _predecessors(active), active, fast_critic, AGENTS
    )
    assert fast_critic.calls == 1
    (fast_q1.sum() + fast_q2.sum()).backward()
    fast_grads = {c: projected_fast[c].grad.detach().clone() for c in coalitions}

    projected_legacy = {
        c: projected_fast[c].detach().clone().requires_grad_(True) for c in coalitions
    }
    legacy_critic = CountingCritic()
    legacy_q1, legacy_q2 = _legacy_agent_values(state, projected_legacy, active, legacy_critic)
    assert legacy_critic.calls == 4
    torch.testing.assert_close(fast_q1.detach(), legacy_q1.detach())
    torch.testing.assert_close(fast_q2.detach(), legacy_q2.detach())
    (legacy_q1.sum() + legacy_q2.sum()).backward()
    for coalition in coalitions:
        torch.testing.assert_close(fast_grads[coalition], projected_legacy[coalition].grad)


def test_training_algorithm_wires_vectorized_hot_path():
    algorithm_path = Path(__file__).parents[1] / "algorithms" / "matd3_shapley_q" / "algorithm.py"
    assert algorithm_path.exists(), "Matd3ShapleyQ algorithm module is missing"
    source = algorithm_path.read_text(encoding="utf-8")
    assert "batched_agent_shapley_values(" in source
    assert source.count(".project_many(") >= 2
    assert "self.projector(state, joint_actions)" not in source
