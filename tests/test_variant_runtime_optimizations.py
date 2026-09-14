from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from algorithms.runtime_helpers import diagnostics_due, forward_policy_actions, select_single_dispatch_actions


class CountingActor(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)
        self.calls: list[bool] = []

    def forward(self, observation: torch.Tensor, *, collect_diagnostics: bool = True) -> torch.Tensor:
        self.calls.append(bool(collect_diagnostics))
        batch = 1 if observation.ndim == 1 else observation.shape[0]
        return torch.full((batch, 1), self.value, dtype=observation.dtype, device=observation.device)


class FakeAlgorithm:
    def __init__(self) -> None:
        self.agent_ids = ["mg1", "mg2", "mg3"]
        self.device = torch.device("cpu")
        self.actors = nn.ModuleDict(OrderedDict([
            ("mg1", CountingActor(-0.2)),
            ("mg2", CountingActor(0.1)),
            ("mg3", CountingActor(0.4)),
        ]))
        self.rng = np.random.default_rng(42)


def test_policy_action_cache_forwards_each_actor_once():
    model = FakeAlgorithm()
    observations = {mg: torch.ones(8, 5) for mg in model.agent_ids}

    actions = forward_policy_actions(
        model.actors,
        observations,
        model.agent_ids,
        collect_diagnostics=False,
    )

    assert set(actions) == set(model.agent_ids)
    for mg in model.agent_ids:
        assert model.actors[mg].calls == [False]
        assert actions[mg].shape == (8, 1)


def test_rollout_select_actions_disables_hot_path_diagnostics():
    model = FakeAlgorithm()
    observations = {mg: np.ones(5, dtype=np.float32) for mg in model.agent_ids}

    actions = select_single_dispatch_actions(model, observations, noise_std=0.0)

    assert set(actions) == set(model.agent_ids)
    assert np.allclose(actions["mg1"], [-0.2])
    assert np.allclose(actions["mg2"], [0.1])
    assert np.allclose(actions["mg3"], [0.4])
    for mg in model.agent_ids:
        assert model.actors[mg].calls == [False]


def test_diagnostics_due_is_sparse_and_can_be_disabled():
    assert not diagnostics_due(update_steps=98, interval=100)
    assert diagnostics_due(update_steps=100, interval=100)
    assert diagnostics_due(update_steps=200, interval=100)
    assert not diagnostics_due(update_steps=100, interval=0)
