from __future__ import annotations

from collections import OrderedDict
import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import algorithms.runtime_helpers as runtime_helpers
from algorithms.runtime_helpers import (
    diagnostics_due,
    forward_policy_actions,
    optimized_single_dispatch_update,
    select_single_dispatch_actions,
)
from utils import torch_runtime


class CountingActor(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)
        self.calls: list[bool] = []
        self.storage_ptrs: list[int] = []

    def forward(self, observation: torch.Tensor, *, collect_diagnostics: bool = True) -> torch.Tensor:
        self.calls.append(bool(collect_diagnostics))
        self.storage_ptrs.append(observation.untyped_storage().data_ptr())
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


def test_rollout_select_actions_uses_one_packed_observation_storage_and_disables_diagnostics():
    model = FakeAlgorithm()
    observations = {mg: np.ones(5, dtype=np.float32) for mg in model.agent_ids}

    actions = select_single_dispatch_actions(model, observations, noise_std=0.0)

    assert set(actions) == set(model.agent_ids)
    assert np.allclose(actions["mg1"], [-0.2])
    assert np.allclose(actions["mg2"], [0.1])
    assert np.allclose(actions["mg3"], [0.4])
    assert len({model.actors[mg].storage_ptrs[0] for mg in model.agent_ids}) == 1
    for mg in model.agent_ids:
        assert model.actors[mg].calls == [False]


def test_diagnostics_due_is_sparse_and_can_be_disabled():
    assert not diagnostics_due(update_steps=98, interval=100)
    assert diagnostics_due(update_steps=100, interval=100)
    assert diagnostics_due(update_steps=200, interval=100)
    assert not diagnostics_due(update_steps=100, interval=0)


def test_experimental_variants_actually_install_runtime_hot_path():
    from algorithms.matd3_transformer.algorithm import Matd3Transformer
    from algorithms.pnn.algorithm import PNN
    from algorithms.pnn_transformer.algorithm import PnnTransformer

    assert hasattr(runtime_helpers, "optimized_pnn_update")
    for cls in (Matd3Transformer, PNN, PnnTransformer):
        assert getattr(cls, "_runtime_optimizations_installed", False)
        assert cls.select_actions is select_single_dispatch_actions
    assert Matd3Transformer.update is optimized_single_dispatch_update
    assert PnnTransformer.update is optimized_single_dispatch_update
    assert PNN.update is runtime_helpers.optimized_pnn_update


def test_packed_replay_sampler_transfers_agent_fields_as_shared_tensors():
    assert hasattr(runtime_helpers, "sample_replay_batch_packed")
    ids = ["mg1", "mg2", "mg3"]
    storage = []
    for step in range(8):
        storage.append(SimpleNamespace(
            observations={mg: np.full(5, step + i, dtype=np.float32) for i, mg in enumerate(ids)},
            global_state=np.full(7, step, dtype=np.float32),
            actions={mg: np.asarray([0.1 * (i + 1)], dtype=np.float32) for i, mg in enumerate(ids)},
            team_reward=float(step),
            next_observations={mg: np.full(5, step + i + 1, dtype=np.float32) for i, mg in enumerate(ids)},
            next_global_state=np.full(7, step + 1, dtype=np.float32),
            done=False,
        ))
    replay = SimpleNamespace(storage=storage, rng=np.random.default_rng(11), __len__=lambda: 8)
    replay.__class__ = type("FakeReplay", (), {
        "__len__": lambda self: len(self.storage),
    })

    batch = runtime_helpers.sample_replay_batch_packed(replay, 4, ids, torch.device("cpu"))

    assert batch["global_state"].shape == (4, 7)
    assert batch["team_reward"].shape == (4, 1)
    assert len({batch["observations"][mg].untyped_storage().data_ptr() for mg in ids}) == 1
    assert len({batch["actions"][mg].untyped_storage().data_ptr() for mg in ids}) == 1
    assert len({batch["next_observations"][mg].untyped_storage().data_ptr() for mg in ids}) == 1


def test_torch_runtime_applies_requested_threads_even_for_cuda(monkeypatch):
    calls: dict[str, int] = {}
    monkeypatch.setattr(torch_runtime.torch, "set_num_threads", lambda value: calls.__setitem__("intra", value))
    monkeypatch.setattr(torch_runtime.torch, "set_num_interop_threads", lambda value: calls.__setitem__("interop", value))
    monkeypatch.setattr(torch_runtime.torch, "get_num_threads", lambda: calls.get("intra", 99))
    monkeypatch.setattr(torch_runtime.torch, "get_num_interop_threads", lambda: calls.get("interop", 77))

    result = torch_runtime.configure_torch_runtime(
        {"runtime": {"cpu_threads": 4, "interop_threads": 1}},
        "cuda",
    )

    assert calls == {"intra": 4, "interop": 1}
    assert result == {"cpu_threads": 4, "interop_threads": 1}


class TrainActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.calls: list[bool] = []
        self._parameter_count = sum(p.numel() for p in self.parameters())

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    def forward(self, observation: torch.Tensor, *, collect_diagnostics: bool = True) -> torch.Tensor:
        self.calls.append(bool(collect_diagnostics))
        return torch.tanh(self.linear(observation))


class TinyTwinCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q1_network = nn.Linear(10, 1)
        self.q2_network = nn.Linear(10, 1)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor):
        x = torch.cat([state, joint_action], dim=-1)
        return self.q1_network(x), self.q2_network(x)

    def q1(self, state: torch.Tensor, joint_action: torch.Tensor):
        return self.q1_network(torch.cat([state, joint_action], dim=-1))


class FixedReplay:
    def __init__(self, batch: dict[str, object]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 16

    def sample(self, batch_size: int, agent_ids: list[str], device: torch.device):
        return self.batch


class UpdateHarness:
    def __init__(self) -> None:
        torch.manual_seed(7)
        self.agent_ids = ["mg1", "mg2", "mg3"]
        self.device = torch.device("cpu")
        self.actors = nn.ModuleDict({mg: TrainActor() for mg in self.agent_ids})
        self.target_actors = copy.deepcopy(self.actors)
        self.critics = nn.ModuleDict({mg: TinyTwinCritic() for mg in self.agent_ids})
        self.target_critics = copy.deepcopy(self.critics)
        self.actor_optimizers = {
            mg: torch.optim.Adam(self.actors[mg].parameters(), lr=1e-3)
            for mg in self.agent_ids
        }
        self.critic_optimizers = {
            mg: torch.optim.Adam(self.critics[mg].parameters(), lr=1e-3)
            for mg in self.agent_ids
        }
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_delay = 2
        self.target_policy_noise = 0.0
        self.target_noise_clip = 0.0
        self.actor_max_grad_norm = 1.0
        self.diagnostics_interval = 100
        self._runtime_variant = "matd3_transformer"
        self.update_steps = 1
        self.rng = np.random.default_rng(42)

        batch_size = 4
        observations = {mg: torch.randn(batch_size, 3) for mg in self.agent_ids}
        next_observations = {mg: torch.randn(batch_size, 3) for mg in self.agent_ids}
        actions = {mg: torch.zeros(batch_size, 1) for mg in self.agent_ids}
        self.replay = FixedReplay({
            "observations": observations,
            "next_observations": next_observations,
            "actions": actions,
            "global_state": torch.randn(batch_size, 4),
            "next_global_state": torch.randn(batch_size, 4),
            "team_reward": torch.randn(batch_size, 1),
            "done": torch.zeros(batch_size, 1),
        })

    def project_actions(self, state: torch.Tensor, raw: torch.Tensor):
        effective = torch.cat([raw, torch.zeros_like(raw)], dim=-1)
        plans = []
        for _ in self.agent_ids:
            zero = torch.zeros(raw.shape[0], dtype=raw.dtype, device=raw.device)
            plans.append({
                "mask": zero.bool(),
                "forced_dispatch": zero.bool(),
                "local_priority_reserve": zero,
            })
        return effective, plans

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.mul_(1.0 - self.tau).add_(self.tau * parameter.data)

    def actor_parameter_counts(self):
        return {mg: self.actors[mg].parameter_count for mg in self.agent_ids}


def test_optimized_update_reuses_one_current_actor_forward_per_agent():
    model = UpdateHarness()
    metrics = optimized_single_dispatch_update(model, batch_size=4)

    assert "actor_loss" in metrics
    assert "actor_diagnostics" not in metrics
    for mg in model.agent_ids:
        assert model.actors[mg].calls == [False]
        assert model.target_actors[mg].calls == [False]
