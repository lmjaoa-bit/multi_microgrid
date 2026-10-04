from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from algorithms.runtime_helpers import FastReplayBuffer, sample_replay_batch_packed


@dataclass
class AuxiliaryBatch:
    observations: dict
    global_state: torch.Tensor
    mask: torch.Tensor
    soc_after: dict
    ess_net_kw: dict
    context_hashes: list[str]


class PlanReplayBuffer(FastReplayBuffer):
    def __init__(self, capacity, auxiliary_capacity, seed, auxiliary_seed):
        super().__init__(capacity, seed)
        if isinstance(auxiliary_capacity, bool) or int(auxiliary_capacity) != auxiliary_capacity or auxiliary_capacity <= 0:
            raise ValueError('auxiliary_capacity must be a positive integer')
        self.auxiliary_capacity = int(auxiliary_capacity)
        self.auxiliary_rng = np.random.default_rng(auxiliary_seed)
        self.labels = []
        self._label_hashes = set()
        self.pending_context = None
        self._pending_label = None

    def bind_context(self, context, label):
        if self.pending_context is not None:
            raise ValueError('Previous planning context has not been consumed')
        if label is not None and label.context_hash != context.context_hash:
            raise ValueError('Label/context hash mismatch')
        self.pending_context, self._pending_label = context, label

    def add(self, observations, global_state, actions, team_reward,
            next_observations, next_global_state, done):
        context, label = self.pending_context, self._pending_label
        self.pending_context, self._pending_label = None, None
        if context is not None:
            if not np.array_equal(global_state, context.global_state) or any(
                not np.array_equal(observations[mg], context.observations[mg]) for mg in context.agent_ids):
                raise ValueError('Transition does not match captured context')
        super().add(observations, global_state, actions, team_reward,
                    next_observations, next_global_state, done)
        if label is not None and label.valid and label.context_hash not in self._label_hashes and len(self.labels) < self.auxiliary_capacity:
            self.labels.append(label)
            self._label_hashes.add(label.context_hash)

    def sample(self, batch_size, agent_ids, device):
        return sample_replay_batch_packed(self, batch_size, agent_ids, device)

    def sample_auxiliary(self, batch_size, agent_ids, device):
        if not self.labels:
            return None
        indices = self.auxiliary_rng.choice(len(self.labels), min(int(batch_size), len(self.labels)), replace=False)
        labels = [self.labels[int(i)] for i in indices]
        tensor = lambda values: torch.as_tensor(np.array(values), dtype=torch.float32, device=device)
        return AuxiliaryBatch(
            {mg: tensor([x.observations[mg] for x in labels]) for mg in agent_ids},
            tensor([x.global_state for x in labels]), tensor([x.mask for x in labels]),
            {mg: tensor([x.soc_after[mg] for x in labels]) for mg in agent_ids},
            {mg: tensor([x.ess_net_kw[mg] for x in labels]) for mg in agent_ids},
            [x.context_hash for x in labels])
