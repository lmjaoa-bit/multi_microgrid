from __future__ import annotations

from collections import deque

import numpy as np

from algorithms.runtime_helpers import FastReplayBuffer


class CounterfactualReplayBuffer(FastReplayBuffer):
    """Actual transitions remain separate from bounded, recent auxiliary labels."""

    def __init__(self, capacity, label_capacity, seed, label_callback):
        super().__init__(capacity, seed)
        self.label_capacity = int(label_capacity)
        self.label_callback = label_callback
        self.by_agent = {}
        self.label_rng = np.random.default_rng(seed + 390001)
        self.pending_context = None

    @property
    def labels(self):
        return [label for buffer in self.by_agent.values() for label in buffer]

    def add(self, observations, global_state, actions, team_reward,
            next_observations, next_global_state, done):
        context, self.pending_context = self.pending_context, None
        if context is not None:
            if not np.array_equal(global_state, context.state) or any(
                not np.array_equal(observations[mg], context.observations[mg])
                for mg in observations
            ):
                raise ValueError("Transition does not match captured training context")
        super().add(observations, global_state, actions, team_reward,
                    next_observations, next_global_state, done)
        if context is not None:
            for label in self.label_callback(context, actions):
                self.by_agent.setdefault(label.agent, deque(maxlen=self.label_capacity)).append(label)

    def sample_labels(self, agent, size, step, max_age, *, approved_only=False):
        labels = [label for label in self.by_agent.get(agent, ())
                  if 0 <= step - label.step <= max_age
                  and (not approved_only or label.actor_approved)]
        if not labels:
            return []
        indices = self.label_rng.choice(len(labels), min(int(size), len(labels)), replace=False)
        return [labels[int(i)] for i in indices]
