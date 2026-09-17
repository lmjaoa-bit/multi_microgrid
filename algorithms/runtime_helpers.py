from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy

import numpy as np
import torch

from algorithms.matd3.replay_buffer import Transition


def _pack_agent_arrays(
    batch,
    agent_ids: Sequence[str],
    attribute: str,
) -> np.ndarray | None:
    rows = []
    expected_shape = None
    for transition in batch:
        mapping = getattr(transition, attribute)
        row = [np.asarray(mapping[mg], dtype=np.float32) for mg in agent_ids]
        shapes = {item.shape for item in row}
        if len(shapes) != 1:
            return None
        if expected_shape is None:
            expected_shape = row[0].shape
        elif row[0].shape != expected_shape:
            return None
        rows.append(np.stack(row, axis=0))
    return np.stack(rows, axis=0)


def _sample_transitions(replay, batch_size: int):
    indices = replay.rng.choice(len(replay), size=int(batch_size), replace=False)
    if hasattr(replay, "_transitions_for_indices"):
        return replay._transitions_for_indices(indices)
    return [replay.storage[int(i)] for i in indices]


def sample_replay_batch_packed(
    replay,
    batch_size: int,
    agent_ids: Sequence[str],
    device: torch.device,
) -> dict[str, object]:
    """Sample one replay batch while minimizing host-to-device transfers.

    Agent observations/actions are packed into [B, A, ...] arrays before conversion to
    tensors, then exposed as zero-copy tensor views per agent. This preserves the replay
    sampling distribution while reducing Python and CUDA transfer overhead.
    """
    batch = _sample_transitions(replay, batch_size)
    to_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)

    obs_np = _pack_agent_arrays(batch, agent_ids, "observations")
    next_obs_np = _pack_agent_arrays(batch, agent_ids, "next_observations")
    actions_np = _pack_agent_arrays(batch, agent_ids, "actions")

    if obs_np is None:
        observations = {
            mg: to_tensor(np.stack([x.observations[mg] for x in batch], axis=0))
            for mg in agent_ids
        }
    else:
        obs_tensor = to_tensor(obs_np)
        observations = {mg: obs_tensor[:, i] for i, mg in enumerate(agent_ids)}

    if next_obs_np is None:
        next_observations = {
            mg: to_tensor(np.stack([x.next_observations[mg] for x in batch], axis=0))
            for mg in agent_ids
        }
    else:
        next_obs_tensor = to_tensor(next_obs_np)
        next_observations = {mg: next_obs_tensor[:, i] for i, mg in enumerate(agent_ids)}

    if actions_np is None:
        actions = {
            mg: to_tensor(np.stack([x.actions[mg] for x in batch], axis=0))
            for mg in agent_ids
        }
    else:
        action_tensor = to_tensor(actions_np)
        actions = {mg: action_tensor[:, i] for i, mg in enumerate(agent_ids)}

    return {
        "observations": observations,
        "global_state": to_tensor(np.stack([x.global_state for x in batch], axis=0)),
        "actions": actions,
        "team_reward": to_tensor(np.asarray([[x.team_reward] for x in batch], dtype=np.float32)),
        "next_observations": next_observations,
        "next_global_state": to_tensor(np.stack([x.next_global_state for x in batch], axis=0)),
        "done": to_tensor(np.asarray([[float(x.done)] for x in batch], dtype=np.float32)),
    }


class FastReplayBuffer:
    """O(1) ring-buffer replay used only by experimental runtime-optimized variants."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.storage: list[Transition] = []
        self._next_index = 0
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_replay(cls, replay) -> "FastReplayBuffer":
        capacity = getattr(getattr(replay, "storage", None), "maxlen", None)
        if capacity is None:
            capacity = getattr(replay, "capacity", None)
        if capacity is None:
            raise ValueError("Cannot infer replay capacity")
        result = cls(int(capacity))
        result.rng.bit_generator.state = copy.deepcopy(replay.rng.bit_generator.state)
        for transition in replay.storage:
            result._append_transition(transition)
        return result

    def __len__(self) -> int:
        return len(self.storage)

    def _append_transition(self, transition: Transition) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
            self._next_index = len(self.storage) % self.capacity
        else:
            self.storage[self._next_index] = transition
            self._next_index = (self._next_index + 1) % self.capacity

    def add(
        self,
        observations: Mapping[str, np.ndarray],
        global_state: np.ndarray,
        actions: Mapping[str, np.ndarray],
        team_reward: float,
        next_observations: Mapping[str, np.ndarray],
        next_global_state: np.ndarray,
        done: bool,
    ) -> None:
        self._append_transition(Transition(
            {k: np.asarray(v, dtype=np.float32).copy() for k, v in observations.items()},
            np.asarray(global_state, dtype=np.float32).copy(),
            {k: np.asarray(v, dtype=np.float32).copy() for k, v in actions.items()},
            float(team_reward),
            {k: np.asarray(v, dtype=np.float32).copy() for k, v in next_observations.items()},
            np.asarray(next_global_state, dtype=np.float32).copy(),
            bool(done),
        ))

    def _logical_get(self, index: int) -> Transition:
        size = len(self.storage)
        if not 0 <= index < size:
            raise IndexError(index)
        if size < self.capacity:
            return self.storage[index]
        return self.storage[(self._next_index + index) % self.capacity]

    def _transitions_for_indices(self, indices) -> list[Transition]:
        return [self._logical_get(int(i)) for i in indices]

    def sample(
        self,
        batch_size: int,
        agent_ids: list[str],
        device: torch.device,
    ) -> dict[str, object]:
        return sample_replay_batch_packed(self, batch_size, agent_ids, device)
