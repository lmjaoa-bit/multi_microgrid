"""Replay for the fixed-baseline cooperation-surplus objective."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ShapleyTransition:
    observations: dict[str, np.ndarray]
    global_state: np.ndarray
    actions: dict[str, np.ndarray]
    team_reward: float
    next_observations: dict[str, np.ndarray]
    next_global_state: np.ndarray
    done: bool
    independent_reward: float
    cooperation_surplus_reward: float
    independent_return_to_go: float
    next_independent_return_to_go: float


class ShapleyReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Replay capacity must be a positive integer")
        self.storage: deque[ShapleyTransition] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)
        self._agent_ids: frozenset[str] | None = None
        self._array_shapes: dict[str, dict[str, tuple[int, ...]]] | None = None
        self._state_shape: tuple[int, ...] | None = None

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, observations: Mapping[str, np.ndarray], global_state: np.ndarray,
            actions: Mapping[str, np.ndarray], team_reward: float,
            next_observations: Mapping[str, np.ndarray], next_global_state: np.ndarray,
            done: bool, *, independent_reward: float, cooperation_surplus_reward: float,
            independent_return_to_go: float, next_independent_return_to_go: float) -> None:
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("done must be a boolean")

        scalar_inputs = dict(
            team_reward=team_reward,
            independent_reward=independent_reward,
            cooperation_surplus_reward=cooperation_surplus_reward,
            independent_return_to_go=independent_return_to_go,
            next_independent_return_to_go=next_independent_return_to_go,
        )
        values: dict[str, float] = {}
        for name, value in scalar_inputs.items():
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    normalized = np.float32(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite float32 scalar") from exc
            if normalized.ndim != 0 or not np.isfinite(normalized):
                raise ValueError(f"{name} must be a finite float32 scalar")
            values[name] = float(normalized)
        if not np.isclose(values["cooperation_surplus_reward"],
                          values["team_reward"] - values["independent_reward"],
                          rtol=1e-6, atol=1e-6):
            raise ValueError("cooperation_surplus_reward must equal team_reward - independent_reward")

        mappings = {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
        }
        if any(not isinstance(value, Mapping) for value in mappings.values()):
            raise ValueError("observations, actions and next_observations must be mappings")
        key_sets = {name: frozenset(value) for name, value in mappings.items()}
        candidate_agents = key_sets["observations"]
        if not candidate_agents or any(keys != candidate_agents for keys in key_sets.values()):
            raise ValueError("observations, actions and next_observations must have identical agent keys")
        if self._agent_ids is not None and candidate_agents != self._agent_ids:
            raise ValueError("observations, actions and next_observations agent keys changed")

        normalized_mappings: dict[str, dict[str, np.ndarray]] = {}
        candidate_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
        for field, mapping in mappings.items():
            normalized_mappings[field] = {}
            candidate_shapes[field] = {}
            for agent, raw_value in mapping.items():
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        array = np.asarray(raw_value, dtype=np.float32)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"{field}[{agent!r}] must be a finite vector") from exc
                if array.ndim != 1 or array.size == 0:
                    raise ValueError(f"{field}[{agent!r}] must be a non-empty vector")
                if not np.isfinite(array).all():
                    raise ValueError(f"{field}[{agent!r}] contains non-finite float32 values")
                normalized_mappings[field][agent] = array.copy()
                candidate_shapes[field][agent] = array.shape

        for agent in candidate_agents:
            if candidate_shapes["observations"][agent] != candidate_shapes["next_observations"][agent]:
                raise ValueError(f"next_observations[{agent!r}] shape does not match observations")
        if self._array_shapes is not None and candidate_shapes != self._array_shapes:
            changed = next(
                field for field in candidate_shapes
                if candidate_shapes[field] != self._array_shapes[field]
            )
            raise ValueError(f"{changed} shape does not match replay schema")

        states = {}
        for field, raw_value in (("global_state", global_state),
                                 ("next_global_state", next_global_state)):
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    array = np.asarray(raw_value, dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{field} must be a finite state vector") from exc
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"{field} must be a non-empty state vector")
            if not np.isfinite(array).all():
                raise ValueError(f"{field} contains non-finite float32 state values")
            states[field] = array.copy()
        if states["global_state"].shape != states["next_global_state"].shape:
            raise ValueError("next_global_state shape does not match global_state")
        if self._state_shape is not None and states["global_state"].shape != self._state_shape:
            raise ValueError("global_state shape does not match replay state schema")

        # Do not establish or mutate the replay schema until all validation passes.
        if self._agent_ids is None:
            self._agent_ids = candidate_agents
            self._array_shapes = candidate_shapes
            self._state_shape = states["global_state"].shape
        self.storage.append(ShapleyTransition(
            observations=normalized_mappings["observations"],
            global_state=states["global_state"],
            actions=normalized_mappings["actions"],
            next_observations=normalized_mappings["next_observations"],
            next_global_state=states["next_global_state"],
            done=bool(done),
            **values,
        ))

    def sample(self, batch_size: int, agent_ids: Sequence[str], device: torch.device) -> dict:
        if type(batch_size) is not int or batch_size < 1 or batch_size > len(self.storage):
            raise ValueError("batch_size must be a positive integer no larger than replay size")
        if self._agent_ids is None or frozenset(agent_ids) != self._agent_ids:
            raise ValueError("agent_ids do not match replay schema")
        indices = self.rng.choice(len(self.storage), size=int(batch_size), replace=False)
        batch = [self.storage[int(i)] for i in indices]
        def tensor(values):
            return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)
        result = {
            field: {mg: tensor([getattr(row, field)[mg] for row in batch]) for mg in agent_ids}
            for field in ("observations", "next_observations", "actions")
        }
        for field in ("global_state", "next_global_state"):
            result[field] = tensor([getattr(row, field) for row in batch])
        for field in ("team_reward", "independent_reward", "cooperation_surplus_reward",
                      "independent_return_to_go", "next_independent_return_to_go", "done"):
            result[field] = tensor([[float(getattr(row, field))] for row in batch])
        return result
