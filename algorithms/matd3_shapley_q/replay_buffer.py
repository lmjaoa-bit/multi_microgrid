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
        self.storage: deque[ShapleyTransition] = deque(maxlen=int(capacity))
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, observations: Mapping[str, np.ndarray], global_state: np.ndarray,
            actions: Mapping[str, np.ndarray], team_reward: float,
            next_observations: Mapping[str, np.ndarray], next_global_state: np.ndarray,
            done: bool, *, independent_reward: float, cooperation_surplus_reward: float,
            independent_return_to_go: float, next_independent_return_to_go: float) -> None:
        values = dict(team_reward=float(team_reward), independent_reward=float(independent_reward),
                      cooperation_surplus_reward=float(cooperation_surplus_reward),
                      independent_return_to_go=float(independent_return_to_go),
                      next_independent_return_to_go=float(next_independent_return_to_go))
        bad = [name for name, value in values.items() if not np.isfinite(value)]
        if bad:
            raise ValueError(f"Non-finite Shapley replay values: {bad}")
        if not np.isclose(values["cooperation_surplus_reward"],
                          values["team_reward"] - values["independent_reward"],
                          rtol=1e-6, atol=1e-6):
            raise ValueError("cooperation_surplus_reward must equal team_reward - independent_reward")
        self.storage.append(ShapleyTransition(
            observations={k: np.asarray(v, dtype=np.float32).copy() for k, v in observations.items()},
            global_state=np.asarray(global_state, dtype=np.float32).copy(),
            actions={k: np.asarray(v, dtype=np.float32).copy() for k, v in actions.items()},
            next_observations={k: np.asarray(v, dtype=np.float32).copy() for k, v in next_observations.items()},
            next_global_state=np.asarray(next_global_state, dtype=np.float32).copy(),
            done=bool(done), **values,
        ))

    def sample(self, batch_size: int, agent_ids: Sequence[str], device: torch.device) -> dict:
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
