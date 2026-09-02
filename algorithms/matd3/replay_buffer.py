from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    observations: dict[str, np.ndarray]
    global_state: np.ndarray
    actions: dict[str, np.ndarray]
    team_reward: float
    next_observations: dict[str, np.ndarray]
    next_global_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.storage: deque[Transition] = deque(maxlen=int(capacity))
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, observations: Mapping[str, np.ndarray], global_state: np.ndarray,
            actions: Mapping[str, np.ndarray], team_reward: float,
            next_observations: Mapping[str, np.ndarray], next_global_state: np.ndarray,
            done: bool) -> None:
        self.storage.append(Transition(
            {k: np.asarray(v, dtype=np.float32).copy() for k, v in observations.items()},
            np.asarray(global_state, dtype=np.float32).copy(),
            {k: np.asarray(v, dtype=np.float32).copy() for k, v in actions.items()},
            float(team_reward),
            {k: np.asarray(v, dtype=np.float32).copy() for k, v in next_observations.items()},
            np.asarray(next_global_state, dtype=np.float32).copy(), bool(done),
        ))

    def sample(self, batch_size: int, agent_ids: list[str], device: torch.device) -> dict[str, object]:
        indices = self.rng.choice(len(self.storage), size=int(batch_size), replace=False)
        batch = [self.storage[int(i)] for i in indices]
        tensor = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
        return {
            "observations": {mg: tensor([x.observations[mg] for x in batch]) for mg in agent_ids},
            "global_state": tensor([x.global_state for x in batch]),
            "actions": {mg: tensor([x.actions[mg] for x in batch]) for mg in agent_ids},
            "team_reward": tensor([[x.team_reward] for x in batch]),
            "next_observations": {mg: tensor([x.next_observations[mg] for x in batch]) for mg in agent_ids},
            "next_global_state": tensor([x.next_global_state for x in batch]),
            "done": tensor([[float(x.done)] for x in batch]),
        }
