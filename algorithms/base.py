from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ReplayBufferProtocol(Protocol):
    def __len__(self) -> int: ...

    def add(
        self,
        observations: Mapping[str, np.ndarray],
        global_state: np.ndarray,
        actions: Mapping[str, np.ndarray],
        team_reward: float,
        next_observations: Mapping[str, np.ndarray],
        next_global_state: np.ndarray,
        done: bool,
    ) -> None: ...


@runtime_checkable
class MultiAgentAlgorithm(Protocol):
    """Small contract required by the shared experiment pipeline."""

    replay: ReplayBufferProtocol

    def select_actions(
        self,
        observations: Mapping[str, np.ndarray],
        noise_std: float = 0.0,
    ) -> dict[str, np.ndarray]: ...

    def update(self, batch_size: int) -> dict[str, Any]: ...

    def save(self, path: str | Path) -> None: ...

    def load(self, path: str | Path) -> None: ...


@runtime_checkable
class RecoverableAlgorithmProtocol(Protocol):
    """Optional opaque training-state contract used only by resumable variants."""

    def save_training_state(
        self,
        directory: Path,
        progress: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def load_training_state(self, directory: Path) -> dict[str, Any]: ...
