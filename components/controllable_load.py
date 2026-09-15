from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd


class ControllableLoad(Protocol):
    """Small common surface for heterogeneous flexible electrical loads."""

    def reset(self, episode_frame: pd.DataFrame, base_load_kw: np.ndarray) -> None: ...

    def state(self, t: int) -> dict[str, float]: ...

    def feasible_bounds(self, t: int) -> tuple[float, float]: ...

    def plan(self, raw_action: float, t: int) -> dict[str, Any]: ...

    def commit(
        self,
        plan: Mapping[str, Any],
        delivered_fraction: float = 1.0,
    ) -> dict[str, float]: ...


def clipped_raw(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("DR action must be finite")
    return float(np.clip(result, -1.0, 1.0))


def raw_to_unit_interval(value: float) -> float:
    return 0.5 * (clipped_raw(value) + 1.0)


def delivered_fraction(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("delivered_fraction must be finite")
    return float(np.clip(result, 0.0, 1.0))
