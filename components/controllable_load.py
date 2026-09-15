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


def build_controllable_load(
    config: Mapping[str, Any],
    *,
    delta_t: float,
    episode_length: int,
) -> ControllableLoad:
    """Build one physical DR model from a microgrid ``dr`` configuration."""
    kind = str(config.get("type", ""))
    if kind == "industrial_mill":
        from components.industrial_mill import IndustrialMillDR
        return IndustrialMillDR(config, delta_t=delta_t, episode_length=episode_length)
    if kind == "commercial_hvac":
        from components.commercial_hvac import CommercialHVACDR
        return CommercialHVACDR(config, delta_t=delta_t, episode_length=episode_length)
    if kind == "ev_charging":
        from components.ev_charging import EVChargingDR
        return EVChargingDR(config, delta_t=delta_t, episode_length=episode_length)
    raise ValueError(f"Unsupported controllable load type: {kind!r}")
