from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AdaptiveDRMap:
    action: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    reference: torch.Tensor
    available: torch.Tensor
    mandatory_floor: torch.Tensor
    recovery_anchor: torch.Tensor
    recovery_gate: torch.Tensor


def smoothstep(value: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    if not edge1 > edge0:
        raise ValueError("smoothstep requires edge1 > edge0")
    unit = ((value - float(edge0)) / (float(edge1) - float(edge0))).clamp(0.0, 1.0)
    return unit.square() * (3.0 - 2.0 * unit)


def residual_map(
    latent: torch.Tensor,
    lower: torch.Tensor,
    reference: torch.Tensor,
    upper: torch.Tensor,
    *,
    interval_epsilon: float,
) -> torch.Tensor:
    low = torch.minimum(lower, upper)
    high = torch.maximum(lower, upper)
    ref = torch.maximum(low, torch.minimum(reference, high))
    z = latent.clamp(-1.0, 1.0)
    mapped = ref + torch.relu(z) * (high - ref) - torch.relu(-z) * (ref - low)
    degenerate = (high - low) <= float(interval_epsilon)
    return torch.where(degenerate, low + 0.0 * latent, mapped)


def validate_adaptive_dr_config(
    config: Mapping[str, Any],
    *,
    forecast_horizon: int,
    ev_max_steps: int,
    episode_horizon_hours: float,
) -> None:
    if not config.get("exploration_in_latent_space", True):
        raise ValueError("V2 requires exploration_in_latent_space=true")
    if not config.get("target_smoothing_in_latent_space", True):
        raise ValueError("V2 requires target_smoothing_in_latent_space=true")
    if float(config.get("interval_epsilon", 1e-6)) <= 0:
        raise ValueError("interval_epsilon must be positive")

    hvac = config.get("hvac", {})
    if not 0 <= float(hvac.get("recovery_pressure_start", 0.35)) < float(hvac.get("recovery_pressure_full", 0.85)):
        raise ValueError("recovery_pressure_start must be below recovery_pressure_full")
    if not 0 <= float(hvac.get("recovery_full_hours", 2.0)) < float(hvac.get("recovery_start_hours", 6.0)):
        raise ValueError("recovery_full_hours must be below recovery_start_hours")
    if float(hvac.get("recovery_start_hours", 6.0)) > float(episode_horizon_hours):
        raise ValueError("recovery_start_hours exceeds episode horizon")

    ev = config.get("ev", {})
    for field in ("price_weight", "net_load_weight"):
        if float(ev.get(field, 0.0)) < 0:
            raise ValueError(f"{field} must be nonnegative")
    if float(ev.get("softmax_temperature", 1.0)) <= 0:
        raise ValueError("softmax_temperature must be positive")
    if bool(ev.get("economic_reference_enabled", True)) and forecast_horizon < ev_max_steps:
        raise ValueError("forecast horizon is shorter than EV availability window")

