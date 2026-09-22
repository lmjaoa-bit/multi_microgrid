"""Peak-risk configuration and observation-only physical features.

The peak-risk token never sees realized future loads or the centralized state.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

SLUG = "matd3_hierarchical_transformer_peak_risk"
PEAK_RISK_CONTRACT_VERSION = "1.0"
_DEFAULTS: dict[str, Any] = {
    "enabled": True, "token_enabled": True, "critic_enabled": True,
    "contract_version": PEAK_RISK_CONTRACT_VERSION,
    "critic_hidden_dims": [64, 64], "critic_lr": 0.0003,
    "critic_update_interval": 1, "critic_loss": "smooth_l1",
    "critic_max_grad_norm": 10.0, "conservative_twin_mode": "max",
    "use_discount": False, "critic_warmup_updates": 1000,
    "actor_start_update": 5000, "actor_weight_initial": 0.0,
    "actor_weight_final": 0.02, "actor_weight_ramp_updates": 15000,
    "reuse_parent_projection": True, "diagnostics_only_on_schedule": True,
}

def validate_peak_risk_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete, versioned peak-risk contract (no unknown keys)."""
    unknown = set(config) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown peak_risk fields: {sorted(unknown)}")
    result = dict(_DEFAULTS)
    result.update(config)
    for key in ("enabled", "token_enabled", "critic_enabled", "use_discount",
                "reuse_parent_projection", "diagnostics_only_on_schedule"):
        if type(result[key]) is not bool:
            raise ValueError(f"peak_risk.{key} must be a boolean")
    if result["contract_version"] != PEAK_RISK_CONTRACT_VERSION:
        raise ValueError("Unsupported peak_risk.contract_version")
    if result["use_discount"] or result["conservative_twin_mode"] != "max":
        raise ValueError("peak_risk max-return contract requires use_discount=false and conservative_twin_mode=max")
    if result["critic_loss"] != "smooth_l1" or not result["reuse_parent_projection"]:
        raise ValueError("peak_risk requires smooth_l1 and reuse_parent_projection=true")
    if not result["diagnostics_only_on_schedule"]:
        raise ValueError("peak_risk diagnostics_only_on_schedule must be true")
    if not isinstance(result["critic_hidden_dims"], (list, tuple)) or not result["critic_hidden_dims"]:
        raise ValueError("peak_risk.critic_hidden_dims must be a non-empty list")
    for width in result["critic_hidden_dims"]:
        if type(width) is not int or width <= 0:
            raise ValueError("peak_risk.critic_hidden_dims values must be positive integers")
    for key in ("critic_update_interval", "critic_warmup_updates", "actor_start_update",
                "actor_weight_ramp_updates"):
        val = result[key]
        if type(val) is not int or val < 0 or key in ("critic_update_interval", "actor_weight_ramp_updates") and val == 0:
            raise ValueError(f"peak_risk.{key} must be a valid nonnegative integer")
    for key in ("critic_lr", "critic_max_grad_norm", "actor_weight_initial", "actor_weight_final"):
        val = result[key]
        if type(val) not in (int, float) or not math.isfinite(val):
            raise ValueError(f"peak_risk.{key} must be finite and numeric")
        if val < 0 or key in ("critic_lr", "critic_max_grad_norm") and val == 0:
            raise ValueError(f"peak_risk.{key} must be positive where required")
    if result["actor_weight_initial"] != 0.0:
        raise ValueError("peak_risk.actor_weight_initial must be zero in contract 1.0")
    if result["enabled"] and result["actor_weight_final"] > 0 and not result["critic_enabled"]:
        raise ValueError("peak_risk.critic_enabled is required for actor regularization")
    if not result["enabled"] and (result["token_enabled"] or result["critic_enabled"] or result["actor_weight_final"] > 0):
        raise ValueError("peak_risk.enabled=false requires all peak features disabled")
    return result


def peak_actor_weight(update_step: int, start: int, ramp: int, final: float) -> float:
    if ramp <= 0:
        raise ValueError("peak_risk.actor_weight_ramp_updates must be positive")
    return float(final) * min(1.0, max(0.0, (int(update_step) - int(start)) / int(ramp)))


def max_return_target(current: torch.Tensor, next1: torch.Tensor,
                      next2: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    for name, tensor in (("current", current), ("next1", next1), ("next2", next2), ("done", done)):
        if tensor.ndim != 2 or tensor.shape[1] != 1 or tensor.shape != current.shape:
            raise ValueError(f"peak target {name} shape mismatch")
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"peak target {name} is nonfinite")
    return torch.where(done >= 0.5, current, torch.maximum(current, torch.maximum(next1, next2)))


def system_actual_load_kw(plans: Sequence[Mapping[str, torch.Tensor]]) -> torch.Tensor:
    if len(plans) != 3:
        raise ValueError("Expected projected plans from all three microgrids")
    loads = [plan["actual_load_kw"].reshape(-1, 1) for plan in plans]
    if not all(torch.isfinite(load).all() for load in loads):
        raise FloatingPointError("Projected actual_load_kw is nonfinite")
    if any(load.shape != loads[0].shape for load in loads):
        raise ValueError("Projected actual_load_kw has inconsistent batch sizes")
    return torch.stack(loads, dim=0).sum(dim=0)


class PeakRiskFeatureExtractor(nn.Module):
    """Fourteen normalized features from the active agent's current observation."""

    def __init__(self, schema: Sequence[str], agent_ids: Sequence[str],
                 active_agent: str, *, forecast_horizon: int = 8) -> None:
        super().__init__()
        self.schema = tuple(schema)
        self.active_agent = str(active_agent)
        self.forecast_horizon = int(forecast_horizon)
        names = {name: index for index, name in enumerate(schema)}
        self.peers = tuple(mg for mg in agent_ids if mg != active_agent)
        kind = ("industrial_mill" if "silo_target_gap" in names else
                "commercial_hvac" if "terminal_air_gap" in names else
                "ev_charging" if "hours_to_departure" in names else None)
        if kind is None:
            raise ValueError(f"{active_agent}: missing categorical device observation")
        self.kind = kind
        need = ["load", "pv", "wind", "previous_system_buy_demand",
                "available_discharge_ratio", "time_remaining"]
        need += [f"{mg}_deficit" for mg in self.peers]
        need += [f"{mg}_surplus" for mg in self.peers]
        for k in range(1, self.forecast_horizon + 1):
            need += [f"load_t+{k}", f"pv_t+{k}", f"wind_t+{k}"]
        bounds = {"industrial_mill": ("mill_feasible_min", "mill_feasible_max", "production_pressure"),
                  "commercial_hvac": ("hvac_feasible_min", "hvac_feasible_max", "thermal_recovery_pressure"),
                  "ev_charging": ("ev_feasible_min", "ev_feasible_max", "deadline_pressure")}
        need += list(bounds[kind])
        missing = sorted(set(need) - set(names))
        if missing:
            raise ValueError(f"{active_agent}: missing peak-risk observation fields: {missing}")
        self.indices = names
        self.bounds = bounds[kind]
        self.register_buffer("forecast_indices", torch.tensor(
            [[names[f"{component}_t+{k}"] for component in ("load", "pv", "wind")]
             for k in range(1, self.forecast_horizon + 1)], dtype=torch.long), persistent=False)
        valid_names = [f"forecast_valid_t+{k}" for k in range(1, self.forecast_horizon + 1)]
        self.register_buffer("valid_indices", torch.tensor(
            [names.get(name, -1) for name in valid_names], dtype=torch.long), persistent=False)
        self.register_buffer("step_indices", torch.arange(
            1, self.forecast_horizon + 1, dtype=torch.float32), persistent=False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != len(self.schema):
            raise ValueError("peak-risk observation shape mismatch")
        x = observation
        ix = self.indices
        local = x[:, ix["load"]] - x[:, ix["pv"]] - x[:, ix["wind"]]
        peer_deficit = sum((x[:, ix[f"{mg}_deficit"]] for mg in self.peers), torch.zeros_like(local))
        peer_surplus = sum((x[:, ix[f"{mg}_surplus"]] for mg in self.peers), torch.zeros_like(local))
        forecasts = x[:, self.forecast_indices.flatten()].reshape(x.shape[0], self.forecast_horizon, 3)
        forecast_net = forecasts[:, :, 0] - forecasts[:, :, 1] - forecasts[:, :, 2]
        if (self.valid_indices >= 0).any():
            presence = self.valid_indices >= 0
            safe_indices = self.valid_indices.clamp_min(0)
            flags = x[:, safe_indices] >= 0.5
            valid = torch.where(presence.unsqueeze(0), flags, torch.ones_like(flags))
        else:
            valid = torch.ones_like(forecast_net, dtype=torch.bool)
        valid = valid & torch.isfinite(forecast_net)
        valid_count = valid.sum(dim=1)
        fallback = local.unsqueeze(1).expand_as(forecast_net)
        finite = torch.where(valid, forecast_net, fallback)
        mean = finite.sum(dim=1) / float(self.forecast_horizon)
        # Invalid forecast positions do not alter the valid-forecast mean/max.
        mean = torch.where(valid_count > 0, torch.where(valid, forecast_net, torch.zeros_like(forecast_net)).sum(dim=1)
                           / valid_count.clamp_min(1), local)
        max_values = torch.where(valid, forecast_net, torch.full_like(forecast_net, -torch.inf))
        maximum, max_pos = max_values.max(dim=1)
        maximum = torch.where(valid_count > 0, maximum, local)
        max_pos = torch.where(valid_count > 0, max_pos, torch.zeros_like(max_pos))
        forecast_fraction = valid.float().mean(dim=1)
        lower, upper, pressure = (x[:, ix[name]] for name in self.bounds)
        urgency = {"industrial_mill": 0.0, "commercial_hvac": 0.5, "ev_charging": 1.0}[self.kind]
        features = torch.stack([
            local,
            local + peer_deficit - peer_surplus,
            x[:, ix["previous_system_buy_demand"]],
            mean,
            maximum,
            maximum - local,
            self.step_indices[max_pos] / float(self.forecast_horizon),
            lower, upper, pressure,
            x[:, ix["available_discharge_ratio"]],
            peer_deficit - peer_surplus,
            forecast_fraction,
            torch.full_like(local, urgency),
        ], dim=1)
        if not torch.isfinite(features).all():
            raise FloatingPointError("peak-risk token features are nonfinite")
        return features
