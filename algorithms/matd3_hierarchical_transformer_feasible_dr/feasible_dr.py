from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


def latest_feasible_ev_floor(
    remaining_energy: torch.Tensor,
    hours_to_departure: torch.Tensor,
    *,
    max_charge_kw: float,
    efficiency: float,
    delta_t: float,
    upper: torch.Tensor,
    reliability_margin: float = 0.0,
) -> torch.Tensor:
    """Compute the minimum EV charging floor needed to preserve deadline feasibility."""
    steps = torch.clamp(torch.ceil(hours_to_departure / delta_t), min=0.0)
    future_steps = torch.clamp(steps - 1.0, min=0.0)
    future_energy = efficiency * max_charge_kw * future_steps * delta_t
    current_kw = torch.clamp(
        (remaining_energy - future_energy) / (efficiency * delta_t),
        min=0.0,
    )
    normalized = current_kw / max_charge_kw + reliability_margin
    return torch.minimum(torch.clamp(normalized, min=0.0), upper)


class FeasibleDRTransform(nn.Module):
    """Maps unconstrained latent DR preference [-1, 1] into feasible environment actions."""

    def __init__(
        self,
        dr_type: str,
        schema: Sequence[str],
        *,
        enabled: bool = True,
        interval_epsilon: float = 1e-6,
        delta_t: float = 1.0,
        episode_length: int = 24,
        max_charge_kw: float = 93.6,
        efficiency: float = 0.93,
        fleet_capacity_kwh: float = 780.0,
        catchup_floor_enabled: bool = True,
        reliability_margin: float = 0.0,
        recovery_anchor_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.dr_type = str(dr_type)
        self.schema = tuple(schema)
        self.enabled = bool(enabled)
        self.interval_epsilon = float(interval_epsilon)
        self.delta_t = float(delta_t)
        self.episode_length = int(episode_length)
        self.max_charge_kw = float(max_charge_kw)
        self.efficiency = float(efficiency)
        self.fleet_capacity_kwh = float(fleet_capacity_kwh)
        self.catchup_floor_enabled = bool(catchup_floor_enabled)
        self.reliability_margin = float(reliability_margin)
        self.recovery_anchor_enabled = bool(recovery_anchor_enabled)

        names = {name: idx for idx, name in enumerate(self.schema)}

        def idx_or_default(name: str) -> int:
            return names.get(name, -1)

        # Schema positions are Python metadata, not tensor state. Keeping them as
        # integers avoids moving slice boundaries onto CUDA with the module.
        self.hvac_min_idx = idx_or_default("hvac_feasible_min")
        self.hvac_max_idx = idx_or_default("hvac_feasible_max")
        self.ev_min_idx = idx_or_default("ev_feasible_min")
        self.ev_max_idx = idx_or_default("ev_feasible_max")
        self.ev_avail_idx = idx_or_default("ev_available")
        self.rem_energy_idx = idx_or_default("remaining_energy")
        self.hours_dep_idx = idx_or_default("hours_to_departure")
        self.deadline_pressure_idx = idx_or_default("deadline_pressure")
        self.recovery_pressure_idx = idx_or_default("thermal_recovery_pressure")

    @classmethod
    def for_test(
        cls,
        dr_type: str,
        schema: Sequence[str],
        *,
        enabled: bool = True,
        interval_epsilon: float = 1e-6,
        delta_t: float = 1.0,
        episode_length: int = 24,
        max_charge_kw: float = 100.0,
        efficiency: float = 0.95,
        fleet_capacity_kwh: float = 100.0,
        catchup_floor_enabled: bool = True,
        reliability_margin: float = 0.0,
    ) -> FeasibleDRTransform:
        return cls(
            dr_type,
            schema,
            enabled=enabled,
            interval_epsilon=interval_epsilon,
            delta_t=delta_t,
            episode_length=episode_length,
            max_charge_kw=max_charge_kw,
            efficiency=efficiency,
            fleet_capacity_kwh=fleet_capacity_kwh,
            catchup_floor_enabled=catchup_floor_enabled,
            reliability_margin=reliability_margin,
        )

    @classmethod
    def from_config(
        cls,
        dr_type: str,
        schema: Sequence[str],
        feasible_config: Mapping[str, Any],
        mg_config: Mapping[str, Any] | None = None,
        *,
        delta_t: float = 1.0,
        episode_length: int = 24,
    ) -> FeasibleDRTransform:
        enabled = bool(feasible_config.get("enabled", True))
        agents_cfg = feasible_config.get("agents", {})
        # If explicitly disabled for this microgrid type or ID
        if dr_type == "industrial_mill":
            enabled = enabled and bool(agents_cfg.get("mg1", False))
        elif dr_type == "commercial_hvac":
            enabled = enabled and bool(agents_cfg.get("mg2", True))
        elif dr_type == "ev_charging":
            enabled = enabled and bool(agents_cfg.get("mg3", True))

        interval_epsilon = float(feasible_config.get("interval_epsilon", 1e-6))
        ev_cfg = feasible_config.get("ev", {})
        hvac_cfg = feasible_config.get("hvac", {})
        if dr_type == "commercial_hvac":
            enabled = enabled and bool(hvac_cfg.get("interval_mapping_enabled", True))

        max_charge_kw = 93.6
        efficiency = 0.93
        fleet_capacity_kwh = 780.0
        if mg_config and "dr" in mg_config:
            dr_spec = mg_config["dr"]
            max_charge_kw = float(dr_spec.get("max_charge_kw", max_charge_kw))
            efficiency = float(dr_spec.get("eta_charge", efficiency))
            fleet_capacity_kwh = float(dr_spec.get("fleet_capacity_kwh", fleet_capacity_kwh))

        return cls(
            dr_type=dr_type,
            schema=schema,
            enabled=enabled,
            interval_epsilon=interval_epsilon,
            delta_t=delta_t,
            episode_length=episode_length,
            max_charge_kw=max_charge_kw,
            efficiency=efficiency,
            fleet_capacity_kwh=fleet_capacity_kwh,
            catchup_floor_enabled=bool(ev_cfg.get("catchup_floor_enabled", True)),
            reliability_margin=float(ev_cfg.get("reliability_margin", 0.0)),
            recovery_anchor_enabled=bool(hvac_cfg.get("recovery_anchor_enabled", False)),
        )

    def forward(self, observation: torch.Tensor, latent_dr: torch.Tensor) -> torch.Tensor:
        """Map latent DR preference [B, 1] into environment-compatible raw DR action [B, 1]."""
        if not self.enabled:
            return latent_dr.clamp(-1.0, 1.0)

        clamped_latent = latent_dr.clamp(-1.0, 1.0)
        preference = (clamped_latent + 1.0) * 0.5

        if self.dr_type == "commercial_hvac":
            if self.hvac_min_idx < 0 or self.hvac_max_idx < 0:
                return clamped_latent
            lower = observation[:, self.hvac_min_idx : self.hvac_min_idx + 1]
            upper = observation[:, self.hvac_max_idx : self.hvac_max_idx + 1]
            width = torch.clamp(upper - lower, min=0.0)
            is_degenerate = width <= self.interval_epsilon
            mapped = lower + preference * width
            return torch.where(is_degenerate, lower + 0.0 * latent_dr, mapped)

        if self.dr_type == "ev_charging":
            if self.ev_min_idx < 0 or self.ev_max_idx < 0:
                return clamped_latent
            lower = observation[:, self.ev_min_idx : self.ev_min_idx + 1]
            upper = observation[:, self.ev_max_idx : self.ev_max_idx + 1]

            if self.ev_avail_idx >= 0:
                available = observation[:, self.ev_avail_idx : self.ev_avail_idx + 1]
                upper = torch.where(available < 0.5, torch.zeros_like(upper), upper)
                lower = torch.where(available < 0.5, torch.zeros_like(lower), lower)

            if self.catchup_floor_enabled and self.rem_energy_idx >= 0 and self.hours_dep_idx >= 0:
                rem_norm = observation[:, self.rem_energy_idx : self.rem_energy_idx + 1]
                hours_norm = observation[:, self.hours_dep_idx : self.hours_dep_idx + 1]
                remaining_kwh = rem_norm * self.fleet_capacity_kwh
                hours_to_dep = hours_norm * (self.episode_length * self.delta_t)
                floor = latest_feasible_ev_floor(
                    remaining_kwh,
                    hours_to_dep,
                    max_charge_kw=self.max_charge_kw,
                    efficiency=self.efficiency,
                    delta_t=self.delta_t,
                    upper=upper,
                    reliability_margin=self.reliability_margin,
                )
            else:
                floor = torch.zeros_like(lower)

            effective_lower = torch.clamp(torch.maximum(lower, floor), max=upper)
            width = torch.clamp(upper - effective_lower, min=0.0)
            is_degenerate = width <= self.interval_epsilon
            mapped_unit = effective_lower + preference * width
            effective_unit = torch.where(is_degenerate, effective_lower + 0.0 * latent_dr, mapped_unit)
            return effective_unit * 2.0 - 1.0

        # Unconfigured / MG1 default:
        return clamped_latent

    def scalar_diagnostics(
        self, observation: torch.Tensor, latent_dr: torch.Tensor
    ) -> dict[str, float]:
        """Compute scalar diagnostics over an observation batch."""
        with torch.no_grad():
            clamped = latent_dr.clamp(-1.0, 1.0)
            pref = (clamped + 1.0) * 0.5
            mapped = self.forward(observation, latent_dr)
            diagnostics: dict[str, float] = {
                "dr_latent_mean": float(latent_dr.mean().cpu()),
                "dr_latent_std": float(latent_dr.std().cpu()) if latent_dr.numel() > 1 else 0.0,
                "dr_requested_mean": float(mapped.mean().cpu()),
                "dr_lower_boundary_rate": float((pref <= 0.05).float().mean().cpu()),
                "dr_upper_boundary_rate": float((pref >= 0.95).float().mean().cpu()),
            }

            if self.dr_type == "commercial_hvac" and self.hvac_min_idx >= 0 and self.hvac_max_idx >= 0:
                low = observation[:, self.hvac_min_idx : self.hvac_min_idx + 1]
                high = observation[:, self.hvac_max_idx : self.hvac_max_idx + 1]
                diagnostics.update({
                    "dr_feasible_min_mean": float(low.mean().cpu()),
                    "dr_feasible_max_mean": float(high.mean().cpu()),
                    "dr_interval_width_mean": float((high - low).mean().cpu()),
                })
                if self.recovery_pressure_idx >= 0:
                    diagnostics["hvac_terminal_recovery_pressure"] = float(
                        observation[:, self.recovery_pressure_idx].mean().cpu()
                    )

            elif self.dr_type == "ev_charging" and self.ev_min_idx >= 0 and self.ev_max_idx >= 0:
                low = observation[:, self.ev_min_idx : self.ev_min_idx + 1]
                high = observation[:, self.ev_max_idx : self.ev_max_idx + 1]
                diagnostics.update({
                    "dr_feasible_min_mean": float(low.mean().cpu()),
                    "dr_feasible_max_mean": float(high.mean().cpu()),
                    "dr_interval_width_mean": float((high - low).mean().cpu()),
                })
                if self.catchup_floor_enabled and self.rem_energy_idx >= 0 and self.hours_dep_idx >= 0:
                    rem_norm = observation[:, self.rem_energy_idx : self.rem_energy_idx + 1]
                    hours_norm = observation[:, self.hours_dep_idx : self.hours_dep_idx + 1]
                    floor = latest_feasible_ev_floor(
                        rem_norm * self.fleet_capacity_kwh,
                        hours_norm * (self.episode_length * self.delta_t),
                        max_charge_kw=self.max_charge_kw,
                        efficiency=self.efficiency,
                        delta_t=self.delta_t,
                        upper=high,
                        reliability_margin=self.reliability_margin,
                    )
                    diagnostics.update({
                        "ev_mandatory_floor_mean": float(floor.mean().cpu()),
                        "ev_mandatory_floor_active_rate": float((floor > low + 1e-4).float().mean().cpu()),
                    })
                if self.deadline_pressure_idx >= 0:
                    diagnostics["ev_deadline_pressure_mean"] = float(
                        observation[:, self.deadline_pressure_idx].mean().cpu()
                    )

            return diagnostics
