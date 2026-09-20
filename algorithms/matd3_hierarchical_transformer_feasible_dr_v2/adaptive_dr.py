from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Mapping

import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer_feasible_dr.feasible_dr import (
    FeasibleDRTransform,
    latest_feasible_ev_floor,
)


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


def hvac_recovery_anchor(
    *,
    air_gap_c: torch.Tensor,
    mass_gap_c: torch.Tensor,
    remaining_hours: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    c_air_kwh_per_c: float,
    c_mass_kwh_per_c: float,
    cop_heat: float,
    cop_cool: float,
    heating_max_kw: float,
    cooling_max_kw: float,
    delta_t: float,
) -> torch.Tensor:
    hours = remaining_hours.clamp_min(float(delta_t))
    thermal_kw = (
        air_gap_c * float(c_air_kwh_per_c)
        + mass_gap_c * float(c_mass_kwh_per_c)
    ) / hours
    electric_kw = torch.where(
        thermal_kw >= 0,
        thermal_kw / float(cop_heat),
        thermal_kw / float(cop_cool),
    )
    normalized = torch.where(
        electric_kw >= 0,
        electric_kw / float(heating_max_kw),
        electric_kw / float(cooling_max_kw),
    )
    return torch.maximum(lower, torch.minimum(normalized, upper))


def hvac_recovery_gate(
    pressure: torch.Tensor,
    remaining_hours: torch.Tensor,
    *,
    pressure_start: float,
    pressure_full: float,
    start_hours: float,
    full_hours: float,
) -> torch.Tensor:
    pressure_gate = smoothstep(pressure, pressure_start, pressure_full)
    time_gate = 1.0 - smoothstep(remaining_hours, full_hours, start_hours)
    return pressure_gate * time_gate


class AdaptiveDRTransform(FeasibleDRTransform):
    """Maps latent DR actions into feasible actions with adaptive references."""

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
        c_air_kwh_per_c: float = 15.0,
        c_mass_kwh_per_c: float = 250.0,
        cop_heat: float = 3.3,
        cop_cool: float = 3.2,
        heating_max_kw: float = 35.0,
        cooling_max_kw: float = 35.0,
        recovery_pressure_start: float = 0.35,
        recovery_pressure_full: float = 0.85,
        recovery_start_hours: float = 6.0,
        recovery_full_hours: float = 2.0,
        recovery_anchor_enabled: bool = True,
        economic_reference_enabled: bool = True,
        price_weight: float = 2.0,
        net_load_weight: float = 1.0,
        softmax_temperature: float = 1.0,
    ) -> None:
        super().__init__(
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
            recovery_anchor_enabled=recovery_anchor_enabled,
        )
        self.c_air_kwh_per_c = float(c_air_kwh_per_c)
        self.c_mass_kwh_per_c = float(c_mass_kwh_per_c)
        self.cop_heat = float(cop_heat)
        self.cop_cool = float(cop_cool)
        self.heating_max_kw = float(heating_max_kw)
        self.cooling_max_kw = float(cooling_max_kw)
        self.recovery_pressure_start = float(recovery_pressure_start)
        self.recovery_pressure_full = float(recovery_pressure_full)
        self.recovery_start_hours = float(recovery_start_hours)
        self.recovery_full_hours = float(recovery_full_hours)
        self.recovery_anchor_enabled = bool(recovery_anchor_enabled)
        self.economic_reference_enabled = bool(economic_reference_enabled)
        self.price_weight = float(price_weight)
        self.net_load_weight = float(net_load_weight)
        self.softmax_temperature = float(softmax_temperature)

        names = {name: idx for idx, name in enumerate(self.schema)}
        self.air_gap_idx = names.get("terminal_air_gap", -1)
        self.mass_gap_idx = names.get("terminal_mass_gap", -1)
        self.time_remaining_idx = names.get("time_remaining", -1)

    @classmethod
    def for_hvac_test(
        cls,
        schema: Sequence[str],
        *,
        c_air_kwh_per_c: float = 15.0,
        c_mass_kwh_per_c: float = 250.0,
        cop_heat: float = 3.3,
        cop_cool: float = 3.2,
        heating_max_kw: float = 35.0,
        cooling_max_kw: float = 35.0,
        delta_t: float = 1.0,
        episode_length: int = 24,
        recovery_pressure_start: float = 0.35,
        recovery_pressure_full: float = 0.85,
        recovery_start_hours: float = 6.0,
        recovery_full_hours: float = 2.0,
        recovery_anchor_enabled: bool = True,
        interval_epsilon: float = 1e-6,
    ) -> AdaptiveDRTransform:
        return cls(
            dr_type="commercial_hvac",
            schema=schema,
            enabled=True,
            interval_epsilon=interval_epsilon,
            delta_t=delta_t,
            episode_length=episode_length,
            c_air_kwh_per_c=c_air_kwh_per_c,
            c_mass_kwh_per_c=c_mass_kwh_per_c,
            cop_heat=cop_heat,
            cop_cool=cop_cool,
            heating_max_kw=heating_max_kw,
            cooling_max_kw=cooling_max_kw,
            recovery_pressure_start=recovery_pressure_start,
            recovery_pressure_full=recovery_pressure_full,
            recovery_start_hours=recovery_start_hours,
            recovery_full_hours=recovery_full_hours,
            recovery_anchor_enabled=recovery_anchor_enabled,
        )

    def map_with_details(
        self, observation: torch.Tensor, latent_dr: torch.Tensor
    ) -> AdaptiveDRMap:
        clamped_latent = latent_dr.clamp(-1.0, 1.0)
        zeros = torch.zeros_like(clamped_latent)
        ones = torch.ones_like(clamped_latent)

        if not self.enabled:
            return AdaptiveDRMap(
                action=clamped_latent,
                lower=-ones,
                upper=ones,
                reference=zeros,
                available=ones,
                mandatory_floor=zeros,
                recovery_anchor=zeros,
                recovery_gate=zeros,
            )

        preference = (clamped_latent + 1.0) * 0.5

        if self.dr_type == "commercial_hvac":
            if self.hvac_min_idx < 0 or self.hvac_max_idx < 0:
                return AdaptiveDRMap(
                    action=clamped_latent,
                    lower=-ones,
                    upper=ones,
                    reference=zeros,
                    available=ones,
                    mandatory_floor=zeros,
                    recovery_anchor=zeros,
                    recovery_gate=zeros,
                )
            lower = observation[:, self.hvac_min_idx : self.hvac_min_idx + 1]
            upper = observation[:, self.hvac_max_idx : self.hvac_max_idx + 1]
            width = torch.clamp(upper - lower, min=0.0)
            is_degenerate = width <= self.interval_epsilon
            mapped = lower + preference * width
            free_action = torch.where(is_degenerate, lower + 0.0 * latent_dr, mapped)

            zero_t = torch.zeros_like(lower)
            if self.recovery_anchor_enabled:
                air_gap = (
                    observation[:, self.air_gap_idx : self.air_gap_idx + 1] * 40.0
                    if self.air_gap_idx >= 0
                    else zero_t
                )
                mass_gap = (
                    observation[:, self.mass_gap_idx : self.mass_gap_idx + 1] * 40.0
                    if self.mass_gap_idx >= 0
                    else zero_t
                )
                rem_hours = (
                    observation[:, self.time_remaining_idx : self.time_remaining_idx + 1]
                    * (self.episode_length * self.delta_t)
                    if self.time_remaining_idx >= 0
                    else torch.full_like(lower, self.episode_length * self.delta_t)
                )
                pressure = (
                    observation[:, self.recovery_pressure_idx : self.recovery_pressure_idx + 1]
                    if self.recovery_pressure_idx >= 0
                    else zero_t
                )
                anchor = hvac_recovery_anchor(
                    air_gap_c=air_gap,
                    mass_gap_c=mass_gap,
                    remaining_hours=rem_hours,
                    lower=lower,
                    upper=upper,
                    c_air_kwh_per_c=self.c_air_kwh_per_c,
                    c_mass_kwh_per_c=self.c_mass_kwh_per_c,
                    cop_heat=self.cop_heat,
                    cop_cool=self.cop_cool,
                    heating_max_kw=self.heating_max_kw,
                    cooling_max_kw=self.cooling_max_kw,
                    delta_t=self.delta_t,
                )
                gate = hvac_recovery_gate(
                    pressure,
                    rem_hours,
                    pressure_start=self.recovery_pressure_start,
                    pressure_full=self.recovery_pressure_full,
                    start_hours=self.recovery_start_hours,
                    full_hours=self.recovery_full_hours,
                )
                action = (1.0 - gate) * free_action + gate * anchor
                return AdaptiveDRMap(
                    action=action,
                    lower=lower,
                    upper=upper,
                    reference=anchor,
                    available=torch.ones_like(lower),
                    mandatory_floor=zero_t,
                    recovery_anchor=anchor,
                    recovery_gate=gate,
                )
            else:
                return AdaptiveDRMap(
                    action=free_action,
                    lower=lower,
                    upper=upper,
                    reference=zero_t,
                    available=torch.ones_like(lower),
                    mandatory_floor=zero_t,
                    recovery_anchor=zero_t,
                    recovery_gate=zero_t,
                )

        # Fallback / other DR types
        v1_action = super().forward(observation, latent_dr)
        return AdaptiveDRMap(
            action=v1_action,
            lower=-ones,
            upper=ones,
            reference=zeros,
            available=ones,
            mandatory_floor=zeros,
            recovery_anchor=zeros,
            recovery_gate=zeros,
        )

    def forward(self, observation: torch.Tensor, latent_dr: torch.Tensor) -> torch.Tensor:
        return self.map_with_details(observation, latent_dr).action



