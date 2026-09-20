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


def economic_slot_weights(
    *,
    prices: torch.Tensor,
    system_net_load_kw: torch.Tensor,
    valid: torch.Tensor,
    price_weight: float,
    net_load_weight: float,
    temperature: float,
) -> torch.Tensor:
    mask = valid.bool()
    large = torch.finfo(prices.dtype).max
    small = torch.finfo(prices.dtype).min

    p_min = torch.where(mask, prices, large).min(dim=-1, keepdim=True).values
    p_max = torch.where(mask, prices, small).max(dim=-1, keepdim=True).values
    n_min = torch.where(mask, system_net_load_kw, large).min(dim=-1, keepdim=True).values
    n_max = torch.where(mask, system_net_load_kw, small).max(dim=-1, keepdim=True).values

    p_score = (prices - p_min) / (p_max - p_min).clamp_min(1e-6)
    n_score = (system_net_load_kw - n_min) / (n_max - n_min).clamp_min(1e-6)
    logits = (-float(price_weight) * p_score - float(net_load_weight) * n_score) / float(temperature)
    logits = logits.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    return torch.where(mask, weights, torch.zeros_like(weights))


def ev_economic_reference(
    *,
    remaining_battery_kwh: torch.Tensor,
    efficiency: float,
    delta_t: float,
    max_charge_kw: float,
    current_weight: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    remaining_grid_kwh = remaining_battery_kwh / float(efficiency)
    reference_kw = remaining_grid_kwh * current_weight / float(delta_t)
    reference_unit = reference_kw / float(max_charge_kw)
    return torch.maximum(lower, torch.minimum(reference_unit, upper))


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
        normalizers_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
        active_agent: str = "mg3",
        forecast_horizon: int = 8,
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
        self.normalizers_by_microgrid = normalizers_by_microgrid or {}
        self.active_agent = str(active_agent)
        self.forecast_horizon = int(forecast_horizon)

        names = {name: idx for idx, name in enumerate(self.schema)}
        self.names = names
        self.air_gap_idx = names.get("terminal_air_gap", -1)
        self.mass_gap_idx = names.get("terminal_mass_gap", -1)
        self.time_remaining_idx = names.get("time_remaining", -1)

        self.load_idx = names.get("load", -1)
        self.pv_idx = names.get("pv", -1)
        self.wind_idx = names.get("wind", -1)
        self.price_idx = names.get("grid_buy_price", names.get("price", -1))

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

    @classmethod
    def from_config(
        cls,
        dr_type: str,
        schema: Sequence[str],
        adaptive_config: Mapping[str, Any],
        mg_config: Mapping[str, Any] | None = None,
        *,
        normalizers_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
        active_agent: str = "mg3",
        forecast_horizon: int = 8,
        delta_t: float = 1.0,
        episode_length: int = 24,
    ) -> AdaptiveDRTransform:
        enabled = bool(adaptive_config.get("enabled", True))
        agents_cfg = adaptive_config.get("agents", {})
        if dr_type == "industrial_mill":
            enabled = enabled and bool(agents_cfg.get("mg1", False))
        elif dr_type == "commercial_hvac":
            enabled = enabled and bool(agents_cfg.get("mg2", True))
        elif dr_type == "ev_charging":
            enabled = enabled and bool(agents_cfg.get("mg3", True))

        interval_epsilon = float(adaptive_config.get("interval_epsilon", 1e-6))
        ev_cfg = adaptive_config.get("ev", {})
        hvac_cfg = adaptive_config.get("hvac", {})

        if dr_type == "commercial_hvac":
            enabled = enabled and bool(hvac_cfg.get("interval_mapping_enabled", True))

        max_charge_kw = 93.6
        efficiency = 0.93
        fleet_capacity_kwh = 780.0
        c_air = 15.0
        c_mass = 250.0
        cop_h = 3.3
        cop_c = 3.2
        heat_max = 35.0
        cool_max = 35.0

        if mg_config and "dr" in mg_config:
            dr_spec = mg_config["dr"]
            max_charge_kw = float(dr_spec.get("max_charge_kw", max_charge_kw))
            efficiency = float(dr_spec.get("eta_charge", efficiency))
            fleet_capacity_kwh = float(dr_spec.get("fleet_capacity_kwh", fleet_capacity_kwh))
            c_air = float(dr_spec.get("c_air_kwh_per_c", c_air))
            c_mass = float(dr_spec.get("c_mass_kwh_per_c", c_mass))
            cop_h = float(dr_spec.get("cop_heat", cop_h))
            cop_c = float(dr_spec.get("cop_cool", cop_c))
            heat_max = float(dr_spec.get("heating_max_kw", heat_max))
            cool_max = float(dr_spec.get("cooling_max_kw", cool_max))

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
            c_air_kwh_per_c=c_air,
            c_mass_kwh_per_c=c_mass,
            cop_heat=cop_h,
            cop_cool=cop_c,
            heating_max_kw=heat_max,
            cooling_max_kw=cool_max,
            recovery_pressure_start=float(hvac_cfg.get("recovery_pressure_start", 0.35)),
            recovery_pressure_full=float(hvac_cfg.get("recovery_pressure_full", 0.85)),
            recovery_start_hours=float(hvac_cfg.get("recovery_start_hours", 6.0)),
            recovery_full_hours=float(hvac_cfg.get("recovery_full_hours", 2.0)),
            recovery_anchor_enabled=bool(hvac_cfg.get("recovery_anchor_enabled", True)),
            economic_reference_enabled=bool(ev_cfg.get("economic_reference_enabled", True)),
            price_weight=float(ev_cfg.get("price_weight", 2.0)),
            net_load_weight=float(ev_cfg.get("net_load_weight", 1.0)),
            softmax_temperature=float(ev_cfg.get("softmax_temperature", 1.0)),
            normalizers_by_microgrid=normalizers_by_microgrid,
            active_agent=active_agent,
            forecast_horizon=forecast_horizon,
        )

    def reconstruct_economic_signals(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_norm = self.normalizers_by_microgrid.get(self.active_agent, {})
        load_scale = float(local_norm.get("load", 1.0))
        pv_scale = float(local_norm.get("pv", 1.0))
        wind_scale = float(local_norm.get("wind", 1.0))
        price_scale = float(local_norm.get("price", 1.0))

        if self.normalizers_by_microgrid:
            peers = [p for p in self.normalizers_by_microgrid if p != self.active_agent]
        else:
            peers = [
                name[:-9]
                for name in self.names
                if name.endswith("_net_load") and not name.startswith("load")
            ]

        available_slots = [0]
        for k in range(1, self.forecast_horizon + 1):
            if f"price_t+{k}" in self.names or f"load_t+{k}" in self.names:
                available_slots.append(k)

        price_list = []
        net_load_list = []
        valid_list = []

        avail = (
            observation[:, self.ev_avail_idx : self.ev_avail_idx + 1] > 0.5
            if self.ev_avail_idx >= 0
            else torch.ones(observation.shape[0], 1, dtype=torch.bool, device=observation.device)
        )
        hours_dep = (
            observation[:, self.hours_dep_idx : self.hours_dep_idx + 1]
            * (self.episode_length * self.delta_t)
            if self.hours_dep_idx >= 0
            else torch.full((observation.shape[0], 1), self.episode_length * self.delta_t, device=observation.device)
        )
        rem_steps = torch.clamp(torch.ceil(hours_dep / self.delta_t), min=0.0)

        zero_t = torch.zeros(observation.shape[0], 1, device=observation.device)

        for slot in available_slots:
            if slot == 0:
                p = (
                    observation[:, self.price_idx : self.price_idx + 1] * price_scale
                    if self.price_idx >= 0
                    else zero_t
                )
                l_kw = observation[:, self.load_idx : self.load_idx + 1] * load_scale if self.load_idx >= 0 else zero_t
                pv_kw = observation[:, self.pv_idx : self.pv_idx + 1] * pv_scale if self.pv_idx >= 0 else zero_t
                w_kw = observation[:, self.wind_idx : self.wind_idx + 1] * wind_scale if self.wind_idx >= 0 else zero_t
                net_kw = l_kw - pv_kw - w_kw

                for peer in peers:
                    p_scale = float(self.normalizers_by_microgrid.get(peer, {}).get("power", 1.0))
                    p_idx = self.names.get(f"{peer}_net_load", -1)
                    if p_idx >= 0:
                        net_kw = net_kw + observation[:, p_idx : p_idx + 1] * p_scale

                v = avail & (rem_steps > 0)
            else:
                p_idx = self.names.get(f"price_t+{slot}", -1)
                p = (
                    observation[:, p_idx : p_idx + 1] * price_scale
                    if p_idx >= 0
                    else zero_t
                )
                l_idx = self.names.get(f"load_t+{slot}", -1)
                pv_idx = self.names.get(f"pv_t+{slot}", -1)
                w_idx = self.names.get(f"wind_t+{slot}", -1)

                l_kw = observation[:, l_idx : l_idx + 1] * load_scale if l_idx >= 0 else zero_t
                pv_kw = observation[:, pv_idx : pv_idx + 1] * pv_scale if pv_idx >= 0 else zero_t
                w_kw = observation[:, w_idx : w_idx + 1] * wind_scale if w_idx >= 0 else zero_t
                net_kw = l_kw - pv_kw - w_kw

                for peer in peers:
                    p_scale = float(self.normalizers_by_microgrid.get(peer, {}).get("power", 1.0))
                    p_idx = self.names.get(f"{peer}_net_load_t+{slot}", -1)
                    if p_idx >= 0:
                        net_kw = net_kw + observation[:, p_idx : p_idx + 1] * p_scale

                v = avail & (rem_steps > slot)
                fc_v_idx = self.names.get(f"forecast_valid_t+{slot}", -1)
                if fc_v_idx >= 0:
                    v = v & (observation[:, fc_v_idx : fc_v_idx + 1] > 0.5)

            price_list.append(p)
            net_load_list.append(net_kw)
            valid_list.append(v)

        return (
            torch.cat(price_list, dim=-1),
            torch.cat(net_load_list, dim=-1),
            torch.cat(valid_list, dim=-1),
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

        if self.dr_type == "ev_charging":
            if self.ev_min_idx < 0 or self.ev_max_idx < 0:
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

            avail = (
                observation[:, self.ev_avail_idx : self.ev_avail_idx + 1] > 0.5
                if self.ev_avail_idx >= 0
                else torch.ones(observation.shape[0], 1, dtype=torch.bool, device=observation.device)
            )

            lower = observation[:, self.ev_min_idx : self.ev_min_idx + 1]
            upper = observation[:, self.ev_max_idx : self.ev_max_idx + 1]

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
                remaining_kwh = torch.zeros_like(lower)
                floor = torch.zeros_like(lower)

            effective_lower = torch.clamp(torch.maximum(lower, floor), max=upper)

            if self.economic_reference_enabled:
                prices, sys_net, valid = self.reconstruct_economic_signals(observation)
                has_valid = valid.any(dim=-1, keepdim=True)
                safe_valid = torch.where(has_valid, valid, torch.ones_like(valid))
                weights = economic_slot_weights(
                    prices=prices,
                    system_net_load_kw=sys_net,
                    valid=safe_valid,
                    price_weight=self.price_weight,
                    net_load_weight=self.net_load_weight,
                    temperature=self.softmax_temperature,
                )
                current_weight = weights[:, 0:1]
                reference = ev_economic_reference(
                    remaining_battery_kwh=remaining_kwh,
                    efficiency=self.efficiency,
                    delta_t=self.delta_t,
                    max_charge_kw=self.max_charge_kw,
                    current_weight=current_weight,
                    lower=effective_lower,
                    upper=upper,
                )
            else:
                reference = (effective_lower + upper) * 0.5

            unit_action = residual_map(
                clamped_latent,
                effective_lower,
                reference,
                upper,
                interval_epsilon=self.interval_epsilon,
            )
            raw_action = unit_action * 2.0 - 1.0

            action = torch.where(avail, raw_action, -torch.ones_like(raw_action))
            final_lower = torch.where(avail, effective_lower, torch.zeros_like(effective_lower))
            final_upper = torch.where(avail, upper, torch.zeros_like(upper))
            final_ref = torch.where(avail, reference, torch.zeros_like(reference))
            final_floor = torch.where(avail, floor, torch.zeros_like(floor))

            return AdaptiveDRMap(
                action=action,
                lower=final_lower,
                upper=final_upper,
                reference=final_ref,
                available=avail.float(),
                mandatory_floor=final_floor,
                recovery_anchor=torch.zeros_like(action),
                recovery_gate=torch.zeros_like(action),
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




