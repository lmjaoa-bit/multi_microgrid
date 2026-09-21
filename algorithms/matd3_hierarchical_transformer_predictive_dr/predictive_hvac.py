from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


def smoothstep(value: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    """Standard cubic Hermite interpolation clamped to [0, 1]."""
    if abs(float(edge1) - float(edge0)) < 1e-12:
        return torch.where(
            value >= float(edge1),
            torch.ones_like(value),
            torch.zeros_like(value),
        )
    u = torch.clamp((value - float(edge0)) / (float(edge1) - float(edge0)), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def validate_predictive_hvac_config(
    config: Mapping[str, Any],
    *,
    forecast_horizon: int,
) -> None:
    """Validate predictive HVAC DR configuration against design specification."""
    rec_h = float(config.get("recovery_horizon_hours", 6.0))
    if rec_h <= 0:
        raise ValueError("recovery_horizon_hours must be positive")
    if rec_h > float(forecast_horizon):
        raise ValueError(
            f"recovery_horizon_hours ({rec_h}) exceeds forecast_horizon ({forecast_horizon})"
        )

    trans_full = float(config.get("reference_transition_full_hours", 4.0))
    if not 0.0 <= trans_full < rec_h:
        raise ValueError(
            f"reference_transition_full_hours ({trans_full}) must be in [0, recovery_horizon_hours)"
        )

    final_air_h = float(config.get("final_air_priority_hours", 2.0))
    if not 0.0 < final_air_h < rec_h:
        raise ValueError(
            f"final_air_priority_hours ({final_air_h}) must be in (0, recovery_horizon_hours)"
        )

    early_air = float(config.get("early_air_weight", 1.0))
    early_mass = float(config.get("early_mass_weight", 1.0))
    final_air = float(config.get("final_air_weight", 10.0))
    final_mass = float(config.get("final_mass_weight", 0.1))

    for name, w in [
        ("early_air_weight", early_air),
        ("early_mass_weight", early_mass),
        ("final_air_weight", final_air),
        ("final_mass_weight", final_mass),
    ]:
        if w < 0:
            raise ValueError(f"{name} must be nonnegative")

    if final_air <= final_mass:
        raise ValueError(
            f"final_air_weight ({final_air}) must be strictly greater than final_mass_weight ({final_mass})"
        )

    mass_assist = float(config.get("mass_assist_extra_c", 1.0))
    if mass_assist < 0:
        raise ValueError("mass_assist_extra_c must be nonnegative")

    solar_mode = str(config.get("solar_forecast_mode", "zero"))
    if solar_mode != "zero":
        raise ValueError(f"unsupported solar_forecast_mode: {solar_mode!r}")

    fallback_mode = str(config.get("invalid_forecast_fallback", "air_only"))
    if fallback_mode != "air_only":
        raise ValueError(f"unsupported invalid_forecast_fallback: {fallback_mode!r}")

    if not bool(config.get("residual_reference_mapping", True)):
        raise ValueError("residual_reference_mapping must be true")

    if not bool(config.get("detach_physical_reference", True)):
        raise ValueError("detach_physical_reference must be true")


@dataclass
class PredictiveHVACMapResult:
    action: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    center: torch.Tensor
    reference: torch.Tensor
    engagement: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class PredictiveHVACTransform(nn.Module):
    """Finite-horizon 2R2C predictive HVAC reference with reference-centered residual mapping."""

    def __init__(
        self,
        schema: Sequence[str],
        *,
        enabled: bool = True,
        recovery_horizon_hours: float = 6.0,
        reference_transition_full_hours: float = 4.0,
        final_air_priority_hours: float = 2.0,
        early_air_weight: float = 1.0,
        early_mass_weight: float = 1.0,
        final_air_weight: float = 10.0,
        final_mass_weight: float = 0.1,
        mass_assist_extra_c: float = 1.0,
        interval_epsilon: float = 1e-6,
        r_out_air: float = 0.22,
        r_air_mass: float = 0.08,
        c_air: float = 15.0,
        c_mass: float = 250.0,
        cop_heat: float = 3.3,
        cop_cool: float = 3.2,
        heating_max_kw: float = 35.0,
        cooling_max_kw: float = 35.0,
        occupied_hours: Sequence[int] = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
        occupied_comfort_c: tuple[float, float] = (21.0, 25.0),
        unoccupied_comfort_c: tuple[float, float] = (16.0, 30.0),
        terminal_air_tolerance_c: float = 1.0,
        terminal_mass_tolerance_c: float = 0.3,
        internal_gain_occupied_fraction: float = 0.50,
        internal_gain_unoccupied_fraction: float = 0.20,
        load_scale: float = 1.0,
        delta_t: float = 1.0,
        episode_length: int = 24,
        forecast_horizon: int = 8,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.recovery_horizon_hours = float(recovery_horizon_hours)
        self.reference_transition_full_hours = float(reference_transition_full_hours)
        self.final_air_priority_hours = float(final_air_priority_hours)
        self.early_air_weight = float(early_air_weight)
        self.early_mass_weight = float(early_mass_weight)
        self.final_air_weight = float(final_air_weight)
        self.final_mass_weight = float(final_mass_weight)
        self.mass_assist_extra_c = float(mass_assist_extra_c)
        self.interval_epsilon = float(interval_epsilon)

        self.r_out_air = float(r_out_air)
        self.r_air_mass = float(r_air_mass)
        self.c_air = float(c_air)
        self.c_mass = float(c_mass)
        self.cop_heat = float(cop_heat)
        self.cop_cool = float(cop_cool)
        self.heating_max_kw = float(heating_max_kw)
        self.cooling_max_kw = float(cooling_max_kw)

        self.occupied_hours = set(occupied_hours)
        self.occupied_comfort_c = (float(occupied_comfort_c[0]), float(occupied_comfort_c[1]))
        self.unoccupied_comfort_c = (float(unoccupied_comfort_c[0]), float(unoccupied_comfort_c[1]))
        self.terminal_air_tolerance_c = float(terminal_air_tolerance_c)
        self.terminal_mass_tolerance_c = float(terminal_mass_tolerance_c)
        self.internal_gain_occupied_fraction = float(internal_gain_occupied_fraction)
        self.internal_gain_unoccupied_fraction = float(internal_gain_unoccupied_fraction)
        self.load_scale = float(load_scale)
        self.delta_t = float(delta_t)
        self.episode_length = int(episode_length)
        self.forecast_horizon = int(forecast_horizon)

        names = {name: idx for idx, name in enumerate(schema)}
        self.names = dict(names)
        self.t_air_idx = names.get("t_air", -1)
        self.t_mass_idx = names.get("t_mass", -1)
        self.t_out_idx = names.get("t_out", -1)
        self.terminal_air_gap_idx = names.get("terminal_air_gap", -1)
        self.terminal_mass_gap_idx = names.get("terminal_mass_gap", -1)
        self.time_remaining_idx = names.get("time_remaining", -1)
        self.hvac_min_idx = names.get("hvac_feasible_min", -1)
        self.hvac_max_idx = names.get("hvac_feasible_max", -1)
        self.hour_sin_idx = names.get("hour_sin", -1)
        self.hour_cos_idx = names.get("hour_cos", -1)
        self.load_idx = names.get("load", -1)

        required_indices = [
            ("t_air", self.t_air_idx),
            ("t_mass", self.t_mass_idx),
            ("t_out", self.t_out_idx),
            ("terminal_air_gap", self.terminal_air_gap_idx),
            ("terminal_mass_gap", self.terminal_mass_gap_idx),
            ("time_remaining", self.time_remaining_idx),
            ("hvac_feasible_min", self.hvac_min_idx),
            ("hvac_feasible_max", self.hvac_max_idx),
        ]
        missing = [name for name, idx in required_indices if idx < 0]
        if missing:
            raise ValueError(f"Missing required schema fields in observation: {missing}")

    @classmethod
    def from_config(
        cls,
        schema: Sequence[str],
        config: Mapping[str, Any],
        mg_config: Mapping[str, Any] | None = None,
        *,
        normalizers_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
        forecast_horizon: int = 8,
        delta_t: float = 1.0,
        episode_length: int = 24,
    ) -> PredictiveHVACTransform:
        pred_cfg = config.get("predictive_hvac", {})
        validate_predictive_hvac_config(pred_cfg, forecast_horizon=forecast_horizon)

        hvac_dev = (mg_config or {}).get("dr", {})
        load_scale = float(
            (normalizers_by_microgrid or {}).get("mg2", {}).get("load", 1.0)
        )

        return cls(
            schema=schema,
            enabled=bool(pred_cfg.get("enabled", True)),
            recovery_horizon_hours=float(pred_cfg.get("recovery_horizon_hours", 6.0)),
            reference_transition_full_hours=float(
                pred_cfg.get("reference_transition_full_hours", 4.0)
            ),
            final_air_priority_hours=float(
                pred_cfg.get("final_air_priority_hours", 2.0)
            ),
            early_air_weight=float(pred_cfg.get("early_air_weight", 1.0)),
            early_mass_weight=float(pred_cfg.get("early_mass_weight", 1.0)),
            final_air_weight=float(pred_cfg.get("final_air_weight", 10.0)),
            final_mass_weight=float(pred_cfg.get("final_mass_weight", 0.1)),
            mass_assist_extra_c=float(pred_cfg.get("mass_assist_extra_c", 1.0)),
            interval_epsilon=float(config.get("interval_epsilon", 1e-6)),
            r_out_air=float(hvac_dev.get("r_out_air_c_per_kw", 0.22)),
            r_air_mass=float(hvac_dev.get("r_air_mass_c_per_kw", 0.08)),
            c_air=float(hvac_dev.get("c_air_kwh_per_c", 15.0)),
            c_mass=float(hvac_dev.get("c_mass_kwh_per_c", 250.0)),
            cop_heat=float(hvac_dev.get("cop_heat", 3.3)),
            cop_cool=float(hvac_dev.get("cop_cool", 3.2)),
            heating_max_kw=float(hvac_dev.get("heating_max_kw", 35.0)),
            cooling_max_kw=float(hvac_dev.get("cooling_max_kw", 35.0)),
            occupied_hours=tuple(int(x) for x in hvac_dev.get("occupied_hours", range(7, 19))),
            occupied_comfort_c=tuple(float(x) for x in hvac_dev.get("occupied_comfort_c", (21.0, 25.0))),
            unoccupied_comfort_c=tuple(float(x) for x in hvac_dev.get("unoccupied_comfort_c", (16.0, 30.0))),
            terminal_air_tolerance_c=float(hvac_dev.get("terminal_air_tolerance_c", 1.0)),
            terminal_mass_tolerance_c=float(hvac_dev.get("terminal_mass_tolerance_c", 0.3)),
            internal_gain_occupied_fraction=float(hvac_dev.get("internal_gain_occupied_fraction", 0.50)),
            internal_gain_unoccupied_fraction=float(hvac_dev.get("internal_gain_unoccupied_fraction", 0.20)),
            load_scale=load_scale,
            delta_t=float(delta_t),
            episode_length=int(episode_length),
            forecast_horizon=int(forecast_horizon),
        )

    def _approximate_current_hour(self, observation: torch.Tensor) -> torch.Tensor:
        """Estimate current integer clock hour in [0, 23] from diurnal encoding or time remaining."""
        if self.hour_sin_idx >= 0 and self.hour_cos_idx >= 0:
            sin_val = observation[:, self.hour_sin_idx]
            cos_val = observation[:, self.hour_cos_idx]
            angle = torch.atan2(sin_val, cos_val)
            angle = torch.where(angle < 0, angle + 2 * torch.pi, angle)
            hour_float = angle * 24.0 / (2 * torch.pi)
            return torch.round(hour_float) % 24
        elif self.time_remaining_idx >= 0:
            rem = observation[:, self.time_remaining_idx] * float(self.episode_length)
            hour_float = float(self.episode_length) - rem
            return torch.clamp(torch.round(hour_float), 0.0, float(self.episode_length - 1))
        return torch.zeros(observation.shape[0], device=observation.device)

    def _is_hour_occupied(self, hour: torch.Tensor) -> torch.Tensor:
        """Vectorized occupancy check based on configured occupied hours."""
        is_occ = torch.zeros_like(hour, dtype=torch.bool)
        for h in self.occupied_hours:
            is_occ = is_occ | (hour.long() == int(h))
        return is_occ

    def _state_comfort_bounds(self, hour: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Comfort band for state at clock hour. Hour 24 (terminal) treated as unoccupied."""
        is_terminal = hour >= float(self.episode_length)
        is_occ = self._is_hour_occupied(hour % 24) & (~is_terminal)
        low = torch.where(
            is_occ,
            torch.full_like(hour, self.occupied_comfort_c[0]),
            torch.full_like(hour, self.unoccupied_comfort_c[0]),
        )
        high = torch.where(
            is_occ,
            torch.full_like(hour, self.occupied_comfort_c[1]),
            torch.full_like(hour, self.unoccupied_comfort_c[1]),
        )
        return low, high

    def _predict_reference(
        self, observation: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Pure vectorized finite-horizon 2R2C rollout producing reference signed power."""
        batch_size = observation.shape[0]
        device = observation.device

        t_air = observation[:, self.t_air_idx : self.t_air_idx + 1] * 40.0
        t_mass = observation[:, self.t_mass_idx : self.t_mass_idx + 1] * 40.0
        t_out = observation[:, self.t_out_idx : self.t_out_idx + 1] * 40.0
        air_gap = observation[:, self.terminal_air_gap_idx : self.terminal_air_gap_idx + 1] * 40.0
        mass_gap = observation[:, self.terminal_mass_gap_idx : self.terminal_mass_gap_idx + 1] * 40.0

        t_air_init = t_air + air_gap
        t_mass_init = t_mass + mass_gap

        time_rem = (
            observation[:, self.time_remaining_idx : self.time_remaining_idx + 1]
            * float(self.episode_length * self.delta_t)
        )
        current_hour = self._approximate_current_hour(observation).unsqueeze(-1)

        # Base load at current slot
        base_load_kw = (
            observation[:, self.load_idx : self.load_idx + 1] * self.load_scale
            if self.load_idx >= 0
            else torch.zeros(batch_size, 1, device=device)
        )

        steps_rem = torch.clamp(
            torch.ceil(time_rem / self.delta_t),
            min=0.0,
            max=float(self.recovery_horizon_hours / self.delta_t),
        ).long()

        max_h = int(torch.clamp(steps_rem.max(), max=self.forecast_horizon).item())
        if max_h <= 0:
            free_center = (lower + upper) * 0.5
            zero_diag = torch.zeros(batch_size, 1, device=device)
            return free_center, {
                "hvac_predictive_reference": free_center,
                "hvac_air_target_next_c": t_air,
                "hvac_mass_required_air_c": t_mass,
                "hvac_air_weight": torch.full_like(zero_diag, self.final_air_weight),
                "hvac_mass_weight": zero_diag,
                "hvac_mass_control_steps": zero_diag,
                "hvac_mass_rate_requested_cph": zero_diag,
                "hvac_mass_rate_clipped_cph": zero_diag,
                "hvac_predicted_terminal_air_gap_c": air_gap,
                "hvac_predicted_terminal_mass_gap_c": mass_gap,
                "hvac_final_air_priority_veto": zero_diag,
                "hvac_forecast_fallback": zero_diag,
                "hvac_nonfinite_fallback": zero_diag,
            }

        sim_air = t_air.clone()
        sim_mass = t_mass.clone()

        first_p_ref = torch.zeros(batch_size, 1, device=device)
        first_air_target = torch.zeros(batch_size, 1, device=device)
        first_mass_req_air = torch.zeros(batch_size, 1, device=device)
        first_w_air = torch.zeros(batch_size, 1, device=device)
        first_w_mass = torch.zeros(batch_size, 1, device=device)
        first_mass_ctrl_steps = torch.zeros(batch_size, 1, device=device)
        first_mass_rate_req = torch.zeros(batch_size, 1, device=device)
        first_mass_rate_clip = torch.zeros(batch_size, 1, device=device)
        first_veto_applied = torch.zeros(batch_size, 1, device=device)
        forecast_fallback_count = torch.zeros(batch_size, 1, device=device)

        rec_horizon_steps = float(self.recovery_horizon_hours / self.delta_t)
        air_priority_steps = float(self.final_air_priority_hours / self.delta_t)

        last_valid_tout = t_out.clone()
        last_valid_load = base_load_kw.clone()

        for step in range(max_h):
            h_active = (steps_rem - step) > 0  # [batch_size, 1] bool
            h_val = torch.clamp(steps_rem - step, min=1.0).float()
            step_hour = current_hour + step

            # Forecast retrieval
            if step == 0:
                tout_k = t_out
                load_k = base_load_kw
            else:
                temp_idx = self.names.get(f"temperature_t+{step}", -1)
                load_idx = self.names.get(f"load_t+{step}", -1)
                valid_idx = self.names.get(f"forecast_valid_t+{step}", -1)

                is_valid = (
                    observation[:, valid_idx : valid_idx + 1] > 0.5
                    if valid_idx >= 0
                    else torch.ones(batch_size, 1, dtype=torch.bool, device=device)
                )
                forecast_fallback_count = forecast_fallback_count + (~is_valid).float()

                tout_val = observation[:, temp_idx : temp_idx + 1] * 40.0 if temp_idx >= 0 else last_valid_tout
                load_val = observation[:, load_idx : load_idx + 1] * self.load_scale if load_idx >= 0 else last_valid_load

                tout_k = torch.where(is_valid, tout_val, last_valid_tout)
                load_k = torch.where(is_valid, load_val, last_valid_load)
                last_valid_tout = tout_k
                last_valid_load = load_k

            is_occ_k = self._is_hour_occupied(step_hour % 24)
            q_internal_k = load_k * torch.where(
                is_occ_k,
                float(self.internal_gain_occupied_fraction),
                float(self.internal_gain_unoccupied_fraction),
            )

            # Smooth air-only recovery target
            t_air_next_air = sim_air + (t_air_init - sim_air) / torch.clamp_min(h_val, 1.0)

            # Recovery phase factor r
            denom = max(rec_horizon_steps - air_priority_steps, 1e-6)
            r = torch.clamp((h_val - air_priority_steps) / denom, 0.0, 1.0)

            w_air = self.final_air_weight + r * (self.early_air_weight - self.final_air_weight)
            w_mass = torch.where(
                h_val <= 1.0,
                torch.zeros_like(h_val),
                self.final_mass_weight + r * (self.early_mass_weight - self.final_mass_weight),
            )

            # Assist interval
            assist_margin = self.terminal_air_tolerance_c + r * self.mass_assist_extra_c
            comfort_low_next, comfort_high_next = self._state_comfort_bounds(step_hour + 1)
            lower_assist = torch.maximum(t_air_init - assist_margin, comfort_low_next)
            upper_assist = torch.minimum(t_air_init + assist_margin, comfort_high_next)

            # Mass controllability
            has_mass_authority = h_val >= 2.0
            mass_ctrl_steps = torch.clamp(h_val - 1.0, min=1.0)
            mass_rate_requested = (t_mass_init - sim_mass) / (mass_ctrl_steps * self.delta_t)

            t_air_assist = torch.where(
                t_mass_init - sim_mass >= 0.0,
                upper_assist,
                lower_assist,
            )
            mass_rate_limit = ((t_air_assist - sim_mass) / self.r_air_mass) / self.c_mass
            mass_direction = torch.sign(t_mass_init - sim_mass)
            directed_rate_avail = torch.clamp_min(mass_direction * mass_rate_limit, 0.0)
            mass_rate_clipped = mass_direction * torch.minimum(
                torch.abs(mass_rate_requested),
                directed_rate_avail,
            )

            t_air_req_mass = torch.clamp(
                sim_mass + self.r_air_mass * (self.c_mass * mass_rate_clipped),
                min=lower_assist,
                max=upper_assist,
            )

            # Combine targets
            t_air_req_mass_eff = torch.where(has_mass_authority, t_air_req_mass, t_air_next_air)
            w_mass_eff = torch.where(has_mass_authority, w_mass, torch.zeros_like(w_mass))

            w_sum = torch.clamp_min(w_air + w_mass_eff, 1e-6)
            t_air_next_comb = (w_air * t_air_next_air + w_mass_eff * t_air_req_mass_eff) / w_sum

            # Final air priority veto
            in_priority_phase = h_val <= air_priority_steps
            worse_error = (
                torch.abs(t_air_init - t_air_next_comb)
                > torch.abs(t_air_init - t_air_next_air) + 1e-6
            )
            veto_mask = in_priority_phase & worse_error
            t_air_next_ref = torch.where(veto_mask, t_air_next_air, t_air_next_comb)

            # Final assist and comfort clamping
            t_air_next_ref = torch.clamp(t_air_next_ref, min=lower_assist, max=upper_assist)

            # Invert 2R2C equation for reference thermal and electric power
            q_hvac_ref = (
                self.c_air * (t_air_next_ref - sim_air) / self.delta_t
                - (tout_k - sim_air) / self.r_out_air
                - (sim_mass - sim_air) / self.r_air_mass
                - q_internal_k
            )

            p_ref_elec = torch.where(
                q_hvac_ref >= 0.0,
                q_hvac_ref / self.cop_heat,
                q_hvac_ref / self.cop_cool,
            )
            p_ref_norm = torch.where(
                p_ref_elec >= 0.0,
                p_ref_elec / self.heating_max_kw,
                p_ref_elec / self.cooling_max_kw,
            )

            if step == 0:
                p_ref_step = torch.clamp(p_ref_norm, lower, upper)
                first_p_ref = p_ref_step
                first_air_target = t_air_next_ref
                first_mass_req_air = t_air_req_mass
                first_w_air = w_air
                first_w_mass = w_mass_eff
                first_mass_ctrl_steps = mass_ctrl_steps
                first_mass_rate_req = mass_rate_requested
                first_mass_rate_clip = mass_rate_clipped
                first_veto_applied = veto_mask.float()
            else:
                p_ref_step = torch.clamp(p_ref_norm, -1.0, 1.0)

            # Roll states forward for remaining horizon
            q_sim = torch.where(
                p_ref_step >= 0.0,
                p_ref_step * self.heating_max_kw * self.cop_heat,
                p_ref_step * self.cooling_max_kw * self.cop_cool,
            )
            air_rate = (
                (tout_k - sim_air) / self.r_out_air
                + (sim_mass - sim_air) / self.r_air_mass
                + q_internal_k
                + q_sim
            ) / self.c_air
            mass_rate = ((sim_air - sim_mass) / self.r_air_mass) / self.c_mass

            next_sim_air = sim_air + self.delta_t * air_rate
            next_sim_mass = sim_mass + self.delta_t * mass_rate

            sim_air = torch.where(h_active, next_sim_air, sim_air)
            sim_mass = torch.where(h_active, next_sim_mass, sim_mass)

        pred_terminal_air_gap = t_air_init - sim_air
        pred_terminal_mass_gap = t_mass_init - sim_mass

        # Nonfinite check and air-only fallback
        nonfinite_mask = ~(
            torch.isfinite(first_p_ref)
            & torch.isfinite(pred_terminal_air_gap)
            & torch.isfinite(pred_terminal_mass_gap)
        )
        nonfinite_count = nonfinite_mask.float()

        if nonfinite_mask.any():
            h_fallback = torch.clamp_min(steps_rem.float(), 1.0)
            air_target_fb = t_air + (t_air_init - t_air) / h_fallback
            q_fb = (
                self.c_air * (air_target_fb - t_air) / self.delta_t
                - (t_out - t_air) / self.r_out_air
                - (t_mass - t_air) / self.r_air_mass
                - base_load_kw * float(self.internal_gain_unoccupied_fraction)
            )
            p_fb = torch.where(q_fb >= 0.0, q_fb / self.cop_heat, q_fb / self.cop_cool)
            p_norm_fb = torch.where(
                p_fb >= 0.0,
                p_fb / self.heating_max_kw,
                p_fb / self.cooling_max_kw,
            )
            first_p_ref = torch.where(
                nonfinite_mask,
                torch.clamp(p_norm_fb, lower, upper),
                first_p_ref,
            )

        return first_p_ref, {
            "hvac_predictive_reference": first_p_ref,
            "hvac_air_target_next_c": first_air_target,
            "hvac_mass_required_air_c": first_mass_req_air,
            "hvac_air_weight": first_w_air,
            "hvac_mass_weight": first_w_mass,
            "hvac_mass_control_steps": first_mass_ctrl_steps,
            "hvac_mass_rate_requested_cph": first_mass_rate_req,
            "hvac_mass_rate_clipped_cph": first_mass_rate_clip,
            "hvac_predicted_terminal_air_gap_c": pred_terminal_air_gap,
            "hvac_predicted_terminal_mass_gap_c": pred_terminal_mass_gap,
            "hvac_final_air_priority_veto": first_veto_applied,
            "hvac_forecast_fallback": (forecast_fallback_count > 0).float(),
            "hvac_nonfinite_fallback": nonfinite_count,
        }

    def map_with_details(
        self, observation: torch.Tensor, latent: torch.Tensor
    ) -> PredictiveHVACMapResult:
        """Map latent preference z in [-1, 1] to physically feasible action using predictive reference."""
        batch_size = observation.shape[0]
        device = observation.device

        clamped_z = torch.clamp(latent, -1.0, 1.0)
        lower = observation[:, self.hvac_min_idx : self.hvac_min_idx + 1]
        upper = observation[:, self.hvac_max_idx : self.hvac_max_idx + 1]
        width = torch.clamp(upper - lower, min=0.0)
        is_degenerate = width <= self.interval_epsilon

        free_center = (lower + upper) * 0.5
        time_rem = (
            observation[:, self.time_remaining_idx : self.time_remaining_idx + 1]
            * float(self.episode_length * self.delta_t)
        )

        if not self.enabled:
            # Standard V1/V2 free feasible action
            preference = (clamped_z + 1.0) * 0.5
            mapped = lower + preference * width
            action = torch.where(is_degenerate, lower + 0.0 * latent, mapped)
            action = torch.clamp(action, lower, upper)
            zeros = torch.zeros(batch_size, 1, device=device)
            return PredictiveHVACMapResult(
                action=action,
                lower=lower,
                upper=upper,
                center=free_center,
                reference=free_center,
                engagement=zeros,
                diagnostics={},
            )

        # Engagement factor smooth transition
        # Outside recovery horizon: engagement == 0.0
        # Below reference_transition_full_hours: engagement == 1.0
        engagement = 1.0 - smoothstep(
            time_rem,
            self.reference_transition_full_hours,
            self.recovery_horizon_hours,
        )

        # Calculate physical predictive reference
        p_ref, diags = self._predict_reference(observation, lower, upper)

        # Detach physical reference from actor gradient graph as required by specification
        p_ref = p_ref.detach()

        center = (1.0 - engagement) * free_center + engagement * p_ref
        center = torch.clamp(center, lower, upper)

        # Reference-centered residual action mapping:
        # z in [-1, 0) -> lower + (z + 1) * (center - lower) = center + z * (center - lower)
        # z in [0, 1]  -> center + z * (upper - center)
        lower_branch = center + clamped_z * (center - lower)
        upper_branch = center + clamped_z * (upper - center)
        action_mapped = torch.where(clamped_z < 0.0, lower_branch, upper_branch)

        # Degenerate interval protection
        final_action = torch.where(is_degenerate, lower + 0.0 * latent, action_mapped)
        final_action = torch.clamp(final_action, lower, upper)

        diags["hvac_reference_engagement"] = engagement

        return PredictiveHVACMapResult(
            action=final_action,
            lower=lower,
            upper=upper,
            center=center,
            reference=p_ref,
            engagement=engagement,
            diagnostics=diags,
        )

    def forward(self, observation: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning only the final action tensor."""
        return self.map_with_details(observation, latent).action
