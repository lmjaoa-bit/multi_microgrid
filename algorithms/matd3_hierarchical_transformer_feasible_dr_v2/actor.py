from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer_feasible_dr.actor import FeasibleDRActor
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import AdaptiveDRTransform


class AdaptiveDRActor(FeasibleDRActor):
    """Hierarchical Transformer Actor with adaptive DR references (V2)."""

    def __init__(
        self,
        schema: Sequence[str],
        agent_ids: Sequence[str],
        active_agent: str,
        mg_config: Mapping[str, Any] | None = None,
        *,
        adaptive_config: Mapping[str, Any],
        normalizers_by_microgrid: Mapping[str, Mapping[str, float]],
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        forecast_horizon: int = 8,
        hidden_dim: int = 64,
        delta_t: float = 1.0,
        episode_length: int = 24,
    ) -> None:
        super().__init__(
            schema,
            agent_ids,
            active_agent,
            mg_config=mg_config,
            feasible_config=adaptive_config,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            hidden_dim=hidden_dim,
            delta_t=delta_t,
            episode_length=episode_length,
        )
        self.active_agent = str(active_agent)
        self.adaptive_config = adaptive_config
        self.normalizers_by_microgrid = normalizers_by_microgrid
        self.delta_t = float(delta_t)
        self.episode_length = int(episode_length)
        self.forecast_horizon = int(forecast_horizon)

        if self.active_agent in ("mg2", "mg3"):
            self.adaptive_dr = AdaptiveDRTransform.from_config(
                self.dr_type,
                self.schema,
                adaptive_config,
                mg_config=mg_config,
                normalizers_by_microgrid=normalizers_by_microgrid,
                active_agent=self.active_agent,
                forecast_horizon=self.forecast_horizon,
                delta_t=self.delta_t,
                episode_length=self.episode_length,
            )
        else:
            self.adaptive_dr = None

        self.last_diagnostics: dict[str, float] = {}

    def map_latent_actions(
        self,
        observation: torch.Tensor,
        latent_actions: torch.Tensor,
        *,
        collect_diagnostics: bool = False,
    ) -> torch.Tensor:
        ess = latent_actions[:, 0:1].clamp(-1.0, 1.0)
        latent_dr = latent_actions[:, 1:2]

        if self.adaptive_dr is None:
            # MG1: exact V1 forward path
            dr = self.feasible_dr(observation, latent_dr)
            return torch.cat([ess, dr], dim=-1)

        details = self.adaptive_dr.map_with_details(observation, latent_dr)
        dr = details.action

        if collect_diagnostics:
            self._record_adaptive_diagnostics(observation, latent_dr, details)

        return torch.cat([ess, dr], dim=-1)

    def forward(self, observation: torch.Tensor, collect_diagnostics: bool = False) -> torch.Tensor:
        latent = self.forward_latent(observation, collect_diagnostics=collect_diagnostics)
        action = self.map_latent_actions(observation, latent, collect_diagnostics=collect_diagnostics)
        return action

    def _record_adaptive_diagnostics(
        self,
        observation: torch.Tensor,
        latent_dr: torch.Tensor,
        details: Any,
    ) -> None:
        with torch.no_grad():
            pref = (latent_dr.clamp(-1.0, 1.0) + 1.0) * 0.5
            dr = details.action
            lower_b = float((pref <= 0.05).float().mean().item())
            upper_b = float((pref >= 0.95).float().mean().item())

            if self.active_agent == "mg3":
                avail = details.available > 0.5
                if avail.any():
                    eff_lower = dr[avail] - details.lower[avail] <= 1e-4
                    eff_upper = details.upper[avail] - dr[avail] <= 1e-4
                    effective_boundary = float((eff_lower | eff_upper).float().mean().item())
                    avail_boundary = effective_boundary
                else:
                    effective_boundary = 0.0
                    avail_boundary = 0.0
            else:
                eff_lower = dr - details.lower <= 1e-4
                eff_upper = details.upper - dr <= 1e-4
                effective_boundary = float((eff_lower | eff_upper).float().mean().item())
                avail_boundary = 0.0

            diagnostics = {
                "dr_latent_mean": float(latent_dr.mean().item()),
                "dr_latent_std": float(latent_dr.std(unbiased=False).item()) if latent_dr.numel() > 1 else 0.0,
                "dr_mapped_mean": float(dr.mean().item()),
                "dr_feasible_min_mean": float(details.lower.mean().item()),
                "dr_feasible_max_mean": float(details.upper.mean().item()),
                "dr_interval_width_mean": float((details.upper - details.lower).mean().item()),
                "dr_lower_boundary_rate": lower_b,
                "dr_upper_boundary_rate": upper_b,
                "dr_effective_boundary_rate": effective_boundary,
            }

            if self.active_agent == "mg2":
                diagnostics.update({
                    "hvac_recovery_anchor_mean": float(details.recovery_anchor.mean().item()),
                    "hvac_recovery_gate_mean": float(details.recovery_gate.mean().item()),
                    "hvac_recovery_gate_active_rate": float((details.recovery_gate > 0.0).float().mean().item()),
                    "hvac_recovery_gate_full_rate": float((details.recovery_gate >= 0.999).float().mean().item()),
                    "hvac_terminal_air_gap_mean_c": (
                        float((observation[:, self.adaptive_dr.air_gap_idx] * 40.0).mean().item())
                        if self.adaptive_dr.air_gap_idx >= 0 else 0.0
                    ),
                    "hvac_terminal_mass_gap_mean_c": (
                        float((observation[:, self.adaptive_dr.mass_gap_idx] * 40.0).mean().item())
                        if self.adaptive_dr.mass_gap_idx >= 0 else 0.0
                    ),
                    "hvac_terminal_recovery_pressure_mean": (
                        float(observation[:, self.adaptive_dr.recovery_pressure_idx].mean().item())
                        if self.adaptive_dr.recovery_pressure_idx >= 0 else 0.0
                    ),
                })
            elif self.active_agent == "mg3":
                diagnostics.update({
                    "ev_mandatory_floor_mean": float(details.mandatory_floor.mean().item()),
                    "ev_mandatory_floor_active_rate": float((details.mandatory_floor > 1e-6).float().mean().item()),
                    "ev_economic_reference_mean": float(details.reference.mean().item()),
                    "ev_reference_above_floor_mean": float(torch.clamp(details.reference - details.mandatory_floor, min=0.0).mean().item()),
                    "ev_latent_reference_deviation_mean": float((pref - details.reference).abs().mean().item()),
                    "ev_deadline_pressure_mean": (
                        float(observation[:, self.adaptive_dr.deadline_pressure_idx].mean().item())
                        if self.adaptive_dr.deadline_pressure_idx >= 0 else 0.0
                    ),
                    "ev_available_rate": float(details.available.mean().item()),
                    "ev_available_boundary_rate": avail_boundary,
                })

            self.last_diagnostics = diagnostics
