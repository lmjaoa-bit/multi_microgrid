from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer_feasible_dr.actor import FeasibleDRActor
from algorithms.matd3_hierarchical_transformer_predictive_dr.predictive_hvac import (
    PredictiveHVACTransform,
    PredictiveHVACMapResult,
)


class PredictiveDRActor(FeasibleDRActor):
    """Hierarchical Transformer Actor with finite-horizon predictive HVAC reference."""

    def __init__(
        self,
        schema: Sequence[str],
        agent_ids: Sequence[str],
        active_agent: str,
        mg_config: Mapping[str, Any] | None = None,
        *,
        predictive_config: Mapping[str, Any],
        normalizers_by_microgrid: Mapping[str, Mapping[str, float]],
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        forecast_horizon: int = 8,
        hidden_dim: int = 64,
        delta_t: float = 1.0,
        episode_length: int = 24,
    ) -> None:
        feasible_config = dict(predictive_config.get("feasible_dr", {}))
        if "interval_epsilon" not in feasible_config and "interval_epsilon" in predictive_config:
            feasible_config["interval_epsilon"] = predictive_config["interval_epsilon"]

        super().__init__(
            schema,
            agent_ids,
            active_agent,
            mg_config=mg_config,
            feasible_config=feasible_config,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            hidden_dim=hidden_dim,
            delta_t=delta_t,
            episode_length=episode_length,
        )
        self.active_agent = str(active_agent)
        self.predictive_config = predictive_config
        self.normalizers_by_microgrid = normalizers_by_microgrid
        self.delta_t = float(delta_t)
        self.episode_length = int(episode_length)
        self.forecast_horizon = int(forecast_horizon)

        if self.active_agent == "mg2":
            self.predictive_hvac = PredictiveHVACTransform.from_config(
                self.schema,
                predictive_config,
                mg_config=mg_config,
                normalizers_by_microgrid=normalizers_by_microgrid,
                forecast_horizon=self.forecast_horizon,
                delta_t=self.delta_t,
                episode_length=self.episode_length,
            )
        else:
            self.predictive_hvac = None

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

        if self.active_agent == "mg2" and self.predictive_hvac is not None:
            details = self.predictive_hvac.map_with_details(observation, latent_dr)
            dr = details.action
            if collect_diagnostics:
                self._record_predictive_diagnostics(observation, latent_dr, details)
        else:
            # MG1 and MG3 preserve parent behavior exactly
            dr = self.feasible_dr(observation, latent_dr)
            if collect_diagnostics:
                self._record_parent_dr_diagnostics(observation, latent_dr, dr)

        return torch.cat([ess, dr], dim=-1)

    def forward(self, observation: torch.Tensor, collect_diagnostics: bool = False) -> torch.Tensor:
        latent = self.forward_latent(observation, collect_diagnostics=collect_diagnostics)
        action = self.map_latent_actions(observation, latent, collect_diagnostics=collect_diagnostics)
        return action

    def _record_predictive_diagnostics(
        self,
        observation: torch.Tensor,
        latent_dr: torch.Tensor,
        details: PredictiveHVACMapResult,
    ) -> None:
        eps = 1e-4
        lower = details.lower
        upper = details.upper
        width = upper - lower
        action = details.action
        center = details.center

        at_lower = (action <= lower + eps).float().mean().item()
        at_upper = (action >= upper - eps).float().mean().item()
        effective_boundary = ((action <= lower + eps) | (action >= upper - eps)).float().mean().item()

        diags: dict[str, float] = {
            "actor_dr_latent_mean": float(latent_dr.mean().item()),
            "actor_dr_latent_std": float(latent_dr.std().item()) if latent_dr.numel() > 1 else 0.0,
            "actor_dr_mapped_mean": float(action.mean().item()),
            "actor_dr_feasible_min_mean": float(lower.mean().item()),
            "actor_dr_feasible_max_mean": float(upper.mean().item()),
            "actor_dr_interval_width_mean": float(width.mean().item()),
            "actor_dr_lower_boundary_rate": float(at_lower),
            "actor_dr_upper_boundary_rate": float(at_upper),
            "actor_dr_effective_boundary_rate": float(effective_boundary),
            "hvac_predictive_reference_mean": float(details.reference.mean().item()),
            "hvac_reference_engagement_mean": float(details.engagement.mean().item()),
            "hvac_center_mean": float(center.mean().item()),
        }

        for k, v in details.diagnostics.items():
            if isinstance(v, torch.Tensor):
                if v.numel() > 0:
                    val = float(v.float().mean().item())
                    name = k if k.endswith("_rate") else f"{k}_mean"
                    diags[name] = val
            elif isinstance(v, (int, float)):
                diags[k] = float(v)

        self.last_diagnostics = diags

    def _record_parent_dr_diagnostics(
        self,
        observation: torch.Tensor,
        latent_dr: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        eps = 1e-4
        if hasattr(self.feasible_dr, "lower_idx") and hasattr(self.feasible_dr, "upper_idx"):
            low_idx = self.feasible_dr.lower_idx
            up_idx = self.feasible_dr.upper_idx
            if low_idx >= 0 and up_idx >= 0:
                lower = observation[:, low_idx : low_idx + 1]
                upper = observation[:, up_idx : up_idx + 1]
                width = upper - lower
                at_lower = (action <= lower + eps).float().mean().item()
                at_upper = (action >= upper - eps).float().mean().item()
                effective_boundary = ((action <= lower + eps) | (action >= upper - eps)).float().mean().item()
                self.last_diagnostics = {
                    "actor_dr_latent_mean": float(latent_dr.mean().item()),
                    "actor_dr_latent_std": float(latent_dr.std().item()) if latent_dr.numel() > 1 else 0.0,
                    "actor_dr_mapped_mean": float(action.mean().item()),
                    "actor_dr_feasible_min_mean": float(lower.mean().item()),
                    "actor_dr_feasible_max_mean": float(upper.mean().item()),
                    "actor_dr_interval_width_mean": float(width.mean().item()),
                    "actor_dr_lower_boundary_rate": float(at_lower),
                    "actor_dr_upper_boundary_rate": float(at_upper),
                    "actor_dr_effective_boundary_rate": float(effective_boundary),
                }
                return

        self.last_diagnostics = {
            "actor_dr_latent_mean": float(latent_dr.mean().item()),
            "actor_dr_mapped_mean": float(action.mean().item()),
        }
