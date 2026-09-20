from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer.actor import HierarchicalTransformerActor
from algorithms.matd3_hierarchical_transformer_feasible_dr.feasible_dr import FeasibleDRTransform


class FeasibleDRActor(HierarchicalTransformerActor):
    """Hierarchical Transformer Actor that maps latent DR actions into feasible intervals."""

    def __init__(
        self,
        schema: Sequence[str],
        agent_ids: Sequence[str],
        active_agent: str,
        mg_config: Mapping[str, Any] | None = None,
        *,
        feasible_config: Mapping[str, Any] | None = None,
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
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            hidden_dim=hidden_dim,
        )
        feasible_cfg = feasible_config or {}
        self.feasible_dr = FeasibleDRTransform.from_config(
            self.dr_type,
            self.schema,
            feasible_cfg,
            mg_config=mg_config,
            delta_t=delta_t,
            episode_length=episode_length,
        )

    def forward_latent(self, observation: torch.Tensor, collect_diagnostics: bool = False) -> torch.Tensor:
        return super().forward(observation, collect_diagnostics=collect_diagnostics)

    def map_latent_actions(self, observation: torch.Tensor, latent_actions: torch.Tensor) -> torch.Tensor:
        ess = latent_actions[:, 0:1].clamp(-1.0, 1.0)
        dr = self.feasible_dr(observation, latent_actions[:, 1:2])
        return torch.cat([ess, dr], dim=-1)

    def forward(self, observation: torch.Tensor, collect_diagnostics: bool = False) -> torch.Tensor:
        latent = self.forward_latent(observation, collect_diagnostics=collect_diagnostics)
        action = self.map_latent_actions(observation, latent)
        if collect_diagnostics:
            self.last_diagnostics.update(
                self.feasible_dr.scalar_diagnostics(observation, latent[:, 1:2])
            )
        return action
