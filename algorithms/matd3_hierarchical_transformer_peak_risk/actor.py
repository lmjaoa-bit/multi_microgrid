from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer_predictive_dr.actor import PredictiveDRActor
from algorithms.matd3_hierarchical_transformer_peak_risk.peak_risk import PeakRiskFeatureExtractor


class PeakRiskActor(PredictiveDRActor):
    """Predictive DR actor with exactly one additional peak-risk semantic token."""

    def __init__(self, schema: Sequence[str], agent_ids: Sequence[str], active_agent: str,
                 mg_config: Mapping[str, Any] | None = None, *, predictive_config: Mapping[str, Any],
                 normalizers_by_microgrid: Mapping[str, Mapping[str, float]],
                 embed_dim: int = 32, num_heads: int = 4, num_layers: int = 2,
                 forecast_horizon: int = 8, hidden_dim: int = 64,
                 delta_t: float = 1.0, episode_length: int = 24) -> None:
        super().__init__(
            schema, agent_ids, active_agent, mg_config=mg_config,
            predictive_config=predictive_config, normalizers_by_microgrid=normalizers_by_microgrid,
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
            forecast_horizon=forecast_horizon, hidden_dim=hidden_dim,
            delta_t=delta_t, episode_length=episode_length,
        )
        self.peak_features = PeakRiskFeatureExtractor(
            schema, agent_ids, active_agent, forecast_horizon=forecast_horizon)
        self.peak_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, embed_dim),
        )
        self.peak_type_embedding = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.peak_type_embedding, std=0.02)
        self.total_tokens += 1

    def branch_modules(self) -> dict[str, list[nn.Module]]:
        result = super().branch_modules()
        result["peak_encoder"] = [self.peak_encoder]
        return result

    def _build_tokens_from_features(
        self, observation: torch.Tensor, peak_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retain the exact 15-token ordering and normalization of the training path."""
        parent_tokens, parent_mask = super()._build_tokens(observation)
        peak_token = self.peak_encoder(peak_features).unsqueeze(1)
        peak_token = self.token_norm(peak_token + self.peak_type_embedding.view(1, 1, -1))
        tokens = torch.cat([parent_tokens[:, :4], peak_token, parent_tokens[:, 4:]], dim=1)
        valid = torch.zeros((observation.shape[0], 1), dtype=torch.bool, device=observation.device)
        mask = torch.cat([parent_mask[:, :4], valid, parent_mask[:, 4:]], dim=1)
        return tokens, mask

    def _build_tokens(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._build_tokens_from_features(observation, self.peak_features(observation))

    def forward_latent_precomputed(
        self, observation: torch.Tensor, peak_features: torch.Tensor
    ) -> torch.Tensor:
        """Inference-only readout using CPU-prepared observation features.

        This mirrors HierarchicalTransformerActor.forward without transient module
        state, so concurrent inference calls cannot exchange feature tensors.
        The regular forward_latent path remains unchanged for training and targets.
        """
        if observation.ndim != 2 or observation.shape[1] != len(self.schema):
            raise ValueError("peak-risk observation shape mismatch")
        if peak_features.shape != (observation.shape[0], 14):
            raise ValueError("peak-risk precomputed feature shape mismatch")
        if peak_features.device != observation.device or peak_features.dtype != observation.dtype:
            raise ValueError("peak-risk precomputed feature device/dtype mismatch")

        token_tensor, padding_mask = self._build_tokens_from_features(
            observation, peak_features)
        attended = self.transformer(token_tensor, src_key_padding_mask=padding_mask)
        ess_rep, dr_rep = attended[:, 0, :], attended[:, 1, :]
        supply_rep, trade_rep = attended[:, 2, :], attended[:, 3, :]

        ess_feats = observation.index_select(-1, self.ess_idx_tensor)
        ess_skip = self.ess_linear_skip(ess_feats)
        ess_input = torch.cat([ess_rep, supply_rep, trade_rep, ess_skip], dim=-1)
        a_ess = self.ess_head(ess_input)

        dr_feats = observation.index_select(-1, self.dr_idx_tensor)
        dr_skip = self.dr_linear_skip(dr_feats)
        dr_input = torch.cat([dr_rep, supply_rep, trade_rep, dr_skip], dim=-1)
        a_dr = self.dr_head(dr_input)
        return torch.cat([a_ess, a_dr], dim=-1)
