from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer_p2p_value.p2p_auxiliary import (
    P2PPrediction,
    P2PRepresentations,
    P2PValueHead,
    validate_p2p_auxiliary_config,
)
from algorithms.matd3_hierarchical_transformer_predictive_dr.actor import PredictiveDRActor


class P2PValueActor(PredictiveDRActor):
    """Hierarchical Transformer Actor with action-conditioned P2P value auxiliary head."""

    def __init__(
        self,
        schema: Sequence[str],
        agent_ids: Sequence[str],
        active_agent: str,
        mg_config: Mapping[str, Any] | None = None,
        *,
        predictive_config: Mapping[str, Any],
        normalizers_by_microgrid: Mapping[str, Mapping[str, float]],
        p2p_auxiliary_config: Mapping[str, Any],
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
            predictive_config=predictive_config,
            normalizers_by_microgrid=normalizers_by_microgrid,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            hidden_dim=hidden_dim,
            delta_t=delta_t,
            episode_length=episode_length,
        )

        required_fields = [
            "previous_ess_p2p_offer",
            "previous_ess_p2p_accepted",
            "previous_ess_p2p_unmatched",
        ]
        missing = [f for f in required_fields if f not in self.schema]
        if missing:
            raise ValueError(f"Missing required P2P observation fields in schema: {missing}")

        self.p2p_auxiliary_config = validate_p2p_auxiliary_config(
            p2p_auxiliary_config, require_enabled=False
        )
        self.p2p_value_head = P2PValueHead(
            self.embed_dim, int(self.p2p_auxiliary_config["hidden_dim"])
        )

    def forward_latent_with_representations(
        self, observation: torch.Tensor, collect_diagnostics: bool = False
    ) -> tuple[torch.Tensor, P2PRepresentations]:
        if observation.ndim != 2:
            raise ValueError(
                f"Expected observation shape [batch_size, {len(self.schema)}], "
                f"got {observation.shape}"
            )
        token_tensor, padding_mask = self._build_tokens(observation)
        attended = self.transformer(token_tensor, src_key_padding_mask=padding_mask)

        ess_rep = attended[:, 0, :]
        dr_rep = attended[:, 1, :]
        supply_rep = attended[:, 2, :]
        trade_rep = attended[:, 3, :]
        peer_rep = attended[:, 4 : 4 + self.num_peers, :].mean(dim=1)

        ess_feats = observation.index_select(-1, self.ess_idx_tensor)
        ess_skip = self.ess_linear_skip(ess_feats)
        ess_input = torch.cat([ess_rep, supply_rep, trade_rep, ess_skip], dim=-1)
        a_ess = self.ess_head(ess_input)

        dr_feats = observation.index_select(-1, self.dr_idx_tensor)
        dr_skip = self.dr_linear_skip(dr_feats)
        dr_input = torch.cat([dr_rep, supply_rep, trade_rep, dr_skip], dim=-1)
        a_dr = self.dr_head(dr_input)

        if collect_diagnostics:
            self.last_diagnostics = {
                "dispatch_norm": float(torch.norm(a_ess).item()),
                "local_norm": float(torch.norm(a_dr).item()),
                "interact_norm": float(torch.norm(attended).item()),
                "norm_ratio": float(
                    torch.norm(a_ess).item() / max(torch.norm(a_dr).item(), 1e-6)
                ),
            }

        latent = torch.cat([a_ess, a_dr], dim=-1)
        reps = P2PRepresentations(
            ess=ess_rep, supply=supply_rep, trade=trade_rep, peer=peer_rep
        )
        return latent, reps

    def forward_latent(
        self, observation: torch.Tensor, collect_diagnostics: bool = False
    ) -> torch.Tensor:
        latent, _ = self.forward_latent_with_representations(
            observation, collect_diagnostics=collect_diagnostics
        )
        return latent

    def predict_p2p_value(
        self, representations: P2PRepresentations, behavior_ess_action: torch.Tensor
    ) -> P2PPrediction:
        return self.p2p_value_head(representations, behavior_ess_action)

    def branch_modules(self) -> dict[str, list[nn.Module]]:
        branches = super().branch_modules()
        branches["p2p_auxiliary"] = [self.p2p_value_head]
        return branches
