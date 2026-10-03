from __future__ import annotations

import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer.actor import HierarchicalTransformerActor


class DeviceForecastReadoutActor(HierarchicalTransformerActor):
    """T-Q actor retaining base-load forecasts and device-specific forecast reads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.forecast_encoder[0].out_features
        num_heads = self.transformer.layers[0].self_attn.num_heads
        self.forecast_feature_dim += 1
        self.ordered_forecast_feature_names += ("local_base_load",)
        self.register_buffer(
            "forecast_load_indices",
            torch.tensor([self.schema.index(f"load_t+{k}")
                          for k in range(1, self.forecast_horizon + 1)], dtype=torch.long),
            persistent=False,
        )
        self.forecast_encoder = nn.Sequential(
            nn.Linear(self.forecast_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embed_dim),
        )
        self.ess_forecast_readout = nn.MultiheadAttention(
            self.embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.dr_forecast_readout = nn.MultiheadAttention(
            self.embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.ess_forecast_norm = nn.LayerNorm(self.embed_dim)
        self.dr_forecast_norm = nn.LayerNorm(self.embed_dim)
        # Keep the four original head inputs and append one forecast context.
        self.ess_head[0] = nn.Linear(self.embed_dim * 5, hidden_dim)
        self.dr_head[0] = nn.Linear(self.embed_dim * 5, hidden_dim)

    def _build_forecast_features(self, obs: torch.Tensor) -> torch.Tensor:
        features = super()._build_forecast_features(obs)
        # Retain the existing load normalization; this is base load, excluding DR.
        load = obs.index_select(-1, self.forecast_load_indices).unsqueeze(-1)
        load = torch.nan_to_num(load, nan=0.0, posinf=0.0, neginf=0.0)
        load = torch.where(features[:, :, 3:4] >= 0.5, load, torch.zeros_like(load))
        return torch.cat([features, load], dim=-1)

    def _device_forecast_contexts(
        self, attended: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = 4 + self.num_peers
        memory = attended[:, start:, :]
        invalid = padding_mask[:, start:]
        available = ~invalid.all(dim=1)
        memory = torch.where(invalid.unsqueeze(-1), torch.zeros_like(memory), memory)
        safe_mask = invalid.clone()
        # At the last hour H8 may be entirely padded. Temporarily expose one zero
        # key to avoid all-masked attention, then discard its output completely.
        safe_mask[:, 0] = safe_mask[:, 0] & available
        ess, _ = self.ess_forecast_readout(
            attended[:, 0:1, :], memory, memory,
            key_padding_mask=safe_mask, need_weights=False,
        )
        dr, _ = self.dr_forecast_readout(
            attended[:, 1:2, :], memory, memory,
            key_padding_mask=safe_mask, need_weights=False,
        )
        ess = self.ess_forecast_norm(ess.squeeze(1))
        dr = self.dr_forecast_norm(dr.squeeze(1))
        return (
            torch.where(available[:, None], ess, torch.zeros_like(ess)),
            torch.where(available[:, None], dr, torch.zeros_like(dr)),
        )

    def action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """One pre-tanh path shared by executed intents and ESS Q guidance."""
        tokens, padding_mask = self._build_tokens(obs)
        attended = self.transformer(tokens, src_key_padding_mask=padding_mask)
        ess_context, dr_context = self._device_forecast_contexts(attended, padding_mask)
        supply_rep, trade_rep = attended[:, 2, :], attended[:, 3, :]
        ess_skip = self.ess_linear_skip(obs.index_select(-1, self.ess_idx_tensor))
        dr_skip = self.dr_linear_skip(obs.index_select(-1, self.dr_idx_tensor))
        ess_input = torch.cat(
            [attended[:, 0, :], supply_rep, trade_rep, ess_skip, ess_context], dim=-1
        )
        dr_input = torch.cat(
            [attended[:, 1, :], supply_rep, trade_rep, dr_skip, dr_context], dim=-1
        )
        return torch.cat([self.ess_head[:-1](ess_input), self.dr_head[:-1](dr_input)], dim=-1)

    def forward(self, obs: torch.Tensor, collect_diagnostics: bool = False) -> torch.Tensor:
        actions = self.action_logits(obs).tanh()
        if collect_diagnostics:
            self.last_diagnostics = {
                "ess_action_norm": float(actions[:, :1].detach().norm(dim=-1).mean().cpu()),
                "dr_action_norm": float(actions[:, 1:].detach().norm(dim=-1).mean().cpu()),
            }
        return actions

    def branch_modules(self) -> dict[str, list[nn.Module]]:
        groups = super().branch_modules()
        groups.update(
            ess_forecast_readout=[self.ess_forecast_readout, self.ess_forecast_norm],
            dr_forecast_readout=[self.dr_forecast_readout, self.dr_forecast_norm],
        )
        return groups
