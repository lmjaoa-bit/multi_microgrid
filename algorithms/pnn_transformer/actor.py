from __future__ import annotations

from collections.abc import Sequence, Mapping
import numpy as np
import torch
from torch import nn


MARKET_FEATURES = frozenset({
    "grid_buy_price", "grid_sell_price",
    "previous_p2p_net", "previous_grid_net", "previous_system_p2p",
    "previous_ess_p2p_offer", "previous_ess_p2p_accepted", "previous_ess_p2p_unmatched",
    "previous_system_buy_demand", "previous_system_sell_offer", "previous_clearing_ratio",
})


class PnnTransformerDispatchActor(nn.Module):
    """Organic PNN-Transformer Hybrid Single-Dispatch Actor."""

    def __init__(
        self,
        schema: Sequence[str],
        agent_ids: Sequence[str],
        active_agent: str,
        *,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dims: tuple[int, ...] = (128, 64),
        product_scale: float = 0.2,
        forecast_horizon: int = 8,
    ) -> None:
        super().__init__()
        self.schema = tuple(schema)
        if len(set(self.schema)) != len(self.schema):
            raise ValueError("Duplicate observation features in schema")

        self.peers = tuple(mg for mg in agent_ids if mg != active_agent)
        self.forecast_horizon = int(forecast_horizon)
        if self.forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon must be positive, got {self.forecast_horizon}")

        self.embed_dim = int(embed_dim)
        if self.embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({num_heads})")

        self.product_scale = float(product_scale)
        if not 0.0 < self.product_scale <= 1.0:
            raise ValueError(f"product_scale must be in (0, 1], got {self.product_scale}")

        hour_indices: list[int] = []
        local_indices: list[int] = []
        market_indices: list[int] = []
        forecast_indices: list[list[int]] = [[] for _ in range(self.forecast_horizon)]
        schema_map = {name: i for i, name in enumerate(self.schema)}

        for name in ("hour_sin", "hour_cos"):
            if name not in schema_map:
                raise ValueError(f"Missing required hour feature: {name}")
            hour_indices.append(schema_map[name])

        for name in (
            "load", "pv", "wind", "soc",
            "available_charge_ratio", "available_discharge_ratio", "terminal_soc_gap",
            "feasible_charge_min_ratio", "feasible_charge_max_ratio",
            "feasible_discharge_min_ratio", "feasible_discharge_max_ratio",
        ):
            if name not in schema_map:
                raise ValueError(f"Missing required local feature: {name}")
            local_indices.append(schema_map[name])

        agent_id_to_idx = {mg: i for i, mg in enumerate(agent_ids)}
        peer_agent_indices = [agent_id_to_idx[peer] for peer in self.peers]
        self.register_buffer("_peer_agent_indices", torch.tensor(peer_agent_indices, dtype=torch.long), persistent=False)

        peer_indices: list[list[int]] = []
        for peer in self.peers:
            feats = []
            for feat in (f"{peer}_net_load", f"{peer}_soc"):
                if feat not in schema_map:
                    raise ValueError(f"Missing required peer feature: {feat}")
                feats.append(schema_map[feat])
            peer_indices.append(feats)

        for name in (
            "grid_buy_price", "grid_sell_price",
            "previous_p2p_net", "previous_grid_net", "previous_system_p2p",
            "previous_ess_p2p_offer", "previous_ess_p2p_accepted", "previous_ess_p2p_unmatched",
            "previous_system_buy_demand", "previous_system_sell_offer", "previous_clearing_ratio",
        ):
            if name not in schema_map:
                raise ValueError(f"Missing required market feature: {name}")
            market_indices.append(schema_map[name])

        for step_idx, k in enumerate(range(1, self.forecast_horizon + 1)):
            step_feats = [f"load_t+{k}", f"pv_t+{k}", f"wind_t+{k}", f"price_t+{k}"]
            for peer in self.peers:
                step_feats.append(f"{peer}_net_load_t+{k}")
            for feat in step_feats:
                if feat not in schema_map:
                    raise ValueError(f"Missing required forecast feature: {feat}")
                forecast_indices[step_idx].append(schema_map[feat])

        forecast_valid_names = [f"forecast_valid_t+{k}" for k in range(1, self.forecast_horizon + 1)]
        present_valid = [name in schema_map for name in forecast_valid_names]
        if any(present_valid) and not all(present_valid):
            missing = [name for name, present in zip(forecast_valid_names, present_valid) if not present]
            raise ValueError(f"Incomplete forecast validity features: missing {missing}")
        forecast_valid_indices = [schema_map[name] for name in forecast_valid_names] if all(present_valid) else []

        self.register_buffer("_hour_indices", torch.tensor(hour_indices, dtype=torch.long), persistent=False)
        self.register_buffer("_local_indices", torch.tensor(local_indices, dtype=torch.long), persistent=False)
        all_peer_indices = [idx for sub in peer_indices for idx in sub]
        self.register_buffer("_all_peer_indices", torch.tensor(all_peer_indices, dtype=torch.long), persistent=False)
        self.register_buffer("_market_indices", torch.tensor(market_indices, dtype=torch.long), persistent=False)
        all_forecast_indices = [idx for sub in forecast_indices for idx in sub]
        self.register_buffer("_all_forecast_indices", torch.tensor(all_forecast_indices, dtype=torch.long), persistent=False)
        self.register_buffer("_forecast_valid_indices", torch.tensor(forecast_valid_indices, dtype=torch.long), persistent=False)
        self.register_buffer("_step_offsets", torch.arange(1, self.forecast_horizon + 1, dtype=torch.long), persistent=False)

        self.dispatch_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.dispatch_norm = nn.LayerNorm(self.embed_dim)
        self.hour_embedding = nn.Embedding(24, self.embed_dim)
        self.hour_linear = nn.Linear(len(hour_indices), self.embed_dim)
        self.hour_norm = nn.LayerNorm(self.embed_dim)
        self.local_encoder = nn.Sequential(
            nn.Linear(len(local_indices), self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        forecast_feature_dim = len(forecast_indices[0])
        self.forecast_encoder = nn.Sequential(
            nn.Linear(forecast_feature_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.horizon_embedding = nn.Embedding(self.forecast_horizon, self.embed_dim)
        nn.init.normal_(self.horizon_embedding.weight, mean=0.0, std=0.02)
        self.forecast_norm = nn.LayerNorm(self.embed_dim)

        self.peer_encoder = nn.Sequential(
            nn.Linear(2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.peer_id_embedding = nn.Embedding(len(agent_ids), self.embed_dim)
        nn.init.normal_(self.peer_id_embedding.weight, mean=0.0, std=0.02)
        self.peer_norm = nn.LayerNorm(self.embed_dim)
        self.market_encoder = nn.Sequential(
            nn.Linear(len(market_indices), self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )

        self.num_tokens = 4 + self.forecast_horizon + len(self.peers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=self.embed_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.embed_dim),
            enable_nested_tensor=False,
        )

        def signed_encoder(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
            )

        self.pnn_local_encoder = signed_encoder(len(local_indices), self.embed_dim)
        self.pnn_temporal_encoder = signed_encoder(len(all_forecast_indices), self.embed_dim)
        self.pnn_peer_encoder = signed_encoder(len(all_peer_indices), self.embed_dim)
        self.pnn_market_encoder = signed_encoder(len(market_indices), self.embed_dim)
        self.product_norm = nn.LayerNorm(self.embed_dim * 3)

        fused_dim = self.embed_dim * 5
        layers: list[nn.Module] = []
        width = fused_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()])
            width = hidden
        self.dispatch_backbone = nn.Sequential(*layers)

        final_linear = nn.Linear(width, 1)
        nn.init.uniform_(final_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(final_linear.bias, -3e-3, 3e-3)
        self.dispatch_head = nn.Sequential(final_linear, nn.Tanh())

        self._parameter_count = sum(p.numel() for p in self.parameters())
        self.last_diagnostics: dict[str, float] = {}

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    def _decode_hour_index(self, hour_sin: torch.Tensor, hour_cos: torch.Tensor) -> torch.Tensor:
        angle = torch.atan2(hour_sin, hour_cos)
        angle = torch.where(angle < 0.0, angle + 2.0 * np.pi, angle)
        return torch.round(angle / (2.0 * np.pi / 24.0)).long() % 24

    def forward_components(
        self, observation: torch.Tensor, *, collect_diagnostics: bool = True
    ) -> dict[str, torch.Tensor]:
        """Forward pass computing and exposing all component representations."""
        is_single = observation.ndim == 1
        if is_single:
            observation = observation.unsqueeze(0)
        if observation.shape[-1] != len(self.schema):
            raise ValueError(f"Observation schema mismatch: expected {len(self.schema)}, got {observation.shape[-1]}")

        batch_size = observation.shape[0]
        hour_feats = observation.index_select(-1, self._hour_indices)
        local_feats = observation.index_select(-1, self._local_indices)
        market_feats = observation.index_select(-1, self._market_indices)
        peer_raw_feats = observation.index_select(-1, self._all_peer_indices)
        raw_forecast = observation.index_select(-1, self._all_forecast_indices)
        if self._forecast_valid_indices.numel() > 0:
            forecast_valid = observation.index_select(-1, self._forecast_valid_indices) > 0.5
        else:
            forecast_valid = torch.ones(
                (batch_size, self.forecast_horizon), dtype=torch.bool, device=observation.device
            )

        dispatch_token = self.dispatch_norm(self.dispatch_token.expand(batch_size, 1, self.embed_dim))
        hour_sin = hour_feats[:, 0]
        hour_cos = hour_feats[:, 1]
        hour_idx = self._decode_hour_index(hour_sin, hour_cos)
        hour_token = self.hour_norm(self.hour_embedding(hour_idx) + self.hour_linear(hour_feats)).unsqueeze(1)

        local_rep = self.local_encoder(local_feats)
        local_token = local_rep.unsqueeze(1)

        f_all = raw_forecast.reshape(batch_size, self.forecast_horizon, -1)
        f_all = f_all.masked_fill((~forecast_valid).unsqueeze(-1), 0.0)
        forecast_raw_feats = f_all.reshape(batch_size, -1)
        f_tokens = self.forecast_encoder(f_all)
        future_hour_idx = (hour_idx.unsqueeze(-1) + self._step_offsets) % 24
        future_hour_emb = self.hour_embedding(future_hour_idx)
        horizon_ids = torch.arange(self.forecast_horizon, device=observation.device, dtype=torch.long)
        horizon_emb = self.horizon_embedding(horizon_ids)
        forecast_tokens = self.forecast_norm(f_tokens + horizon_emb.unsqueeze(0) + future_hour_emb)
        forecast_tokens = forecast_tokens.masked_fill((~forecast_valid).unsqueeze(-1), 0.0)

        peer_all = peer_raw_feats.reshape(batch_size, len(self.peers), 2)
        peer_encoded = self.peer_encoder(peer_all)
        peer_spatial_emb = self.peer_id_embedding(self._peer_agent_indices)
        peer_tokens = self.peer_norm(peer_encoded + peer_spatial_emb.unsqueeze(0))
        market_token = self.market_encoder(market_feats).unsqueeze(1)

        tokens = torch.cat([dispatch_token, hour_token, local_token, forecast_tokens, peer_tokens, market_token], dim=1)
        attention_padding_mask = torch.zeros(
            (batch_size, self.num_tokens), dtype=torch.bool, device=observation.device
        )
        attention_padding_mask[:, 3:3 + self.forecast_horizon] = ~forecast_valid
        attended = self.transformer(tokens, src_key_padding_mask=attention_padding_mask)
        dispatch_rep = attended[:, 0, :]

        e_pnn_local = self.pnn_local_encoder(local_feats)
        e_pnn_temporal = self.pnn_temporal_encoder(forecast_raw_feats)
        e_pnn_peer = self.pnn_peer_encoder(peer_raw_feats)
        e_pnn_market = self.pnn_market_encoder(market_feats)
        p_temporal = e_pnn_local * e_pnn_temporal
        p_spatial = e_pnn_local * e_pnn_peer
        p_market = e_pnn_local * e_pnn_market
        raw_products = torch.cat([p_temporal, p_spatial, p_market], dim=-1)
        normed_products = self.product_norm(raw_products)
        product_signals = self.product_scale * normed_products

        fused = torch.cat([dispatch_rep, product_signals, local_rep], dim=-1)
        features = self.dispatch_backbone(fused)
        action = self.dispatch_head(features)

        if collect_diagnostics:
            with torch.no_grad():
                d_norm = float(dispatch_rep.norm(dim=-1).mean().detach().cpu())
                p_norm = float(product_signals.norm(dim=-1).mean().detach().cpu())
                l_norm = float(local_rep.norm(dim=-1).mean().detach().cpu())
                t_prod_norm = float(p_temporal.norm(dim=-1).mean().detach().cpu())
                s_prod_norm = float(p_spatial.norm(dim=-1).mean().detach().cpu())
                m_prod_norm = float(p_market.norm(dim=-1).mean().detach().cpu())
                self.last_diagnostics = {
                    "dispatch_norm": d_norm,
                    "product_norm": p_norm,
                    "local_norm": l_norm,
                    "temporal_product_norm": t_prod_norm,
                    "spatial_product_norm": s_prod_norm,
                    "market_product_norm": m_prod_norm,
                    "product_to_dispatch_ratio": float(p_norm / max(d_norm, 1e-8)),
                    "local_to_dispatch_ratio": float(l_norm / max(d_norm, 1e-8)),
                    "forecast_valid_fraction": float(forecast_valid.float().mean().detach().cpu()),
                }

        common = {
            "action": action,
            "dispatch_rep": dispatch_rep,
            "product_signals": product_signals,
            "local_rep": local_rep,
            "fused": fused,
            "attended": attended,
            "tokens": tokens,
            "forecast_tokens": forecast_tokens,
            "forecast_valid": forecast_valid,
            "attention_padding_mask": attention_padding_mask,
        }
        if is_single:
            return {name: value.squeeze(0) for name, value in common.items()}
        return common

    def forward(
        self, observation: torch.Tensor, *, collect_diagnostics: bool = True
    ) -> torch.Tensor:
        """Forward pass producing continuous single dispatch action in [-1, 1]."""
        return self.forward_components(
            observation, collect_diagnostics=collect_diagnostics
        )["action"]

    def product_diagnostics(self) -> Mapping[str, float]:
        return dict(self.last_diagnostics)

    def branch_modules(self) -> dict[str, tuple[nn.Module, ...] | tuple[nn.Parameter, ...]]:
        return {
            "dispatch_token": (self.dispatch_token, self.dispatch_norm),
            "hour_tokenizer": (self.hour_embedding, self.hour_linear, self.hour_norm),
            "local_tokenizer": (self.local_encoder,),
            "forecast_tokenizer": (self.forecast_encoder, self.horizon_embedding, self.forecast_norm),
            "peer_tokenizer": (self.peer_encoder, self.peer_id_embedding, self.peer_norm),
            "market_tokenizer": (self.market_encoder,),
            "transformer": (self.transformer,),
            "pnn_encoders": (self.pnn_local_encoder, self.pnn_temporal_encoder, self.pnn_peer_encoder, self.pnn_market_encoder),
            "product_norm": (self.product_norm,),
            "backbone": (self.dispatch_backbone,),
            "head": (self.dispatch_head,),
        }
