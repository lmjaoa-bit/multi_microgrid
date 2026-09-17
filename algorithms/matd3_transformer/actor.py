from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import torch
from torch import nn


MARKET_FEATURES = frozenset({
    "grid_buy_price", "grid_sell_price",
    "previous_p2p_net", "previous_grid_net", "previous_system_p2p",
    "previous_ess_p2p_offer", "previous_ess_p2p_accepted", "previous_ess_p2p_unmatched",
    "previous_system_buy_demand", "previous_system_sell_offer", "previous_clearing_ratio",
})


class TransformerDispatchActor(nn.Module):
    """Spatio-Temporal Transformer Dispatch Actor.

    Architecture:
      1. Learnable [DISPATCH] Decision Token:
         A query token that attends across all spatial and temporal tokens.
      2. Temporal & Diurnal Tokenizers:
         Discrete 24h periodic embedding for current hour and absolute clock hours of future forecasts.
      3. Future Forecast Horizon Embedding:
         Explicit, learnable position embedding for horizons t+1 ... t+H (H=forecast_horizon).
         Preserves bidirectional self-attention (no causal masking) across forecast tokens.
      4. True Local Physical Residual Branch:
         Direct output of local_encoder(local_feats) bypasses the Transformer entirely and is
         concatenated with dispatch_rep (torch.cat([dispatch_rep, local_rep], dim=-1)) without .detach().
         Simultaneously, local_rep is fed as the LOCAL token to the Transformer.
      5. No Artificial Attention Scaling:
         dispatch_rep = attended[..., 0, :] without manual scaling.
    """

    def __init__(
        self,
        schema: Sequence[str],
        agent_ids: Sequence[str],
        active_agent: str,
        *,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dims: tuple[int, ...] = (64, 32),
        forecast_horizon: int = 4,
        attention_scale: float | None = None,  # accepted for backward compatibility only, ignored
    ) -> None:
        super().__init__()
        self.schema = tuple(schema)
        if len(set(self.schema)) != len(self.schema):
            raise ValueError("Duplicate observation features in schema")

        self.peers = tuple(mg for mg in agent_ids if mg != active_agent)
        self.forecast_horizon = int(forecast_horizon)
        if self.forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon must be positive, got {self.forecast_horizon}")

        # 1. Map feature indices
        hour_indices: list[int] = []
        local_indices: list[int] = []
        market_indices: list[int] = []
        forecast_indices: list[list[int]] = [[] for _ in range(self.forecast_horizon)]

        schema_map = {name: i for i, name in enumerate(self.schema)}

        # Hour indices
        for name in ("hour_sin", "hour_cos"):
            if name not in schema_map:
                raise ValueError(f"Missing required hour feature: {name}")
            hour_indices.append(schema_map[name])

        # Local features
        for name in (
            "load", "pv", "wind", "soc",
            "available_charge_ratio", "available_discharge_ratio", "terminal_soc_gap",
            "feasible_charge_min_ratio", "feasible_charge_max_ratio",
            "feasible_discharge_min_ratio", "feasible_discharge_max_ratio",
        ):
            if name not in schema_map:
                raise ValueError(f"Missing required local feature: {name}")
            local_indices.append(schema_map[name])

        # Disaggregated Peer current features: 2 features [net_load, soc] per peer
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

        # Market features
        for name in (
            "grid_buy_price", "grid_sell_price",
            "previous_p2p_net", "previous_grid_net", "previous_system_p2p",
            "previous_ess_p2p_offer", "previous_ess_p2p_accepted", "previous_ess_p2p_unmatched",
            "previous_system_buy_demand", "previous_system_sell_offer", "previous_clearing_ratio",
        ):
            if name not in schema_map:
                raise ValueError(f"Missing required market feature: {name}")
            market_indices.append(schema_map[name])

        # Forecast features for steps k = 1, ..., forecast_horizon
        for step_idx, k in enumerate(range(1, self.forecast_horizon + 1)):
            step_feats = [f"load_t+{k}", f"pv_t+{k}", f"wind_t+{k}", f"price_t+{k}"]
            for peer in self.peers:
                step_feats.append(f"{peer}_net_load_t+{k}")
            for feat in step_feats:
                if feat not in schema_map:
                    raise ValueError(f"Missing required forecast feature: {feat}")
                forecast_indices[step_idx].append(schema_map[feat])

        # Register index buffers for fast torch.index_select
        self.register_buffer("_hour_indices", torch.tensor(hour_indices, dtype=torch.long), persistent=False)
        self.register_buffer("_local_indices", torch.tensor(local_indices, dtype=torch.long), persistent=False)
        all_peer_indices = [idx for sub in peer_indices for idx in sub]
        self.register_buffer("_all_peer_indices", torch.tensor(all_peer_indices, dtype=torch.long), persistent=False)
        self.register_buffer("_market_indices", torch.tensor(market_indices, dtype=torch.long), persistent=False)
        all_forecast_indices = [idx for sub in forecast_indices for idx in sub]
        self.register_buffer("_all_forecast_indices", torch.tensor(all_forecast_indices, dtype=torch.long), persistent=False)
        self.register_buffer("_step_offsets", torch.arange(1, self.forecast_horizon + 1, dtype=torch.long), persistent=False)

        self.embed_dim = int(embed_dim)
        self.embed_dim = int(embed_dim)
        if self.embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({num_heads})")

        # 2. Tokenizers with Pre-Attention LayerNorm Balancing
        # Decision Query Token [1, 1, embed_dim]
        self.dispatch_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.dispatch_norm = nn.LayerNorm(self.embed_dim)

        # Hour Token: discrete 24h table + continuous projection + LayerNorm
        self.hour_embedding = nn.Embedding(24, self.embed_dim)
        self.hour_linear = nn.Linear(len(hour_indices), self.embed_dim)
        self.hour_norm = nn.LayerNorm(self.embed_dim)

        # Local Physical Residual Encoder (Linear + LayerNorm without activation to preserve signed direction)
        self.local_encoder = nn.Sequential(
            nn.Linear(len(local_indices), self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        # Forecast Token: shared encoder + horizon embedding + LayerNorm balancing
        forecast_feature_dim = len(forecast_indices[0])
        self.forecast_encoder = nn.Sequential(
            nn.Linear(forecast_feature_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.horizon_embedding = nn.Embedding(self.forecast_horizon, self.embed_dim)
        nn.init.normal_(self.horizon_embedding.weight, mean=0.0, std=0.02)
        self.forecast_norm = nn.LayerNorm(self.embed_dim)

        # Disaggregated Peer Token: shared encoder + spatial identity + LayerNorm balancing
        self.peer_encoder = nn.Sequential(
            nn.Linear(2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.peer_id_embedding = nn.Embedding(len(agent_ids), self.embed_dim)
        nn.init.normal_(self.peer_id_embedding.weight, mean=0.0, std=0.02)
        self.peer_norm = nn.LayerNorm(self.embed_dim)

        # Market Token Encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(len(market_indices), self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )

        # 3. Deepened 2-Layer Transformer Encoder (Pre-LN)
        # Sequence: 1 dispatch + 1 hour + 1 local + H forecast + num_peers peer + 1 market
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

        # 4. Bilinear Physical Interaction & Dispatch Backbone
        # Bilinear interaction between attended dispatch representation and signed local physical state
        self.interact_norm = nn.LayerNorm(self.embed_dim)

        # Fused representation: dispatch_rep (embed_dim) + local_rep (embed_dim) + interact (embed_dim) = 3 * embed_dim
        fused_dim = self.embed_dim * 3
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
        """Decodes hour index in [0, 23] from sine and cosine features."""
        angle = torch.atan2(hour_sin, hour_cos)
        angle = torch.where(angle < 0.0, angle + 2.0 * np.pi, angle)
        hour_idx = torch.round(angle / (2.0 * np.pi / 24.0)).long() % 24
        return hour_idx

    def forward_components(
        self,
        observation: torch.Tensor,
        collect_diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass exposing internal representations for verification and diagnostics."""
        is_single = observation.ndim == 1
        if is_single:
            observation = observation.unsqueeze(0)

        if observation.shape[-1] != len(self.schema):
            raise ValueError(f"Observation schema mismatch: expected {len(self.schema)}, got {observation.shape[-1]}")

        batch_size = observation.shape[0]

        # Extract feature groups
        hour_feats = observation.index_select(-1, self._hour_indices)
        local_feats = observation.index_select(-1, self._local_indices)
        market_feats = observation.index_select(-1, self._market_indices)

        # 0. Dispatch decision query token [B, 1, embed_dim] with LayerNorm balancing
        dispatch_token = self.dispatch_norm(self.dispatch_token.expand(batch_size, 1, self.embed_dim))

        # 1. Hour Token (current clock hour) with LayerNorm balancing
        hour_sin = hour_feats[:, 0]
        hour_cos = hour_feats[:, 1]
        hour_idx = self._decode_hour_index(hour_sin, hour_cos)
        hour_token = self.hour_norm(self.hour_embedding(hour_idx) + self.hour_linear(hour_feats)).unsqueeze(1)

        # 2. Local physical representation [B, embed_dim]
        # Feeds directly into Transformer (as LOCAL token) AND bypasses as residual skip connection
        local_rep = self.local_encoder(local_feats)
        local_token = local_rep.unsqueeze(1)

        # 3. Forecast Tokens with explicit Horizon Embedding + Diurnal Clock-Hour Embedding + LayerNorm balancing [B, H, embed_dim]
        f_all = observation.index_select(-1, self._all_forecast_indices).reshape(batch_size, self.forecast_horizon, -1)
        f_tokens = self.forecast_encoder(f_all)
        future_hour_idx = (hour_idx.unsqueeze(-1) + self._step_offsets) % 24
        future_hour_emb = self.hour_embedding(future_hour_idx)
        horizon_ids = torch.arange(self.forecast_horizon, device=observation.device, dtype=torch.long)
        horizon_emb = self.horizon_embedding(horizon_ids)
        forecast_tokens = self.forecast_norm(f_tokens + horizon_emb.unsqueeze(0) + future_hour_emb)

        # 4. Disaggregated Peer Tokens with Spatial Identity Embeddings + LayerNorm balancing [B, num_peers, embed_dim]
        peer_all = observation.index_select(-1, self._all_peer_indices).reshape(batch_size, len(self.peers), 2)
        peer_encoded = self.peer_encoder(peer_all)
        peer_spatial_emb = self.peer_id_embedding(self._peer_agent_indices)
        peer_tokens = self.peer_norm(peer_encoded + peer_spatial_emb.unsqueeze(0))

        # 5. Market Token [B, 1, embed_dim] (already normalized in market_encoder)
        market_token = self.market_encoder(market_feats).unsqueeze(1)

        # Assemble balanced sequence of tokens: [B, 4 + H + num_peers, embed_dim]
        # Order: 0: DISPATCH, 1: HOUR, 2: LOCAL, 3..(2+H): FORECASTS, (3+H)..(2+H+P): PEERS, (3+H+P): MARKET
        tokens = torch.cat([dispatch_token, hour_token, local_token, forecast_tokens, peer_tokens, market_token], dim=1)

        # Multi-Head Self-Attention (2 Layers)
        attended = self.transformer(tokens)

        # Pooled dispatch representation from [DISPATCH] query token (Token 0)
        dispatch_rep = attended[:, 0, :]

        # Second-order physical bilinear interaction: global market strategy crossed with signed local physical status
        interact = self.interact_norm(dispatch_rep * local_rep)

        # Residual fusion: global dispatch_rep + local physical constraint local_rep + physical interaction interact
        fused = torch.cat([dispatch_rep, local_rep, interact], dim=-1)
        features = self.dispatch_backbone(fused)
        action = self.dispatch_head(features)

        # Diagnostics are opt-in on training hot paths because converting these
        # tensors to Python floats synchronizes CUDA with the host.
        if collect_diagnostics:
            d_norm = float(dispatch_rep.norm(dim=-1).mean().detach().cpu())
            l_norm = float(local_rep.norm(dim=-1).mean().detach().cpu())
            i_norm = float(interact.norm(dim=-1).mean().detach().cpu())
            ratio = float(d_norm / (l_norm + 1e-8))
            self.last_diagnostics = {
                "dispatch_norm": d_norm,
                "local_norm": l_norm,
                "interact_norm": i_norm,
                "norm_ratio": ratio,
            }

        if is_single:
            return {
                "action": action.squeeze(0),
                "dispatch_rep": dispatch_rep.squeeze(0),
                "local_rep": local_rep.squeeze(0),
                "interact": interact.squeeze(0),
                "fused": fused.squeeze(0),
                "attended": attended.squeeze(0),
                "tokens": tokens.squeeze(0),
                "forecast_tokens": forecast_tokens.squeeze(0),
                "peer_tokens": peer_tokens.squeeze(0),
            }
        return {
            "action": action,
            "dispatch_rep": dispatch_rep,
            "local_rep": local_rep,
            "interact": interact,
            "fused": fused,
            "attended": attended,
            "tokens": tokens,
            "forecast_tokens": forecast_tokens,
            "peer_tokens": peer_tokens,
        }

    def forward(
        self,
        observation: torch.Tensor,
        collect_diagnostics: bool = True,
    ) -> torch.Tensor:
        """Forward pass generating continuous single dispatch intent in [-1, 1]."""
        return self.forward_components(
            observation, collect_diagnostics=collect_diagnostics
        )["action"]

    def branch_modules(self) -> dict[str, tuple[nn.Module, ...] | tuple[nn.Parameter, ...]]:
        return {
            "dispatch_token": (self.dispatch_token,),
            "hour_tokenizer": (self.hour_embedding, self.hour_linear, self.hour_norm),
            "local_tokenizer": (self.local_encoder,),
            "forecast_tokenizer": (self.forecast_encoder, self.horizon_embedding, self.forecast_norm),
            "peer_tokenizer": (self.peer_encoder, self.peer_id_embedding, self.peer_norm),
            "market_tokenizer": (self.market_encoder,),
            "transformer": (self.transformer,),
            "interact_norm": (self.interact_norm,),
            "backbone": (self.dispatch_backbone,),
            "head": (self.dispatch_head,),
        }
