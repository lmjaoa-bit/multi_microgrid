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

DR_FEATURES = frozenset({
    # MG1 industrial mill
    "silo_ratio", "silo_target_gap", "time_remaining",
    "required_avg_production", "production_pressure",
    "mill_feasible_min", "mill_feasible_max",
    # MG2 HVAC
    "t_air", "t_mass", "t_out", "comfort_lower_margin",
    "comfort_upper_margin", "occupied", "is_weekend",
    "terminal_air_gap", "terminal_mass_gap", "thermal_recovery_pressure",
    "hvac_feasible_min", "hvac_feasible_max",
    # MG3 EV
    "ev_soc", "ev_soc_gap", "remaining_energy",
    "hours_to_departure", "ev_available",
    "ev_feasible_min", "ev_feasible_max", "deadline_pressure",
})

ESS_FEATURES = (
    "soc",
    "available_charge_ratio", "available_discharge_ratio",
    "terminal_soc_gap", "time_to_terminal",
    "required_avg_power_to_target", "terminal_pressure",
    "feasible_charge_min_ratio", "feasible_charge_max_ratio",
    "feasible_discharge_min_ratio", "feasible_discharge_max_ratio",
)


class TransformerDispatchActor(nn.Module):
    """Spatio-temporal dispatch actor with explicit ESS and DR decision heads.

    P0 architecture:
      1. Future information is tokenized as a microgrid x time grid instead of
         mixing all peers into one token per horizon.
      2. The active microgrid has a rich local future token at each t+k.
      3. Every peer has an independent future energy-summary token at each t+k.
      4. Current local energy, ESS state, DR state, peers, market and clock each
         receive dedicated tokens.
      5. A shared Transformer produces the dispatch representation.
      6. ESS and DR actions are decoded by separate residual branches/heads.

    The centralized critic is intentionally unchanged in this P0 step.
    """

    native_dr_split_heads = True

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
        attention_scale: float | None = None,  # compatibility only; ignored
    ) -> None:
        super().__init__()
        self.schema = tuple(schema)
        if len(set(self.schema)) != len(self.schema):
            raise ValueError("Duplicate observation features in schema")

        self.agent_ids = tuple(agent_ids)
        if active_agent not in self.agent_ids:
            raise ValueError(f"Unknown active_agent {active_agent!r}")
        self.active_agent = str(active_agent)
        self.active_agent_index = self.agent_ids.index(active_agent)
        self.peers = tuple(mg for mg in self.agent_ids if mg != active_agent)
        self.forecast_horizon = int(forecast_horizon)
        if self.forecast_horizon <= 0:
            raise ValueError(
                f"forecast_horizon must be positive, got {self.forecast_horizon}"
            )

        schema_map = {name: i for i, name in enumerate(self.schema)}

        # Current clock.
        hour_indices = []
        for name in ("hour_sin", "hour_cos"):
            if name not in schema_map:
                raise ValueError(f"Missing required hour feature: {name}")
            hour_indices.append(schema_map[name])

        # Current local energy state.
        local_energy_names = ["load", "pv", "wind"]
        for name in (
            "renewable_power", "energy_balance", "deficit_ratio", "surplus_ratio"
        ):
            if name in schema_map:
                local_energy_names.append(name)
        local_energy_indices = [schema_map[name] for name in local_energy_names]

        # ESS state/constraint guidance.
        ess_names = []
        for name in ESS_FEATURES:
            if name in schema_map:
                ess_names.append(name)
        if "soc" not in ess_names:
            raise ValueError("Missing required ESS feature: soc")
        ess_indices = [schema_map[name] for name in ess_names]

        # Heterogeneous controllable-load state.
        dr_names = [name for name in self.schema if name in DR_FEATURES]
        if not dr_names:
            raise ValueError(
                "Transformer P0 requires DR state features for the split DR head"
            )
        dr_indices = [schema_map[name] for name in dr_names]

        # Current peer summaries.
        agent_id_to_idx = {mg: i for i, mg in enumerate(self.agent_ids)}
        peer_agent_indices = [agent_id_to_idx[peer] for peer in self.peers]
        summary_suffixes = (
            "energy_balance", "deficit", "surplus",
            "charge_flex", "discharge_flex", "previous_p2p_net",
        )
        self.peer_representation = (
            "energy_summary"
            if self.peers and f"{self.peers[0]}_energy_balance" in schema_map
            else "legacy_cross_microgrid"
        )
        peer_indices: list[list[int]] = []
        for peer in self.peers:
            if self.peer_representation == "energy_summary":
                feature_names = [f"{peer}_{suffix}" for suffix in summary_suffixes]
            else:
                feature_names = [f"{peer}_net_load", f"{peer}_soc"]
            indices = []
            for name in feature_names:
                if name not in schema_map:
                    raise ValueError(f"Missing required peer feature: {name}")
                indices.append(schema_map[name])
            peer_indices.append(indices)
        self.peer_feature_dim = len(peer_indices[0]) if peer_indices else 0

        # Current market/history state.
        market_names = (
            "grid_buy_price", "grid_sell_price",
            "previous_p2p_net", "previous_grid_net", "previous_system_p2p",
            "previous_ess_p2p_offer", "previous_ess_p2p_accepted",
            "previous_ess_p2p_unmatched", "previous_system_buy_demand",
            "previous_system_sell_offer", "previous_clearing_ratio",
        )
        market_indices = []
        for name in market_names:
            if name not in schema_map:
                raise ValueError(f"Missing required market feature: {name}")
            market_indices.append(schema_map[name])

        # Rich active-microgrid future tokens.
        local_future_indices: list[list[int]] = []
        for k in range(1, self.forecast_horizon + 1):
            names = [
                f"load_t+{k}", f"pv_t+{k}", f"wind_t+{k}", f"price_t+{k}"
            ]
            if f"renewable_power_t+{k}" in schema_map:
                names.extend([
                    f"renewable_power_t+{k}",
                    f"energy_balance_t+{k}",
                    f"deficit_ratio_t+{k}",
                    f"surplus_ratio_t+{k}",
                ])
            for optional in (
                f"temperature_t+{k}",
                f"forecast_valid_t+{k}",
            ):
                if optional in schema_map:
                    names.append(optional)
            indices = []
            for name in names:
                if name not in schema_map:
                    raise ValueError(f"Missing required local forecast feature: {name}")
                indices.append(schema_map[name])
            local_future_indices.append(indices)

        local_future_dim = len(local_future_indices[0])
        if any(len(indices) != local_future_dim for indices in local_future_indices):
            raise ValueError("Local future feature width must be constant across horizons")

        # Peer future tokens are independent for every (peer, horizon) pair.
        peer_future_indices: list[list[list[int]]] = []
        for k in range(1, self.forecast_horizon + 1):
            step: list[list[int]] = []
            for peer in self.peers:
                if self.peer_representation == "energy_summary":
                    names = [
                        f"{peer}_energy_balance_t+{k}",
                        f"{peer}_deficit_t+{k}",
                        f"{peer}_surplus_t+{k}",
                    ]
                else:
                    names = [f"{peer}_net_load_t+{k}"]
                indices = []
                for name in names:
                    if name not in schema_map:
                        raise ValueError(f"Missing required peer forecast feature: {name}")
                    indices.append(schema_map[name])
                step.append(indices)
            peer_future_indices.append(step)

        self.peer_future_feature_dim = (
            len(peer_future_indices[0][0]) if self.peers else 0
        )
        if self.peers:
            for step in peer_future_indices:
                for indices in step:
                    if len(indices) != self.peer_future_feature_dim:
                        raise ValueError(
                            "Peer future feature width must be constant across peers/horizons"
                        )

        # Fast index buffers.
        self.register_buffer(
            "_hour_indices", torch.tensor(hour_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_local_indices", torch.tensor(local_energy_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_ess_indices", torch.tensor(ess_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_dr_indices", torch.tensor(dr_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_market_indices", torch.tensor(market_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_peer_agent_indices",
            torch.tensor(peer_agent_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_all_peer_indices",
            torch.tensor(
                [idx for sub in peer_indices for idx in sub], dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "_all_forecast_indices",
            torch.tensor(
                [idx for sub in local_future_indices for idx in sub],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_all_peer_future_indices",
            torch.tensor(
                [
                    idx
                    for step in peer_future_indices
                    for peer_indices_for_step in step
                    for idx in peer_indices_for_step
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        # Forecast validity is shared by the local microgrid and all peers:
        # when an episode ends, the entire microgrid x time slice is invalid.
        valid_names = [
            f"forecast_valid_t+{k}" for k in range(1, self.forecast_horizon + 1)
        ]
        valid_present = [name in schema_map for name in valid_names]
        if any(valid_present) and not all(valid_present):
            raise ValueError("Forecast validity must be provided for every horizon step")
        self.register_buffer(
            "_forecast_valid_indices",
            torch.tensor(
                [schema_map[name] for name in valid_names] if all(valid_present) else [],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_step_offsets",
            torch.arange(1, self.forecast_horizon + 1, dtype=torch.long),
            persistent=False,
        )

        self.local_energy_feature_names = tuple(local_energy_names)
        self.ess_feature_names = tuple(ess_names)
        self.dr_feature_names = tuple(dr_names)
        self.local_future_feature_dim = local_future_dim

        self.embed_dim = int(embed_dim)
        if self.embed_dim % int(num_heads) != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Query and current-state tokenizers.
        self.dispatch_token = nn.Parameter(
            torch.randn(1, 1, self.embed_dim) * 0.02
        )
        self.dispatch_norm = nn.LayerNorm(self.embed_dim)

        self.hour_embedding = nn.Embedding(24, self.embed_dim)
        self.hour_linear = nn.Linear(len(hour_indices), self.embed_dim)
        self.hour_norm = nn.LayerNorm(self.embed_dim)

        self.local_encoder = nn.Sequential(
            nn.Linear(len(local_energy_indices), self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.ess_encoder = nn.Sequential(
            nn.Linear(len(ess_indices), self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.dr_encoder = nn.Sequential(
            nn.Linear(len(dr_indices), self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )

        # Current peer tokenizers.
        self.peer_encoder = nn.Sequential(
            nn.Linear(self.peer_feature_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.peer_id_embedding = nn.Embedding(len(self.agent_ids), self.embed_dim)
        nn.init.normal_(self.peer_id_embedding.weight, mean=0.0, std=0.02)
        self.peer_norm = nn.LayerNorm(self.embed_dim)

        self.market_encoder = nn.Sequential(
            nn.Linear(len(market_indices), self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )

        # Spatio-temporal future tokenizers.
        self.forecast_encoder = nn.Sequential(
            nn.Linear(self.local_future_feature_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.peer_future_encoder = nn.Sequential(
            nn.Linear(self.peer_future_feature_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.horizon_embedding = nn.Embedding(
            self.forecast_horizon, self.embed_dim
        )
        nn.init.normal_(self.horizon_embedding.weight, mean=0.0, std=0.02)
        self.forecast_norm = nn.LayerNorm(self.embed_dim)
        self.peer_future_norm = nn.LayerNorm(self.embed_dim)

        # Sequence:
        # [DISPATCH, HOUR, LOCAL_ENERGY, ESS, DR, current peers,
        #  (active + peers) x H future grid, MARKET]
        self.num_spatial_entities = 1 + len(self.peers)
        self.num_future_tokens = (
            self.forecast_horizon * self.num_spatial_entities
        )
        self.num_tokens = 6 + len(self.peers) + self.num_future_tokens

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

        # Head-specific residual interactions.
        self.ess_interact_norm = nn.LayerNorm(self.embed_dim)
        self.dr_interact_norm = nn.LayerNorm(self.embed_dim)

        fused_dim = self.embed_dim * 4

        def make_backbone() -> nn.Sequential:
            layers: list[nn.Module] = []
            width = fused_dim
            for hidden in hidden_dims:
                layers.extend([
                    nn.Linear(width, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                ])
                width = hidden
            return nn.Sequential(*layers)

        self.ess_backbone = make_backbone()
        self.dr_backbone = make_backbone()
        final_width = hidden_dims[-1] if hidden_dims else fused_dim

        ess_linear = nn.Linear(final_width, 1)
        dr_linear = nn.Linear(final_width, 1)
        nn.init.uniform_(ess_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(ess_linear.bias, -3e-3, 3e-3)
        nn.init.uniform_(dr_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(dr_linear.bias, -3e-3, 3e-3)
        self.ess_head = nn.Sequential(ess_linear, nn.Tanh())
        self.dr_head = nn.Sequential(dr_linear, nn.Tanh())

        self._parameter_count = sum(p.numel() for p in self.parameters())
        self.last_diagnostics: dict[str, float] = {}

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    def _decode_hour_index(
        self, hour_sin: torch.Tensor, hour_cos: torch.Tensor
    ) -> torch.Tensor:
        angle = torch.atan2(hour_sin, hour_cos)
        angle = torch.where(angle < 0.0, angle + 2.0 * np.pi, angle)
        return torch.round(angle / (2.0 * np.pi / 24.0)).long() % 24

    def forward_components(
        self,
        observation: torch.Tensor,
        collect_diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        is_single = observation.ndim == 1
        if is_single:
            observation = observation.unsqueeze(0)
        if observation.shape[-1] != len(self.schema):
            raise ValueError(
                f"Observation schema mismatch: expected {len(self.schema)}, "
                f"got {observation.shape[-1]}"
            )

        batch_size = observation.shape[0]
        hour_feats = observation.index_select(-1, self._hour_indices)
        local_feats = observation.index_select(-1, self._local_indices)
        ess_feats = observation.index_select(-1, self._ess_indices)
        dr_feats = observation.index_select(-1, self._dr_indices)
        market_feats = observation.index_select(-1, self._market_indices)

        dispatch_token = self.dispatch_norm(
            self.dispatch_token.expand(batch_size, 1, self.embed_dim)
        )

        hour_sin = hour_feats[:, 0]
        hour_cos = hour_feats[:, 1]
        hour_idx = self._decode_hour_index(hour_sin, hour_cos)
        hour_token = self.hour_norm(
            self.hour_embedding(hour_idx) + self.hour_linear(hour_feats)
        ).unsqueeze(1)

        active_spatial = self.peer_id_embedding.weight[
            self.active_agent_index
        ].unsqueeze(0)

        local_rep = self.local_encoder(local_feats)
        local_token = self.peer_norm(
            local_rep + active_spatial
        ).unsqueeze(1)

        ess_rep = self.ess_encoder(ess_feats)
        ess_token = self.peer_norm(
            ess_rep + active_spatial
        ).unsqueeze(1)

        dr_rep = self.dr_encoder(dr_feats)
        dr_token = self.peer_norm(
            dr_rep + active_spatial
        ).unsqueeze(1)

        # Current peer tokens.
        peer_all = observation.index_select(
            -1, self._all_peer_indices
        ).reshape(batch_size, len(self.peers), self.peer_feature_dim)
        peer_encoded = self.peer_encoder(peer_all)
        peer_spatial = self.peer_id_embedding(self._peer_agent_indices)
        peer_tokens = self.peer_norm(
            peer_encoded + peer_spatial.unsqueeze(0)
        )

        # Shared temporal embeddings for the complete future grid.
        future_hour_idx = (
            hour_idx.unsqueeze(-1) + self._step_offsets
        ) % 24
        future_hour_emb = self.hour_embedding(future_hour_idx)
        horizon_ids = torch.arange(
            self.forecast_horizon,
            device=observation.device,
            dtype=torch.long,
        )
        horizon_emb = self.horizon_embedding(horizon_ids)
        temporal_emb = future_hour_emb + horizon_emb.unsqueeze(0)

        # Rich future token for the active microgrid at each horizon.
        local_future_all = observation.index_select(
            -1, self._all_forecast_indices
        ).reshape(
            batch_size,
            self.forecast_horizon,
            self.local_future_feature_dim,
        )
        local_future_tokens = self.forecast_encoder(local_future_all)
        local_future_tokens = self.forecast_norm(
            local_future_tokens
            + temporal_emb
            + active_spatial.unsqueeze(1)
        )

        # Independent peer future tokens: [B, H, P, E].
        peer_future_all = observation.index_select(
            -1, self._all_peer_future_indices
        ).reshape(
            batch_size,
            self.forecast_horizon,
            len(self.peers),
            self.peer_future_feature_dim,
        )
        peer_future_tokens = self.peer_future_encoder(peer_future_all)
        peer_future_tokens = self.peer_future_norm(
            peer_future_tokens
            + temporal_emb.unsqueeze(2)
            + peer_spatial.unsqueeze(0).unsqueeze(0)
        )

        # microgrid x time grid: local first, then each peer, for every horizon.
        spatiotemporal_tokens = torch.cat(
            [local_future_tokens.unsqueeze(2), peer_future_tokens],
            dim=2,
        )
        future_grid = spatiotemporal_tokens.reshape(
            batch_size, self.num_future_tokens, self.embed_dim
        )

        market_token = self.market_encoder(market_feats).unsqueeze(1)

        tokens = torch.cat(
            [
                dispatch_token,
                hour_token,
                local_token,
                ess_token,
                dr_token,
                peer_tokens,
                future_grid,
                market_token,
            ],
            dim=1,
        )
        # PyTorch's padding mask suppresses invalid *keys/values*. Merely
        # filling forecast fields with zeros does NOT remove their learned
        # spatial/temporal embeddings from the attention mechanism.
        if self._forecast_valid_indices.numel():
            valid_by_step = observation.index_select(
                -1, self._forecast_valid_indices
            ) >= 0.5
            future_padding = (~valid_by_step).repeat_interleave(
                self.num_spatial_entities, dim=1
            )
            current_padding = torch.zeros(
                batch_size, 5 + len(self.peers),
                dtype=torch.bool, device=observation.device,
            )
            market_padding = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=observation.device,
            )
            attention_padding_mask = torch.cat(
                [current_padding, future_padding, market_padding], dim=1
            )
            assert attention_padding_mask.shape == tokens.shape[:2]
            attended = self.transformer(
                tokens, src_key_padding_mask=attention_padding_mask
            )
        else:
            # Legacy observations without forecast_valid_t+k retain their
            # historical unmasked behavior.
            valid_by_step = torch.ones(
                batch_size, self.forecast_horizon,
                dtype=torch.bool, device=observation.device,
            )
            attention_padding_mask = torch.zeros(
                tokens.shape[:2], dtype=torch.bool, device=observation.device
            )
            attended = self.transformer(tokens)
        dispatch_rep = attended[:, 0, :]

        ess_interact = self.ess_interact_norm(dispatch_rep * ess_rep)
        dr_interact = self.dr_interact_norm(dispatch_rep * dr_rep)

        ess_fused = torch.cat(
            [dispatch_rep, local_rep, ess_rep, ess_interact], dim=-1
        )
        dr_fused = torch.cat(
            [dispatch_rep, local_rep, dr_rep, dr_interact], dim=-1
        )
        ess_features = self.ess_backbone(ess_fused)
        dr_features = self.dr_backbone(dr_fused)
        ess_action = self.ess_head(ess_features)
        dr_action = self.dr_head(dr_features)
        action = torch.cat([ess_action, dr_action], dim=-1)

        if collect_diagnostics:
            d_norm = float(
                dispatch_rep.norm(dim=-1).mean().detach().cpu()
            )
            l_norm = float(local_rep.norm(dim=-1).mean().detach().cpu())
            ess_norm = float(ess_rep.norm(dim=-1).mean().detach().cpu())
            dr_norm = float(dr_rep.norm(dim=-1).mean().detach().cpu())
            ess_i_norm = float(
                ess_interact.norm(dim=-1).mean().detach().cpu()
            )
            dr_i_norm = float(
                dr_interact.norm(dim=-1).mean().detach().cpu()
            )
            self.last_diagnostics = {
                "dispatch_norm": d_norm,
                "local_norm": l_norm,
                "ess_norm": ess_norm,
                "dr_norm": dr_norm,
                "ess_interact_norm": ess_i_norm,
                "dr_interact_norm": dr_i_norm,
                "interact_norm": 0.5 * (ess_i_norm + dr_i_norm),
                "norm_ratio": d_norm / (l_norm + 1e-8),
            }

        result = {
            "action": action,
            "ess_action": ess_action,
            "dr_action": dr_action,
            "dispatch_rep": dispatch_rep,
            "local_rep": local_rep,
            "ess_rep": ess_rep,
            "dr_rep": dr_rep,
            "ess_interact": ess_interact,
            "dr_interact": dr_interact,
            "interact": 0.5 * (ess_interact + dr_interact),
            "ess_fused": ess_fused,
            "dr_fused": dr_fused,
            "fused": torch.cat([ess_fused, dr_fused], dim=-1),
            "attended": attended,
            "tokens": tokens,
            # Backward-compatible name: active-microgrid future tokens.
            "forecast_tokens": local_future_tokens,
            "local_future_tokens": local_future_tokens,
            "peer_tokens": peer_tokens,
            "peer_future_tokens": peer_future_tokens,
            "spatiotemporal_tokens": spatiotemporal_tokens,
            "attention_padding_mask": attention_padding_mask,
            "valid_forecast_steps": valid_by_step,
        }

        if is_single:
            return {
                name: value.squeeze(0)
                for name, value in result.items()
            }
        return result

    def forward(
        self,
        observation: torch.Tensor,
        collect_diagnostics: bool = True,
    ) -> torch.Tensor:
        """Return [ESS intent, DR intent], both in [-1, 1]."""
        return self.forward_components(
            observation,
            collect_diagnostics=collect_diagnostics,
        )["action"]

    def branch_modules(
        self,
    ) -> dict[str, tuple[nn.Module, ...] | tuple[nn.Parameter, ...]]:
        # Aggregate legacy branch names are retained for existing diagnostics.
        return {
            "dispatch_token": (self.dispatch_token,),
            "hour_tokenizer": (
                self.hour_embedding, self.hour_linear, self.hour_norm,
            ),
            "local_tokenizer": (
                self.local_encoder, self.ess_encoder, self.dr_encoder,
            ),
            "local_energy_tokenizer": (self.local_encoder,),
            "ess_tokenizer": (self.ess_encoder,),
            "dr_tokenizer": (self.dr_encoder,),
            "forecast_tokenizer": (
                self.forecast_encoder,
                self.horizon_embedding,
                self.forecast_norm,
            ),
            "peer_future_tokenizer": (
                self.peer_future_encoder,
                self.peer_future_norm,
            ),
            "peer_tokenizer": (
                self.peer_encoder,
                self.peer_id_embedding,
                self.peer_norm,
            ),
            "market_tokenizer": (self.market_encoder,),
            "transformer": (self.transformer,),
            "interact_norm": (
                self.ess_interact_norm, self.dr_interact_norm,
            ),
            "ess_interact_norm": (self.ess_interact_norm,),
            "dr_interact_norm": (self.dr_interact_norm,),
            "backbone": (self.ess_backbone, self.dr_backbone),
            "ess_backbone": (self.ess_backbone,),
            "dr_backbone": (self.dr_backbone,),
            "head": (self.ess_head, self.dr_head),
            "ess_head": (self.ess_head,),
            "dr_head": (self.dr_head,),
        }
