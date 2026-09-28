from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from components.ev_sessions import cohort_schema


DR_FEATURE_SPECS: dict[str, tuple[str, ...]] = {
    "industrial_mill": (
        "silo_ratio", "silo_target_gap", "time_remaining",
        "required_avg_production", "production_pressure",
        "mill_feasible_min", "mill_feasible_max",
    ),
    "commercial_hvac": (
        "t_air", "t_mass", "t_out", "comfort_lower_margin",
        "comfort_upper_margin", "occupied", "is_weekend",
        "terminal_air_gap", "terminal_mass_gap", "time_remaining",
        "thermal_recovery_pressure", "hvac_feasible_min", "hvac_feasible_max",
    ),
    "ev_charging": (
        "ev_soc", "ev_soc_gap", "remaining_energy", "hours_to_departure",
        "ev_available", "ev_feasible_min", "ev_feasible_max", "deadline_pressure",
    ),
}

ESS_FEATURE_NAMES = (
    "soc", "available_charge_ratio", "available_discharge_ratio",
    "terminal_soc_gap", "time_to_terminal", "required_avg_power_to_target",
    "terminal_pressure", "feasible_charge_min_ratio", "feasible_charge_max_ratio",
    "feasible_discharge_min_ratio", "feasible_discharge_max_ratio",
)

MARKET_FEATURE_NAMES = (
    "grid_buy_price", "grid_sell_price",
    "previous_p2p_net", "previous_grid_net", "previous_system_p2p",
    "previous_ess_p2p_accepted",
)

PEER_SUMMARY_SUFFIXES = (
    "energy_balance", "deficit", "surplus",
    "charge_flex", "discharge_flex", "previous_p2p_net",
)

POWER_NORMALIZER_KEYS = ("load", "pv", "wind", "power")


class HierarchicalTransformerActor(nn.Module):
    """Hierarchical Categorized-Token Spatio-Temporal Transformer Actor.

    Fully vectorized tensor implementation (zero Python loops, zero GPU-CPU stalls).
    Decomposes observation space into 14 physically meaningful semantic tokens:
      1. ESS Token (1 Token, 32d): Battery state, terminal pressure, static capacity/efficiency.
      2. Heterogeneous DR Token (1 Token, 32d): Dedicated device-specific encoder
         (Mill / HVAC / EV) with pacing bias for EV charging to prevent valley rebound peaks.
      3. Current Supply-Demand Token (1 Token, 32d): Physical base balance B = (PV + WT - Load) / S_power.
      4. P2P Trade & Grid Margin Token (1 Token, 32d): Historical clearing, price, and system peak margin.
      5. Peer Energy Summary Tokens (2 Tokens, 2x32d): Public energy balance and flexibility of peer MGs.
      6. Temporal Forecast Tokens (8 Tokens, 8x32d): 8-hour horizon predictions of net load, price, and weather.

    Readout Architecture:
      - Pre-LN Spatio-Temporal Transformer (2 layers, 4 heads, d_model=32)
      - Decoupled Dual Device Heads (ESS Head and DR Head)
      - Bilinear Physical Skip / Gate for tight feasible bounds compliance.
    """

    def __init__(
        self,
        schema: Sequence[str],
        agent_ids: Sequence[str],
        active_agent: str,
        mg_config: Mapping[str, Any] | None = None,
        *,
        power_normalizers: Mapping[str, float],
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        forecast_horizon: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.schema = tuple(schema)
        self.active_agent = str(active_agent)
        self.agent_ids = tuple(agent_ids)
        self.peers = tuple(mg for mg in agent_ids if mg != active_agent)
        self.forecast_horizon = int(forecast_horizon)
        self.embed_dim = int(embed_dim)
        if self.embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        normalizer_values: list[float] = []
        for key in POWER_NORMALIZER_KEYS:
            if key not in power_normalizers:
                raise ValueError(
                    f"{self.active_agent}: power_normalizers missing required key: {key}"
                )
            value = float(power_normalizers[key])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{self.active_agent}: power normalizer {key} must be positive and finite, "
                    f"got {value!r}"
                )
            normalizer_values.append(value)
        self.register_buffer(
            "power_normalizer_tensor",
            torch.tensor(normalizer_values, dtype=torch.float32),
            persistent=False,
        )

        names = {name: idx for idx, name in enumerate(self.schema)}
        def get_idx(name: str) -> int:
            if name not in names:
                raise ValueError(f"Actor for {self.active_agent} missing required feature: {name}")
            return names[name]

        # 1. Hour indices
        self.hour_indices = [get_idx("hour_sin"), get_idx("hour_cos")]
        self.register_buffer("hour_idx_tensor", torch.tensor(self.hour_indices, dtype=torch.long), persistent=False)

        # 2. ESS indices
        self.ess_indices = [get_idx(x) for x in ESS_FEATURE_NAMES]
        self.register_buffer("ess_idx_tensor", torch.tensor(self.ess_indices, dtype=torch.long), persistent=False)

        # Static battery info if available
        battery_params = [0.2, 0.08, 0.08, 0.95, 0.95]
        if mg_config and "battery" in mg_config:
            b = mg_config["battery"]
            battery_params = [
                float(b.get("capacity_kwh", 200.0)) / 1000.0,
                float(b.get("max_charge_kw", 80.0)) / 1000.0,
                float(b.get("max_discharge_kw", 80.0)) / 1000.0,
                float(b.get("eta_charge", 0.95)),
                float(b.get("eta_discharge", 0.95)),
            ]
        self.register_buffer("battery_static", torch.tensor(battery_params, dtype=torch.float32), persistent=False)
        self.ess_encoder = nn.Sequential(
            nn.Linear(len(ESS_FEATURE_NAMES) + len(battery_params), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # 3. Heterogeneous DR indices & encoder
        dr_type = "unknown"
        if mg_config and "dr" in mg_config:
            dr_type = str(mg_config["dr"].get("type", "unknown"))
        else:
            if "silo_ratio" in names:
                dr_type = "industrial_mill"
            elif "t_air" in names:
                dr_type = "commercial_hvac"
            elif "ev_soc" in names:
                dr_type = "ev_charging"
        self.dr_type = dr_type

        if dr_type not in DR_FEATURE_SPECS:
            raise ValueError(f"Unsupported DR type for {self.active_agent}: {dr_type}")

        if dr_type == "ev_charging" and mg_config and mg_config.get("dr", {}).get("cohorts"):
            raw_dr_names = tuple(cohort_schema(mg_config["dr"]))
            self.ev_cohort_mode = True
        else:
            raw_dr_names = DR_FEATURE_SPECS[dr_type]
            self.ev_cohort_mode = False
        self.dr_indices = [get_idx(x) for x in raw_dr_names]
        self.register_buffer("dr_idx_tensor", torch.tensor(self.dr_indices, dtype=torch.long), persistent=False)

        extra_dr_dim = 1 if dr_type == "ev_charging" and not self.ev_cohort_mode else 0
        self.dr_encoder = nn.Sequential(
            nn.Linear(len(raw_dr_names) + extra_dr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # 4. Supply-Demand indices & encoder
        self.supply_indices = [get_idx("load"), get_idx("pv"), get_idx("wind")]
        self.register_buffer("supply_idx_tensor", torch.tensor(self.supply_indices, dtype=torch.long), persistent=False)
        self.supply_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # 5. P2P Trade & System Grid Margin indices & encoder
        self.trade_indices = [get_idx(x) for x in MARKET_FEATURE_NAMES if x in names]
        self.register_buffer("trade_idx_tensor", torch.tensor(self.trade_indices, dtype=torch.long), persistent=False)
        self.trade_encoder = nn.Sequential(
            nn.Linear(len(self.trade_indices) + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # 6. Peer representation indices (Vectorized: [batch, num_peers, peer_dim])
        all_peer_indices: list[int] = []
        is_summary = bool(self.peers and f"{self.peers[0]}_energy_balance" in names)
        for peer in self.peers:
            if is_summary:
                p_inds = [get_idx(f"{peer}_{x}") for x in PEER_SUMMARY_SUFFIXES]
            else:
                p_inds = [get_idx(f"{peer}_net_load"), get_idx(f"{peer}_soc")]
            all_peer_indices.extend(p_inds)
        self.num_peers = len(self.peers)
        self.peer_dim = len(all_peer_indices) // max(1, self.num_peers)
        self.register_buffer("all_peer_indices", torch.tensor(all_peer_indices, dtype=torch.long), persistent=False)
        self.peer_encoder = nn.Sequential(
            nn.Linear(self.peer_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # 7. 8-hour Forecast indices & encoder (Vectorized: [batch, forecast_horizon, 6])
        # Each step has: [load, pv, wind, price, temp_feature, valid_feature]
        all_forecast_indices: list[int] = []
        self.has_temp_by_step: list[bool] = []
        self.has_valid_by_step: list[bool] = []

        dummy_idx = self.hour_indices[0]  # safe fallback index for dummy slicing
        for k in range(1, self.forecast_horizon + 1):
            all_forecast_indices.extend([
                get_idx(f"load_t+{k}"),
                get_idx(f"pv_t+{k}"),
                get_idx(f"wind_t+{k}"),
                get_idx(f"price_t+{k}"),
            ])
            temp_name = f"temperature_t+{k}"
            if temp_name in names:
                all_forecast_indices.append(names[temp_name])
                self.has_temp_by_step.append(True)
            else:
                all_forecast_indices.append(dummy_idx)
                self.has_temp_by_step.append(False)

            valid_name = f"forecast_valid_t+{k}"
            if valid_name in names:
                all_forecast_indices.append(names[valid_name])
                self.has_valid_by_step.append(True)
            else:
                all_forecast_indices.append(dummy_idx)
                self.has_valid_by_step.append(False)

        self.register_buffer("all_forecast_indices", torch.tensor(all_forecast_indices, dtype=torch.long), persistent=False)
        self.register_buffer("temp_mask", torch.tensor(self.has_temp_by_step, dtype=torch.float32).unsqueeze(0).unsqueeze(-1), persistent=False)
        self.register_buffer("valid_mask_present", torch.tensor(self.has_valid_by_step, dtype=torch.bool), persistent=False)

        self.forecast_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Diurnal temporal embeddings
        step_ks = torch.arange(1, self.forecast_horizon + 1, dtype=torch.float32)
        self.register_buffer("step_sin", (step_ks * (2.0 * math.pi / 24.0)).sin().unsqueeze(0).unsqueeze(-1), persistent=False)
        self.register_buffer("step_cos", (step_ks * (2.0 * math.pi / 24.0)).cos().unsqueeze(0).unsqueeze(-1), persistent=False)

        # Token Type Embeddings:
        self.total_tokens = 1 + 1 + 1 + 1 + self.num_peers + self.forecast_horizon
        self.token_type_embedding = nn.Embedding(6, embed_dim)
        type_ids = [0, 1, 2, 3] + [4] * self.num_peers + [5] * self.forecast_horizon
        self.register_buffer("type_ids_tensor", torch.tensor(type_ids, dtype=torch.long), persistent=False)

        # Pre-LN Transformer
        self.token_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Decoupled Dual Readout Heads with Bilinear Physical Skip:
        self.ess_linear_skip = nn.Linear(len(ESS_FEATURE_NAMES), embed_dim)
        self.ess_head = nn.Sequential(
            nn.Linear(embed_dim * 3 + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        self.dr_linear_skip = nn.Linear(len(raw_dr_names), embed_dim)
        self.dr_head = nn.Sequential(
            nn.Linear(embed_dim * 3 + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.last_diagnostics: dict[str, float] = {}

    @property
    def parameter_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def branch_modules(self) -> dict[str, list[nn.Module]]:
        """Return named submodule groups for diagnostic inspection."""
        return {
            "ess_encoder": [self.ess_encoder],
            "dr_encoder": [self.dr_encoder],
            "supply_encoder": [self.supply_encoder],
            "trade_encoder": [self.trade_encoder],
            "peer_encoder": [self.peer_encoder],
            "forecast_encoder": [self.forecast_encoder],
            "transformer": [self.transformer],
            "ess_head": [self.ess_head, self.ess_linear_skip],
            "dr_head": [self.dr_head, self.dr_linear_skip],
        }

    def _physical_net_load(
        self,
        load: torch.Tensor,
        pv: torch.Tensor,
        wind: torch.Tensor,
    ) -> torch.Tensor:
        """Return physical base net load on the shared power scale.

        Input channels remain normalized by their own observation scales. This helper
        restores each physical power before combining them, then applies the existing
        shared power normalizer. Positive values mean a base-load deficit.
        """
        scales = self.power_normalizer_tensor
        return (
            load * scales[0]
            - pv * scales[1]
            - wind * scales[2]
        ) / scales[3]

    def _build_tokens(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = obs.shape[0]

        # 1. ESS Token: [B, 1, embed_dim]
        ess_feats = obs.index_select(-1, self.ess_idx_tensor)
        static_b = self.battery_static.unsqueeze(0).expand(batch_size, -1)
        ess_tok = self.ess_encoder(torch.cat([ess_feats, static_b], dim=-1)).unsqueeze(1)

        # 2. DR Token: [B, 1, embed_dim]
        dr_feats = obs.index_select(-1, self.dr_idx_tensor)
        if self.dr_type == "ev_charging" and not self.ev_cohort_mode:
            rem_energy = dr_feats[:, 2:3]
            hrs_dep = torch.clamp(dr_feats[:, 3:4], min=1.0)
            pace_power = (rem_energy / hrs_dep) / 100.0
            dr_input = torch.cat([dr_feats, pace_power], dim=-1)
        else:
            dr_input = dr_feats
        dr_tok = self.dr_encoder(dr_input).unsqueeze(1)

        # 3. Supply-Demand Token: [B, 1, embed_dim]
        supply_raw = obs.index_select(-1, self.supply_idx_tensor)
        load = supply_raw[:, 0:1]
        pv = supply_raw[:, 1:2]
        wind = supply_raw[:, 2:3]
        # Physical base supply-demand balance; positive means renewable surplus.
        base_balance = -self._physical_net_load(load, pv, wind)
        balance_sign = torch.sign(base_balance)
        supply_input = torch.cat([load, pv, wind, base_balance, balance_sign], dim=-1)
        supply_tok = self.supply_encoder(supply_input).unsqueeze(1)

        # 4. P2P Trade & System Grid Margin Token: [B, 1, embed_dim]
        trade_feats = obs.index_select(-1, self.trade_idx_tensor)
        if "previous_system_p2p" in self.schema:
            sys_p2p = obs[:, self.schema.index("previous_system_p2p"):self.schema.index("previous_system_p2p") + 1]
            sys_margin = 1.0 - torch.clamp(sys_p2p, 0.0, 1.0)
        else:
            sys_margin = torch.ones((batch_size, 1), device=obs.device, dtype=torch.float32)
        trade_input = torch.cat([trade_feats, sys_margin], dim=-1)
        trade_tok = self.trade_encoder(trade_input).unsqueeze(1)

        # 5. Peer Energy Summary Tokens: [B, num_peers, embed_dim]
        peer_raw = obs.index_select(-1, self.all_peer_indices).reshape(batch_size, self.num_peers, self.peer_dim)
        peer_tokens = self.peer_encoder(peer_raw)

        # 6. Temporal Forecast Tokens (Vectorized across all 8 steps in ONE operation): [B, 8, embed_dim]
        f_raw = obs.index_select(-1, self.all_forecast_indices).reshape(batch_size, self.forecast_horizon, 6)
        f_load = f_raw[:, :, 0:1]
        f_pv = f_raw[:, :, 1:2]
        f_wind = f_raw[:, :, 2:3]
        f_price = f_raw[:, :, 3:4]
        # Physical base net load; positive means an electricity deficit.
        f_net_load = self._physical_net_load(f_load, f_pv, f_wind)
        f_temp = f_raw[:, :, 4:5] * self.temp_mask
        f_valid = f_raw[:, :, 5:6]

        hour_feats = obs.index_select(-1, self.hour_idx_tensor)
        sin0 = hour_feats[:, 0:1].unsqueeze(1)
        cos0 = hour_feats[:, 1:2].unsqueeze(1)
        sin_k = sin0 * self.step_cos + cos0 * self.step_sin
        cos_k = cos0 * self.step_cos - sin0 * self.step_sin

        f_input = torch.cat([f_net_load, f_price, f_temp, f_valid, sin_k, cos_k], dim=-1)
        forecast_tokens = self.forecast_encoder(f_input)

        # Assemble all 14 tokens in single tensor concatenation: [B, 14, embed_dim]
        tokens = torch.cat([ess_tok, dr_tok, supply_tok, trade_tok, peer_tokens, forecast_tokens], dim=1)
        tokens = tokens + self.token_type_embedding(self.type_ids_tensor).unsqueeze(0)
        tokens = self.token_norm(tokens)

        # Vectorized padding mask: [B, 14]
        f_invalid = (f_valid < 0.5).squeeze(-1)  # [B, 8]
        non_forecast_valid = torch.zeros((batch_size, 4 + self.num_peers), device=obs.device, dtype=torch.bool)
        padding_mask = torch.cat([non_forecast_valid, f_invalid], dim=1)

        return tokens, padding_mask

    def forward(self, obs: torch.Tensor, collect_diagnostics: bool = False) -> torch.Tensor:
        """Fully vectorized forward pass generating 2-channel dispatch actions [a_ess, a_dr]."""
        if obs.ndim != 2:
            raise ValueError(f"Expected observation shape [batch_size, {len(self.schema)}], got {obs.shape}")

        token_tensor, padding_mask = self._build_tokens(obs)

        # Multi-head Spatio-Temporal Attention
        attended = self.transformer(token_tensor, src_key_padding_mask=padding_mask)

        # Extract semantic representations:
        ess_rep = attended[:, 0, :]
        dr_rep = attended[:, 1, :]
        supply_rep = attended[:, 2, :]
        trade_rep = attended[:, 3, :]

        # Bilinear Physical Residuals
        ess_feats = obs.index_select(-1, self.ess_idx_tensor)
        ess_skip = self.ess_linear_skip(ess_feats)
        ess_input = torch.cat([ess_rep, supply_rep, trade_rep, ess_skip], dim=-1)
        a_ess = self.ess_head(ess_input)

        dr_feats = obs.index_select(-1, self.dr_idx_tensor)
        dr_skip = self.dr_linear_skip(dr_feats)
        dr_input = torch.cat([dr_rep, supply_rep, trade_rep, dr_skip], dim=-1)
        a_dr = self.dr_head(dr_input)

        if collect_diagnostics:
            self.last_diagnostics = {
                "ess_action_norm": float(a_ess.detach().norm(dim=-1).mean().cpu()),
                "dr_action_norm": float(a_dr.detach().norm(dim=-1).mean().cpu()),
            }

        return torch.cat([a_ess, a_dr], dim=-1)
