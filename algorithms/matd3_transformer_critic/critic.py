"""Device-aware action-conditioned centralized twin Transformer critic.

The 37 state tokens are [ESS, DR, supply, trade] for each of three MGs,
one market token, and three MG forecast tokens at each of eight horizons.
Six action queries read the state tokens and jointly condition on all nine
projected physical controls. No future controlled device state is invented.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn


DR_SUFFIXES = {
    "industrial_mill": (
        "silo_ratio", "silo_target_gap", "time_remaining",
        "required_avg_production", "production_pressure", "dr_min", "dr_max",
    ),
    "commercial_hvac": (
        "t_air", "t_mass", "t_out", "comfort_lower_margin",
        "comfort_upper_margin", "occupied", "is_weekend",
        "terminal_air_gap", "terminal_mass_gap", "time_remaining",
        "thermal_recovery_pressure", "dr_min", "dr_max",
    ),
    "ev_charging": (
        "ev_soc", "ev_soc_gap", "remaining_energy", "hours_to_departure",
        "ev_available", "deadline_pressure", "dr_min", "dr_max",
    ),
}

ESS_SUFFIXES = (
    "soc", "terminal_soc_gap", "time_to_terminal",
    "required_avg_power_to_target", "terminal_pressure",
    "available_charge", "available_discharge",
    "feasible_charge_min", "feasible_charge_max",
    "feasible_discharge_min", "feasible_discharge_max",
)
TRADE_SUFFIXES = (
    "previous_p2p_net", "previous_ess_p2p_offer",
    "previous_ess_p2p_accepted", "previous_ess_p2p_unmatched",
)
MARKET_NAMES = (
    "hour_sin", "hour_cos", "previous_system_p2p",
    "previous_system_buy_demand", "previous_system_sell_offer",
    "previous_clearing_ratio",
)


class DeviceActionQ(nn.Module):
    """One independent Q network; the other twin owns a separate encoder."""

    def __init__(
        self,
        schema: Sequence[str],
        microgrids: Sequence[Mapping],
        normalizers: Mapping,
        *,
        horizon: int = 8,
        embed_dim: int = 32,
        heads: int = 4,
        layers: int = 1,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if len(microgrids) != 3 or horizon != 8:
            raise ValueError("Transformer critic V1 requires 3 microgrids and an 8h forecast")
        if embed_dim <= 0 or heads <= 0 or embed_dim % heads or layers < 1:
            raise ValueError("Invalid Transformer embedding/heads/layers")
        self.state_dim = len(schema)
        self.action_dim = len(microgrids) * 3
        names = {name: idx for idx, name in enumerate(schema)}
        def idx(name: str) -> int:
            if name not in names:
                raise ValueError(f"Transformer critic requires global state field: {name}")
            return names[name]

        self.ids = tuple(str(mg["id"]) for mg in microgrids)
        self.horizon = horizon
        self.ess_indices = []
        self.dr_indices = []
        self.supply_indices = []
        self.trade_indices = []
        self.dr_embeddings = nn.ModuleList()
        energy_norms = []
        ess_static = []
        for mg in microgrids:
            name = str(mg["id"])
            kind = str(mg.get("dr", {}).get("type", ""))
            if kind not in DR_SUFFIXES:
                raise ValueError(f"Unsupported DR type for {name}: {kind}")
            self.ess_indices.append(tuple(idx(f"{name}_{x}") for x in ESS_SUFFIXES))
            dr_fields = tuple(idx(f"{name}_{x}") for x in DR_SUFFIXES[kind])
            self.dr_indices.append(dr_fields)
            self.dr_embeddings.append(nn.Sequential(
                nn.Linear(len(dr_fields), embed_dim), nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            ))
            self.supply_indices.append(tuple(idx(f"{name}_{x}") for x in ("load", "pv", "wind")))
            buy = f"{name}_grid_buy_price" if f"{name}_grid_buy_price" in names else "grid_buy_price"
            sell = f"{name}_grid_sell_price" if f"{name}_grid_sell_price" in names else "grid_sell_price"
            self.trade_indices.append(tuple(idx(f"{name}_{x}") for x in TRADE_SUFFIXES) + (idx(buy), idx(sell)))
            norm = normalizers[name]
            energy_norms.append((
                float(norm["load"]), float(norm["pv"]), float(norm["wind"]),
                max(float(norm["power"]), 1e-6),
            ))
            battery = mg["battery"]
            ess_static.append((
                float(battery["capacity_kwh"]) / 1000.0,
                float(battery["max_charge_kw"]) / 1000.0,
                float(battery["max_discharge_kw"]) / 1000.0,
                float(battery["eta_charge"]),
                float(battery["eta_discharge"]),
            ))

        self.register_buffer("energy_norms", torch.tensor(energy_norms, dtype=torch.float32), persistent=False)
        self.register_buffer("ess_static", torch.tensor(ess_static, dtype=torch.float32), persistent=False)
        self.register_buffer("market_indices", torch.tensor([idx(x) for x in MARKET_NAMES], dtype=torch.long), persistent=False)
        self.register_buffer("hour_indices", torch.tensor([idx("hour_sin"), idx("hour_cos")], dtype=torch.long), persistent=False)

        future_index, temp_index, temp_present, horizon_id, mg_id = [], [], [], [], []
        for k in range(1, horizon + 1):
            for j, name in enumerate(self.ids):
                future_index.append((idx(f"{name}_net_load_t+{k}"), idx(f"{name}_price_t+{k}")))
                temperature = f"{name}_temperature_t+{k}"
                temp_present.append(float(temperature in names))
                temp_index.append(idx(temperature) if temperature in names else idx("hour_sin"))
                horizon_id.append(k)
                mg_id.append(j)
        self.register_buffer("future_indices", torch.tensor(future_index, dtype=torch.long), persistent=False)
        self.register_buffer("temp_indices", torch.tensor(temp_index, dtype=torch.long), persistent=False)
        self.register_buffer("temp_present", torch.tensor(temp_present, dtype=torch.float32), persistent=False)
        self.register_buffer("future_valid_indices", torch.tensor(
            [idx(f"forecast_valid_t+{k}") if f"forecast_valid_t+{k}" in names else -1
             for k in range(1, horizon + 1)], dtype=torch.long,
        ), persistent=False)

        type_ids = [x for _ in self.ids for x in range(4)] + [4] + [5] * (3 * horizon)
        mg_ids = [j for j in range(3) for _ in range(4)] + [3] + mg_id
        horizon_ids = [0] * 13 + horizon_id
        self.register_buffer("type_ids", torch.tensor(type_ids, dtype=torch.long), persistent=False)
        self.register_buffer("mg_ids", torch.tensor(mg_ids, dtype=torch.long), persistent=False)
        self.register_buffer("horizon_ids", torch.tensor(horizon_ids, dtype=torch.long), persistent=False)
        k = torch.arange(1, horizon + 1, dtype=torch.float32).repeat_interleave(3)
        self.register_buffer("future_sin", (k * (2 * math.pi / 24)).sin(), persistent=False)
        self.register_buffer("future_cos", (k * (2 * math.pi / 24)).cos(), persistent=False)

        self.ess_embedding = nn.Sequential(nn.Linear(len(ESS_SUFFIXES) + 5, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.supply_embedding = nn.Sequential(nn.Linear(4, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.trade_embedding = nn.Sequential(nn.Linear(6, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.market_embedding = nn.Sequential(nn.Linear(len(MARKET_NAMES), embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.future_embedding = nn.Sequential(nn.Linear(5, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.type_embedding = nn.Embedding(6, embed_dim)
        self.mg_embedding = nn.Embedding(4, embed_dim)
        self.horizon_embedding = nn.Embedding(horizon + 1, embed_dim)
        self.clock_embedding = nn.Linear(2, embed_dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.ess_query_embedding = nn.Linear(11, embed_dim)
        self.dr_query_embedding = nn.Linear(10, embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, heads, dropout=0.0, batch_first=True,
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        widths = [6 * embed_dim + self.state_dim + self.action_dim, *map(int, hidden_dims), 1]
        head_layers: list[nn.Module] = []
        for input_width, output_width in zip(widths[:-2], widths[1:-1]):
            head_layers.extend((nn.Linear(input_width, output_width), nn.ReLU()))
        head_layers.append(nn.Linear(widths[-2], 1))
        self.value_head = nn.Sequential(*head_layers)

    def _state_tokens(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        tokens = []
        for j in range(3):
            ess = state[:, self.ess_indices[j]]
            static = self.ess_static[j].expand(batch, -1)
            tokens.append(self.ess_embedding(torch.cat((ess, static), dim=-1)))
            tokens.append(self.dr_embeddings[j](state[:, self.dr_indices[j]]))
            supply = state[:, self.supply_indices[j]]
            norm = self.energy_norms[j]
            balance = (supply[:, 1] * norm[1] + supply[:, 2] * norm[2]
                       - supply[:, 0] * norm[0]) / norm[3]
            tokens.append(self.supply_embedding(torch.cat((supply, balance[:, None]), dim=-1)))
            tokens.append(self.trade_embedding(state[:, self.trade_indices[j]]))
        tokens.append(self.market_embedding(state.index_select(1, self.market_indices)))
        forecast = state.index_select(1, self.future_indices.flatten()).reshape(batch, -1, 2)
        temperature = state.index_select(1, self.temp_indices) * self.temp_present
        if bool((self.future_valid_indices >= 0).all()):
            valid = (state.index_select(1, self.future_valid_indices) > 0.5)
            valid = valid.repeat_interleave(3, dim=1)
        else:
            valid = torch.ones((batch, self.horizon * 3), device=state.device, dtype=torch.bool)
        features = torch.cat((
            forecast, temperature.unsqueeze(-1),
            self.temp_present.expand(batch, -1).unsqueeze(-1),
            valid.to(state.dtype).unsqueeze(-1),
        ), dim=-1)
        future_tokens = self.future_embedding(features)
        x = torch.cat((torch.stack(tokens, dim=1), future_tokens), dim=1)
        x = (x + self.type_embedding(self.type_ids)[None]
             + self.mg_embedding(self.mg_ids)[None]
             + self.horizon_embedding(self.horizon_ids)[None])
        hour = state.index_select(1, self.hour_indices)
        now = self.clock_embedding(hour)
        sin0, cos0 = hour[:, :1], hour[:, 1:]
        future_clock = torch.stack((
            sin0 * self.future_cos + cos0 * self.future_sin,
            cos0 * self.future_cos - sin0 * self.future_sin,
        ), dim=-1)
        x = x + torch.cat((
            now[:, None].expand(-1, 13, -1),
            self.clock_embedding(future_clock),
        ), dim=1)
        padding = torch.cat((
            torch.zeros((batch, 13), device=state.device, dtype=torch.bool),
            ~valid,
        ), dim=1)
        return x, padding

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"Expected state [B, {self.state_dim}]")
        if joint_action.ndim != 2 or joint_action.shape != (state.shape[0], self.action_dim):
            raise ValueError(f"Expected projected action [B, {self.action_dim}]")
        tokens, padding = self._state_tokens(state)
        encoded = self.state_encoder(tokens, src_key_padding_mask=padding)
        queries = []
        for j in range(3):
            start = j * 3
            queries.append(self.ess_query_embedding(torch.cat(
                (joint_action[:, start:start + 2], joint_action), dim=-1,
            )))
            queries.append(self.dr_query_embedding(torch.cat(
                (joint_action[:, start + 2:start + 3], joint_action), dim=-1,
            )))
        q = torch.stack(queries, dim=1)
        q = (q + self.mg_embedding(torch.arange(3, device=state.device).repeat_interleave(2))[None]
             + self.type_embedding(torch.tensor([0, 1] * 3, device=state.device))[None])
        readout, _ = self.cross_attention(q, encoded, encoded, key_padding_mask=padding, need_weights=False)
        context = self.query_norm(q + readout).flatten(1)
        return self.value_head(torch.cat((context, state, joint_action), dim=-1))


class TwinDeviceTransformerCritic(nn.Module):
    """Independent twin encoders, cross-attention blocks and scalar Q heads."""

    def __init__(self, schema, microgrids, normalizers, **network):
        super().__init__()
        self.q1_network = DeviceActionQ(schema, microgrids, normalizers, **network)
        self.q2_network = DeviceActionQ(schema, microgrids, normalizers, **network)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor):
        return self.q1_network(state, joint_action), self.q2_network(state, joint_action)

    def q1(self, state: torch.Tensor, joint_action: torch.Tensor):
        return self.q1_network(state, joint_action)
