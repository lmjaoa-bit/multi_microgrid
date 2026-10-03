from __future__ import annotations

import re

import torch
from torch import nn

LEAD = re.compile(r"_t\+(\d+)$")


class TemporalValue(nn.Module):
    """Current state/action token plus lossless H24 global forecast tokens."""

    def __init__(self, schema, action_dim=9, *, forecast_horizon=24, embed_dim=48,
                 num_heads=4, num_layers=2, feedforward_dim=128):
        super().__init__()
        if forecast_horizon != 24:
            raise ValueError("Transformer Critic requires forecast_horizon=24 (H24)")
        if min(embed_dim, num_heads, num_layers, feedforward_dim) <= 0 or embed_dim % num_heads:
            raise ValueError("Critic dimensions must be positive and embed_dim divisible by num_heads")
        if len(set(schema)) != len(schema):
            raise ValueError("Critic state schema has duplicate features")
        current, future = [], {}
        for index, name in enumerate(schema):
            match = LEAD.search(name)
            if match:
                future.setdefault(int(match.group(1)), []).append(index)
            else:
                current.append(index)
        if sorted(future) != list(range(1, 25)):
            raise ValueError("Critic state schema must provide the complete H24 horizon")
        channels = [[LEAD.sub("", schema[i]) for i in future[k]] for k in range(1, 25)]
        if any(names != channels[0] for names in channels[1:]):
            raise ValueError("Forecast tokens must have identical ordered channels")
        if "forecast_valid" not in channels[0]:
            raise ValueError("Critic requires forecast validity masks")
        self.validity_column = channels[0].index("forecast_valid")
        self.state_dim = len(schema)
        self.action_dim = action_dim
        self.register_buffer("current_indices", torch.tensor(current, dtype=torch.long))
        self.register_buffer("forecast_indices", torch.tensor([future[k] for k in range(1, 25)], dtype=torch.long))
        self.current = nn.Sequential(nn.Linear(len(current) + action_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.forecast = nn.Linear(len(channels[0]), embed_dim)
        self.position = nn.Parameter(torch.empty(24, embed_dim))
        nn.init.normal_(self.position, std=.02)
        block = nn.TransformerEncoderLayer(
            embed_dim, num_heads, feedforward_dim, dropout=0., activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(embed_dim * 2, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, state, joint_action):
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state shape [batch, {self.state_dim}]")
        if joint_action.shape != (state.shape[0], self.action_dim):
            raise ValueError(f"Expected effective joint action shape [batch, {self.action_dim}]")
        current = self.current(torch.cat([state.index_select(-1, self.current_indices), joint_action], -1))
        forecast = state[:, self.forecast_indices]
        valid = torch.nan_to_num(forecast[:, :, self.validity_column], nan=0., posinf=0., neginf=0.) >= .5
        forecast = torch.where(valid[:, :, None], forecast, torch.zeros_like(forecast))
        forecast = torch.nan_to_num(forecast, nan=0., posinf=0., neginf=0.)
        tokens = torch.cat([current[:, None], self.forecast(forecast) + self.position[None]], 1)
        padding = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], 1)
        output = self.transformer(tokens, src_key_padding_mask=padding)
        # The current/action token is always visible, including the last hour.
        pooled = torch.where(valid[:, :, None], output[:, 1:], torch.zeros_like(output[:, 1:])).sum(1)
        pooled = pooled / valid.sum(1, keepdim=True).clamp_min(1)
        return self.head(torch.cat([output[:, 0], pooled], -1))


class TransformerTwinCritic(nn.Module):
    """Independent Q1 and Q2 encoders, preserving the TD3 critic interface."""

    def __init__(self, schema, action_dim=9, **settings):
        super().__init__()
        self.q1_network = TemporalValue(schema, action_dim, **settings)
        self.q2_network = TemporalValue(schema, action_dim, **settings)

    def forward(self, state, joint_action):
        return self.q1_network(state, joint_action), self.q2_network(state, joint_action)

    def q1(self, state, joint_action):
        return self.q1_network(state, joint_action)
