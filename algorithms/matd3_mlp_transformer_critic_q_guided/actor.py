from __future__ import annotations

import torch
from torch import nn


class SemanticH8Features(nn.Module):
    """Fixed inputs of the ordinary T encoders, flattened without learned attention."""

    def __init__(self, reference_actor):
        super().__init__()
        self.schema = reference_actor.schema
        self.forecast_horizon = reference_actor.forecast_horizon
        self.dr_type = reference_actor.dr_type
        self.ev_cohort_mode = reference_actor.ev_cohort_mode
        self.num_peers = reference_actor.num_peers
        for name in (
            "ess_idx_tensor", "dr_idx_tensor", "supply_idx_tensor", "trade_idx_tensor",
            "all_peer_indices", "all_forecast_indices", "all_peer_forecast_indices",
            "hour_idx_tensor", "power_normalizer_tensor",
            "battery_static", "temp_mask", "step_sin", "step_cos",
        ):
            self.register_buffer(name, getattr(reference_actor, name).detach().clone())
        self.sys_index = self.schema.index("previous_system_p2p")
        self.register_buffer("forecast_load_indices", torch.tensor(
            [self.schema.index(f"load_t+{k}") for k in range(1, 9)],
            dtype=torch.long, device=reference_actor.ess_idx_tensor.device,
        ))

    def forward(self, obs):
        batch = obs.shape[0]
        ess = obs.index_select(-1, self.ess_idx_tensor)
        dr = obs.index_select(-1, self.dr_idx_tensor)
        if self.dr_type == "ev_charging" and not self.ev_cohort_mode:
            dr = torch.cat([dr, (dr[:, 2:3] / dr[:, 3:4].clamp_min(1.)) / 100.], -1)
        supply = obs.index_select(-1, self.supply_idx_tensor)
        scales = self.power_normalizer_tensor
        balance = -(supply[:, 0:1] * scales[0] - supply[:, 1:2] * scales[1]
                    - supply[:, 2:3] * scales[2]) / scales[3]
        supply = torch.cat([supply, balance, torch.sign(balance)], -1)
        trade = obs.index_select(-1, self.trade_idx_tensor)
        trade = torch.cat([trade, 1. - obs[:, self.sys_index:self.sys_index + 1].clamp(0., 1.)], -1)
        peer = obs.index_select(-1, self.all_peer_indices)
        raw = obs.index_select(-1, self.all_forecast_indices).reshape(batch, self.forecast_horizon, 6)
        raw = torch.nan_to_num(raw, nan=0., posinf=0., neginf=0.)
        valid = raw[:, :, 5:6].clamp(0., 1.)
        net = (raw[:, :, 0:1] * scales[0] - raw[:, :, 1:2] * scales[1]
               - raw[:, :, 2:3] * scales[2]) / scales[3]
        hour = obs.index_select(-1, self.hour_idx_tensor)
        sin0, cos0 = hour[:, 0:1, None], hour[:, 1:2, None]
        sin_k = sin0 * self.step_cos + cos0 * self.step_sin
        cos_k = cos0 * self.step_cos - sin0 * self.step_sin
        peer_future = obs.index_select(-1, self.all_peer_forecast_indices).reshape(
            batch, self.forecast_horizon, self.num_peers
        )
        peer_future = torch.nan_to_num(peer_future, nan=0., posinf=0., neginf=0.)
        base_load = obs.index_select(-1, self.forecast_load_indices).unsqueeze(-1)
        base_load = torch.nan_to_num(base_load, nan=0., posinf=0., neginf=0.)
        forecast = torch.cat([
            net, raw[:, :, 3:4], raw[:, :, 4:5] * self.temp_mask,
            valid, sin_k, cos_k, peer_future, base_load,
        ], -1)
        forecast = torch.where((valid >= .5).expand_as(forecast), forecast, torch.zeros_like(forecast))
        return torch.cat([
            ess, self.battery_static[None].expand(batch, -1), dr, supply, trade, peer,
            forecast.flatten(1),
        ], -1)


class SemanticMlpActor(nn.Module):
    """Two-layer MLP on the same H8 non-learning features as ordinary T."""

    def __init__(self, reference_actor):
        super().__init__()
        self.schema = reference_actor.schema
        self.features = SemanticH8Features(reference_actor)
        self.forecast_horizon = self.features.forecast_horizon
        device = reference_actor.ess_idx_tensor.device
        with torch.no_grad():
            input_dim = self.features(torch.zeros(1, len(self.schema), device=device)).shape[-1]
        target_count = sum(p.numel() for p in reference_actor.parameters())
        # Match the diagnosed T-Q device-readout budget without depending on its
        # removable experimental package: two MHA reads, two norms, appended head
        # inputs, and the retained base-load forecast channel.
        embed = reference_actor.embed_dim
        hidden = reference_actor.forecast_encoder[0].out_features
        target_count += 8 * embed * embed + 12 * embed + 2 * hidden * embed + hidden
        # Two hidden layers and two logits: w^2 + (input_dim + 4)w + 2 parameters.
        self.hidden_dim = min(range(8, 513), key=lambda w: abs(w * w + (input_dim + 4) * w + 2 - target_count))
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def action_logits(self, observations):
        if observations.ndim != 2 or observations.shape[-1] != len(self.schema):
            raise ValueError(f"Expected observation shape [batch, {len(self.schema)}]")
        return self.network(self.features(observations))

    def forward(self, observations, collect_diagnostics=False):
        return self.action_logits(observations).tanh()
