from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class PlanOutput:
    actions: torch.Tensor
    soc_after: torch.Tensor
    ess_net_kw: torch.Tensor
    plan_mask: torch.Tensor
    attention: torch.Tensor | None
    forecast_mask: torch.Tensor
    current_soc: torch.Tensor


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed, heads, hidden):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed)
        self.memory_norm = nn.LayerNorm(embed)
        self.attention = nn.MultiheadAttention(embed, heads, dropout=0., batch_first=True)
        self.ff_norm = nn.LayerNorm(embed)
        self.ff = nn.Sequential(nn.Linear(embed, hidden), nn.GELU(), nn.Linear(hidden, embed))

    def forward(self, query, memory, padding, collect):
        norm_memory = self.memory_norm(memory)
        attended, weights = self.attention(self.query_norm(query), norm_memory, norm_memory,
            key_padding_mask=padding, need_weights=collect, average_attn_weights=False)
        query = query + attended
        return query + self.ff(self.ff_norm(query)), weights


class PlanTransformerActor(nn.Module):
    def __init__(self, schema, agent_ids, agent_id, mg_config, power_normalizers, network_config):
        super().__init__()
        self.schema = tuple(schema)
        self.agent_id = agent_id
        self.peers = tuple(mg for mg in agent_ids if mg != agent_id)
        net = network_config
        if int(net.get('forecast_horizon', 24)) != 24 or len(self.peers) != 2:
            raise ValueError('Plan-Transformer requires H24 and two peers')
        embed, heads = int(net.get('embed_dim', 32)), int(net.get('num_heads', 4))
        hidden, count = int(net.get('hidden_dim', 64)), int(net.get('num_layers', 2))
        if embed <= 0 or heads <= 0 or embed % heads or hidden <= 0 or count != 2:
            raise ValueError('Invalid network dimensions; two layers required')
        index = {name: i for i, name in enumerate(schema)}
        self.hour_indices = (index['hour_sin'], index['hour_cos'])
        self.soc_index = index['soc']
        now = [i for i, name in enumerate(schema) if '_t+' not in name and
               not any(name.startswith(peer + '_') for peer in self.peers)]
        self.register_buffer('now_indices', torch.tensor(now), persistent=False)
        self.register_buffer('supply_indices', torch.tensor([index[name] for name in ('load', 'pv', 'wind')]), persistent=False)
        future, temp_present = [], []
        for k in range(1, 25):
            names = [f'{name}_t+{k}' for name in ('load', 'pv', 'wind', 'price')]
            has_temp = f'temperature_t+{k}' in index
            future.append([index[name] for name in names] +
                          [index.get(f'temperature_t+{k}', index[f'load_t+{k}']),
                           index[f'forecast_valid_t+{k}']] +
                          [index[f'{peer}_energy_balance_t+{k}'] for peer in self.peers])
            temp_present.append(float(has_temp))
        self.register_buffer('future_indices', torch.tensor(future), persistent=False)
        self.register_buffer('temperature_present', torch.tensor(temp_present)[None, :], persistent=False)
        peer_features = ('energy_balance', 'deficit', 'surplus', 'charge_flex', 'discharge_flex', 'previous_p2p_net')
        self.register_buffer('peer_indices', torch.tensor([[index[f'{peer}_{f}'] for f in peer_features]
                                                          for peer in self.peers]), persistent=False)
        ratios = [float(power_normalizers[f]) / float(power_normalizers['power']) for f in ('load', 'pv', 'wind')]
        self.register_buffer('power_ratios', torch.tensor(ratios), persistent=False)
        angle = torch.arange(1, 25) * (2 * math.pi / 24)
        self.register_buffer('relative_time', torch.stack([angle.sin(), angle.cos()], -1)[None], persistent=False)
        encoder = lambda size: nn.Sequential(nn.Linear(size, hidden), nn.GELU(), nn.Linear(hidden, embed))
        self.now_encoder = encoder(len(now) + 2)
        self.future_encoder = encoder(10)
        self.peer_encoder = encoder(6)
        self.memory_type = nn.Parameter(torch.zeros(1, 26, embed))
        nn.init.normal_(self.memory_type, std=.02)
        self.layers = nn.ModuleList([CrossAttentionLayer(embed, heads, hidden) for _ in range(count)])
        self.output_norm = nn.LayerNorm(embed)
        self.action_head = nn.Sequential(nn.Linear(embed, hidden), nn.GELU(), nn.Linear(hidden, 2), nn.Tanh())
        self.plan_head = nn.Sequential(nn.Linear(embed, hidden), nn.GELU(), nn.Linear(hidden, 48))
        battery = mg_config['battery']
        self.soc_min, self.soc_max = float(battery['soc_min']), float(battery['soc_max'])
        self.charge_max, self.discharge_max = float(battery['max_charge_kw']), float(battery['max_discharge_kw'])

    def _future_features(self, obs):
        raw = obs[:, self.future_indices]
        valid = raw[:, :, 5] >= .5
        if not torch.isfinite(raw[:, :, 5]).all():
            raise ValueError('Validity flags must be finite')
        safe = torch.where(valid[:, :, None], raw, torch.zeros_like(raw))
        if not torch.isfinite(safe).all():
            raise ValueError('Valid forecast features must be finite')
        power = safe[:, :, :3] * self.power_ratios
        base = power[:, :, 0:1]
        net = base - power[:, :, 1:2] - power[:, :, 2:3]
        temperature = safe[:, :, 4:5] * self.temperature_present[:, :, None]
        features = torch.cat([base, net, safe[:, :, 3:4], temperature,
            self.temperature_present[:, :, None].expand(obs.shape[0], -1, -1),
            safe[:, :, 5:8], self.relative_time.expand(obs.shape[0], -1, -1)], -1)
        return torch.where(valid[:, :, None], features, torch.zeros_like(features))

    def forward_with_plan(self, obs, collect_attention=False):
        if obs.ndim != 2 or obs.shape[-1] != len(self.schema):
            raise ValueError('Invalid observation shape')
        now, peers = obs[:, self.now_indices], obs[:, self.peer_indices]
        if not torch.isfinite(now).all() or not torch.isfinite(peers).all():
            raise ValueError('Current observation must be finite')
        supply = obs[:, self.supply_indices] * self.power_ratios
        renewable = supply[:, 1:].sum(-1, keepdim=True)
        query = self.now_encoder(torch.cat([now, supply[:, :1] - renewable, renewable], -1))[:, None]
        future = self._future_features(obs)
        valid = obs[:, self.future_indices[:, 5]] >= .5
        memory = torch.cat([self.future_encoder(future), self.peer_encoder(peers)], dim=1) + self.memory_type
        padding = torch.cat([~valid, torch.zeros(obs.shape[0], 2, dtype=torch.bool, device=obs.device)], -1)
        attention = []
        for layer in self.layers:
            query, weights = layer(query, memory, padding, collect_attention)
            if collect_attention:
                attention.append(weights[:, :, 0])
        rep = self.output_norm(query[:, 0])
        plan = self.plan_head(rep).reshape(-1, 24, 2)
        soc = self.soc_min + (self.soc_max - self.soc_min) * plan[:, :, 0].sigmoid()
        power = plan[:, :, 1].tanh()
        power = torch.where(power >= 0, power * self.discharge_max, power * self.charge_max)
        hour = torch.remainder(torch.round(torch.atan2(obs[:, self.hour_indices[0]], obs[:, self.hour_indices[1]]) * 24 / (2 * math.pi)), 24)
        mask = torch.arange(24, device=obs.device)[None] + hour[:, None] < 24
        return PlanOutput(self.action_head(rep), soc, power, mask,
                          torch.stack(attention, dim=1) if collect_attention else None,
                          valid, obs[:, self.soc_index:self.soc_index + 1])

    def forward(self, obs):
        return self.forward_with_plan(obs).actions
