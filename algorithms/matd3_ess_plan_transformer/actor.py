from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .features import SemanticMlpActor
from .plan_projection import decode_nominal_plan


@dataclass
class EssPlanOutput:
    actions: torch.Tensor
    intents: torch.Tensor
    request_kw: torch.Tensor
    nominal_soc_after: torch.Tensor
    mask: torch.Tensor


class EssPlanActor(nn.Module):
    """Forecast-conditioned intent sequence; slot zero IS the ESS action."""

    def __init__(self, reference_actor, battery, delta_t=1.):
        super().__init__()
        self.schema = tuple(reference_actor.schema)
        self.battery, self.delta_t = dict(battery), float(delta_t)
        self.dr_actor = SemanticMlpActor(reference_actor)
        self.hidden_dim = self.dr_actor.hidden_dim
        self.soc_index = self.schema.index('soc')
        self.hour_indices = [self.schema.index(n) for n in ('hour_sin', 'hour_cos')]
        self.register_buffer('now_indices', torch.tensor([i for i,n in enumerate(self.schema) if '_t+' not in n]))
        embed = reference_actor.embed_dim
        layer = reference_actor.transformer.layers[0]
        heads = layer.self_attn.num_heads
        hidden = reference_actor.forecast_encoder[0].out_features
        self.now_encoder = nn.Sequential(nn.Linear(len(self.now_indices), hidden), nn.GELU(), nn.Linear(hidden, embed))
        self.future_encoder = nn.Sequential(nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, embed))
        self.peer_encoder = nn.Linear(reference_actor.peer_dim, embed)
        self.plan_queries = nn.Parameter(torch.randn(24, embed)*.02)
        self.reads = nn.ModuleList([nn.MultiheadAttention(embed, heads, dropout=0., batch_first=True)
                                  for _ in reference_actor.transformer.layers])
        self.norms = nn.ModuleList([nn.LayerNorm(embed) for _ in self.reads])
        self.ffns = nn.ModuleList([nn.Sequential(nn.Linear(embed, hidden), nn.GELU(), nn.Linear(hidden, embed))
                                  for _ in self.reads])
        self.intent_head = nn.Linear(embed, 1)

    @property
    def network(self):
        return self.dr_actor.network

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def _logits(self, obs):
        if obs.ndim != 2 or obs.shape[1] != len(self.schema):
            raise ValueError('Invalid observation shape')
        f = self.dr_actor.features
        now = obs.index_select(-1, self.now_indices)
        if not torch.isfinite(now).all():
            raise ValueError('Nonfinite current observation')
        raw = obs.index_select(-1, f.all_forecast_indices).reshape(-1, 24, 6)
        if not torch.isfinite(raw[:, :, 5]).all():
            raise ValueError('Nonfinite forecast validity')
        valid = raw[:, :, 5] >= .5
        hour_number = torch.remainder(torch.round(torch.atan2(obs[:, self.hour_indices[0]],
                          obs[:, self.hour_indices[1]])*24/(2*math.pi)),24)
        valid = valid & (torch.arange(1,25,device=obs.device)[None]+hour_number[:,None] < 24)
        safe = torch.where(valid[:, :, None], raw, torch.zeros_like(raw))
        peer_future = obs.index_select(-1, f.all_peer_forecast_indices).reshape(-1, 24, 2)
        peer_future = torch.where(valid[:, :, None], peer_future, torch.zeros_like(peer_future))
        if not torch.isfinite(safe).all() or not torch.isfinite(peer_future).all():
            raise ValueError('Nonfinite valid forecast')
        scales = f.power_normalizer_tensor
        net = (safe[:, :, 0]*scales[0]-safe[:, :, 1]*scales[1]-safe[:, :, 2]*scales[2])/scales[3]
        hour = obs[:, self.hour_indices]
        sin_k = hour[:, :1]*f.step_cos[:, :, 0]+hour[:, 1:]*f.step_sin[:, :, 0]
        cos_k = hour[:, 1:]*f.step_cos[:, :, 0]-hour[:, :1]*f.step_sin[:, :, 0]
        future = torch.cat([net[:, :, None], safe[:, :, 3:4], safe[:, :, 4:5]*f.temp_mask,
                            safe[:, :, 5:6], sin_k[:, :, None], cos_k[:, :, None],
                            peer_future, safe[:, :, :1]], -1)
        future = torch.where(valid[:, :, None], future, torch.zeros_like(future))
        rep = self.now_encoder(now)
        peer = obs.index_select(-1, f.all_peer_indices).reshape(len(obs), 2, -1)
        memory = torch.cat([rep[:, None], self.peer_encoder(peer), self.future_encoder(future)], 1)
        padding = torch.cat([torch.zeros(len(obs), 3, dtype=torch.bool, device=obs.device), ~valid], 1)
        query = rep[:, None]+self.plan_queries[None]
        for read, norm, ffn in zip(self.reads, self.norms, self.ffns):
            attended = read(query, memory, memory, key_padding_mask=padding, need_weights=False)[0]
            query = norm(query+attended)
            query = query+ffn(query)
        logits = self.intent_head(query).squeeze(-1)
        dr = self.dr_actor.action_logits(obs)[:, 1:2]
        return logits, dr

    def action_logits(self, observations):
        plan, dr = self._logits(observations)
        return torch.cat([plan[:, :1], dr], -1)

    def forward(self, observations, collect_diagnostics=False):
        # RL/inference need no 24-step SOC rollout; they use this same plan slot.
        return self.action_logits(observations).tanh()

    def forward_with_plan(self, observations, collect_diagnostics=False):
        logits, dr = self._logits(observations)
        intents = logits.tanh()
        hour = torch.remainder(torch.round(torch.atan2(observations[:, self.hour_indices[0]],
                  observations[:, self.hour_indices[1]])*24/(2*math.pi)), 24)
        mask = torch.arange(24, device=observations.device)[None]+hour[:, None] < 24
        request, soc = decode_nominal_plan(intents, observations[:, self.soc_index:self.soc_index+1],
                                         mask, self.battery, self.delta_t)
        return EssPlanOutput(torch.cat([intents[:, :1], dr.tanh()], -1), intents, request, soc, mask)
