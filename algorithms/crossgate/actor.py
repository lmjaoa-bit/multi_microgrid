from __future__ import annotations

from collections.abc import Mapping
import torch
from torch import nn


MARKET_FEATURES = frozenset({
    "previous_p2p_net", "previous_grid_net", "previous_system_p2p",
    "previous_ess_p2p_offer", "previous_ess_p2p_accepted", "previous_ess_p2p_unmatched",
    "previous_system_buy_demand", "previous_system_sell_offer", "previous_clearing_ratio",
})


class SingleDispatchActor(nn.Module):
    """One signed physical intent; local/peer gates are features, not extra actions."""

    def __init__(self, schema, agent_ids, active_agent, *, local_encoder_dim=32,
                 peer_encoder_dim=32, market_encoder_dim=16, hidden_dims=(96, 64),
                 gate_strength=.25):
        super().__init__()
        self.schema = tuple(schema)
        if len(set(self.schema)) != len(self.schema):
            raise ValueError("Duplicate observation features")
        peers = tuple(mg + "_" for mg in agent_ids if mg != active_agent)
        groups = {"local": [], "market": [], "peer": []}
        for i, name in enumerate(self.schema):
            key = "market" if name in MARKET_FEATURES else "peer" if name.startswith(peers) else "local"
            groups[key].append(i)
        if any(not group for group in groups.values()):
            raise ValueError("Single-dispatch actor requires local, market and peer observations")
        for name, indices in groups.items():
            self.register_buffer("_" + name + "_indices", torch.tensor(indices), persistent=False)
        def encoder(n, width):
            return nn.Sequential(nn.Linear(n, width), nn.LayerNorm(width), nn.ReLU())
        self.local_encoder = encoder(len(groups["local"]), local_encoder_dim)
        self.market_encoder = encoder(len(groups["market"]), market_encoder_dim)
        self.peer_encoder = encoder(len(groups["peer"]), peer_encoder_dim)
        self.local_gate = nn.Linear(peer_encoder_dim + market_encoder_dim, local_encoder_dim)
        self.peer_gate = nn.Linear(local_encoder_dim + market_encoder_dim, peer_encoder_dim)
        for gate in (self.local_gate, self.peer_gate):
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.gate_strength = float(gate_strength)
        if not 0 < self.gate_strength <= 1:
            raise ValueError("gate_strength must be in (0, 1]")
        width = local_encoder_dim + market_encoder_dim + peer_encoder_dim
        layers = []
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()])
            width = hidden
        self.dispatch_backbone = nn.Sequential(*layers)
        self.dispatch_head = nn.Sequential(nn.Linear(width, 1), nn.Tanh())
        self._parameter_count = sum(p.numel() for p in self.parameters())
        self._gate_values = None
        self._gate_metrics = {}

    @property
    def parameter_count(self):
        return self._parameter_count

    def forward(self, observation):
        if observation.shape[-1] != len(self.schema):
            raise ValueError("Single-dispatch observation schema mismatch")
        local = self.local_encoder(observation.index_select(-1, self._local_indices))
        market = self.market_encoder(observation.index_select(-1, self._market_indices))
        peer = self.peer_encoder(observation.index_select(-1, self._peer_indices))
        lg = torch.sigmoid(self.local_gate(torch.cat([peer, market], -1)))
        pg = torch.sigmoid(self.peer_gate(torch.cat([local, market], -1)))
        local = local * (1 + self.gate_strength * (2 * lg - 1))
        peer = peer * (1 + self.gate_strength * (2 * pg - 1))
        self._gate_values = (lg.detach(), pg.detach())
        return self.dispatch_head(self.dispatch_backbone(torch.cat([local, market, peer], -1)))

    def gate_diagnostics(self) -> Mapping[str, float]:
        if self._gate_values is not None:
            self._gate_metrics = {}
            for name, gate in zip(("local_gate", "peer_gate"), self._gate_values):
                self._gate_metrics.update({
                    name + "_mean": float(gate.mean()),
                    name + "_std": float(gate.std(unbiased=False)),
                    name + "_low_rate": float((gate < .1).float().mean()),
                    name + "_high_rate": float((gate > .9).float().mean()),
                })
            self._gate_values = None
        return dict(self._gate_metrics)

    def branch_modules(self):
        return {
            "local_features": (self.local_encoder,),
            "market_features": (self.market_encoder,),
            "peer_features": (self.peer_encoder,),
            "gates": tuple(m for m in (self.local_gate, self.peer_gate) if m is not None),
            "dispatch": (self.dispatch_backbone, self.dispatch_head),
        }
