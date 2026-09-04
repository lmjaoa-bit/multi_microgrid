from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


MARKET_FEATURES = frozenset({
    "previous_p2p_net",
    "previous_grid_net",
    "previous_system_p2p",
    "previous_ess_p2p_offer",
    "previous_ess_p2p_accepted",
    "previous_ess_p2p_unmatched",
    "previous_system_buy_demand",
    "previous_system_sell_offer",
    "previous_clearing_ratio",
})


def feature_group_indices(
    schema: Sequence[str],
    agent_ids: Sequence[str],
    active_agent: str,
) -> dict[str, tuple[int, ...]]:
    """Partition a public observation into local, market, and peer features."""
    names = tuple(str(name) for name in schema)
    if len(names) != len(set(names)):
        raise ValueError(f"{active_agent}: observation schema contains duplicate names")
    peers = tuple(str(agent) for agent in agent_ids if str(agent) != str(active_agent))
    peer_prefixes = tuple(f"{peer}_" for peer in peers)
    groups: dict[str, list[int]] = {"local": [], "market": [], "peer": []}
    for index, name in enumerate(names):
        if name in MARKET_FEATURES:
            groups["market"].append(index)
        elif peer_prefixes and name.startswith(peer_prefixes):
            groups["peer"].append(index)
        else:
            groups["local"].append(index)
    if any(not groups[name] for name in groups):
        sizes = {name: len(indices) for name, indices in groups.items()}
        raise ValueError(f"{active_agent}: CROSSGATE requires non-empty feature groups, got {sizes}")
    partition = sorted(index for indices in groups.values() for index in indices)
    if partition != list(range(len(names))):
        raise RuntimeError(f"{active_agent}: CROSSGATE feature partition is incomplete")
    return {name: tuple(indices) for name, indices in groups.items()}


def _encoder(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.ReLU(),
    )


def _fusion(input_dim: int, hidden_dims: Sequence[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    width = int(input_dim)
    for hidden in hidden_dims:
        hidden = int(hidden)
        layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()])
        width = hidden
    return nn.Sequential(*layers), width


class CrossGateActor(nn.Module):
    """Local battery controller plus a CrossGate-conditioned ESS offer head.

    The battery action is produced only from local features.  The offer branch
    receives its own local representation, market/clearing context, peer
    context, and the already-computed battery action.  Detaching that action in
    the offer branch prevents the offer loss path from destabilising the local
    battery controller while preserving the sequential conditioning signal.
    """

    def __init__(
        self,
        observation_schema: Sequence[str],
        agent_ids: Sequence[str],
        active_agent: str,
        action_dim: int,
        *,
        local_encoder_dim: int = 32,
        peer_encoder_dim: int = 32,
        market_encoder_dim: int = 16,
        battery_hidden_dims: Sequence[int] = (64,),
        offer_hidden_dims: Sequence[int] = (96, 64),
        gate_strength: float = 0.25,
        detach_battery_for_offer: bool = True,
    ) -> None:
        super().__init__()
        if int(action_dim) != 2:
            raise ValueError("CROSSGATE offer actor requires exactly two actions")
        if local_encoder_dim <= 0 or peer_encoder_dim <= 0 or market_encoder_dim <= 0:
            raise ValueError("CROSSGATE encoder dimensions must be positive")
        if not battery_hidden_dims or any(int(width) <= 0 for width in battery_hidden_dims):
            raise ValueError("CROSSGATE battery_hidden_dims must contain positive widths")
        if not offer_hidden_dims or any(int(width) <= 0 for width in offer_hidden_dims):
            raise ValueError("CROSSGATE offer_hidden_dims must contain positive widths")
        if not 0.0 < float(gate_strength) <= 1.0:
            raise ValueError("CROSSGATE gate_strength must be in (0, 1]")

        self.active_agent = str(active_agent)
        self.observation_schema = tuple(str(name) for name in observation_schema)
        self.gate_strength = float(gate_strength)
        self.detach_battery_for_offer = bool(detach_battery_for_offer)
        groups = feature_group_indices(self.observation_schema, agent_ids, self.active_agent)
        self.feature_groups = groups
        self.register_buffer(
            "_local_indices", torch.tensor(groups["local"], dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "_market_indices", torch.tensor(groups["market"], dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "_peer_indices", torch.tensor(groups["peer"], dtype=torch.long), persistent=False,
        )

        # Do not share the local encoder between the two heads.  This keeps
        # CrossGate/offer gradients out of the battery representation.
        self.battery_local_encoder = _encoder(len(groups["local"]), int(local_encoder_dim))
        self.offer_local_encoder = _encoder(len(groups["local"]), int(local_encoder_dim))
        self.market_encoder = _encoder(len(groups["market"]), int(market_encoder_dim))
        self.peer_encoder = _encoder(len(groups["peer"]), int(peer_encoder_dim))
        self.local_gate = nn.Linear(
            int(peer_encoder_dim) + int(market_encoder_dim), int(local_encoder_dim),
        )
        self.peer_gate = nn.Linear(
            int(local_encoder_dim) + int(market_encoder_dim), int(peer_encoder_dim),
        )
        # A zero-logit gate starts at 0.5. With centered residual gating this is
        # exactly the identity map while retaining non-zero gate gradients.
        nn.init.zeros_(self.local_gate.weight)
        nn.init.zeros_(self.local_gate.bias)
        nn.init.zeros_(self.peer_gate.weight)
        nn.init.zeros_(self.peer_gate.bias)

        self.battery_backbone, battery_output = _fusion(
            int(local_encoder_dim), battery_hidden_dims,
        )
        self.battery_head = nn.Sequential(nn.Linear(battery_output, 1), nn.Tanh())

        offer_input = (
            int(local_encoder_dim) + int(market_encoder_dim) + int(peer_encoder_dim) + 1
        )
        self.offer_backbone, offer_output = _fusion(offer_input, offer_hidden_dims)
        self.offer_head = nn.Sequential(nn.Linear(offer_output, 1), nn.Tanh())
        self._latest_gate_diagnostics: dict[str, float] = {}

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _gate_statistics(prefix: str, gate: torch.Tensor) -> dict[str, float]:
        detached = gate.detach()
        return {
            f"{prefix}_mean": float(detached.mean().item()),
            f"{prefix}_std": float(detached.std(unbiased=False).item()),
            f"{prefix}_low_rate": float((detached < 0.1).float().mean().item()),
            f"{prefix}_high_rate": float((detached > 0.9).float().mean().item()),
        }

    def gate_diagnostics(self) -> Mapping[str, float]:
        return dict(self._latest_gate_diagnostics)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != len(self.observation_schema):
            raise ValueError(
                f"{self.active_agent}: expected observation width {len(self.observation_schema)}, "
                f"got {observation.shape[-1]}"
            )
        local_observation = observation.index_select(-1, self._local_indices)
        battery_local = self.battery_local_encoder(local_observation)
        battery_action = self.battery_head(self.battery_backbone(battery_local))

        local = self.offer_local_encoder(local_observation)
        market = self.market_encoder(observation.index_select(-1, self._market_indices))
        peer = self.peer_encoder(observation.index_select(-1, self._peer_indices))

        local_gate = torch.sigmoid(self.local_gate(torch.cat([peer, market], dim=-1)))
        peer_gate = torch.sigmoid(self.peer_gate(torch.cat([local, market], dim=-1)))
        local_crossed = local * (1.0 + self.gate_strength * (2.0 * local_gate - 1.0))
        peer_crossed = peer * (1.0 + self.gate_strength * (2.0 * peer_gate - 1.0))
        battery_context = battery_action.detach() if self.detach_battery_for_offer else battery_action
        offer_features = torch.cat(
            [local_crossed, market, peer_crossed, battery_context], dim=-1,
        )
        offer_action = self.offer_head(self.offer_backbone(offer_features))

        self._latest_gate_diagnostics = {
            **self._gate_statistics("local_gate", local_gate),
            **self._gate_statistics("peer_gate", peer_gate),
            "battery_action_mean": float(battery_action.detach().mean().item()),
            "offer_action_mean": float(offer_action.detach().mean().item()),
        }
        return torch.cat([battery_action, offer_action], dim=-1)
