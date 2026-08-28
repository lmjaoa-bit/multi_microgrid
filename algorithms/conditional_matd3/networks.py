from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dims: Iterable[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for width in hidden_dims:
        width = int(width)
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    return nn.Sequential(*layers)


class ConditionalActor(nn.Module):
    """Actor with three continuous heads and one binary commitment logit head."""

    action_dim = 4

    def __init__(self, obs_dim: int, hidden_dims: Iterable[int] = (256, 256, 128)) -> None:
        super().__init__()
        hidden_dims = tuple(int(x) for x in hidden_dims)
        if not hidden_dims:
            raise ValueError("Actor hidden_dims cannot be empty.")
        self.encoder = _mlp(obs_dim, hidden_dims)
        latent = hidden_dims[-1]
        self.battery_head = nn.Linear(latent, 1)
        self.generator_setpoint_head = nn.Linear(latent, 1)
        self.commitment_logit_head = nn.Linear(latent, 1)
        self.p2p_bid_head = nn.Linear(latent, 1)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(obs)
        return {
            "battery": torch.tanh(self.battery_head(h)),
            "generator_setpoint": torch.tanh(self.generator_setpoint_head(h)),
            "commitment_logit": self.commitment_logit_head(h),
            "p2p_bid": torch.tanh(self.p2p_bid_head(h)),
        }

    @staticmethod
    def commitment_probability(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(outputs["commitment_logit"])

    @staticmethod
    def hard_action(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        probability = torch.sigmoid(outputs["commitment_logit"])
        commitment = torch.where(probability >= 0.5, torch.ones_like(probability), -torch.ones_like(probability))
        setpoint = torch.where(commitment > 0.0, outputs["generator_setpoint"], -torch.ones_like(outputs["generator_setpoint"]))
        return torch.cat([outputs["battery"], commitment, setpoint, outputs["p2p_bid"]], dim=-1)


class CentralizedCritic(nn.Module):
    """Centralized Q(s, a_1, ..., a_N) for one microgrid's return."""

    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dims: Iterable[int] = (256, 256, 128)) -> None:
        super().__init__()
        hidden_dims = tuple(int(x) for x in hidden_dims)
        if not hidden_dims:
            raise ValueError("Critic hidden_dims cannot be empty.")
        self.net = _mlp(int(state_dim) + int(joint_action_dim), hidden_dims)
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, joint_action], dim=-1)
        return self.output(self.net(x))
