from __future__ import annotations

import torch
from torch import nn


class Actor(nn.Module):
    """Standard deterministic actor with bounded continuous actions."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = observation_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.extend([nn.Linear(width, action_dim), nn.Tanh()])
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)
