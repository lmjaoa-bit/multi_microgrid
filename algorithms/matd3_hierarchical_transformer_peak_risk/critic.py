from __future__ import annotations

import torch
from torch import nn


def _head(width: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    modules: list[nn.Module] = []
    for hidden in hidden_dims:
        modules.extend((nn.Linear(width, hidden), nn.GELU()))
        width = hidden
    modules.extend((nn.Linear(width, 1), nn.Softplus()))
    return nn.Sequential(*modules)


class TwinPeakCritic(nn.Module):
    """Exactly one shared, lightweight, twin cost critic for all microgrids."""

    def __init__(self, global_state_dim: int, joint_action_dim: int,
                 hidden_dims: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        if not hidden_dims or any(int(v) <= 0 for v in hidden_dims):
            raise ValueError("peak critic hidden_dims must be positive")
        width = int(global_state_dim) + int(joint_action_dim)
        self.peak1_network = _head(width, tuple(hidden_dims))
        self.peak2_network = _head(width, tuple(hidden_dims))

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([state, joint_action], dim=-1)
        return self.peak1_network(inputs), self.peak2_network(inputs)

    def peak1(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        return self.peak1_network(torch.cat([state, joint_action], dim=-1))
