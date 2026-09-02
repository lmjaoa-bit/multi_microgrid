from __future__ import annotations

import torch
from torch import nn


def _q_network(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(width, hidden), nn.ReLU()])
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class TwinCritic(nn.Module):
    """Twin centralized critics used by TD3 to reduce overestimation."""

    def __init__(self, global_state_dim: int, joint_action_dim: int,
                 hidden_dims: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        input_dim = global_state_dim + joint_action_dim
        self.q1_network = _q_network(input_dim, hidden_dims)
        self.q2_network = _q_network(input_dim, hidden_dims)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([state, joint_action], dim=-1)
        return self.q1_network(inputs), self.q2_network(inputs)

    def q1(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        return self.q1_network(torch.cat([state, joint_action], dim=-1))
