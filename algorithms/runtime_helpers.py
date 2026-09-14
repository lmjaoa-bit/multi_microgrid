from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn


def diagnostics_due(update_steps: int, interval: int) -> bool:
    """Return True only on configured diagnostic-sampling updates.

    interval <= 0 disables expensive diagnostics entirely.
    """
    return int(interval) > 0 and int(update_steps) % int(interval) == 0


def forward_policy_actions(
    actors: Mapping[str, nn.Module],
    observations: Mapping[str, torch.Tensor],
    agent_ids: Sequence[str],
    *,
    collect_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Forward every actor exactly once for one policy-update batch."""
    return {
        mg: actors[mg](observations[mg], collect_diagnostics=collect_diagnostics)
        for mg in agent_ids
    }


def select_single_dispatch_actions(
    algorithm,
    observations: Mapping[str, np.ndarray],
    noise_std: float = 0.0,
) -> dict[str, np.ndarray]:
    """Experimental single-dispatch rollout with one GPU->CPU synchronization.

    Actor forwards stay on-device until every agent action has been computed. Diagnostics are
    explicitly disabled on this hot path; algorithm updates sample them separately.
    """
    tensors: list[torch.Tensor] = []
    with torch.no_grad():
        for mg in algorithm.agent_ids:
            obs = torch.as_tensor(
                observations[mg], dtype=torch.float32, device=algorithm.device
            ).unsqueeze(0)
            tensors.append(algorithm.actors[mg](obs, collect_diagnostics=False))
        action_matrix = torch.cat(tensors, dim=0).detach().cpu().numpy()

    actions: dict[str, np.ndarray] = {}
    for index, mg in enumerate(algorithm.agent_ids):
        action = action_matrix[index].copy()
        if noise_std > 0.0:
            action += algorithm.rng.normal(0.0, noise_std, size=action.shape)
        actions[mg] = np.clip(action, -1.0, 1.0).astype(np.float32)
    return actions
