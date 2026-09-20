from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithms.diagnostic_sampling import DiagnosticSchedule
from algorithms.matd3_hierarchical_transformer.algorithm import Matd3HierarchicalTransformer
from algorithms.matd3_hierarchical_transformer_feasible_dr.actor import FeasibleDRActor
from algorithms.matd3_hierarchical_transformer_feasible_dr.training import (
    select_feasible_actions,
    update_feasible_dr,
)
from algorithms.runtime_helpers import FastReplayBuffer


class Matd3HierarchicalTransformerFeasibleDR(Matd3HierarchicalTransformer):
    """Hierarchical Transformer with feasible MG2/MG3 DR actions."""

    action_semantics = "single_dispatch_dr_feasible_v1"

    def __init__(
        self,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        super().__init__(env, config, device=device, seed=seed)

        settings = config.get("algorithm", config)
        self.feasible_dr_config = copy.deepcopy(settings.get("feasible_dr", {}))

        net = settings.get("network", {})
        embed_dim = int(net.get("embed_dim", 32))
        num_heads = int(net.get("num_heads", 4))
        num_layers = int(net.get("num_layers", 2))
        forecast_horizon = int(net.get("forecast_horizon", 8))
        hidden_dim = int(net.get("hidden_dim", 64))
        delta_t = float(getattr(env, "delta_t", 1.0))
        episode_length = int(getattr(env, "episode_length", 24))

        torch.manual_seed(seed)
        self.actors = nn.ModuleDict({
            mg: FeasibleDRActor(
                self.observation_schemas[mg],
                self.agent_ids,
                mg,
                mg_config=env.cfg_by_id[mg] if hasattr(env, "cfg_by_id") else None,
                feasible_config=self.feasible_dr_config,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                forecast_horizon=forecast_horizon,
                hidden_dim=hidden_dim,
                delta_t=delta_t,
                episode_length=episode_length,
            )
            for mg in self.agent_ids
        }).to(self.device)

        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.actor_optimizers = {
            mg: torch.optim.Adam(self.actors[mg].parameters(), lr=float(settings.get("actor_lr", 2e-4)))
            for mg in self.agent_ids
        }
        self._actor_parameters = {mg: tuple(actor.parameters()) for mg, actor in self.actors.items()}
        self._actor_branch_parameters = {
            mg: {
                name: tuple(
                    p
                    for item in items
                    for p in (item.parameters() if isinstance(item, nn.Module) else [item])
                )
                for name, items in actor.branch_modules().items()
            }
            for mg, actor in self.actors.items()
        }
        self._actor_layer_parameters = {
            mg: {
                f"layer_{i}": tuple(layer.parameters())
                for i, layer in enumerate(actor.transformer.layers)
            }
            for mg, actor in self.actors.items()
        }

        # Sparse diagnostic schedule & fast replay buffer (without post-mapping noise injection)
        logging_cfg = config.get("logging", {}) if isinstance(config, Mapping) else {}
        interval = logging_cfg.get("diagnostic_interval", 20)
        self.diagnostic_schedule = DiagnosticSchedule(interval, self.policy_delay)
        self._diagnostic_plain_actor = False
        self.replay = FastReplayBuffer.from_replay(self.replay)

    def select_actions(self, observations: Mapping[str, Any], noise_std: float = 0.0) -> dict[str, np.ndarray]:
        return select_feasible_actions(self, observations, noise_std=noise_std)

    def update(self, batch_size: int) -> dict[str, Any]:
        return update_feasible_dr(self, batch_size)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "algorithm_slug": "matd3_hierarchical_transformer_feasible_dr",
                "action_semantics": self.action_semantics,
                "projection_signature": self.projection_signature,
                "feasible_dr_config": self.feasible_dr_config,
                "observation_schemas": self.observation_schemas,
                "state_schema": self.state_schema,
                "actors": self.actors.state_dict(),
                "target_actors": self.target_actors.state_dict(),
                "critics": self.critics.state_dict(),
                "target_critics": self.target_critics.state_dict(),
                "update_steps": self.update_steps,
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        slug = checkpoint.get("algorithm_slug")
        if slug != "matd3_hierarchical_transformer_feasible_dr":
            raise ValueError(
                f"Incompatible algorithm checkpoint: expected matd3_hierarchical_transformer_feasible_dr, got {slug}"
            )
        if checkpoint.get("action_semantics") != self.action_semantics:
            raise ValueError("Incompatible action semantics")
        if checkpoint.get("projection_signature") != self.projection_signature:
            raise ValueError("Projection signature mismatch")

        ckpt_feasible_cfg = checkpoint.get("feasible_dr_config")
        if ckpt_feasible_cfg != self.feasible_dr_config:
            raise ValueError(
                f"Incompatible feasible DR configuration: expected {self.feasible_dr_config}, got {ckpt_feasible_cfg}"
            )

        self.actors.load_state_dict(checkpoint["actors"])
        self.target_actors.load_state_dict(checkpoint["target_actors"])
        self.critics.load_state_dict(checkpoint["critics"])
        self.target_critics.load_state_dict(checkpoint["target_critics"])
        self.update_steps = int(checkpoint.get("update_steps", 0))

