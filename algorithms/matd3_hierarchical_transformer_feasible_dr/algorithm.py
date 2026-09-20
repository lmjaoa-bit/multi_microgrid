from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer.algorithm import Matd3HierarchicalTransformer
from algorithms.matd3_hierarchical_transformer_feasible_dr.actor import FeasibleDRActor


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
