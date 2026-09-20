from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer_feasible_dr.algorithm import (
    Matd3HierarchicalTransformerFeasibleDR,
)
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.actor import AdaptiveDRActor
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
    validate_adaptive_dr_config,
)
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.training import (
    select_adaptive_actions,
    update_adaptive_dr,
)


class Matd3HierarchicalTransformerFeasibleDRV2(Matd3HierarchicalTransformerFeasibleDR):
    """V2 adaptive-reference feasible DR Transformer."""

    action_semantics = "single_dispatch_dr_adaptive_reference_v2"

    def __init__(
        self,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        settings = config.get("algorithm", config)
        self.adaptive_dr_config = copy.deepcopy(settings.get("adaptive_dr", {}))

        net = settings.get("network", {})
        embed_dim = int(net.get("embed_dim", 32))
        num_heads = int(net.get("num_heads", 4))
        num_layers = int(net.get("num_layers", 2))
        forecast_horizon = int(net.get("forecast_horizon", 8))
        hidden_dim = int(net.get("hidden_dim", 64))
        delta_t = float(getattr(env, "delta_t", 1.0))
        episode_length = int(getattr(env, "episode_length", 24))
        episode_horizon_hours = episode_length * delta_t

        ev_max_steps = 7
        if hasattr(env, "cfg_by_id") and "mg3" in env.cfg_by_id:
            mg3_cfg = env.cfg_by_id["mg3"]
            dr_spec = mg3_cfg.get("dr", {}) if isinstance(mg3_cfg, dict) else getattr(mg3_cfg, "dr", {})
            if isinstance(dr_spec, dict):
                dep = dr_spec.get("departure_hour", 7)
                arr = dr_spec.get("available_start_hour", 0)
                ev_max_steps = int(dep - arr)
            elif hasattr(dr_spec, "departure_hour"):
                dep = getattr(dr_spec, "departure_hour", 7)
                arr = getattr(dr_spec, "available_start_hour", 0)
                ev_max_steps = int(dep - arr)

        validate_adaptive_dr_config(
            self.adaptive_dr_config,
            forecast_horizon=forecast_horizon,
            ev_max_steps=ev_max_steps,
            episode_horizon_hours=episode_horizon_hours,
        )

        source_normalizers = getattr(getattr(env, "global_state_builder", None), "norm", {})
        normalizers_by_microgrid: dict[str, dict[str, float]] = {}
        for mg in env.agent_ids:
            if mg not in source_normalizers:
                raise ValueError(f"Missing normalizers for microgrid {mg}")
            normalizers_by_microgrid[mg] = {}
            for field in ("load", "pv", "wind", "power", "price"):
                if field not in source_normalizers[mg]:
                    raise ValueError(f"Missing scale {field} for microgrid {mg}")
                val = float(source_normalizers[mg][field])
                if not np.isfinite(val) or val <= 0.0:
                    raise ValueError(f"Invalid non-positive scale {field}={val} for microgrid {mg}")
                normalizers_by_microgrid[mg][field] = val

        normalizers_json = json.dumps(normalizers_by_microgrid, sort_keys=True, separators=(",", ":"))
        self.normalizers_by_microgrid = normalizers_by_microgrid
        self.normalizers_digest = hashlib.sha256(normalizers_json.encode("utf-8")).hexdigest()

        private_config = copy.deepcopy(dict(config))
        private_algo = private_config.setdefault("algorithm", {})
        private_algo["feasible_dr"] = {
            "enabled": False,
            "exploration_in_latent_space": True,
            "target_smoothing_in_latent_space": True,
            "hvac": {"recovery_anchor_enabled": False},
        }

        super().__init__(env, private_config, device=device, seed=seed)

        self.config = config
        self.algorithm_config = copy.deepcopy(settings)

        torch.manual_seed(seed)
        self.actors = nn.ModuleDict({
            mg: AdaptiveDRActor(
                self.observation_schemas[mg],
                self.agent_ids,
                mg,
                mg_config=env.cfg_by_id[mg] if hasattr(env, "cfg_by_id") else None,
                adaptive_config=self.adaptive_dr_config,
                normalizers_by_microgrid=self.normalizers_by_microgrid,
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

    def select_actions(
        self,
        observations: Mapping[str, Any],
        noise_std: float = 0.0,
    ) -> dict[str, np.ndarray]:
        return select_adaptive_actions(self, observations, noise_std=noise_std)

    def update(self, batch_size: int) -> dict[str, Any]:
        return update_adaptive_dr(self, batch_size)
