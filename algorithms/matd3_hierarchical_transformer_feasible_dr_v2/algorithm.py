from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from algorithms.matd3_hierarchical_transformer_feasible_dr.algorithm import (
    Matd3HierarchicalTransformerFeasibleDR,
)
from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.adaptive_dr import (
    validate_adaptive_dr_config,
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
        forecast_horizon = int(net.get("forecast_horizon", 8))
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

        super().__init__(env, config, device=device, seed=seed)
