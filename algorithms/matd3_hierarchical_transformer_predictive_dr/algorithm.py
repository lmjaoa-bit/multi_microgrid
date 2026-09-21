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
from algorithms.matd3_hierarchical_transformer_predictive_dr.actor import PredictiveDRActor
from algorithms.matd3_hierarchical_transformer_predictive_dr.predictive_hvac import (
    validate_predictive_hvac_config,
)
from algorithms.matd3_hierarchical_transformer_predictive_dr.training import (
    select_predictive_actions,
    update_predictive_dr,
)


class Matd3HierarchicalTransformerPredictiveDr(Matd3HierarchicalTransformerFeasibleDR):
    """Hierarchical Transformer Actor with finite-horizon predictive HVAC reference."""

    action_semantics = "single_dispatch_dr_predictive_reference"
    predictive_contract_version = "1.1"

    def __init__(
        self,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        settings = config.get("algorithm", config)
        self.predictive_hvac_config = copy.deepcopy(settings.get("predictive_hvac", {}))
        requested_feasible_dr_config = copy.deepcopy(settings.get("feasible_dr", {}))
        self.feasible_dr_config = requested_feasible_dr_config

        net = settings.get("network", {})
        embed_dim = int(net.get("embed_dim", 32))
        num_heads = int(net.get("num_heads", 4))
        num_layers = int(net.get("num_layers", 2))
        forecast_horizon = int(net.get("forecast_horizon", 8))
        hidden_dim = int(net.get("hidden_dim", 64))
        delta_t = float(getattr(env, "delta_t", 1.0))
        episode_length = int(getattr(env, "episode_length", 24))

        validate_predictive_hvac_config(
            self.predictive_hvac_config,
            forecast_horizon=forecast_horizon,
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
        self.normalization_signature = self.normalizers_digest

        private_config = copy.deepcopy(dict(config))
        private_algo = private_config.setdefault("algorithm", {})
        private_algo["feasible_dr"] = {
            "enabled": False,
            "exploration_in_latent_space": True,
            "target_smoothing_in_latent_space": True,
            "hvac": {"recovery_anchor_enabled": False},
        }

        super().__init__(env, private_config, device=device, seed=seed)

        # The feasible-DR parent reads the private bootstrap configuration and
        # stores it on self. Restore the user-requested configuration before
        # constructing the predictive actors.
        self.feasible_dr_config = requested_feasible_dr_config
        self.actor_predictive_config = copy.deepcopy(settings)
        self.actor_predictive_config["predictive_hvac"] = self.predictive_hvac_config
        self.actor_predictive_config["feasible_dr"] = self.feasible_dr_config

        actor_kwargs = {
            "predictive_config": self.actor_predictive_config,
            "normalizers_by_microgrid": self.normalizers_by_microgrid,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "forecast_horizon": forecast_horizon,
            "hidden_dim": hidden_dim,
            "delta_t": delta_t,
            "episode_length": episode_length,
        }

        self.actors = nn.ModuleDict({
            mg: PredictiveDRActor(
                self.observation_schemas[mg],
                self.agent_ids,
                mg,
                mg_config=getattr(env, "cfg_by_id", {}).get(mg),
                **actor_kwargs,
            ).to(self.device)
            for mg in self.agent_ids
        })
        self.target_actors = nn.ModuleDict({
            mg: PredictiveDRActor(
                self.observation_schemas[mg],
                self.agent_ids,
                mg,
                mg_config=getattr(env, "cfg_by_id", {}).get(mg),
                **actor_kwargs,
            ).to(self.device)
            for mg in self.agent_ids
        })
        for mg in self.agent_ids:
            self.target_actors[mg].load_state_dict(self.actors[mg].state_dict())
            for param in self.target_actors[mg].parameters():
                param.requires_grad_(False)

        actor_lr = float(settings.get("actor_lr", 2e-4))
        self.actor_optimizers = {
            mg: torch.optim.Adam(self.actors[mg].parameters(), lr=actor_lr)
            for mg in self.agent_ids
        }

    def select_actions(
        self,
        observations: Mapping[str, Any],
        noise_std: float = 0.0,
    ) -> dict[str, np.ndarray]:
        return select_predictive_actions(self, observations, noise_std=noise_std)

    def update(self, batch_size: int) -> dict[str, Any]:
        return update_predictive_dr(self, batch_size)

    @classmethod
    def canonical_predictive_config(cls, cfg: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k in sorted(cfg.keys()):
            val = cfg[k]
            if isinstance(val, Mapping):
                result[k] = cls.canonical_predictive_config(val)
            elif isinstance(val, (int, float, bool, str)) or val is None:
                result[k] = val
            elif isinstance(val, (list, tuple)):
                result[k] = [
                    cls.canonical_predictive_config(x) if isinstance(x, Mapping) else x
                    for x in val
                ]
            else:
                result[k] = str(val)
        return result

    def save(self, path: Path | str) -> None:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        canonical_cfg = self.canonical_predictive_config(self.predictive_hvac_config)
        canonical_feasible_cfg = self.canonical_predictive_config(self.feasible_dr_config)
        state: dict[str, Any] = {
            "algorithm_slug": "matd3_hierarchical_transformer_predictive_dr",
            "action_semantics": self.action_semantics,
            "predictive_hvac_contract_version": self.predictive_contract_version,
            "predictive_hvac_config": canonical_cfg,
            "feasible_dr_config": canonical_feasible_cfg,
            "observation_schema_by_agent": {
                mg: list(self.observation_schemas[mg]) for mg in self.agent_ids
            },
            "normalizers_by_microgrid": self.normalizers_by_microgrid,
            "normalization_signature": self.normalization_signature,
            "actors": {mg: self.actors[mg].state_dict() for mg in self.agent_ids},
            "target_actors": {mg: self.target_actors[mg].state_dict() for mg in self.agent_ids},
            "critics": {mg: self.critics[mg].state_dict() for mg in self.agent_ids},
            "target_critics": {mg: self.target_critics[mg].state_dict() for mg in self.agent_ids},
            "update_steps": self.update_steps,
        }
        torch.save(state, target)

    def load(self, path: Path | str) -> None:
        source = Path(path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Checkpoint not found: {source}")

        state = torch.load(source, map_location=self.device, weights_only=False)

        expected_slug = "matd3_hierarchical_transformer_predictive_dr"
        actual_slug = state.get("algorithm_slug")
        if actual_slug != expected_slug:
            raise ValueError(
                f"Checkpoint algorithm mismatch: expected '{expected_slug}', got '{actual_slug}'"
            )

        expected_version = self.predictive_contract_version
        actual_version = state.get("predictive_hvac_contract_version")
        if actual_version != expected_version:
            raise ValueError(
                f"Checkpoint predictive contract version mismatch: expected '{expected_version}', got '{actual_version}'"
            )

        saved_cfg = state.get("predictive_hvac_config")
        current_cfg = self.canonical_predictive_config(self.predictive_hvac_config)
        if saved_cfg != current_cfg:
            raise ValueError(
                f"Checkpoint predictive_hvac_config mismatch: expected {current_cfg}, got {saved_cfg}"
            )

        saved_feasible_cfg = state.get("feasible_dr_config")
        current_feasible_cfg = self.canonical_predictive_config(self.feasible_dr_config)
        if saved_feasible_cfg != current_feasible_cfg:
            raise ValueError(
                "Checkpoint feasible_dr_config mismatch: "
                f"expected {current_feasible_cfg}, got {saved_feasible_cfg}"
            )

        saved_schema = state.get("observation_schema_by_agent", {})
        for mg in self.agent_ids:
            current_s = list(self.observation_schemas.get(mg, []))
            s_saved = list(saved_schema.get(mg, []))
            if s_saved != current_s:
                raise ValueError(
                    f"Observation schema mismatch for microgrid {mg}: expected {current_s}, got {s_saved}"
                )

        saved_norm = state.get("normalizers_by_microgrid")
        if saved_norm != self.normalizers_by_microgrid:
            raise ValueError(
                f"Normalizers mismatch: expected {self.normalizers_by_microgrid}, got {saved_norm}"
            )

        for mg in self.agent_ids:
            self.actors[mg].load_state_dict(state["actors"][mg])
            self.target_actors[mg].load_state_dict(state["target_actors"][mg])
            self.critics[mg].load_state_dict(state["critics"][mg])
            self.target_critics[mg].load_state_dict(state["target_critics"][mg])
        self.update_steps = int(state.get("update_steps", 0))
