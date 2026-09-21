from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer_p2p_value.actor import P2PValueActor
from algorithms.matd3_hierarchical_transformer_p2p_value.p2p_auxiliary import (
    validate_p2p_auxiliary_config,
)
from algorithms.matd3_hierarchical_transformer_predictive_dr import (
    Matd3HierarchicalTransformerPredictiveDr,
)

ALGORITHM_SLUG = "matd3_hierarchical_transformer_p2p_value"
PARENT_PREDICTIVE_CONTRACT_VERSION = "1.1"
P2P_AUXILIARY_CONTRACT_VERSION = "1.0"


class Matd3HierarchicalTransformerP2PValue(Matd3HierarchicalTransformerPredictiveDr):
    """Hierarchical Transformer with action-conditioned P2P value auxiliary learning."""

    slug = ALGORITHM_SLUG
    action_semantics = "single_dispatch_dr_predictive_p2p_value"
    parent_predictive_contract_version = PARENT_PREDICTIVE_CONTRACT_VERSION
    p2p_auxiliary_contract_version = P2P_AUXILIARY_CONTRACT_VERSION

    def __init__(
        self,
        env: Any,
        config: Mapping[str, Any],
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        settings = config.get("algorithm", {})
        aux_cfg = settings.get("p2p_auxiliary", {})
        self.p2p_auxiliary_config = validate_p2p_auxiliary_config(aux_cfg, require_enabled=True)

        super().__init__(env, config, device=device, seed=seed)
        if self.predictive_contract_version != PARENT_PREDICTIVE_CONTRACT_VERSION:
            raise ValueError(
                f"Expected parent predictive contract version {PARENT_PREDICTIVE_CONTRACT_VERSION}, "
                f"got {self.predictive_contract_version}"
            )

        self.delta_t = float(getattr(env, "delta_t", 1.0))
        self.episode_length = int(getattr(env, "episode_length", 24))

        source_normalizers = getattr(getattr(env, "global_state_builder", None), "norm", {})
        normalizers_by_microgrid: dict[str, dict[str, float]] = {}
        for mg in env.agent_ids:
            if mg not in source_normalizers:
                raise ValueError(f"Missing normalizers for microgrid {mg}")
            normalizers_by_microgrid[mg] = {}
            for field in ("load", "pv", "wind", "power", "price", "trade"):
                if field not in source_normalizers[mg]:
                    raise ValueError(f"Missing scale {field} for microgrid {mg}")
                val = float(source_normalizers[mg][field])
                if not np.isfinite(val) or val <= 0.0:
                    raise ValueError(f"Invalid non-positive scale {field}={val} for microgrid {mg}")
                normalizers_by_microgrid[mg][field] = val

        self.normalizers_by_microgrid = normalizers_by_microgrid
        normalizers_json = json.dumps(normalizers_by_microgrid, sort_keys=True, separators=(",", ":"))
        self.normalization_signature = hashlib.sha256(normalizers_json.encode("utf-8")).hexdigest()

        torch.manual_seed(seed)
        net = settings.get("network", {})
        embed_dim = int(net.get("embed_dim", 32))
        num_heads = int(net.get("num_heads", 4))
        num_layers = int(net.get("num_layers", 2))
        forecast_horizon = int(net.get("forecast_horizon", 8))
        hidden_dim = int(net.get("hidden_dim", 64))

        actor_kwargs = {
            "predictive_config": self.actor_predictive_config,
            "normalizers_by_microgrid": self.normalizers_by_microgrid,
            "p2p_auxiliary_config": self.p2p_auxiliary_config,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "forecast_horizon": forecast_horizon,
            "hidden_dim": hidden_dim,
            "delta_t": self.delta_t,
            "episode_length": self.episode_length,
        }

        self.actors = nn.ModuleDict({
            mg: P2PValueActor(
                self.observation_schemas[mg],
                self.agent_ids,
                mg,
                mg_config=getattr(env, "cfg_by_id", {}).get(mg),
                **actor_kwargs,
            ).to(self.device)
            for mg in self.agent_ids
        })
        self.target_actors = nn.ModuleDict({
            mg: P2PValueActor(
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
        self._actor_parameters = {
            mg: tuple(actor.parameters()) for mg, actor in self.actors.items()
        }
        self._actor_branch_parameters = {
            mg: {
                name: tuple(
                    parameter
                    for module in modules
                    for parameter in module.parameters()
                )
                for name, modules in actor.branch_modules().items()
            }
            for mg, actor in self.actors.items()
        }
        self._actor_layer_parameters = {
            mg: {
                f"layer_{index}": tuple(layer.parameters())
                for index, layer in enumerate(actor.transformer.layers)
            }
            for mg, actor in self.actors.items()
        }

    def update(self, batch_size: int) -> dict[str, Any]:
        from algorithms.matd3_hierarchical_transformer_p2p_value.training import update_p2p_value

        return update_p2p_value(self, batch_size)

    def save(self, path: Path | str) -> None:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        canonical_hvac = self.canonical_predictive_config(self.predictive_hvac_config)
        canonical_feasible = self.canonical_predictive_config(self.feasible_dr_config)
        canonical_aux = self.canonical_predictive_config(self.p2p_auxiliary_config)
        state: dict[str, Any] = {
            "algorithm_slug": ALGORITHM_SLUG,
            "action_semantics": self.action_semantics,
            "parent_predictive_contract_version": self.parent_predictive_contract_version,
            "predictive_hvac_config": canonical_hvac,
            "feasible_dr_config": canonical_feasible,
            "p2p_auxiliary_contract_version": self.p2p_auxiliary_contract_version,
            "p2p_auxiliary_config": canonical_aux,
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

        actual_slug = state.get("algorithm_slug")
        if actual_slug != ALGORITHM_SLUG:
            raise ValueError(
                f"Checkpoint algorithm_slug mismatch: expected '{ALGORITHM_SLUG}', got '{actual_slug}'"
            )

        actual_semantics = state.get("action_semantics")
        if actual_semantics != self.action_semantics:
            raise ValueError(
                f"Checkpoint action_semantics mismatch: expected '{self.action_semantics}', got '{actual_semantics}'"
            )

        actual_parent_version = state.get("parent_predictive_contract_version")
        if actual_parent_version != self.parent_predictive_contract_version:
            raise ValueError(
                f"Checkpoint parent_predictive_contract_version mismatch: expected '{self.parent_predictive_contract_version}', got '{actual_parent_version}'"
            )

        actual_aux_version = state.get("p2p_auxiliary_contract_version")
        if actual_aux_version != self.p2p_auxiliary_contract_version:
            raise ValueError(
                f"Checkpoint p2p_auxiliary_contract_version mismatch: expected '{self.p2p_auxiliary_contract_version}', got '{actual_aux_version}'"
            )

        saved_hvac = state.get("predictive_hvac_config")
        cur_hvac = self.canonical_predictive_config(self.predictive_hvac_config)
        if saved_hvac != cur_hvac:
            raise ValueError(
                f"Checkpoint predictive_hvac_config mismatch: expected {cur_hvac}, got {saved_hvac}"
            )

        saved_feasible = state.get("feasible_dr_config")
        cur_feasible = self.canonical_predictive_config(self.feasible_dr_config)
        if saved_feasible != cur_feasible:
            raise ValueError(
                f"Checkpoint feasible_dr_config mismatch: expected {cur_feasible}, got {saved_feasible}"
            )

        saved_aux = state.get("p2p_auxiliary_config")
        cur_aux = self.canonical_predictive_config(self.p2p_auxiliary_config)
        if saved_aux != cur_aux:
            raise ValueError(
                f"Checkpoint p2p_auxiliary_config mismatch: expected {cur_aux}, got {saved_aux}"
            )

        saved_schema = state.get("observation_schema_by_agent", {})
        for mg in self.agent_ids:
            cur_s = list(self.observation_schemas.get(mg, []))
            s_saved = list(saved_schema.get(mg, []))
            if s_saved != cur_s:
                raise ValueError(
                    f"Observation observation_schema_by_agent mismatch for microgrid {mg}: expected {cur_s}, got {s_saved}"
                )

        saved_norm = state.get("normalizers_by_microgrid")
        if saved_norm != self.normalizers_by_microgrid:
            raise ValueError(
                f"Checkpoint normalizers_by_microgrid mismatch: expected {self.normalizers_by_microgrid}, got {saved_norm}"
            )

        for mg in self.agent_ids:
            self.actors[mg].load_state_dict(state["actors"][mg])
            self.target_actors[mg].load_state_dict(state["target_actors"][mg])
            self.critics[mg].load_state_dict(state["critics"][mg])
            self.target_critics[mg].load_state_dict(state["target_critics"][mg])

        self.update_steps = int(state.get("update_steps", 0))
