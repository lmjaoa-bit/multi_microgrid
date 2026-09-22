"""Isolated peak-risk successor to the corrected predictive-DR Transformer."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer_predictive_dr.algorithm import (
    Matd3HierarchicalTransformerPredictiveDr,
)
from algorithms.matd3_hierarchical_transformer_predictive_dr.actor import PredictiveDRActor
from algorithms.matd3_hierarchical_transformer_peak_risk.actor import PeakRiskActor
from algorithms.matd3_hierarchical_transformer_peak_risk.critic import TwinPeakCritic
from algorithms.matd3_hierarchical_transformer_peak_risk.peak_risk import (
    PEAK_RISK_CONTRACT_VERSION, SLUG, validate_peak_risk_config,
)
from algorithms.matd3_hierarchical_transformer_peak_risk.training import update_peak_risk


class Matd3HierarchicalTransformerPeakRisk(Matd3HierarchicalTransformerPredictiveDr):
    """Three decentralized Transformer actors and one shared twin system-load cost critic."""

    peak_risk_contract_version = PEAK_RISK_CONTRACT_VERSION

    def __init__(self, env: Any, config: Mapping[str, Any],
                 device: str = "cpu", seed: int = 42) -> None:
        settings = config.get("algorithm", config)
        if settings.get("name", SLUG) != SLUG:
            raise ValueError(f"algorithm.name must equal {SLUG}")
        self.peak_risk_config = validate_peak_risk_config(settings.get("peak_risk", {}))
        if "p2p_auxiliary" in settings:
            raise ValueError("Peak-risk algorithm must not inherit p2p_auxiliary")
        self.peak_risk_config_fingerprint = hashlib.sha256(json.dumps(
            self.peak_risk_config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        super().__init__(env, config, device=device, seed=seed)

        # Re-create actors only when the token is enabled: a zero-token extension
        # changes attention and is not equivalent to the parent's 14-token path.
        self.peak_token_enabled = bool(self.peak_risk_config["enabled"]
                                       and self.peak_risk_config["token_enabled"])
        if self.peak_token_enabled:
            net = settings.get("network", {})
            actor_kwargs = {
                "predictive_config": self.actor_predictive_config,
                "normalizers_by_microgrid": self.normalizers_by_microgrid,
                "embed_dim": int(net.get("embed_dim", 32)),
                "num_heads": int(net.get("num_heads", 4)),
                "num_layers": int(net.get("num_layers", 2)),
                "forecast_horizon": int(net.get("forecast_horizon", 8)),
                "hidden_dim": int(net.get("hidden_dim", 64)),
                "delta_t": float(getattr(env, "delta_t", 1.0)),
                "episode_length": int(getattr(env, "episode_length", 24)),
            }
            actors = nn.ModuleDict({
                mg: PeakRiskActor(
                    self.observation_schemas[mg], self.agent_ids, mg,
                    mg_config=getattr(env, "cfg_by_id", {}).get(mg), **actor_kwargs,
                ) for mg in self.agent_ids
            }).to(self.device)
            self.actors = actors
            self.target_actors = copy.deepcopy(actors).to(self.device)
            for param in self.target_actors.parameters():
                param.requires_grad_(False)
            self.actor_optimizers = {
                mg: torch.optim.Adam(self.actors[mg].parameters(),
                                     lr=float(settings.get("actor_lr", 2e-4)))
                for mg in self.agent_ids
            }
            # The predictive parent has already cached gradient-diagnostic
            # parameter sets for its superseded actors; refresh all three.
            self._actor_parameters = {
                mg: tuple(actor.parameters()) for mg, actor in self.actors.items()
            }
            self._actor_branch_parameters = {
                mg: {
                    name: tuple(p for module in modules for p in module.parameters())
                    for name, modules in actor.branch_modules().items()
                } for mg, actor in self.actors.items()
            }
            self._actor_layer_parameters = {
                mg: {
                    f"layer_{idx}": tuple(layer.parameters())
                    for idx, layer in enumerate(actor.transformer.layers)
                } for mg, actor in self.actors.items()
            }

        self.system_peak_scale_kw = float(sum(
            self.normalizers_by_microgrid[mg]["load"] +
            float(self.projector.dr[mg]["scale_kw"]) for mg in self.agent_ids
        ))
        if not math.isfinite(self.system_peak_scale_kw) or self.system_peak_scale_kw <= 0:
            raise ValueError("system_peak_scale_kw must be positive and finite")
        self.peak_critic_enabled = bool(self.peak_risk_config["enabled"]
                                        and self.peak_risk_config["critic_enabled"])
        self.peak_critic_updates = 0
        if self.peak_critic_enabled:
            self.peak_critic = TwinPeakCritic(
                int(env.global_state_dim), len(self.agent_ids) * self.effective_action_dim,
                tuple(self.peak_risk_config["critic_hidden_dims"]),
            ).to(self.device)
            self.target_peak_critic = copy.deepcopy(self.peak_critic).to(self.device)
            for parameter in self.target_peak_critic.parameters():
                parameter.requires_grad_(False)
            self.peak_optimizer = torch.optim.Adam(
                self.peak_critic.parameters(), lr=self.peak_risk_config["critic_lr"])
        else:
            self.peak_critic = None
            self.target_peak_critic = None
            self.peak_optimizer = None

    def _peak_contract(self) -> dict[str, Any]:
        return {
            "algorithm_slug": SLUG,
            "peak_risk_contract_version": self.peak_risk_contract_version,
            "predictive_hvac_contract_version": self.predictive_contract_version,
            "action_semantics": self.action_semantics,
            "projection_signature": self.projection_signature,
            "observation_schema_by_agent": {
                mg: list(self.observation_schemas[mg]) for mg in self.agent_ids
            },
            "state_schema": list(self.state_schema),
            "normalization_signature": self.normalization_signature,
            "normalizers_by_microgrid": self.normalizers_by_microgrid,
            "peak_risk_config_fingerprint": self.peak_risk_config_fingerprint,
            "peak_risk_config": self.peak_risk_config,
            "system_peak_scale_kw": self.system_peak_scale_kw,
            "predictive_hvac_config": self.canonical_predictive_config(self.predictive_hvac_config),
            "feasible_dr_config": self.canonical_predictive_config(self.feasible_dr_config),
        }

    def _training_fingerprints(self) -> dict[str, Any]:
        fingerprints = super()._training_fingerprints()
        fingerprints.update({
            "peak_risk_contract_version": self.peak_risk_contract_version,
            "peak_risk_config_fingerprint": self.peak_risk_config_fingerprint,
            "system_peak_scale_kw": self.system_peak_scale_kw,
            "observation_schema_by_agent": {
                mg: list(self.observation_schemas[mg]) for mg in self.agent_ids
            },
        })
        return fingerprints

    def _extra_training_state(self) -> dict[str, Any]:
        result = {"peak_critic_enabled": self.peak_critic_enabled,
                  "peak_critic_updates": self.peak_critic_updates,
                  "peak_contract": self._peak_contract()}
        if self.peak_critic_enabled:
            result.update({
                "peak_critic": self.peak_critic.state_dict(),
                "target_peak_critic": self.target_peak_critic.state_dict(),
                "peak_optimizer": self.peak_optimizer.state_dict(),
            })
        return result

    def _load_extra_training_state(self, state: Mapping[str, Any]) -> None:
        self._validate_peak_contract(state.get("peak_contract", {}), source="recovery")
        if state.get("peak_critic_enabled") is not self.peak_critic_enabled:
            raise ValueError("recovery peak_critic_enabled mismatch")
        count = state.get("peak_critic_updates")
        if type(count) is not int or count < 0:
            raise ValueError("recovery peak_critic_updates is invalid")
        if self.peak_critic_enabled:
            for key in ("peak_critic", "target_peak_critic", "peak_optimizer"):
                if key not in state:
                    raise ValueError(f"recovery missing {key}")
            self.peak_critic.load_state_dict(state["peak_critic"])
            self.target_peak_critic.load_state_dict(state["target_peak_critic"])
            self.peak_optimizer.load_state_dict(state["peak_optimizer"])
        self.peak_critic_updates = count

    def _validate_peak_contract(self, saved: Mapping[str, Any], *, source: str) -> None:
        if not isinstance(saved, Mapping):
            raise ValueError(f"{source} missing peak contract")
        for key, expected in self._peak_contract().items():
            if key not in saved or saved[key] != expected:
                raise ValueError(f"{source} {key} mismatch: expected {expected!r}, got {saved.get(key)!r}")

    def update(self, batch_size: int) -> dict[str, Any]:
        if not self.peak_critic_enabled:
            return super().update(batch_size)
        return update_peak_risk(self, batch_size)

    def save(self, path: Path | str) -> None:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        content = self._peak_contract()
        content.update({
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "update_steps": self.update_steps,
            "peak_critic_updates": self.peak_critic_updates,
            "peak_critic": self.peak_critic.state_dict() if self.peak_critic_enabled else None,
            "target_peak_critic": self.target_peak_critic.state_dict() if self.peak_critic_enabled else None,
        })
        torch.save(content, target)

    def load(self, path: Path | str) -> None:
        checkpoint = torch.load(Path(path).resolve(), map_location=self.device, weights_only=False)
        self._validate_peak_contract(checkpoint, source="checkpoint")
        for field in ("actors", "target_actors", "critics", "target_critics"):
            if field not in checkpoint:
                raise ValueError(f"checkpoint missing {field}")
        if self.peak_critic_enabled:
            for field in ("peak_critic", "target_peak_critic"):
                if checkpoint.get(field) is None:
                    raise ValueError(f"checkpoint missing {field}")
        # All immutable contracts are checked before changing any network state.
        self.actors.load_state_dict(checkpoint["actors"], strict=True)
        self.target_actors.load_state_dict(checkpoint["target_actors"], strict=True)
        self.critics.load_state_dict(checkpoint["critics"], strict=True)
        self.target_critics.load_state_dict(checkpoint["target_critics"], strict=True)
        if self.peak_critic_enabled:
            self.peak_critic.load_state_dict(checkpoint["peak_critic"], strict=True)
            self.target_peak_critic.load_state_dict(checkpoint["target_peak_critic"], strict=True)
        self.update_steps = int(checkpoint["update_steps"])
        self.peak_critic_updates = int(checkpoint["peak_critic_updates"])
