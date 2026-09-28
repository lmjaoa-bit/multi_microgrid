from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithms.matd3 import MATD3
from algorithms.matd3_forecast_context.actor import (
    POWER_NORMALIZER_KEYS,
    ForecastContextActor,
)
from core.joint_action_projection import JointActionProjector


class Matd3ForecastContext(MATD3):
    """MATD3 ordinary-T forecast-context ablation implementation."""

    ALGORITHM_SLUG = "matd3_forecast_context"
    ACTOR_FEATURE_SEMANTICS = "physical_power_balance_v1"
    FORECAST_CONTEXT_VERSION = "forecast_context_v1"
    MASK_SEMANTICS = "valid_lt_half_padding_v1"
    LOCAL_BALANCE_SIGN = "positive_deficit"
    PEER_BALANCE_SIGN = "positive_surplus"

    GROUP_MATRIX = {
        "A": (8, False),
        "B": (8, True),
        "C": (24, False),
        "D": (24, True),
    }

    @classmethod
    def from_env(
        cls,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
        seed: int | None = None,
    ) -> "Matd3ForecastContext":
        return cls(
            env,
            config,
            device=device,
            seed=env.seed if seed is None else seed,
        )

    @classmethod
    def _parse_study_config(
        cls,
        config: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], str, int, bool, tuple[str, ...]]:
        if not isinstance(config, Mapping):
            raise ValueError("Algorithm config must be a mapping")
        settings = config.get("algorithm")
        if not isinstance(settings, Mapping):
            raise ValueError("Algorithm config is missing algorithm mapping")
        if settings.get("name") != cls.ALGORITHM_SLUG:
            raise ValueError(
                f"algorithm.name must equal {cls.ALGORITHM_SLUG!r}"
            )

        experiment = config.get("experiment")
        if not isinstance(experiment, Mapping):
            raise ValueError("Forecast context config is missing experiment mapping")
        group = experiment.get("group")
        if group not in cls.GROUP_MATRIX:
            raise ValueError("experiment.group must be one of A, B, C, D")
        if experiment.get("network_family") != "transformer":
            raise ValueError("experiment.network_family must equal 'transformer'")

        net = settings.get("network")
        if not isinstance(net, Mapping):
            raise ValueError("algorithm.network must be a mapping")

        horizon = net.get("forecast_horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, Integral):
            raise ValueError("algorithm.network.forecast_horizon must be integer 8 or 24")
        horizon = int(horizon)
        if horizon not in (8, 24):
            raise ValueError("algorithm.network.forecast_horizon must be 8 or 24")

        include_peer = net.get("include_peer_forecast")
        if not isinstance(include_peer, bool):
            raise ValueError(
                "algorithm.network.include_peer_forecast must be a YAML boolean"
            )

        peer_features_raw = net.get("peer_forecast_features")
        if isinstance(peer_features_raw, (str, bytes)) or not isinstance(
            peer_features_raw, Sequence
        ):
            raise ValueError("peer_forecast_features must be [energy_balance]")
        peer_features = tuple(peer_features_raw)
        if peer_features != ("energy_balance",):
            raise ValueError("peer_forecast_features must be [energy_balance]")

        if net.get("forecast_context_version") != cls.FORECAST_CONTEXT_VERSION:
            raise ValueError(
                f"forecast_context_version must equal {cls.FORECAST_CONTEXT_VERSION!r}"
            )
        if net.get("feature_semantics") != cls.ACTOR_FEATURE_SEMANTICS:
            raise ValueError(
                "Unsupported actor feature semantics: "
                f"expected {cls.ACTOR_FEATURE_SEMANTICS!r}, "
                f"got {net.get('feature_semantics')!r}"
            )

        expected_horizon, expected_peer = cls.GROUP_MATRIX[str(group)]
        if (horizon, include_peer) != (expected_horizon, expected_peer):
            raise ValueError(
                f"experiment.group {group} requires "
                f"forecast_horizon={expected_horizon} and "
                f"include_peer_forecast={expected_peer}"
            )
        return settings, net, str(group), horizon, include_peer, peer_features

    def __init__(
        self,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        if getattr(env, "ess_sharing_control_mode", "") != "single_dispatch":
            raise ValueError("This algorithm requires single_dispatch ESS control")
        if int(env.action_dim) != 2:
            raise ValueError(
                "Forecast context Transformer requires raw actions [ESS intent, DR intent]"
            )
        if len(tuple(env.agent_ids)) != 3 or len(set(env.agent_ids)) != 3:
            raise ValueError("Forecast context study requires exactly three microgrids")

        (
            settings,
            net,
            self.experiment_group,
            self.forecast_horizon,
            self.include_peer_forecast,
            self.peer_forecast_features,
        ) = self._parse_study_config(config)
        self.actor_feature_semantics = self.ACTOR_FEATURE_SEMANTICS

        allowed = {
            "actor_lr",
            "critic_lr",
            "gamma",
            "tau",
            "policy_delay",
            "target_policy_noise",
            "target_noise_clip",
            "replay_capacity",
        }
        kwargs = {key: settings[key] for key in allowed if key in settings}

        # Keep the ordinary-T construction order so group A has the same seeded
        # initialization path as the locked reference implementation.
        super().__init__(
            env.agent_ids,
            env.observation_dims,
            env.global_state_dim,
            3,
            hidden_dims=tuple(settings.get("hidden_dims", (128, 128))),
            device=device,
            seed=seed,
            **kwargs,
        )

        self.raw_action_dim = int(env.action_dim)
        self.effective_action_dim = 3
        self.action_dim = self.raw_action_dim

        self.observation_schemas = {
            mg: tuple(env.obs_builder.schema(mg))
            for mg in self.agent_ids
        }
        self.state_schema = tuple(env.global_state_builder.schema())
        self.projector = JointActionProjector(
            env.config,
            self.state_schema,
            env.global_state_builder.norm,
            env.global_state_builder.price_norm,
        )
        projection_payload = {
            "semantics": self.projector.semantics,
            "config": env.config,
            "schema": self.state_schema,
            "normalizers": env.global_state_builder.norm,
            "price_norm": env.global_state_builder.price_norm,
        }
        self.projection_signature = hashlib.sha256(
            json.dumps(
                projection_payload,
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

        raw_normalizers = getattr(env.obs_builder, "norm", None)
        if not isinstance(raw_normalizers, Mapping):
            raise ValueError("Observation builder is missing actor power normalizers")
        self.actor_feature_normalizers: dict[str, dict[str, float]] = {}
        for mg in self.agent_ids:
            source = raw_normalizers.get(mg)
            if not isinstance(source, Mapping):
                raise ValueError(f"{mg}: actor power normalizers are missing")
            normalized: dict[str, float] = {}
            for key in POWER_NORMALIZER_KEYS:
                if key not in source:
                    raise ValueError(f"{mg}: actor power normalizer missing key {key}")
                value = float(source[key])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"{mg}: actor power normalizer {key} must be positive and finite"
                    )
                normalized[key] = value
            self.actor_feature_normalizers[mg] = normalized

        self.embed_dim = int(net.get("embed_dim", 32))
        self.num_heads = int(net.get("num_heads", 4))
        self.num_layers = int(net.get("num_layers", 2))
        self.hidden_dim = int(net.get("hidden_dim", 64))

        self.actors = nn.ModuleDict(
            {
                mg: ForecastContextActor(
                    self.observation_schemas[mg],
                    self.agent_ids,
                    mg,
                    mg_config=(
                        env.cfg_by_id[mg]
                        if hasattr(env, "cfg_by_id")
                        else None
                    ),
                    power_normalizers=self.actor_feature_normalizers[mg],
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers,
                    forecast_horizon=self.forecast_horizon,
                    hidden_dim=self.hidden_dim,
                    include_peer_forecast=self.include_peer_forecast,
                    peer_forecast_features=self.peer_forecast_features,
                )
                for mg in self.agent_ids
            }
        ).to(self.device)

        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.actor_optimizers = {
            mg: torch.optim.Adam(
                self.actors[mg].parameters(),
                lr=float(settings.get("actor_lr", 2e-4)),
            )
            for mg in self.agent_ids
        }
        self.actor_max_grad_norm = float(settings.get("actor_max_grad_norm", 1.0))
        if (
            not np.isfinite(self.actor_max_grad_norm)
            or self.actor_max_grad_norm <= 0.0
        ):
            raise ValueError("actor_max_grad_norm must be positive and finite")

        self._rebuild_actor_parameter_caches()

    def _rebuild_actor_parameter_caches(self) -> None:
        self._actor_parameters = {
            mg: tuple(actor.parameters())
            for mg, actor in self.actors.items()
        }
        self._actor_branch_parameters = {
            mg: {
                name: tuple(
                    parameter
                    for item in items
                    for parameter in (
                        item.parameters()
                        if isinstance(item, nn.Module)
                        else [item]
                    )
                )
                for name, items in actor.branch_modules().items()
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

    def actor_parameter_counts(self) -> dict[str, int]:
        return {
            mg: self.actors[mg].parameter_count
            for mg in self.agent_ids
        }

    def project_actions(
        self,
        state: torch.Tensor,
        joint_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        return self.projector(state, joint_raw)

    def _actor_gradient_diagnostics(
        self,
        mg: str,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        squared = {
            id(parameter): (
                parameter.grad.detach().square().sum()
                if parameter.grad is not None
                else torch.zeros((), device=self.device)
            )
            for parameter in self._actor_parameters[mg]
        }

        def norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
            terms = [squared[id(parameter)] for parameter in parameters]
            return (
                float(torch.stack(terms).sum().sqrt().cpu())
                if terms
                else 0.0
            )

        total_norm = norm(self._actor_parameters[mg])
        branch_norms = {
            name: norm(parameters)
            for name, parameters in self._actor_branch_parameters[mg].items()
        }
        layer_norms = {
            name: norm(parameters)
            for name, parameters in self._actor_layer_parameters[mg].items()
        }
        return total_norm, branch_norms, layer_norms

    def update(self, batch_size: int) -> dict[str, Any]:
        """Reference grouped update; plugin installation replaces it with sampled diagnostics."""
        if len(self.replay) < batch_size:
            return {}

        batch = self.replay.sample(batch_size, self.agent_ids, self.device)
        state = batch["global_state"]
        next_state = batch["next_global_state"]
        raw = torch.stack(
            [batch["actions"][mg] for mg in self.agent_ids],
            dim=1,
        )
        effective, _ = self.project_actions(state, raw)

        with torch.no_grad():
            target_raw = []
            for mg in self.agent_ids:
                action = self.target_actors[mg](
                    batch["next_observations"][mg]
                )
                noise = (
                    torch.randn_like(action)
                    .mul(self.target_policy_noise)
                    .clamp(
                        -self.target_noise_clip,
                        self.target_noise_clip,
                    )
                )
                target_raw.append(
                    (action + noise).clamp(-1.0, 1.0)
                )
            target_effective, _ = self.project_actions(
                next_state,
                torch.stack(target_raw, dim=1),
            )
            target_q = {}
            for mg in self.agent_ids:
                q1, q2 = self.target_critics[mg](
                    next_state,
                    target_effective.flatten(1),
                )
                target_q[mg] = batch["team_reward"] + (
                    1.0 - batch["done"]
                ) * self.gamma * torch.minimum(q1, q2)

        metrics: dict[str, Any] = {}
        records: dict[str, float] = {}
        q_stats: dict[str, dict[str, float]] = {}
        for mg in self.agent_ids:
            q1, q2 = self.critics[mg](state, effective.flatten(1))
            loss1 = nn.functional.mse_loss(q1, target_q[mg])
            loss2 = nn.functional.mse_loss(q2, target_q[mg])
            loss = loss1 + loss2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{mg}: nonfinite Critic loss")
            self.critic_optimizers[mg].zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = self._gradient_norm(self.critics[mg])
            if not np.isfinite(grad_norm):
                raise FloatingPointError(f"{mg}: nonfinite Critic gradient")
            self.critic_optimizers[mg].step()

            q_min = torch.minimum(q1.detach(), q2.detach())
            td = target_q[mg] - q_min
            records[mg] = float(loss.detach())
            q_stats[mg] = {
                "critic_loss": records[mg],
                "critic_grad_norm": grad_norm,
                "q_mean": float(q_min.mean()),
                "q_std": float(q_min.std(unbiased=False)),
                "td_error_mean": float(td.mean()),
                "td_error_rmse": float(td.square().mean().sqrt()),
            }

        metrics.update(
            critic_loss=float(np.mean(list(records.values()))),
            critic_losses=records,
            critic_q_diagnostics=q_stats,
        )
        for stat in (
            "critic_grad_norm",
            "q_mean",
            "q_std",
            "td_error_mean",
            "td_error_rmse",
        ):
            metrics[stat] = float(
                np.mean([q_stats[mg][stat] for mg in self.agent_ids])
            )

        self.update_steps += 1
        if self.update_steps % self.policy_delay == 0:
            losses: dict[str, float] = {}
            grads: dict[str, float] = {}
            branches: dict[str, dict[str, float]] = {}
            diagnostics: dict[str, dict[str, float]] = {}
            with torch.no_grad():
                cached_actions = {
                    mg: self.actors[mg](batch["observations"][mg])
                    for mg in self.agent_ids
                }

            for active in self.agent_ids:
                actor = self.actors[active]
                critic = self.critics[active]
                optimizer = self.actor_optimizers[active]
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
                try:
                    active_action = actor(batch["observations"][active])
                    joint = [
                        active_action if mg == active else cached_actions[mg]
                        for mg in self.agent_ids
                    ]
                    projected, _ = self.project_actions(
                        state,
                        torch.stack(joint, dim=1),
                    )
                    loss = -critic.q1(
                        state,
                        projected.flatten(1),
                    ).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"{active}: nonfinite Actor loss"
                        )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    before, branch, _ = self._actor_gradient_diagnostics(active)
                    grads[active] = before
                    branches[active] = branch
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(),
                        self.actor_max_grad_norm,
                    )
                    after = self._gradient_norm(actor)
                    optimizer.step()
                    losses[active] = float(loss.detach())
                    diagnostics[active] = {
                        "total_grad_norm": float(before),
                        "post_clip_grad_norm": float(after),
                        "clip_applied": float(
                            before > self.actor_max_grad_norm
                        ),
                        **{
                            f"{name}_grad_norm": float(value)
                            for name, value in branch.items()
                        },
                    }
                    self._soft_update(
                        actor,
                        self.target_actors[active],
                    )
                    self._soft_update(
                        critic,
                        self.target_critics[active],
                    )
                finally:
                    for parameter in critic.parameters():
                        parameter.requires_grad_(True)

            metrics.update(
                actor_loss=float(np.mean(list(losses.values()))),
                actor_losses=losses,
                actor_grad_norm=float(np.mean(list(grads.values()))),
                actor_grad_norms=grads,
                actor_branch_grad_norms=branches,
                actor_diagnostics=diagnostics,
            )
        return metrics

    def _validate_actor_feature_metadata(
        self,
        checkpoint: Mapping[str, Any],
    ) -> None:
        semantics = checkpoint.get("actor_feature_semantics")
        if semantics != self.ACTOR_FEATURE_SEMANTICS:
            raise ValueError(
                "Actor feature semantics mismatch: "
                f"expected {self.ACTOR_FEATURE_SEMANTICS!r}, got {semantics!r}"
            )

        saved = checkpoint.get("actor_feature_normalizers")
        if not isinstance(saved, Mapping):
            raise ValueError(
                "Checkpoint actor feature normalizers are missing or invalid"
            )
        if set(saved.keys()) != set(self.agent_ids):
            raise ValueError(
                "Checkpoint actor feature normalizer agent set mismatch"
            )
        expected_keys = set(POWER_NORMALIZER_KEYS)
        for mg in self.agent_ids:
            mg_saved = saved[mg]
            if not isinstance(mg_saved, Mapping):
                raise ValueError(
                    f"{mg}: checkpoint actor feature normalizers are invalid"
                )
            if set(mg_saved.keys()) != expected_keys:
                raise ValueError(
                    f"{mg}: checkpoint actor feature normalizer keys mismatch"
                )
            for key in POWER_NORMALIZER_KEYS:
                try:
                    saved_value = np.float64(mg_saved[key])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"{mg}: checkpoint actor feature normalizer {key} is invalid"
                    ) from exc
                current_value = np.float64(
                    self.actor_feature_normalizers[mg][key]
                )
                if (
                    not np.isfinite(saved_value)
                    or saved_value <= 0.0
                    or not np.isclose(
                        saved_value,
                        current_value,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"{mg}: checkpoint actor feature normalizer {key} mismatch: "
                        f"expected {current_value!r}, got {saved_value!r}"
                    )

    @staticmethod
    def _canonical_contract_json(contract: Mapping[str, Any]) -> str:
        return json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def _contract_signature(cls, contract: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            cls._canonical_contract_json(contract).encode("utf-8")
        ).hexdigest()

    def _forecast_context_contract(self) -> dict[str, Any]:
        return {
            "horizon": int(self.forecast_horizon),
            "include_peer_forecast": bool(self.include_peer_forecast),
            "peer_forecast_features": list(self.peer_forecast_features),
            "agent_order": list(self.agent_ids),
            "per_agent": {
                mg: {
                    "peer_order": list(self.actors[mg].peers),
                    "ordered_forecast_feature_names": list(
                        self.actors[mg].ordered_forecast_feature_names
                    ),
                    "local_source_schema_names": [
                        list(step)
                        for step in self.actors[mg].local_source_schema_names
                    ],
                    "peer_source_schema_names": [
                        list(step)
                        for step in self.actors[mg].peer_source_schema_names
                    ],
                }
                for mg in self.agent_ids
            },
            "power_normalizers_by_agent": copy.deepcopy(
                self.actor_feature_normalizers
            ),
            "mask_semantics": self.MASK_SEMANTICS,
            "local_balance_sign": self.LOCAL_BALANCE_SIGN,
            "peer_balance_sign": self.PEER_BALANCE_SIGN,
            "architecture": {
                "embed_dim": int(self.embed_dim),
                "num_heads": int(self.num_heads),
                "num_layers": int(self.num_layers),
                "hidden_dim": int(self.hidden_dim),
            },
        }

    @staticmethod
    def _first_difference(
        expected: Any,
        actual: Any,
        path: str = "forecast_context_contract",
    ) -> str | None:
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                return path
            if set(expected.keys()) != set(actual.keys()):
                return f"{path}.keys"
            for key in expected:
                difference = Matd3ForecastContext._first_difference(
                    expected[key],
                    actual[key],
                    f"{path}.{key}",
                )
                if difference:
                    return difference
            return None
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(expected) != len(actual):
                return path
            for index, (left, right) in enumerate(zip(expected, actual)):
                difference = Matd3ForecastContext._first_difference(
                    left,
                    right,
                    f"{path}[{index}]",
                )
                if difference:
                    return difference
            return None
        return None if expected == actual else path

    def _validate_contract_normalizers(
        self,
        saved: Any,
    ) -> None:
        if not isinstance(saved, Mapping):
            raise ValueError(
                "forecast_context_contract.power_normalizers_by_agent is invalid"
            )
        if set(saved.keys()) != set(self.agent_ids):
            raise ValueError(
                "forecast_context_contract.power_normalizers_by_agent agent set mismatch"
            )
        for mg in self.agent_ids:
            values = saved[mg]
            if not isinstance(values, Mapping):
                raise ValueError(
                    f"forecast_context_contract.power_normalizers_by_agent.{mg} is invalid"
                )
            if set(values.keys()) != set(POWER_NORMALIZER_KEYS):
                raise ValueError(
                    f"forecast_context_contract.power_normalizers_by_agent.{mg} keys mismatch"
                )
            for key in POWER_NORMALIZER_KEYS:
                try:
                    saved_value = np.float64(values[key])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"forecast_context_contract normalizer {mg}.{key} is invalid"
                    ) from exc
                current_value = np.float64(
                    self.actor_feature_normalizers[mg][key]
                )
                if (
                    not np.isfinite(saved_value)
                    or saved_value <= 0.0
                    or not np.isclose(
                        saved_value,
                        current_value,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ):
                    raise ValueError(
                        "forecast_context_contract normalizer mismatch at "
                        f"{mg}.{key}: expected {current_value!r}, got {saved_value!r}"
                    )

    def _validate_forecast_context_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
    ) -> None:
        if checkpoint.get("algorithm_slug") != self.ALGORITHM_SLUG:
            raise ValueError(
                "Incompatible algorithm checkpoint: expected "
                f"{self.ALGORITHM_SLUG}, got {checkpoint.get('algorithm_slug')}"
            )
        if (
            checkpoint.get("forecast_context_version")
            != self.FORECAST_CONTEXT_VERSION
        ):
            raise ValueError(
                "Checkpoint forecast_context_version is missing or incompatible"
            )
        saved_contract = checkpoint.get("forecast_context_contract")
        if not isinstance(saved_contract, Mapping):
            raise ValueError(
                "Checkpoint forecast_context_contract is missing or invalid"
            )
        saved_signature = checkpoint.get("forecast_context_signature")
        if not isinstance(saved_signature, str):
            raise ValueError(
                "Checkpoint forecast_context_signature is missing or invalid"
            )
        computed_signature = self._contract_signature(saved_contract)
        if saved_signature != computed_signature:
            raise ValueError(
                "Checkpoint forecast_context_signature does not match its contract"
            )

        current_contract = self._forecast_context_contract()
        self._validate_contract_normalizers(
            saved_contract.get("power_normalizers_by_agent")
        )

        saved_non_float = copy.deepcopy(dict(saved_contract))
        current_non_float = copy.deepcopy(current_contract)
        saved_non_float.pop("power_normalizers_by_agent", None)
        current_non_float.pop("power_normalizers_by_agent", None)
        difference = self._first_difference(
            current_non_float,
            saved_non_float,
        )
        if difference:
            raise ValueError(
                f"Forecast context checkpoint contract mismatch at {difference}"
            )

    def save(self, path: str | Path) -> None:
        contract = self._forecast_context_contract()
        torch.save(
            {
                "algorithm_slug": self.ALGORITHM_SLUG,
                "forecast_context_version": self.FORECAST_CONTEXT_VERSION,
                "forecast_context_contract": contract,
                "forecast_context_signature": self._contract_signature(contract),
                "actor_feature_semantics": self.actor_feature_semantics,
                "actor_feature_normalizers": copy.deepcopy(
                    self.actor_feature_normalizers
                ),
                "action_semantics": self.projector.semantics,
                "projection_signature": self.projection_signature,
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
        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint payload must be a mapping")

        # All compatibility checks happen before the first load_state_dict call.
        self._validate_actor_feature_metadata(checkpoint)
        self._validate_forecast_context_checkpoint(checkpoint)
        if checkpoint.get("action_semantics") != self.projector.semantics:
            raise ValueError("Incompatible action semantics")
        if checkpoint.get("projection_signature") != self.projection_signature:
            raise ValueError("Projection signature mismatch")
        if checkpoint.get("state_schema") != self.state_schema:
            raise ValueError("Checkpoint state schema does not match")
        if checkpoint.get("observation_schemas") != self.observation_schemas:
            raise ValueError("Checkpoint observation schema does not match")

        self.actors.load_state_dict(checkpoint["actors"])
        self.target_actors.load_state_dict(checkpoint["target_actors"])
        self.critics.load_state_dict(checkpoint["critics"])
        self.target_critics.load_state_dict(checkpoint["target_critics"])
        self.update_steps = int(checkpoint.get("update_steps", 0))
