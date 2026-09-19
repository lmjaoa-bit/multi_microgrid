from __future__ import annotations

import copy
import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from algorithms.matd3.critic import TwinCritic
from core.joint_action_projection import JointActionProjector


ACTION_SEMANTICS = "single_dispatch_dr_v1"
RAW_ACTION_DIM = 2
EFFECTIVE_ACTION_DIM = 3


def _new_linear_like(
    old: nn.Linear,
    *,
    in_features: int | None = None,
    out_features: int | None = None,
) -> nn.Linear:
    reference = old.weight
    layer = nn.Linear(
        int(old.in_features if in_features is None else in_features),
        int(old.out_features if out_features is None else out_features),
        bias=old.bias is not None,
        device=reference.device,
        dtype=reference.dtype,
    )
    return layer


def _expand_linear_input(
    old: nn.Linear,
    new_in_features: int,
    old_to_new: list[int],
) -> nn.Linear:
    if len(old_to_new) != old.in_features:
        raise ValueError("Linear input mapping length does not match the source layer")
    layer = _new_linear_like(old, in_features=new_in_features)
    with torch.no_grad():
        if old.bias is not None:
            layer.bias.copy_(old.bias)
        for old_index, new_index in enumerate(old_to_new):
            layer.weight[:, int(new_index)].copy_(old.weight[:, old_index])
    return layer


def _expand_action_head(actor: nn.Module) -> None:
    head = getattr(actor, "dispatch_head", None)
    if not isinstance(head, nn.Sequential) or not head or not isinstance(head[0], nn.Linear):
        raise TypeError(f"{type(actor).__name__} must expose dispatch_head beginning with nn.Linear")
    old = head[0]
    if old.out_features == RAW_ACTION_DIM:
        return
    if old.out_features != 1:
        raise ValueError(f"Unsupported historical action width: {old.out_features}")
    layer = _new_linear_like(old, out_features=RAW_ACTION_DIM)
    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
    if layer.bias is not None:
        nn.init.uniform_(layer.bias, -3e-3, 3e-3)
    with torch.no_grad():
        layer.weight[0].copy_(old.weight[0])
        if old.bias is not None:
            layer.bias[0].copy_(old.bias[0])
    actor.dispatch_head[0] = layer


def _schema_names(schema: tuple[str, ...], indices: torch.Tensor) -> list[str]:
    return [schema[int(index)] for index in indices.detach().cpu().tolist()]


def _feature_mapping(old_names: list[str], new_names: list[str]) -> list[int]:
    index = {name: i for i, name in enumerate(new_names)}
    missing = [name for name in old_names if name not in index]
    if missing:
        raise ValueError(f"DR adapter lost historical features: {missing}")
    return [index[name] for name in old_names]


def _adapt_transformer_features(actor: nn.Module, schema: tuple[str, ...], *, pnn: bool) -> None:
    old_local_names = _schema_names(schema, actor._local_indices)
    old_forecast_names = _schema_names(schema, actor._all_forecast_indices)

    excluded_indices = set(actor._hour_indices.detach().cpu().tolist())
    excluded_indices.update(actor._market_indices.detach().cpu().tolist())
    excluded_indices.update(actor._all_peer_indices.detach().cpu().tolist())
    excluded_indices.update(i for i, name in enumerate(schema) if "_t+" in name)
    new_local_names = [name for i, name in enumerate(schema) if i not in excluded_indices]

    horizon = int(actor.forecast_horizon)
    new_forecast_names: list[str] = []
    for k in range(1, horizon + 1):
        suffix = f"_t+{k}"
        step_names = [name for name in schema if name.endswith(suffix)]
        if not step_names:
            raise ValueError(f"No forecast fields found for horizon t+{k}")
        new_forecast_names.extend(step_names)

    schema_index = {name: i for i, name in enumerate(schema)}
    local_mapping = _feature_mapping(old_local_names, new_local_names)
    forecast_mapping = _feature_mapping(old_forecast_names, new_forecast_names)

    actor.local_encoder[0] = _expand_linear_input(
        actor.local_encoder[0], len(new_local_names), local_mapping
    )
    actor.forecast_encoder[0] = _expand_linear_input(
        actor.forecast_encoder[0], len(new_forecast_names) // horizon,
        _feature_mapping(
            old_forecast_names[: len(old_forecast_names) // horizon],
            new_forecast_names[: len(new_forecast_names) // horizon],
        ),
    )
    actor._local_indices = torch.tensor(
        [schema_index[name] for name in new_local_names],
        dtype=torch.long,
        device=actor._local_indices.device,
    )
    actor._all_forecast_indices = torch.tensor(
        [schema_index[name] for name in new_forecast_names],
        dtype=torch.long,
        device=actor._all_forecast_indices.device,
    )

    if pnn:
        actor.pnn_local_encoder[0] = _expand_linear_input(
            actor.pnn_local_encoder[0], len(new_local_names), local_mapping
        )
        actor.pnn_temporal_encoder[0] = _expand_linear_input(
            actor.pnn_temporal_encoder[0], len(new_forecast_names), forecast_mapping
        )


def _adapt_actor(actor: nn.Module, schema: tuple[str, ...], actor_kind: str) -> None:
    if actor_kind == "pnn":
        _expand_action_head(actor)
    elif actor_kind == "transformer":
        # New Transformer actors may natively consume the full DR-aware schema
        # and emit the two raw actions either with split heads or one shared
        # 2-D head. Historical actors still use the adapter path.
        native_two_action = bool(
            getattr(actor, "native_dr_split_heads", False)
            or getattr(actor, "native_dr_action_head", False)
        )
        if not native_two_action:
            _adapt_transformer_features(actor, schema, pnn=False)
            _expand_action_head(actor)
    elif actor_kind == "pnn_transformer":
        _adapt_transformer_features(actor, schema, pnn=True)
        _expand_action_head(actor)
    else:
        raise ValueError(f"Unknown DR actor kind: {actor_kind}")
    if hasattr(actor, "_parameter_count"):
        actor._parameter_count = sum(parameter.numel() for parameter in actor.parameters())


def _branch_parameters(actor: nn.Module) -> dict[str, tuple[nn.Parameter, ...]]:
    result: dict[str, tuple[nn.Parameter, ...]] = {}
    for name, items in actor.branch_modules().items():
        parameters: list[nn.Parameter] = []
        for item in items:
            if isinstance(item, nn.Module):
                parameters.extend(item.parameters())
            elif isinstance(item, nn.Parameter):
                parameters.append(item)
            else:
                raise TypeError(f"Unsupported branch item {type(item)!r} in {name}")
        result[name] = tuple(parameters)
    return result


def _rebuild_actor_diagnostics(algorithm) -> None:
    algorithm._actor_parameters = {
        mg: tuple(actor.parameters()) for mg, actor in algorithm.actors.items()
    }
    algorithm._actor_branch_parameters = {
        mg: _branch_parameters(actor) for mg, actor in algorithm.actors.items()
    }
    algorithm._actor_layer_parameters = {
        mg: {
            name: tuple(module.parameters())
            for name, module in actor.named_children()
            if any(True for _ in module.parameters())
        }
        for mg, actor in algorithm.actors.items()
    }


def _projection_signature(env, projector: JointActionProjector, state_schema: tuple[str, ...]) -> str:
    payload = {
        "semantics": projector.semantics,
        "config": env.config,
        "schema": state_schema,
        "normalizers": env.global_state_builder.norm,
        "price_norm": env.global_state_builder.price_norm,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _adapt_algorithm_instance(algorithm, env, config: Mapping, actor_kind: str) -> None:
    if env.ess_sharing_control_mode != "single_dispatch":
        raise ValueError("DR algorithm variants require single_dispatch ESS control")
    if str(getattr(env, "action_design", "")) != "single_dispatch_dr" or int(env.action_dim) != 2:
        raise ValueError("DR algorithm variants require raw actions [ESS intent, DR intent]")

    settings = config.get("algorithm", config)
    algorithm.observation_schemas = {
        mg: tuple(env.obs_builder.schema(mg)) for mg in algorithm.agent_ids
    }
    algorithm.state_schema = tuple(env.global_state_builder.schema())

    for mg in algorithm.agent_ids:
        _adapt_actor(algorithm.actors[mg], algorithm.observation_schemas[mg], actor_kind)
    algorithm.target_actors = copy.deepcopy(algorithm.actors).to(algorithm.device)
    algorithm.actor_optimizers = {
        mg: torch.optim.Adam(
            algorithm.actors[mg].parameters(), lr=float(settings.get("actor_lr", 2e-4))
        )
        for mg in algorithm.agent_ids
    }
    _rebuild_actor_diagnostics(algorithm)

    hidden_dims = tuple(settings.get("hidden_dims", (128, 128)))
    joint_effective_dim = EFFECTIVE_ACTION_DIM * len(algorithm.agent_ids)
    algorithm.critics = nn.ModuleDict({
        mg: TwinCritic(env.global_state_dim, joint_effective_dim, hidden_dims)
        for mg in algorithm.agent_ids
    }).to(algorithm.device)
    algorithm.target_critics = copy.deepcopy(algorithm.critics).to(algorithm.device)
    algorithm.critic_optimizers = {
        mg: torch.optim.Adam(
            algorithm.critics[mg].parameters(), lr=float(settings.get("critic_lr", 3e-4))
        )
        for mg in algorithm.agent_ids
    }

    algorithm.projector = JointActionProjector(
        env.config,
        algorithm.state_schema,
        env.global_state_builder.norm,
        env.global_state_builder.price_norm,
    )
    algorithm.projection_signature = _projection_signature(
        env, algorithm.projector, algorithm.state_schema
    )
    algorithm.raw_action_dim = RAW_ACTION_DIM
    algorithm.effective_action_dim = EFFECTIVE_ACTION_DIM
    algorithm.action_dim = RAW_ACTION_DIM


def _checkpoint_payload(algorithm, slug: str) -> dict:
    return {
        "algorithm_slug": slug,
        "action_semantics": ACTION_SEMANTICS,
        "projection_signature": algorithm.projection_signature,
        "observation_schemas": algorithm.observation_schemas,
        "state_schema": algorithm.state_schema,
        "actors": algorithm.actors.state_dict(),
        "target_actors": algorithm.target_actors.state_dict(),
        "critics": algorithm.critics.state_dict(),
        "target_critics": algorithm.target_critics.state_dict(),
        "update_steps": algorithm.update_steps,
    }


def install_dr_environment_adapter(cls, *, slug: str, actor_kind: str):
    """Adapt one historical experimental algorithm to the heterogeneous ESS+DR environment.

    The historical representation architecture is retained. Only the environment-facing action
    interface, DR observation inputs, physical projection, critic action width, and checkpoint
    semantics are migrated to the canonical ``single_dispatch_dr`` contract.
    """
    marker = f"_dr_environment_adapter_{slug}"
    if getattr(cls, marker, False):
        return cls

    original_init = cls.__init__

    @wraps(original_init)
    def wrapped_init(self, env, config, device="cpu", seed=42):
        original_init(self, env, config, device=device, seed=seed)
        _adapt_algorithm_instance(self, env, config, actor_kind)

    def save(self, path):
        torch.save(_checkpoint_payload(self, slug), Path(path))

    def load(self, path):
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm_slug") != slug:
            raise ValueError("Checkpoint belongs to a different algorithm")
        if checkpoint.get("action_semantics") != ACTION_SEMANTICS:
            raise ValueError("Checkpoint belongs to the legacy pre-DR action space")
        if checkpoint.get("projection_signature") != self.projection_signature:
            raise ValueError("Checkpoint physical projection/configuration does not match")
        if checkpoint.get("state_schema") != self.state_schema:
            raise ValueError("Checkpoint state schema does not match")
        if checkpoint.get("observation_schemas") != self.observation_schemas:
            raise ValueError("Checkpoint observation schema does not match")
        for name in ("actors", "target_actors", "critics", "target_critics"):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.update_steps = int(checkpoint.get("update_steps", 0))

    cls.__init__ = wrapped_init
    cls.save = save
    cls.load = load
    setattr(cls, marker, True)
    return cls
