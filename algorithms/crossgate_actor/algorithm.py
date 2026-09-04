from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from algorithms.crossgate_actor.actor import CrossGateActor as CrossGateActorNetwork
from algorithms.matd3 import MATD3


class CrossGateActor(MATD3):
    """MATD3 with isolated actor-side cross-microgrid feature gating."""

    def __init__(
        self,
        agent_ids: list[str],
        observation_dims: Mapping[str, int],
        observation_schemas: Mapping[str, Sequence[str]],
        global_state_dim: int,
        action_dim: int = 2,
        hidden_dims: tuple[int, ...] = (128, 128),
        actor_lr: float = 3e-4,
        local_encoder_dim: int = 32,
        peer_encoder_dim: int = 32,
        market_encoder_dim: int = 16,
        battery_hidden_dims: tuple[int, ...] = (64,),
        offer_hidden_dims: tuple[int, ...] = (96, 64),
        gate_strength: float = 0.25,
        detach_battery_for_offer: bool = True,
        **matd3_kwargs: Any,
    ) -> None:
        super().__init__(
            agent_ids=agent_ids,
            observation_dims=observation_dims,
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            actor_lr=actor_lr,
            **matd3_kwargs,
        )
        schemas = {mg: tuple(observation_schemas[mg]) for mg in self.agent_ids}
        for mg in self.agent_ids:
            if len(schemas[mg]) != int(observation_dims[mg]):
                raise ValueError(f"{mg}: observation schema and dimension do not match")
        self.observation_schemas = schemas
        self.actors = nn.ModuleDict({
            mg: CrossGateActorNetwork(
                schemas[mg], self.agent_ids, mg, action_dim,
                local_encoder_dim=local_encoder_dim,
                peer_encoder_dim=peer_encoder_dim,
                market_encoder_dim=market_encoder_dim,
                battery_hidden_dims=battery_hidden_dims,
                offer_hidden_dims=offer_hidden_dims,
                gate_strength=gate_strength,
                detach_battery_for_offer=detach_battery_for_offer,
            )
            for mg in self.agent_ids
        }).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.actor_optimizers = {
            mg: torch.optim.Adam(self.actors[mg].parameters(), lr=actor_lr)
            for mg in self.agent_ids
        }

    @classmethod
    def from_env(
        cls,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
        seed: int | None = None,
    ) -> "CrossGateActor":
        settings = config.get("algorithm", config)
        allowed = {
            "hidden_dims", "actor_lr", "critic_lr", "gamma", "tau", "policy_delay",
            "target_policy_noise", "target_noise_clip", "replay_capacity",
        }
        kwargs = {key: settings[key] for key in allowed if key in settings}
        if "hidden_dims" in kwargs:
            kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])
        crossgate = settings.get("crossgate", {})
        battery_hidden_dims = tuple(crossgate.get("battery_hidden_dims", (64,)))
        offer_hidden_dims = tuple(crossgate.get("offer_hidden_dims", (96, 64)))
        schemas = {mg: tuple(env.obs_builder.schema(mg)) for mg in env.agent_ids}
        return cls(
            env.agent_ids,
            env.observation_dims,
            schemas,
            env.global_state_dim,
            env.action_dim,
            seed=env.seed if seed is None else int(seed),
            device=device,
            local_encoder_dim=int(crossgate.get("local_encoder_dim", 32)),
            peer_encoder_dim=int(crossgate.get("peer_encoder_dim", 32)),
            market_encoder_dim=int(crossgate.get("market_encoder_dim", 16)),
            battery_hidden_dims=battery_hidden_dims,
            offer_hidden_dims=offer_hidden_dims,
            gate_strength=float(crossgate.get("gate_strength", 0.25)),
            detach_battery_for_offer=bool(crossgate.get("detach_battery_for_offer", True)),
            **kwargs,
        )

    def gate_diagnostics(self) -> dict[str, dict[str, float]]:
        return {
            mg: dict(self.actors[mg].gate_diagnostics())
            for mg in self.agent_ids
        }

    def actor_parameter_counts(self) -> dict[str, int]:
        return {mg: int(self.actors[mg].parameter_count) for mg in self.agent_ids}

    def update(self, batch_size: int) -> dict[str, Any]:
        metrics = super().update(batch_size)
        if not metrics:
            return metrics
        by_agent = self.gate_diagnostics()
        metrics["crossgate_gate_diagnostics"] = by_agent
        for feature in (
            "local_gate_mean", "local_gate_std", "local_gate_low_rate", "local_gate_high_rate",
            "peer_gate_mean", "peer_gate_std", "peer_gate_low_rate", "peer_gate_high_rate",
        ):
            values = [stats[feature] for stats in by_agent.values() if feature in stats]
            metrics[f"crossgate_{feature}"] = float(np.mean(values)) if values else None
        counts = self.actor_parameter_counts()
        metrics["actor_parameter_counts"] = counts
        metrics["actor_parameter_count"] = float(np.mean(list(counts.values())))
        return metrics
