"""Isolated MATD3 variant: only its centralized twin critics are replaced."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithms.diagnostic_sampling import DiagnosticSchedule, _select_actions, fast_plain_matd3_update
from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from algorithms.matd3_transformer_critic.critic import TwinDeviceTransformerCritic
from algorithms.runtime_helpers import FastReplayBuffer


SLUG = "matd3_transformer_critic"
NETWORK_VERSION = "device_action_cross_attention_37x6_v1"


class Matd3TransformerCritic(Matd3Dispatch):
    """Same raw actors, projected actions, rewards and workflow as MATD3 dispatch."""

    def __init__(self, env, config, device="cpu", seed=42):
        super().__init__(env, config, device=device, seed=seed)
        settings = config.get("algorithm", config)
        network = dict(settings.get("network", {}))
        horizon = int(network.get("forecast_horizon", 8))
        if int(env.global_state_builder.future_h) != horizon:
            raise ValueError("Transformer critic horizon must match the environment forecast horizon")
        self.network_config = {
            "horizon": horizon,
            "embed_dim": int(network.get("embed_dim", 32)),
            "heads": int(network.get("num_heads", 4)),
            "layers": int(network.get("num_layers", 1)),
            "hidden_dims": tuple(network.get("hidden_dims", (128, 128))),
        }
        self.critics = nn.ModuleDict({
            mg: TwinDeviceTransformerCritic(
                self.state_schema,
                env.config["microgrids"],
                env.global_state_builder.norm,
                **self.network_config,
            ) for mg in self.agent_ids
        }).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device)
        self.critic_optimizers = {
            mg: torch.optim.Adam(
                self.critics[mg].parameters(),
                lr=float(settings.get("critic_lr", 3e-4)),
            ) for mg in self.agent_ids
        }
        # Use the repository's existing fast replay and sparse-diagnostic update.
        # No attention weights or auxiliary rollouts are computed during normal updates.
        self.replay = FastReplayBuffer.from_replay(self.replay)
        self.diagnostic_schedule = DiagnosticSchedule(
            int(config.get("logging", {}).get("diagnostic_interval", 20)),
            self.policy_delay,
        )
        self._diagnostic_variant = SLUG
        self._diagnostic_plain_actor = True

    # Explicit methods protect the experimental variant from plugin import order:
    # loading the existing MATD3 plugin can monkeypatch its parent class.
    select_actions = _select_actions

    @staticmethod
    def _module_grad_norm(module: nn.Module) -> float:
        squared = sum(
            float(parameter.grad.detach().square().sum())
            for parameter in module.parameters() if parameter.grad is not None
        )
        return math.sqrt(squared)

    def update(self, batch_size):
        metrics = fast_plain_matd3_update(self, batch_size)
        if not metrics:
            return metrics
        # The shared pipeline persists q_stats as q_statistics per episode.
        # Gradients are already computed by the normal TD update; sample only
        # at the existing diagnostic cadence, not every environment step.
        for mg in self.agent_ids:
            critic = self.critics[mg]
            q_stats = metrics["q_stats"][mg]
            q_stats["state_encoder_grad_norm"] = self._module_grad_norm(
                critic.q1_network.state_encoder
            )
            q_stats["action_cross_attention_grad_norm"] = self._module_grad_norm(
                critic.q1_network.cross_attention
            )
        return metrics

    def save(self, path):
        torch.save({
            "algorithm_slug": SLUG,
            "network_version": NETWORK_VERSION,
            "network_config": self.network_config,
            "action_semantics": self.projector.semantics,
            "projection_signature": self.projection_signature,
            "observation_schemas": self.observation_schemas,
            "state_schema": self.state_schema,
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "update_steps": self.update_steps,
        }, Path(path))

    def load(self, path):
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        expected = {
            "algorithm_slug": SLUG,
            "network_version": NETWORK_VERSION,
            "network_config": self.network_config,
            "action_semantics": self.projector.semantics,
            "projection_signature": self.projection_signature,
            "observation_schemas": self.observation_schemas,
            "state_schema": self.state_schema,
        }
        for name, value in expected.items():
            if checkpoint.get(name) != value:
                raise ValueError(f"Checkpoint {name} does not match this algorithm/configuration")
        for name in ("actors", "target_actors", "critics", "target_critics"):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.update_steps = int(checkpoint.get("update_steps", 0))
