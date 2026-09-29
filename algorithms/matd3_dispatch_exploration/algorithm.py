from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from algorithms.matd3_dispatch.plugin import Matd3Dispatch


class Matd3DispatchExploration(Matd3Dispatch):
    """Dispatch baseline with delayed learning and occasional random ESS intent."""

    SLUG = "matd3_dispatch_exploration"

    def __init__(self, env, config, device="cpu", seed=42):
        super().__init__(env, config, device=device, seed=seed)
        settings = config.get("algorithm", config)
        self.learning_starts = int(settings.get("learning_starts", 1000))
        if not 0 <= self.learning_starts <= self.replay.capacity:
            raise ValueError("learning_starts must be between 0 and replay_capacity")
        self.dispatch_random_action_prob = float(
            settings.get("dispatch_random_action_prob", 0.10)
        )
        if not np.isfinite(self.dispatch_random_action_prob) or not 0 <= self.dispatch_random_action_prob <= 1:
            raise ValueError("dispatch_random_action_prob must be in [0, 1]")

    def select_actions(self, observations, noise_std=0.0):
        actions = super().select_actions(observations, noise_std=noise_std)
        if noise_std > 0 and self.dispatch_random_action_prob > 0:
            for mg in self.agent_ids:
                if self.rng.random() < self.dispatch_random_action_prob:
                    actions[mg][0] = np.float32(self.rng.uniform(-1.0, 1.0))
        return actions

    def update(self, batch_size):
        if len(self.replay) < self.learning_starts:
            return {}
        return super().update(batch_size)

    def save(self, path):
        torch.save({
            "algorithm_slug": self.SLUG,
            "action_semantics": self.projector.semantics,
            "projection_signature": self.projection_signature,
            "observation_schemas": self.observation_schemas,
            "state_schema": self.state_schema,
            "reward_v2_contract": copy.deepcopy(self.reward_v2_contract),
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "update_steps": self.update_steps,
        }, Path(path))

    def load(self, path):
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm_slug") != self.SLUG:
            raise ValueError("Checkpoint belongs to a different algorithm")
        if self.reward_v2_contract is not None:
            saved_contract = checkpoint.get("reward_v2_contract")
            if saved_contract is None or saved_contract != self.reward_v2_contract:
                raise ValueError("Reward V2 checkpoint contract mismatch")
        if checkpoint.get("action_semantics") != self.projector.semantics:
            raise ValueError("Incompatible action semantics")
        if checkpoint.get("projection_signature") != self.projection_signature:
            raise ValueError("Checkpoint physical projection/configuration does not match")
        if checkpoint.get("state_schema") != self.state_schema:
            raise ValueError("Checkpoint state schema does not match")
        if checkpoint.get("observation_schemas") != self.observation_schemas:
            raise ValueError("Checkpoint observation schema does not match")
        for name in ("actors", "target_actors", "critics", "target_critics"):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.update_steps = int(checkpoint.get("update_steps", 0))
