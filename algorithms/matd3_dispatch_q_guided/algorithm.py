from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch

from algorithms.matd3_dispatch.plugin import Matd3Dispatch


class Matd3DispatchQGuided(Matd3Dispatch):
    """Recover ineffective dispatch intents using agreement of the twin critics."""

    SLUG = "matd3_dispatch_q_guided"

    def __init__(self, env, config, device="cpu", seed=42):
        super().__init__(env, config, device=device, seed=seed)
        settings = config.get("algorithm", config)
        self.learning_starts = int(settings.get("learning_starts", 1000))
        if not 0 <= self.learning_starts <= self.replay.capacity:
            raise ValueError("learning_starts must be between 0 and replay_capacity")
        self.dispatch_random_action_prob = float(settings.get("dispatch_random_action_prob", 0.10))
        if not math.isfinite(self.dispatch_random_action_prob) or not 0 <= self.dispatch_random_action_prob <= 1:
            raise ValueError("dispatch_random_action_prob must be in [0, 1]")
        self.dispatch_q_guidance_weight = float(settings.get("dispatch_q_guidance_weight", 0.01))
        self.dispatch_q_guidance_margin = float(settings.get("dispatch_q_guidance_margin", 0.01))
        for name in ("dispatch_q_guidance_weight", "dispatch_q_guidance_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        # Finite logit targets; these are probes, not a discrete action space.
        self.dispatch_candidates = torch.tensor(
            [-0.9, -0.5, 0.0, 0.5, 0.9], device=self.device
        )

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

    def actor_auxiliary_loss(self, active, state, joint_raw, observations):
        index = self.agent_ids.index(active)
        zero = joint_raw[:, index, 0].sum() * 0.0
        metrics = {
            "dispatch_q_guidance_trigger_rate": zero.detach(),
            "dispatch_q_guidance_loss": zero.detach(),
            "dispatch_q_guidance_q_gain": zero.detach(),
        }
        if self.dispatch_q_guidance_weight == 0:
            return zero, metrics

        with torch.no_grad():
            raw = joint_raw.detach()
            intent = raw[:, index, 0]
            charge = state[:, self.projector.index[f"{active}_feasible_charge_max"]]
            discharge = state[:, self.projector.index[f"{active}_feasible_discharge_max"]]
            blocked = ((intent > 0) & (discharge <= 1e-8)) | ((intent < 0) & (charge <= 1e-8))
            if not blocked.any():
                return zero, metrics
            selected_state, selected_raw = state[blocked], raw[blocked]
            effective, _ = self.project_actions(selected_state, selected_raw)
            q1, q2 = self.critics[active](selected_state, effective.flatten(1))

            count, candidates = selected_state.shape[0], self.dispatch_candidates.numel()
            probes = selected_raw[:, None].expand(-1, candidates, -1, -1).clone()
            probes[:, :, index, 0] = self.dispatch_candidates
            probe_state = selected_state[:, None].expand(-1, candidates, -1).reshape(
                count * candidates, -1
            )
            probe_effective, _ = self.project_actions(
                probe_state, probes.reshape(count * candidates, len(self.agent_ids), 2)
            )
            candidate_q1, candidate_q2 = self.critics[active](probe_state, probe_effective.flatten(1))
            # Both critics must improve over their own current-action estimate.
            gains = torch.minimum(
                candidate_q1.reshape(count, candidates) - q1,
                candidate_q2.reshape(count, candidates) - q2,
            )
            gain, choice = gains.max(dim=1)
            approved = gain > self.dispatch_q_guidance_margin
            if not approved.any():
                return zero, metrics
            targets = torch.atanh(self.dispatch_candidates[choice])

        # Regress pre-tanh dispatch logits so saturation cannot erase this signal.
        # Only this auxiliary objective bypasses tanh; physical execution is unchanged.
        logits = self.actors[active].network[:-1](observations[blocked])[:, 0]
        loss = self.dispatch_q_guidance_weight * (
            logits[approved] - targets[approved]
        ).square().sum() / state.shape[0]
        metrics.update(
            dispatch_q_guidance_trigger_rate=approved.float().sum() / state.shape[0],
            dispatch_q_guidance_loss=loss.detach(),
            dispatch_q_guidance_q_gain=gain[approved].mean(),
        )
        return loss, metrics

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
