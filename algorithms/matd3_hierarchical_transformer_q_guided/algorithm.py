from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from algorithms.matd3_hierarchical_transformer.algorithm import (
    Matd3HierarchicalTransformer,
)
from core.reward_v2 import stable_contract_hash


class Matd3HierarchicalTransformerQGuided(Matd3HierarchicalTransformer):
    """Ordinary T with conservative Q-guided ESS-logit recovery."""

    SLUG = "matd3_hierarchical_transformer_q_guided"

    def __init__(self, env, config, device="cpu", seed=42):
        super().__init__(env, config, device=device, seed=seed)
        settings = config.get("algorithm", config)
        self.learning_starts = int(settings.get("learning_starts", 1000))
        if not 0 <= self.learning_starts <= self.replay.capacity:
            raise ValueError("learning_starts must be between 0 and replay_capacity")

        self.dispatch_random_action_prob = float(
            settings.get("dispatch_random_action_prob", 0.10)
        )
        if (
            not math.isfinite(self.dispatch_random_action_prob)
            or not 0.0 <= self.dispatch_random_action_prob <= 1.0
        ):
            raise ValueError("dispatch_random_action_prob must be in [0, 1]")

        self.dispatch_q_guidance_weight = float(
            settings.get("dispatch_q_guidance_weight", 0.01)
        )
        self.dispatch_q_guidance_margin = float(
            settings.get("dispatch_q_guidance_margin", 0.01)
        )
        for name in ("dispatch_q_guidance_weight", "dispatch_q_guidance_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        self.dispatch_candidates = torch.tensor(
            [-0.9, -0.5, 0.0, 0.5, 0.9], dtype=torch.float32, device=self.device
        )

        # The inherited implementation builds this contract for ordinary T.
        # Give the experiment its own identity and a matching integrity hash.
        if self.reward_v2_contract is not None:
            contract = copy.deepcopy(self.reward_v2_contract)
            contract.pop("signature", None)
            contract["algorithm_slug"] = self.SLUG
            contract["signature"] = stable_contract_hash(contract)
            self.reward_v2_contract = contract

    def select_actions(self, observations, noise_std: float = 0.0):
        actions = super().select_actions(observations, noise_std=noise_std)
        if noise_std > 0.0 and self.dispatch_random_action_prob > 0.0:
            for mg in self.agent_ids:
                if self.rng.random() < self.dispatch_random_action_prob:
                    # ESS dispatch is action channel zero; leave the DR channel alone.
                    actions[mg][0] = np.float32(self.rng.uniform(-1.0, 1.0))
        return actions

    def update(self, batch_size: int) -> dict[str, Any]:
        if len(self.replay) < self.learning_starts:
            return {}
        return super().update(batch_size)

    @staticmethod
    def _ess_pre_tanh_logit(actor, observations: torch.Tensor) -> torch.Tensor:
        """Recompute the ESS head before tanh, preserving the Transformer graph."""
        tokens, padding_mask = actor._build_tokens(observations)
        attended = actor.transformer(tokens, src_key_padding_mask=padding_mask)
        ess_rep = attended[:, 0, :]
        supply_rep = attended[:, 2, :]
        trade_rep = attended[:, 3, :]
        ess_features = observations.index_select(-1, actor.ess_idx_tensor)
        ess_skip = actor.ess_linear_skip(ess_features)
        ess_input = torch.cat([ess_rep, supply_rep, trade_rep, ess_skip], dim=-1)
        return actor.ess_head[:-1](ess_input).squeeze(-1)

    def actor_auxiliary_loss(
        self,
        active: str,
        state: torch.Tensor,
        joint_raw: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Nudge only blocked ESS intents when both critics approve a probe."""
        index = self.agent_ids.index(active)
        zero = joint_raw[:, index, 0].sum() * 0.0
        metrics = {
            "dispatch_q_guidance_trigger_rate": zero.detach(),
            "dispatch_q_guidance_loss": zero.detach(),
            "dispatch_q_guidance_q_gain": zero.detach(),
        }
        if self.dispatch_q_guidance_weight == 0.0:
            return zero, metrics

        with torch.no_grad():
            raw = joint_raw.detach()
            intent = raw[:, index, 0]
            charge = state[:, self.projector.index[f"{active}_feasible_charge_max"]]
            discharge = state[:, self.projector.index[f"{active}_feasible_discharge_max"]]
            blocked = ((intent > 0.0) & (discharge <= 1e-8)) | (
                (intent < 0.0) & (charge <= 1e-8)
            )
            if not blocked.any():
                return zero, metrics

            selected_state = state[blocked]
            selected_raw = raw[blocked]
            current_effective, _ = self.project_actions(selected_state, selected_raw)
            current_q1, current_q2 = self.critics[active](
                selected_state, current_effective.flatten(1)
            )

            batch_count = selected_state.shape[0]
            candidate_count = self.dispatch_candidates.numel()
            probes = selected_raw[:, None].expand(
                -1, candidate_count, -1, -1
            ).clone()
            probes[:, :, index, 0] = self.dispatch_candidates
            probe_state = selected_state[:, None].expand(
                -1, candidate_count, -1
            ).reshape(batch_count * candidate_count, -1)
            probe_effective, _ = self.project_actions(
                probe_state,
                probes.reshape(batch_count * candidate_count, len(self.agent_ids), 2),
            )
            candidate_q1, candidate_q2 = self.critics[active](
                probe_state, probe_effective.flatten(1)
            )
            gains = torch.minimum(
                candidate_q1.reshape(batch_count, candidate_count) - current_q1,
                candidate_q2.reshape(batch_count, candidate_count) - current_q2,
            )
            gain, choice = gains.max(dim=1)
            approved = gain > self.dispatch_q_guidance_margin
            if not approved.any():
                return zero, metrics
            targets = torch.atanh(self.dispatch_candidates[choice])

        # The auxiliary gradient bypasses tanh but does not bypass physical
        # execution: only the ESS output logit is nudged toward an approved probe.
        logits = self._ess_pre_tanh_logit(
            self.actors[active], observations[blocked]
        )
        loss = self.dispatch_q_guidance_weight * (
            logits[approved] - targets[approved]
        ).square().sum() / state.shape[0]
        metrics.update(
            dispatch_q_guidance_trigger_rate=(
                approved.float().sum() / state.shape[0]
            ),
            dispatch_q_guidance_loss=loss.detach(),
            dispatch_q_guidance_q_gain=gain[approved].mean(),
        )
        return loss, metrics

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "algorithm_slug": self.SLUG,
                "actor_feature_semantics": self.actor_feature_semantics,
                "actor_feature_normalizers": copy.deepcopy(
                    self.actor_feature_normalizers
                ),
                "action_semantics": self.projector.semantics,
                "projection_signature": self.projection_signature,
                "observation_schemas": self.observation_schemas,
                "state_schema": self.state_schema,
                "actor_forecast_context_version": self.actor_forecast_context_version,
                "peer_forecast_features": list(self.peer_forecast_features),
                "reward_v2_contract": copy.deepcopy(self.reward_v2_contract),
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
            Path(path), map_location=self.device, weights_only=False
        )
        if checkpoint.get("algorithm_slug") != self.SLUG:
            raise ValueError("Checkpoint belongs to a different algorithm")
        if self.reward_v2_contract is not None:
            saved_contract = checkpoint.get("reward_v2_contract")
            if saved_contract is None or saved_contract != self.reward_v2_contract:
                raise ValueError("Reward V2 checkpoint contract mismatch")
            if (
                checkpoint.get("actor_forecast_context_version")
                != self.actor_forecast_context_version
            ):
                raise ValueError("Actor forecast context version mismatch")
        self._validate_actor_feature_metadata(checkpoint)
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
