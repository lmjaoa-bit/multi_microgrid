from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

# The formal plugin installs the shared projected TD3 update and diagnostic hook.
from algorithms.matd3_hierarchical_transformer.plugin import Matd3HierarchicalTransformer
from algorithms.matd3_hierarchical_transformer.actor import HierarchicalTransformerActor
from algorithms.matd3_mlp_dueling_critic_h24_q_guided.actor import SemanticMlpActor
from algorithms.matd3_mlp_dueling_critic_h24_q_guided.critic import DuelingTwinCritic, resolve_critic_config
from core.reward_v2 import stable_contract_hash


class Matd3MlpDuelingCriticH24QGuided(Matd3HierarchicalTransformer):
    """H24 semantic MLP actors and dueling twin critics with ESS recovery."""

    SLUG = "matd3_mlp_dueling_critic_h24_q_guided"

    def __init__(self, env, config, device="cpu", seed=42):
        config = copy.deepcopy(config)
        settings = config.get("algorithm", config)
        actor_settings = settings.setdefault("network", {})
        actor_settings.setdefault("include_peer_forecast", True)
        actor_settings.setdefault("forecast_horizon", 24)
        horizon = actor_settings["forecast_horizon"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon != 24:
            raise ValueError("MLP Actor requires integer forecast_horizon=24 (H24)")
        if not actor_settings.get("include_peer_forecast", True):
            raise ValueError("MLP Actor requires the current peer forecast features")
        # The formal parent admits peer forecasts only with H8. Use that path
        # solely to initialize shared projection/replay/update machinery, then
        # construct H24 feature references locally. No parent policy is trained.
        parent_config = copy.deepcopy(config)
        parent_config.get("algorithm", parent_config)["network"]["forecast_horizon"] = 8
        super().__init__(env, parent_config, device=device, seed=seed)
        reference_settings = {
            key: int(actor_settings.get(key, default))
            for key, default in (("embed_dim", 32), ("num_heads", 4),
                                 ("num_layers", 2), ("hidden_dim", 64))
        }
        self.actors = nn.ModuleDict({
            mg: SemanticMlpActor(HierarchicalTransformerActor(
                self.observation_schemas[mg], self.agent_ids, mg,
                mg_config=env.cfg_by_id[mg],
                power_normalizers=self.actor_feature_normalizers[mg],
                forecast_horizon=24, include_peer_forecast=True,
                peer_forecast_features=self.peer_forecast_features,
                **reference_settings,
            ))
            for mg in self.agent_ids
        }).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.actor_optimizers = {mg: torch.optim.Adam(actor.parameters(), lr=float(settings.get("actor_lr", 2e-4)))
                                 for mg, actor in self.actors.items()}
        self._actor_parameters = {mg: tuple(actor.parameters()) for mg, actor in self.actors.items()}
        self._actor_branch_parameters = {mg: {"joint": params} for mg, params in self._actor_parameters.items()}
        self._actor_layer_parameters = {
            mg: {f"linear_{i}": tuple(layer.parameters()) for i, layer in enumerate(actor.network) if isinstance(layer, nn.Linear)}
            for mg, actor in self.actors.items()
        }
        self._diagnostic_plain_actor = True
        self._diagnostic_variant = self.SLUG
        self.critic_network_config = resolve_critic_config(settings.get("critic_network", {}))
        self.critics = nn.ModuleDict({
            mg: DuelingTwinCritic(self.state_schema, len(self.agent_ids) * 3, **self.critic_network_config)
            for mg in self.agent_ids
        }).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device)
        self.critic_optimizers = {mg: torch.optim.Adam(critic.parameters(), lr=float(settings.get("critic_lr", 3e-4)))
                                  for mg, critic in self.critics.items()}
        self.actor_network_config = {
            "architecture": "semantic_mlp_h24_v1", "forecast_horizon": 24,
            "feature_semantics": self.actor_feature_semantics,
            "hidden_dims_by_agent": {mg: [actor.hidden_dim, actor.hidden_dim] for mg, actor in self.actors.items()},
            "include_peer_forecast": True, "include_base_load_forecast": True,
            "peer_forecast_features": list(self.peer_forecast_features),
            "parameter_budget_reference": "t_q_device_forecast_readout_h24_v1",
        }
        self.actor_forecast_context_version = "semantic_mlp_peer_balance_base_load_h24_v1"
        self.learning_starts = int(settings.get("learning_starts", 1000))
        if not 0 <= self.learning_starts <= self.replay.capacity:
            raise ValueError("learning_starts must be between 0 and replay_capacity")
        self.dispatch_random_action_prob = float(settings.get("dispatch_random_action_prob", .10))
        if not math.isfinite(self.dispatch_random_action_prob) or not 0 <= self.dispatch_random_action_prob <= 1:
            raise ValueError("dispatch_random_action_prob must be in [0, 1]")
        self.dispatch_q_guidance_weight = float(settings.get("dispatch_q_guidance_weight", .01))
        self.dispatch_q_guidance_margin = float(settings.get("dispatch_q_guidance_margin", .01))
        for name in ("dispatch_q_guidance_weight", "dispatch_q_guidance_margin"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.dispatch_candidates = torch.tensor([-.9, -.5, 0., .5, .9], device=self.device)
        if self.reward_v2_contract is not None:
            contract = copy.deepcopy(self.reward_v2_contract)
            contract.pop("signature", None)
            contract.update(algorithm_slug=self.SLUG, actor_forecast_context_version=self.actor_forecast_context_version,
                            actor_network_config=self.actor_network_config, critic_network_config=self.critic_network_config,
                            critic_feature_semantics="full_state_h24_dueling_v_transformer_a_mlp_zero_ref_v1")
            contract["signature"] = stable_contract_hash(contract)
            self.reward_v2_contract = contract

    def select_actions(self, observations, noise_std=0.):
        actions = super().select_actions(observations, noise_std=noise_std)
        if noise_std > 0 and self.dispatch_random_action_prob > 0:
            for mg in self.agent_ids:
                if self.rng.random() < self.dispatch_random_action_prob:
                    actions[mg][0] = np.float32(self.rng.uniform(-1., 1.))
        return actions

    def update(self, batch_size):
        if len(self.replay) < self.learning_starts:
            return {}
        return super().update(batch_size)

    def actor_auxiliary_loss(self, active, state, joint_raw, observations):
        loss, metrics = self._ess_q_guidance_loss(active, state, joint_raw, observations)
        # Shared due() has already advanced this stateful cursor before entering
        # the Actor phase. Observe that advance once, then reuse it for all agents.
        boundary = self.diagnostic_schedule.next_boundary
        previous = getattr(self, "_dueling_diagnostic_boundary", self.diagnostic_schedule.interval)
        if boundary != previous:
            self._dueling_diagnostic_boundary = boundary
            self._dueling_diagnostic_step = self.update_steps
        if getattr(self, "_dueling_diagnostic_step", -1) == self.update_steps:
            metrics.update(self._dueling_critic_diagnostics(active, state, joint_raw))
        return loss, metrics

    @torch.no_grad()
    def _dueling_critic_diagnostics(self, active, state, joint_raw):
        effective, _ = self.project_actions(state.detach(), joint_raw.detach())
        action = effective.flatten(1).detach()
        critic = self.critics[active]
        parts = critic.decompose(state.detach(), action)
        value, advantage, q = parts["v1"], parts["adv1"], parts["q1"]
        # Sample std matches the study definition; singleton diagnostics use
        # population std so their zero variance remains finite.
        correction = len(state) > 1
        value_centered = value - value.mean()
        advantage_centered = advantage - advantage.mean()
        correlation = (value_centered * advantage_centered).mean() / (
            value.std(unbiased=False) * advantage.std(unbiased=False) + 1e-8
        )
        with torch.enable_grad():
            differentiable_action = action.clone().requires_grad_(True)
            gradient = torch.autograd.grad(
                critic.q1(state.detach(), differentiable_action).sum(), differentiable_action,
            )[0].detach()
        index = self.agent_ids.index(active) * 3
        return {
            "critic_v_mean": value.mean(),
            "critic_adv_std": advantage.std(unbiased=False),
            "critic_adv_abs_mean": advantage.abs().mean(),
            "critic_adv_share": advantage.std(unbiased=correction) / (q.std(unbiased=correction) + 1e-8),
            "critic_dqda_dispatch_rms": gradient[:, index:index + 2].square().mean().sqrt(),
            "critic_dqda_dr_rms": gradient[:, index + 2].square().mean().sqrt(),
            "critic_v_adv_corr": correlation,
        }

    def _ess_q_guidance_loss(self, active, state, joint_raw, observations):
        """Existing ESS-only conservative Q recovery, applied to MLP logits."""
        index = self.agent_ids.index(active)
        zero = joint_raw[:, index, 0].sum() * 0.
        metrics = {name: zero.detach() for name in
                   ("dispatch_q_guidance_trigger_rate", "dispatch_q_guidance_loss", "dispatch_q_guidance_q_gain")}
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
            count, candidates = len(selected_state), self.dispatch_candidates.numel()
            probes = selected_raw[:, None].expand(-1, candidates, -1, -1).clone()
            probes[:, :, index, 0] = self.dispatch_candidates
            probe_state = selected_state[:, None].expand(-1, candidates, -1).reshape(count * candidates, -1)
            probe_effective, _ = self.project_actions(probe_state, probes.reshape(count * candidates, len(self.agent_ids), 2))
            # Baseline and probes share the same state-only value context. Reuse
            # it within this call; never cache across updates or different states.
            actions = torch.cat([effective.flatten(1)[:, None],
                                 probe_effective.reshape(count, candidates, -1)], dim=1)
            scores1, scores2 = self.critics[active].q_candidates(selected_state, actions)
            gains = torch.minimum(scores1[:, 1:] - scores1[:, :1],
                                  scores2[:, 1:] - scores2[:, :1])
            gain, choice = gains.max(1)
            approved = gain > self.dispatch_q_guidance_margin
            if not approved.any():
                return zero, metrics
            targets = torch.atanh(self.dispatch_candidates[choice])
        logits = self.actors[active].action_logits(observations[blocked])[:, 0]
        loss = self.dispatch_q_guidance_weight * (logits[approved] - targets[approved]).square().sum() / len(state)
        metrics.update(dispatch_q_guidance_trigger_rate=approved.float().sum() / len(state),
                       dispatch_q_guidance_loss=loss.detach(), dispatch_q_guidance_q_gain=gain[approved].mean())
        return loss, metrics

    def _checkpoint_metadata(self):
        return {
            "algorithm_slug": self.SLUG,
            "actor_feature_semantics": self.actor_feature_semantics,
            "actor_feature_normalizers": copy.deepcopy(self.actor_feature_normalizers),
            "actor_network_config": copy.deepcopy(self.actor_network_config),
            "critic_network_config": copy.deepcopy(self.critic_network_config),
            "action_semantics": self.projector.semantics,
            "projection_signature": self.projection_signature,
            "observation_schemas": self.observation_schemas,
            "state_schema": self.state_schema,
            "actor_forecast_context_version": self.actor_forecast_context_version,
            "peer_forecast_features": list(self.peer_forecast_features),
            "reward_v2_contract": copy.deepcopy(self.reward_v2_contract),
        }

    def save(self, path):
        checkpoint = self._checkpoint_metadata()
        checkpoint.update({name: getattr(self, name).state_dict() for name in
                           ("actors", "target_actors", "critics", "target_critics")})
        checkpoint["update_steps"] = self.update_steps
        torch.save(checkpoint, Path(path))

    def load(self, path):
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        for name, expected in self._checkpoint_metadata().items():
            if checkpoint.get(name) != expected:
                raise ValueError(f"Checkpoint structure/contract mismatch: {name}")
        self._validate_actor_feature_metadata(checkpoint)
        # Preflight all networks, including fixed-feature buffers, before any write.
        for name in ("actors", "target_actors", "critics", "target_critics"):
            module = getattr(self, name)
            expected = module.state_dict()
            saved = checkpoint.get(name, {})
            parameters = dict(module.named_parameters())
            if saved.keys() != expected.keys():
                raise ValueError(f"Checkpoint state keys mismatch: {name}")
            for key, value in expected.items():
                tensor = saved[key]
                if not isinstance(tensor, torch.Tensor) or tensor.shape != value.shape or tensor.dtype != value.dtype:
                    raise ValueError(f"Checkpoint state shape/dtype mismatch: {name}.{key}")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Checkpoint nonfinite state: {name}.{key}")
                if key not in parameters and not torch.equal(tensor, value):
                    raise ValueError(f"Checkpoint fixed state mismatch: {name}.{key}")
        update_steps = checkpoint.get("update_steps", 0)
        if isinstance(update_steps, bool) or not isinstance(update_steps, int) or update_steps < 0:
            raise ValueError("Checkpoint update_steps must be a non-negative integer")
        for name in ("actors", "target_actors", "critics", "target_critics"):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.update_steps = update_steps
