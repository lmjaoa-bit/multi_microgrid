from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithms.matd3 import MATD3
from algorithms.crossgate.actor import SingleDispatchActor
from core.single_dispatch import SingleDispatchProjector


class Crossgate(MATD3):
    """signed-dispatch study; critics learn on projected physical commands."""

    @classmethod
    def from_env(cls, env, config, device="cpu", seed=None):
        return cls(env, config, device=device, seed=env.seed if seed is None else seed)

    def __init__(self, env, config, device="cpu", seed=42):
        if env.ess_sharing_control_mode != "single_dispatch":
            raise ValueError("This algorithm requires single_dispatch control")
        settings = config.get("algorithm", config)
        allowed = {"actor_lr", "critic_lr", "gamma", "tau", "policy_delay",
                   "target_policy_noise", "target_noise_clip", "replay_capacity"}
        kwargs = {key: settings[key] for key in allowed if key in settings}
        super().__init__(env.agent_ids, env.observation_dims, env.global_state_dim, 2,
                         hidden_dims=tuple(settings.get("hidden_dims", (128, 128))),
                         device=device, seed=seed, **kwargs)
        self.observation_schemas = {mg: tuple(env.obs_builder.schema(mg)) for mg in self.agent_ids}
        self.state_schema = tuple(env.global_state_builder.schema())
        self.projector = SingleDispatchProjector(env.config, self.state_schema,
            env.global_state_builder.norm, env.global_state_builder.price_norm)
        signature = dict(config=env.config, schema=self.state_schema,
                         normalizers=env.global_state_builder.norm,
                         price_norm=env.global_state_builder.price_norm)
        self.projection_signature = hashlib.sha256(
            json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()
        self.action_dim = 1  # Critics retain two projected physical controls per agent.
        net = settings.get("network", {})
        self.actors = nn.ModuleDict({
            mg: SingleDispatchActor(
                self.observation_schemas[mg], self.agent_ids, mg,
                local_encoder_dim=int(net.get("local_encoder_dim", 32)),
                peer_encoder_dim=int(net.get("peer_encoder_dim", 32)),
                market_encoder_dim=int(net.get("market_encoder_dim", 16)),
                hidden_dims=tuple(net.get("hidden_dims", (96, 64))),
                gate_strength=float(net.get("gate_strength", .25)),
            ) for mg in self.agent_ids
        }).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.actor_optimizers = {
            mg: torch.optim.Adam(self.actors[mg].parameters(), lr=float(settings.get("actor_lr", 2e-4)))
            for mg in self.agent_ids
        }
        self.actor_max_grad_norm = float(settings.get("actor_max_grad_norm", 1.0))
        if not np.isfinite(self.actor_max_grad_norm) or self.actor_max_grad_norm <= 0:
            raise ValueError("actor_max_grad_norm must be positive and finite")
        # Architecture is fixed after construction; cache parameter membership,
        # not gradient values. load_state_dict copies into the same Parameters.
        self._actor_parameters = {mg: tuple(actor.parameters()) for mg, actor in self.actors.items()}
        self._actor_branch_parameters = {
            mg: {name: tuple(p for module in modules for p in module.parameters())
                 for name, modules in actor.branch_modules().items()}
            for mg, actor in self.actors.items()
        }
        self._actor_layer_parameters = {
            mg: {name: tuple(module.parameters()) for name, module in actor.named_children()
                 if any(True for _ in module.parameters())}
            for mg, actor in self.actors.items()
        }

    def _actor_gradient_diagnostics(self, mg):
        # Read each tensor norm once and reuse it for total/branch/layer views.
        squared = {id(p): float(p.grad.detach().norm(2)) ** 2 if p.grad is not None else 0.0
                   for p in self._actor_parameters[mg]}
        def norm(parameters):
            return sum(squared[id(p)] for p in parameters) ** 0.5
        return (norm(self._actor_parameters[mg]),
                {name: norm(params) for name, params in self._actor_branch_parameters[mg].items()},
                {name: norm(params) for name, params in self._actor_layer_parameters[mg].items()})

    def actor_parameter_counts(self):
        return {mg: self.actors[mg].parameter_count for mg in self.agent_ids}

    def gate_diagnostics(self):
        return {mg: dict(self.actors[mg].gate_diagnostics()) for mg in self.agent_ids}

    @staticmethod
    def _modules_grad_norm(modules):
        return sum(float(p.grad.detach().square().sum()) for m in modules
                   for p in m.parameters() if p.grad is not None) ** 0.5

    def project_actions(self, state, joint_raw):
        return self.projector(state, joint_raw)

    def update(self, batch_size):
        if len(self.replay) < batch_size:
            return {}
        batch = self.replay.sample(batch_size, self.agent_ids, self.device)
        state, next_state = batch["global_state"], batch["next_global_state"]
        # Replay preserves raw controls and physical state. Effective controls are
        # reconstructed using the same decoder as rollout and targets.
        raw = torch.stack([batch["actions"][mg] for mg in self.agent_ids], dim=1)
        effective, _ = self.project_actions(state, raw)
        with torch.no_grad():
            target_raw = []
            for mg in self.agent_ids:
                action = self.target_actors[mg](batch["next_observations"][mg])
                noise = torch.randn_like(action).mul(self.target_policy_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip)
                target_raw.append((action + noise).clamp(-1, 1))
            target_effective, _ = self.project_actions(next_state, torch.stack(target_raw, dim=1))
            target_q = {}
            for mg in self.agent_ids:
                q1, q2 = self.target_critics[mg](next_state, target_effective.flatten(1))
                target_q[mg] = batch["team_reward"] + (
                    1 - batch["done"]) * self.gamma * torch.minimum(q1, q2)

        metrics = {}
        records, q_stats = {}, {}
        for mg in self.agent_ids:
            q1, q2 = self.critics[mg](state, effective.flatten(1))
            c1 = nn.functional.mse_loss(q1, target_q[mg])
            c2 = nn.functional.mse_loss(q2, target_q[mg])
            loss = c1 + c2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{mg}: nonfinite Critic loss")
            self.critic_optimizers[mg].zero_grad(set_to_none=True)
            loss.backward()
            norm = self._gradient_norm(self.critics[mg])
            if not np.isfinite(norm):
                raise FloatingPointError(f"{mg}: nonfinite Critic gradient")
            self.critic_optimizers[mg].step()
            td = target_q[mg] - torch.minimum(q1, q2)
            q_min = torch.minimum(q1.detach(), q2.detach())
            q_stats[mg] = dict(mean=float(q_min.mean()), min=float(q_min.min()),
                              max=float(q_min.max()), std=float(q_min.std(unbiased=False)))
            records[mg] = dict(critic_loss=float(loss.detach()), critic1_loss=float(c1.detach()),
                critic2_loss=float(c2.detach()), critic_grad_norm=norm,
                q1_mean=float(q1.detach().mean()), q2_mean=float(q2.detach().mean()),
                q1_q2_gap=float((q1.detach().mean() - q2.detach().mean()).abs()),
                target_q_mean=float(target_q[mg].mean()), td_error_mean=float(td.detach().mean()),
                td_error_std=float(td.detach().std(unbiased=False)))
        for key in next(iter(records.values())):
            metrics[key] = float(np.mean([r[key] for r in records.values()]))
        metrics["critic_losses"] = {mg: r["critic_loss"] for mg, r in records.items()}
        metrics["critic_grad_norms"] = {mg: r["critic_grad_norm"] for mg, r in records.items()}
        metrics["critic1_losses"] = {mg: r["critic1_loss"] for mg, r in records.items()}
        metrics["critic2_losses"] = {mg: r["critic2_loss"] for mg, r in records.items()}
        metrics["q_stats"] = q_stats

        self.update_steps += 1
        if self.update_steps % self.policy_delay == 0:
            losses, grads, branches, diagnostic, layers = {}, {}, {}, {}, {}
            for active_index, active in enumerate(self.agent_ids):
                critic = self.critics[active]
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
                try:
                    parts = []
                    for mg in self.agent_ids:
                        # Re-evaluate after each preceding actor update, preserving
                        # sequential semantics, but do not build inactive graphs.
                        with torch.set_grad_enabled(mg == active):
                            action = self.actors[mg](batch["observations"][mg])
                        parts.append(action)
                    active_action = parts[active_index]
                    active_action.retain_grad()
                    policy_effective, plans = self.project_actions(state, torch.stack(parts, dim=1))
                    loss = -critic.q1(state, policy_effective.flatten(1)).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{active}: nonfinite Actor loss")
                    self.actor_optimizers[active].zero_grad(set_to_none=True)
                    loss.backward()
                    actor = self.actors[active]
                    before, branch, layer = self._actor_gradient_diagnostics(active)
                    if not np.isfinite(before):
                        raise FloatingPointError(f"{active}: nonfinite Actor gradient")
                    grads[active], branches[active] = before, branch
                    layers[active] = layer
                    action_grad = active_action.grad.detach() * batch_size
                    # No separate offer action exists. Do not mask the policy's
                    # local-charge/discharge learning when ESS selling is unavailable.
                    plan = plans[active_index]
                    parameters = self._actor_parameters[active]
                    original = [p.detach().clone() for p in parameters]
                    torch.nn.utils.clip_grad_norm_(parameters, self.actor_max_grad_norm,
                                                   error_if_nonfinite=True)
                    after = self._gradient_norm(actor)
                    self.actor_optimizers[active].step()
                    step_norm = sum(float((p.detach() - old).square().sum())
                                    for p, old in zip(parameters, original)) ** 0.5
                    param_norm = sum(float(old.square().sum()) for old in original) ** 0.5
                    diagnostic[active] = {
                        "total_grad_norm": before, "post_clip_grad_norm": after,
                        "local_feature_grad_norm": branch["local_features"],
                        "market_feature_grad_norm": branch["market_features"],
                        "peer_feature_grad_norm": branch["peer_features"],
                        "gate_grad_norm": branch["gates"], "dispatch_grad_norm": branch["dispatch"],
                        "clip_applied": float(before > self.actor_max_grad_norm),
                        "dispatch_action_grad_rms": float(action_grad[:, 0].square().mean().sqrt()),
                        "dispatch_saturation_rate": float((active_action.detach().abs() >= .95).float().mean()),
                        "offer_eligible_rate": float(plan["mask"].detach().float().mean()),
                        "local_override_kw_max": float(plan["local_priority_reserve"].detach().abs().max()),
                        "optimizer_step_norm": step_norm,
                        "relative_optimizer_step_norm": step_norm / max(param_norm, 1e-20),
                    }
                    losses[active] = float(loss.detach())
                    self._soft_update(actor, self.target_actors[active])
                    self._soft_update(critic, self.target_critics[active])
                finally:
                    for parameter in critic.parameters():
                        parameter.requires_grad_(True)
            metrics.update(actor_loss=float(np.mean(list(losses.values()))), actor_losses=losses,
                actor_grad_norm=float(np.mean(list(grads.values()))), actor_grad_norms=grads,
                actor_branch_grad_norms=branches, actor_layer_grad_norms=layers,
                actor_diagnostics=diagnostic)
        counts = self.actor_parameter_counts()
        metrics["actor_parameter_counts"] = counts
        metrics["actor_parameter_count"] = float(np.mean(list(counts.values())))
        gates = self.gate_diagnostics()
        metrics["crossgate_gate_diagnostics"] = gates
        for key in next(iter(gates.values()), {}):
            metrics[f"crossgate_{key}"] = float(np.mean([g[key] for g in gates.values()]))
        return metrics

    def save(self, path):
        torch.save({
            "algorithm_slug": "crossgate", "action_semantics": "single_dispatch_v1",
            "projection_signature": self.projection_signature,
            "observation_schemas": self.observation_schemas, "state_schema": self.state_schema,
            "actors": self.actors.state_dict(), "target_actors": self.target_actors.state_dict(),
            "critics": self.critics.state_dict(), "target_critics": self.target_critics.state_dict(),
            "update_steps": self.update_steps,
        }, Path(path))

    def load(self, path):
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm_slug") != "crossgate" or checkpoint.get("action_semantics") != "single_dispatch_v1":
            raise ValueError("Checkpoint belongs to a different algorithm/action space")
        if checkpoint.get("projection_signature") != self.projection_signature:
            raise ValueError("Checkpoint physical projection/configuration does not match")
        if checkpoint.get("state_schema") != self.state_schema:
            raise ValueError("Checkpoint state schema does not match")
        if checkpoint.get("observation_schemas") != self.observation_schemas:
            raise ValueError("Checkpoint observation schema does not match")
        for name in ("actors", "target_actors", "critics", "target_critics"):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.update_steps = int(checkpoint.get("update_steps", 0))
