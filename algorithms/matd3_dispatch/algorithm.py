from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithms.matd3 import MATD3
from algorithms.matd3.actor import Actor
from core.joint_action_projection import JointActionProjector


class Matd3Dispatch(MATD3):
    """Flat CTDE MATD3 for the heterogeneous ESS + DR environment.

    Actors store and emit the two raw controls ``[ESS intent, DR intent]``.  The
    centralized critics never consume those raw controls directly: rollout,
    replay updates, target actions and policy gradients all pass through the
    same differentiable projector and therefore use three effective physical
    controls per microgrid: ``[ESS-local, ESS-trade, DR-effective]``.
    """

    @classmethod
    def from_env(cls, env, config, device="cpu", seed=None):
        return cls(env, config, device=device, seed=env.seed if seed is None else seed)

    def __init__(self, env, config, device="cpu", seed=42):
        if env.ess_sharing_control_mode != "single_dispatch":
            raise ValueError("This algorithm requires single_dispatch ESS control")
        if int(env.action_dim) != 2:
            raise ValueError("MATD3 DR baseline requires raw actions [ESS intent, DR intent]")

        settings = config.get("algorithm", config)
        allowed = {"actor_lr", "critic_lr", "gamma", "tau", "policy_delay",
                   "target_policy_noise", "target_noise_clip", "replay_capacity"}
        kwargs = {key: settings[key] for key in allowed if key in settings}

        # The base class creates the centralized twin critics.  They see three
        # projected physical controls per agent, not the two raw actor outputs.
        super().__init__(env.agent_ids, env.observation_dims, env.global_state_dim, 3,
                         hidden_dims=tuple(settings.get("hidden_dims", (128, 128))),
                         device=device, seed=seed, **kwargs)
        self.raw_action_dim = int(env.action_dim)
        self.effective_action_dim = 3
        self.action_dim = self.raw_action_dim
        self.observation_schemas = {mg: tuple(env.obs_builder.schema(mg)) for mg in self.agent_ids}
        self.state_schema = tuple(env.global_state_builder.schema())
        self.projector = JointActionProjector(
            env.config,
            self.state_schema,
            env.global_state_builder.norm,
            env.global_state_builder.price_norm,
        )
        signature = dict(
            semantics=self.projector.semantics,
            config=env.config,
            schema=self.state_schema,
            normalizers=env.global_state_builder.norm,
            price_norm=env.global_state_builder.price_norm,
        )
        self.projection_signature = hashlib.sha256(
            json.dumps(signature, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Rebuild only the actors because the base constructor used the
        # effective critic action dimension (3) for its temporary actors.
        hidden = tuple(settings.get("hidden_dims", (128, 128)))
        self.actors = nn.ModuleDict({
            mg: Actor(env.observation_dims[mg], self.raw_action_dim, hidden)
            for mg in self.agent_ids
        }).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.actor_optimizers = {
            mg: torch.optim.Adam(self.actors[mg].parameters(), lr=float(settings.get("actor_lr", 2e-4)))
            for mg in self.agent_ids
        }
        self.actor_max_grad_norm = float(settings.get("actor_max_grad_norm", 1.0))
        if not np.isfinite(self.actor_max_grad_norm) or self.actor_max_grad_norm <= 0:
            raise ValueError("actor_max_grad_norm must be positive and finite")

        self._actor_parameters = {mg: tuple(actor.parameters()) for mg, actor in self.actors.items()}
        self._actor_branch_parameters = {
            mg: {"joint": tuple(actor.parameters())}
            for mg, actor in self.actors.items()
        }
        self._actor_layer_parameters = {
            mg: {name: tuple(module.parameters()) for name, module in actor.named_children()
                 if any(True for _ in module.parameters())}
            for mg, actor in self.actors.items()
        }

    def _actor_gradient_diagnostics(self, mg):
        squared = {id(p): float(p.grad.detach().norm(2)) ** 2 if p.grad is not None else 0.0
                   for p in self._actor_parameters[mg]}

        def norm(parameters):
            return sum(squared[id(p)] for p in parameters) ** 0.5

        return (
            norm(self._actor_parameters[mg]),
            {name: norm(params) for name, params in self._actor_branch_parameters[mg].items()},
            {name: norm(params) for name, params in self._actor_layer_parameters[mg].items()},
        )

    def actor_parameter_counts(self):
        return {mg: sum(p.numel() for p in actor.parameters())
                for mg, actor in self.actors.items()}

    def project_actions(self, state, joint_raw):
        return self.projector(state, joint_raw)

    def update(self, batch_size):
        if len(self.replay) < batch_size:
            return {}
        batch = self.replay.sample(batch_size, self.agent_ids, self.device)
        state, next_state = batch["global_state"], batch["next_global_state"]

        raw = torch.stack([batch["actions"][mg] for mg in self.agent_ids], dim=1)
        effective, _ = self.project_actions(state, raw)

        with torch.no_grad():
            target_raw = []
            for mg in self.agent_ids:
                action = self.target_actors[mg](batch["next_observations"][mg])
                noise = torch.randn_like(action).mul(self.target_policy_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip
                )
                target_raw.append((action + noise).clamp(-1, 1))
            target_effective, _ = self.project_actions(next_state, torch.stack(target_raw, dim=1))
            target_q = {}
            for mg in self.agent_ids:
                q1, q2 = self.target_critics[mg](next_state, target_effective.flatten(1))
                target_q[mg] = batch["team_reward"] + (
                    1 - batch["done"]
                ) * self.gamma * torch.minimum(q1, q2)

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
            grad_norm = self._gradient_norm(self.critics[mg])
            if not np.isfinite(grad_norm):
                raise FloatingPointError(f"{mg}: nonfinite Critic gradient")
            self.critic_optimizers[mg].step()
            td = target_q[mg] - torch.minimum(q1, q2)
            q_min = torch.minimum(q1.detach(), q2.detach())
            q_stats[mg] = dict(
                mean=float(q_min.mean()), min=float(q_min.min()),
                max=float(q_min.max()), std=float(q_min.std(unbiased=False)),
            )
            records[mg] = dict(
                critic_loss=float(loss.detach()),
                critic1_loss=float(c1.detach()),
                critic2_loss=float(c2.detach()),
                critic_grad_norm=grad_norm,
                q1_mean=float(q1.detach().mean()),
                q2_mean=float(q2.detach().mean()),
                q1_q2_gap=float((q1.detach().mean() - q2.detach().mean()).abs()),
                target_q_mean=float(target_q[mg].mean()),
                td_error_mean=float(td.detach().mean()),
                td_error_std=float(td.detach().std(unbiased=False)),
            )
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
                        with torch.set_grad_enabled(mg == active):
                            action = self.actors[mg](batch["observations"][mg])
                        parts.append(action)
                    active_action = parts[active_index]
                    active_action.retain_grad()
                    policy_effective, plans = self.project_actions(
                        state, torch.stack(parts, dim=1)
                    )
                    loss = -critic.q1(state, policy_effective.flatten(1)).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{active}: nonfinite Actor loss")
                    self.actor_optimizers[active].zero_grad(set_to_none=True)
                    loss.backward()
                    actor = self.actors[active]
                    before, branch, layer = self._actor_gradient_diagnostics(active)
                    if not np.isfinite(before):
                        raise FloatingPointError(f"{active}: nonfinite Actor gradient")
                    grads[active], branches[active], layers[active] = before, branch, layer
                    action_grad = active_action.grad.detach() * batch_size
                    plan = plans[active_index]
                    parameters = self._actor_parameters[active]
                    original = [p.detach().clone() for p in parameters]
                    torch.nn.utils.clip_grad_norm_(
                        parameters, self.actor_max_grad_norm, error_if_nonfinite=True
                    )
                    after = self._gradient_norm(actor)
                    self.actor_optimizers[active].step()
                    step_norm = sum(float((p.detach() - old).square().sum())
                                    for p, old in zip(parameters, original)) ** 0.5
                    param_norm = sum(float(old.square().sum()) for old in original) ** 0.5
                    diagnostic[active] = {
                        "total_grad_norm": before,
                        "post_clip_grad_norm": after,
                        "joint_network_grad_norm": branch["joint"],
                        "clip_applied": float(before > self.actor_max_grad_norm),
                        "dispatch_action_grad_rms": float(action_grad[:, 0].square().mean().sqrt()),
                        "dr_action_grad_rms": float(action_grad[:, 1].square().mean().sqrt()),
                        "dispatch_saturation_rate": float(
                            (active_action.detach()[:, 0].abs() >= .95).float().mean()
                        ),
                        "dr_saturation_rate": float(
                            (active_action.detach()[:, 1].abs() >= .95).float().mean()
                        ),
                        "dr_projection_rate": float(plan["dr_projected"].detach().float().mean()),
                        "offer_eligible_rate": float(plan["mask"].detach().float().mean()),
                        "forced_dispatch_rate": float(plan["forced_dispatch"].detach().float().mean()),
                        "local_override_kw_max": float(
                            plan["local_priority_reserve"].detach().abs().max()
                        ),
                        "optimizer_step_norm": step_norm,
                        "relative_optimizer_step_norm": step_norm / max(param_norm, 1e-20),
                    }
                    losses[active] = float(loss.detach())
                    self._soft_update(actor, self.target_actors[active])
                    self._soft_update(critic, self.target_critics[active])
                finally:
                    for parameter in critic.parameters():
                        parameter.requires_grad_(True)
            metrics.update(
                actor_loss=float(np.mean(list(losses.values()))),
                actor_losses=losses,
                actor_grad_norm=float(np.mean(list(grads.values()))),
                actor_grad_norms=grads,
                actor_branch_grad_norms=branches,
                actor_layer_grad_norms=layers,
                actor_diagnostics=diagnostic,
            )
        counts = self.actor_parameter_counts()
        metrics["actor_parameter_counts"] = counts
        metrics["actor_parameter_count"] = float(np.mean(list(counts.values())))
        return metrics

    def save(self, path):
        torch.save({
            "algorithm_slug": "matd3_dispatch",
            "action_semantics": "single_dispatch_dr_v1",
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
        if (checkpoint.get("algorithm_slug") != "matd3_dispatch"
                or checkpoint.get("action_semantics") != "single_dispatch_dr_v1"):
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
