from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithms.matd3 import MATD3
from algorithms.pnn_transformer.actor import PnnTransformerDispatchActor
from core.single_dispatch import SingleDispatchProjector


class PnnTransformer(MATD3):
    """Organic PNN-Transformer Hybrid Dispatch Controller.

    Integrates:
      - 2-layer Pre-LN Spatio-Temporal Transformer with explicit forecast horizon
        and 24h periodic diurnal embeddings for macro inter-temporal planning.
      - Signed second-order Hadamard pairwise product network (PNN) capturing physical
        bilinear energy arbitrage (temporal, spatial, market interactions).
      - Local physical residual bypass ensuring battery feasibility and state constraints
        directly influence single-dispatch decisions.
      - Single dispatch physical control projection for centralized critics.
    """

    @classmethod
    def from_env(cls, env, config, device="cpu", seed=None):
        return cls(env, config, device=device, seed=env.seed if seed is None else seed)

    def __init__(self, env, config, device="cpu", seed=42):
        if env.ess_sharing_control_mode != "single_dispatch":
            raise ValueError("This algorithm requires single_dispatch control")
        settings = config.get("algorithm", config)
        allowed = {
            "actor_lr", "critic_lr", "gamma", "tau", "policy_delay",
            "target_policy_noise", "target_noise_clip", "replay_capacity",
        }
        kwargs = {key: settings[key] for key in allowed if key in settings}
        super().__init__(
            env.agent_ids,
            env.observation_dims,
            env.global_state_dim,
            2,
            hidden_dims=tuple(settings.get("hidden_dims", (128, 128))),
            device=device,
            seed=seed,
            **kwargs,
        )
        self.observation_schemas = {mg: tuple(env.obs_builder.schema(mg)) for mg in self.agent_ids}
        self.state_schema = tuple(env.global_state_builder.schema())
        self.projector = SingleDispatchProjector(
            env.config,
            self.state_schema,
            env.global_state_builder.norm,
            env.global_state_builder.price_norm,
        )
        signature = dict(
            config=env.config,
            schema=self.state_schema,
            normalizers=env.global_state_builder.norm,
            price_norm=env.global_state_builder.price_norm,
        )
        self.projection_signature = hashlib.sha256(
            json.dumps(signature, sort_keys=True, default=str).encode()
        ).hexdigest()
        self.action_dim = 1  # Critics retain two projected physical controls per agent.

        net = settings.get("network", {})
        embed_dim = int(net.get("embed_dim", 32))
        num_heads = int(net.get("num_heads", 4))
        num_layers = int(net.get("num_layers", 2))
        hidden_dims = tuple(net.get("hidden_dims", (128, 64)))
        forecast_horizon = int(net.get("forecast_horizon", getattr(getattr(env, "obs_builder", None), "future_h", 4)))

        raw_scale = net.get("product_scale", 0.2)
        if isinstance(raw_scale, Mapping):
            product_scales = {mg: float(raw_scale.get(mg, 0.2)) for mg in self.agent_ids}
        else:
            product_scales = {mg: float(raw_scale) for mg in self.agent_ids}

        self.actors = nn.ModuleDict({
            mg: PnnTransformerDispatchActor(
                self.observation_schemas[mg],
                self.agent_ids,
                mg,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                hidden_dims=hidden_dims,
                product_scale=product_scales[mg],
                forecast_horizon=forecast_horizon,
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

        self._actor_parameters = {mg: tuple(actor.parameters()) for mg, actor in self.actors.items()}
        self._actor_branch_parameters = {
            mg: {
                name: tuple(
                    p
                    for item in items
                    for p in (item.parameters() if isinstance(item, nn.Module) else [item])
                )
                for name, items in actor.branch_modules().items()
            }
            for mg, actor in self.actors.items()
        }
        self._actor_layer_parameters = {
            mg: {name: tuple(module.parameters()) for name, module in actor.named_children()
                 if any(True for _ in module.parameters())}
            for mg, actor in self.actors.items()
        }

    def _actor_gradient_diagnostics(self, mg):
        squared = {
            id(p): float(p.grad.detach().norm(2)) ** 2 if p.grad is not None else 0.0
            for p in self._actor_parameters[mg]
        }
        def norm(parameters):
            return sum(squared[id(p)] for p in parameters) ** 0.5
        return (
            norm(self._actor_parameters[mg]),
            {name: norm(params) for name, params in self._actor_branch_parameters[mg].items()},
            {name: norm(params) for name, params in self._actor_layer_parameters[mg].items()},
        )

    def actor_parameter_counts(self):
        return {mg: self.actors[mg].parameter_count for mg in self.agent_ids}

    def product_diagnostics(self):
        return {mg: self.actors[mg].product_diagnostics() for mg in self.agent_ids}

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
            norm = self._gradient_norm(self.critics[mg])
            if not np.isfinite(norm):
                raise FloatingPointError(f"{mg}: nonfinite Critic gradient")
            self.critic_optimizers[mg].step()
            td = target_q[mg] - torch.minimum(q1, q2)
            q_min = torch.minimum(q1.detach(), q2.detach())
            records[mg] = float(loss.detach())
            td_detached = td.detach()
            q_stats[mg] = {
                "critic_loss": records[mg],
                "critic_grad_norm": norm,
                "q_mean": float(q_min.mean()),
                "q_std": float(q_min.std(unbiased=False)),
                "td_error_mean": float(td_detached.mean()),
                "td_error_rmse": float(td_detached.square().mean().sqrt()),
            }

        metrics.update(
            critic_loss=float(np.mean(list(records.values()))),
            critic_losses=records,
            critic_q_diagnostics=q_stats,
        )
        for stat in ("critic_grad_norm", "q_mean", "q_std", "td_error_mean", "td_error_rmse"):
            metrics[f"{stat}"] = float(np.mean([q_stats[mg][stat] for mg in self.agent_ids]))

        self.update_steps += 1
        if self.update_steps % self.policy_delay == 0:
            losses, grads, branches, layers, diagnostic = {}, {}, {}, {}, {}
            with torch.no_grad():
                cached_actions = {mg: self.actors[mg](batch["observations"][mg]) for mg in self.agent_ids}

            for active in self.agent_ids:
                actor = self.actors[active]
                critic = self.critics[active]
                optimizer = self.actor_optimizers[active]
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
                try:
                    active_action = actor(batch["observations"][active])
                    active_action.retain_grad()
                    joint = [active_action if mg == active else cached_actions[mg] for mg in self.agent_ids]
                    projected, plans = self.project_actions(state, torch.stack(joint, dim=1))
                    active_index = self.agent_ids.index(active)
                    plan = plans[active_index]
                    projected_for_q = projected.clone()
                    projected_for_q[:, active_index, :] = projected[:, active_index, :]
                    q1 = critic.q1(state, projected_for_q.flatten(1))
                    loss = -q1.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{active}: nonfinite Actor loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    before, branch, layer = self._actor_gradient_diagnostics(active)
                    grads[active], branches[active], layers[active] = before, branch, layer
                    action_grad = (
                        active_action.grad.detach() * batch_size
                        if active_action.grad is not None
                        else torch.zeros_like(active_action)
                    )
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), self.actor_max_grad_norm)
                    after = self._gradient_norm(actor)
                    pre_step = {name: param.detach().clone() for name, param in actor.named_parameters()}
                    optimizer.step()
                    step_sq = sum(
                        float((param.detach() - pre_step[name]).square().sum())
                        for name, param in actor.named_parameters()
                    )
                    param_sq = sum(float(param.detach().square().sum()) for param in actor.parameters())
                    step_norm, param_norm = step_sq ** 0.5, param_sq ** 0.5
                    diag_map = actor.last_diagnostics
                    diagnostic[active] = {
                        "total_grad_norm": before,
                        "post_clip_grad_norm": after,
                        "dispatch_token_grad_norm": branch.get("dispatch_token", 0.0),
                        "hour_tokenizer_grad_norm": branch.get("hour_tokenizer", 0.0),
                        "local_tokenizer_grad_norm": branch.get("local_tokenizer", 0.0),
                        "forecast_tokenizer_grad_norm": branch.get("forecast_tokenizer", 0.0),
                        "peer_tokenizer_grad_norm": branch.get("peer_tokenizer", 0.0),
                        "market_tokenizer_grad_norm": branch.get("market_tokenizer", 0.0),
                        "transformer_grad_norm": branch.get("transformer", 0.0),
                        "pnn_encoders_grad_norm": branch.get("pnn_encoders", 0.0),
                        "product_norm_grad_norm": branch.get("product_norm", 0.0),
                        "backbone_grad_norm": branch.get("backbone", 0.0),
                        "head_grad_norm": branch.get("head", 0.0),
                        "clip_applied": float(before > self.actor_max_grad_norm),
                        "dispatch_action_grad_rms": float(action_grad[:, 0].square().mean().sqrt()),
                        "dispatch_saturation_rate": float((active_action.detach().abs() >= 0.95).float().mean()),
                        "offer_eligible_rate": float(plan["mask"].detach().float().mean()),
                        "local_override_kw_max": float(plan["local_priority_reserve"].detach().abs().max()),
                        "optimizer_step_norm": step_norm,
                        "relative_optimizer_step_norm": step_norm / max(param_norm, 1e-20),
                        "transformer_dispatch_norm": diag_map.get("dispatch_norm", 0.0),
                        "pnn_product_norm": diag_map.get("product_norm", 0.0),
                        "local_residual_norm": diag_map.get("local_norm", 0.0),
                        "temporal_product_norm": diag_map.get("temporal_product_norm", 0.0),
                        "spatial_product_norm": diag_map.get("spatial_product_norm", 0.0),
                        "market_product_norm": diag_map.get("market_product_norm", 0.0),
                        "product_to_dispatch_ratio": diag_map.get("product_to_dispatch_ratio", 0.0),
                        "local_to_dispatch_ratio": diag_map.get("local_to_dispatch_ratio", 0.0),
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
            "algorithm_slug": "pnn_transformer",
            "action_semantics": "single_dispatch_v1",
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
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        slug = checkpoint.get("algorithm_slug")
        if slug not in {"pnn_transformer", "matd3_transformer", "pnn", "matd3_dispatch"}:
            raise ValueError(f"Incompatible algorithm checkpoint: expected pnn_transformer, got {slug}")
        if checkpoint.get("action_semantics") != "single_dispatch_v1":
            raise ValueError("Incompatible action semantics")
        if checkpoint.get("projection_signature") != self.projection_signature:
            raise ValueError("Projection signature mismatch")
        self.actors.load_state_dict(checkpoint["actors"])
        self.target_actors.load_state_dict(checkpoint["target_actors"])
        self.critics.load_state_dict(checkpoint["critics"])
        self.target_critics.load_state_dict(checkpoint["target_critics"])
        self.update_steps = int(checkpoint.get("update_steps", 0))

