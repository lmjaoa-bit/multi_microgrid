from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from algorithms.matd3 import MATD3
from algorithms.matd3_hierarchical_transformer.actor import HierarchicalTransformerActor
from core.joint_action_projection import JointActionProjector


class Matd3HierarchicalTransformer(MATD3):
    """MATD3 with Hierarchical Categorized-Token Spatio-Temporal Transformer Actors.

    Architecture:
      - 3 independent, decentralized Actors (one per microgrid).
      - 14 categorized semantic tokens (ESS, DR, Supply, Trade, Peer1, Peer2, 8xForecast).
      - 8-hour lookahead forecast horizon with diurnal periodic embeddings.
      - Heterogeneous device encoders for Mill, HVAC, and EV with pacing bias.
      - Centralized Twin Critic (CTDE) with physical joint-action projection (3 effective actions/agent).
    """

    @classmethod
    def from_env(
        cls,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
        seed: int | None = None,
    ) -> Matd3HierarchicalTransformer:
        return cls(env, config, device=device, seed=env.seed if seed is None else seed)

    def __init__(
        self,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        if getattr(env, "ess_sharing_control_mode", "") != "single_dispatch":
            raise ValueError("This algorithm requires single_dispatch ESS control")
        if int(env.action_dim) != 2:
            raise ValueError("Hierarchical Transformer requires raw actions [ESS intent, DR intent]")

        settings = config.get("algorithm", config)
        allowed = {
            "actor_lr", "critic_lr", "gamma", "tau", "policy_delay",
            "target_policy_noise", "target_noise_clip", "replay_capacity",
        }
        kwargs = {key: settings[key] for key in allowed if key in settings}

        # Centralized Twin Critics receive 3 effective projected physical controls per agent
        super().__init__(
            env.agent_ids,
            env.observation_dims,
            env.global_state_dim,
            3,
            hidden_dims=tuple(settings.get("hidden_dims", (128, 128))),
            device=device,
            seed=seed,
            **kwargs,
        )

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

        net = settings.get("network", {})
        embed_dim = int(net.get("embed_dim", 32))
        num_heads = int(net.get("num_heads", 4))
        num_layers = int(net.get("num_layers", 2))
        forecast_horizon = int(net.get("forecast_horizon", 8))
        hidden_dim = int(net.get("hidden_dim", 64))

        # 3 Independent Actors (one per microgrid)
        self.actors = nn.ModuleDict({
            mg: HierarchicalTransformerActor(
                self.observation_schemas[mg],
                self.agent_ids,
                mg,
                mg_config=env.cfg_by_id[mg] if hasattr(env, "cfg_by_id") else None,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                forecast_horizon=forecast_horizon,
                hidden_dim=hidden_dim,
            )
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
            mg: {
                f"layer_{i}": tuple(layer.parameters())
                for i, layer in enumerate(actor.transformer.layers)
            }
            for mg, actor in self.actors.items()
        }

    def actor_parameter_counts(self) -> dict[str, int]:
        return {mg: self.actors[mg].parameter_count for mg in self.agent_ids}

    def project_actions(self, state: torch.Tensor, joint_raw: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        return self.projector(state, joint_raw)

    def _actor_gradient_diagnostics(self, mg: str) -> tuple[float, dict[str, float], dict[str, float]]:
        squared = {
            id(p): p.grad.detach().square().sum() if p.grad is not None else torch.zeros((), device=self.device)
            for p in self._actor_parameters[mg]
        }

        def norm(params: tuple[torch.nn.Parameter, ...]) -> float:
            terms = [squared[id(p)] for p in params]
            return float(torch.stack(terms).sum().sqrt().cpu()) if terms else 0.0

        total_norm = norm(self._actor_parameters[mg])
        branch_norms = {name: norm(params) for name, params in self._actor_branch_parameters[mg].items()}
        layer_norms = {name: norm(params) for name, params in self._actor_layer_parameters[mg].items()}
        return total_norm, branch_norms, layer_norms

    def update(self, batch_size: int) -> dict[str, Any]:
        if len(self.replay) < batch_size:
            return {}

        batch = self.replay.sample(batch_size, self.agent_ids, self.device)
        state, next_state = batch["global_state"], batch["next_global_state"]
        raw = torch.stack([batch["actions"][mg] for mg in self.agent_ids], dim=1)
        effective, _ = self.project_actions(state, raw)

        # Target Q computation
        with torch.no_grad():
            target_raw = []
            for mg in self.agent_ids:
                action = self.target_actors[mg](batch["next_observations"][mg])
                noise = torch.randn_like(action).mul(self.target_policy_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip
                )
                target_raw.append((action + noise).clamp(-1.0, 1.0))
            target_effective, _ = self.project_actions(next_state, torch.stack(target_raw, dim=1))
            target_q = {}
            for mg in self.agent_ids:
                q1, q2 = self.target_critics[mg](next_state, target_effective.flatten(1))
                target_q[mg] = batch["team_reward"] + (1.0 - batch["done"]) * self.gamma * torch.minimum(q1, q2)

        # Critic update
        metrics: dict[str, Any] = {}
        records: dict[str, float] = {}
        q_stats: dict[str, dict[str, float]] = {}
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

        # Delayed policy update
        self.update_steps += 1
        if self.update_steps % self.policy_delay == 0:
            losses, grads, branches, diagnostic = {}, {}, {}, {}
            with torch.no_grad():
                cached_actions = {mg: self.actors[mg](batch["observations"][mg]) for mg in self.agent_ids}

            for active in self.agent_ids:
                actor = self.actors[active]
                critic = self.critics[active]
                optimizer = self.actor_optimizers[active]

                for param in critic.parameters():
                    param.requires_grad_(False)
                try:
                    active_action = actor(batch["observations"][active])
                    joint = [active_action if mg == active else cached_actions[mg] for mg in self.agent_ids]
                    projected, _ = self.project_actions(state, torch.stack(joint, dim=1))
                    active_index = self.agent_ids.index(active)
                    projected_for_q = projected.clone()
                    projected_for_q[:, active_index, :] = projected[:, active_index, :]
                    q1 = critic.q1(state, projected_for_q.flatten(1))
                    loss = -q1.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{active}: nonfinite Actor loss")

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    before, branch = self._actor_gradient_diagnostics(active)
                    grads[active], branches[active] = before, branch
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), self.actor_max_grad_norm)
                    after = self._gradient_norm(actor)
                    optimizer.step()

                    losses[active] = float(loss.detach())
                    diagnostic[active] = {
                        "total_grad_norm": float(before),
                        "post_clip_grad_norm": float(after),
                        "clip_applied": float(before > self.actor_max_grad_norm),
                        **{f"{k}_grad_norm": float(v) for k, v in branch.items()},
                    }
                    self._soft_update(actor, self.target_actors[active])
                    self._soft_update(critic, self.target_critics[active])
                finally:
                    for param in critic.parameters():
                        param.requires_grad_(True)

            metrics.update(
                actor_loss=float(np.mean(list(losses.values()))),
                actor_losses=losses,
                actor_grad_norm=float(np.mean(list(grads.values()))),
                actor_grad_norms=grads,
                actor_branch_grad_norms=branches,
                actor_diagnostics=diagnostic,
            )

        return metrics

    def save(self, path: str | Path) -> None:
        torch.save({
            "algorithm_slug": "matd3_hierarchical_transformer",
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

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        slug = checkpoint.get("algorithm_slug")
        if slug != "matd3_hierarchical_transformer":
            raise ValueError(f"Incompatible algorithm checkpoint: expected matd3_hierarchical_transformer, got {slug}")
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
