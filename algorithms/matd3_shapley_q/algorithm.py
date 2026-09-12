"""Exact-predecessor Shapley critics for fixed-baseline cooperation surplus."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithms.matd3.critic import TwinCritic
from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from algorithms.matd3_shapley_q.coalition import enumerate_predecessor_coalitions, shapley_weight
from algorithms.matd3_shapley_q.independent_baseline import IndependentBaselineEvaluator
from algorithms.matd3_shapley_q.replay_buffer import ShapleyReplayBuffer


class CoalitionAwareProjector:
    """Keep every local dispatch; only coalition members can exchange offers."""

    def __init__(self, projector, agent_ids):
        self.projector = projector
        self.agent_ids = tuple(agent_ids)

    def __call__(self, state, raw, coalition):
        coalition = frozenset(coalition)
        if not coalition.issubset(self.agent_ids):
            raise ValueError("Unknown agent in coalition")
        effective, plans = self.projector(state, raw)
        channels = []
        for i, mg in enumerate(self.agent_ids):
            plan = plans[i]
            member = mg in coalition
            peer_cap = torch.zeros_like(plan["local"])
            if member and self.projector.enabled:
                for j, other in enumerate(plans):
                    if j != i and self.agent_ids[j] in coalition:
                        peer_cap = peer_cap + torch.where(
                            other["buy"] + 1e-10 >= plan["sell"],
                            other["demand"].clamp(max=self.projector.pair_capacity),
                            torch.zeros_like(peer_cap))
            cap = torch.where(plan["charging"] | (plan["deficit"] - plan["local"] > 1e-7),
                              torch.zeros_like(peer_cap), torch.minimum(plan["remaining"], peer_cap))
            offer = torch.minimum(plan["uncapped_offer"], cap)
            scale = max(self.projector.batteries[mg]["max_charge_kw"],
                        self.projector.batteries[mg]["max_discharge_kw"])
            channels.append(torch.stack([effective[:, i, 0], offer / scale], dim=-1))
            plan.update(in_coalition=member, cap=cap, offer=offer,
                        mask=cap > 1e-9, peer_cap=peer_cap)
        return torch.stack(channels, dim=1), plans


class TwinMarginalCritic(TwinCritic):
    """Centralized marginal heads conditioned on predecessor membership.

    Both before/after coalition physical controls are visible; local operation
    remains present even for nonmembers. Heads learn marginal values, not
    independent copies of the Grand return.
    """

    def __init__(self, state_dim, agent_count, hidden_dims):
        super().__init__(state_dim + agent_count, 4 * agent_count, hidden_dims)

    def forward(self, state, before, after, predecessor_mask):
        return super().forward(torch.cat([state, predecessor_mask], dim=-1),
                               torch.cat([before.flatten(1), after.flatten(1)], dim=-1))


class Matd3ShapleyQ(Matd3Dispatch):
    """Three decentralized actors and three centralized twin marginal critics."""

    def __init__(self, env, config, device="cpu", seed=42):
        super().__init__(env, config, device=device, seed=seed)
        if len(self.agent_ids) != 3:
            raise ValueError("MATD3-ShapleyQ requires exactly three agents")
        settings = config.get("algorithm", config)
        self.coalition_projector = CoalitionAwareProjector(self.projector, self.agent_ids)
        self.critics = nn.ModuleDict({
            mg: TwinMarginalCritic(env.global_state_dim, len(self.agent_ids),
                                  tuple(settings.get("hidden_dims", (128, 128))))
            for mg in self.agent_ids
        }).to(self.device)
        self.target_critics = copy.deepcopy(self.critics)
        self.critic_optimizers = {
            mg: torch.optim.Adam(critic.parameters(), lr=float(settings.get("critic_lr", 3e-4)))
            for mg, critic in self.critics.items()
        }
        self.replay = ShapleyReplayBuffer(settings.get("replay_capacity", 200_000), seed)
        self.baseline_evaluator = IndependentBaselineEvaluator(env, config, device=str(self.device))
        self.baseline_config = dict(mode=self.baseline_evaluator.mode,
            checkpoint=self.baseline_evaluator.checkpoint_path, disable_p2p=True, freeze_parameters=True)
        self.predecessors = {
            mg: tuple((coalition, shapley_weight(len(coalition), len(self.agent_ids)))
                      for coalition in enumerate_predecessor_coalitions(mg, self.agent_ids))
            for mg in self.agent_ids
        }

    def on_episode_start(self, env, **_):
        self.baseline_evaluator.reset_episode(env)

    def get_independent_step_data(self, t):
        return self.baseline_evaluator.get_step_data(t)

    @staticmethod
    def _conservative_shapley_sum(q1, q2, agent_ids):
        return sum(torch.minimum(q1[mg], q2[mg]) for mg in agent_ids)

    def _agent_shapley_values(self, state, raw, active, critic, projected=None):
        projected = {} if projected is None else projected
        first, second = [], []
        for predecessor, weight in self.predecessors[active]:
            joined = predecessor | {active}
            for coalition in (predecessor, joined):
                if coalition not in projected:
                    projected[coalition] = self.coalition_projector(state, raw, coalition)[0]
            mask = state.new_tensor([mg in predecessor for mg in self.agent_ids]).expand(state.shape[0], -1)
            q1, q2 = critic(state, projected[predecessor], projected[joined], mask)
            first.append(weight * q1)
            second.append(weight * q2)
        return sum(first), sum(second)

    def _shapley_values(self, state, raw, critics):
        q1, q2, projected = {}, {}, {}
        for active in self.agent_ids:
            q1[active], q2[active] = self._agent_shapley_values(
                state, raw, active, critics[active], projected)
        return q1, q2

    def update(self, batch_size):
        if len(self.replay) < batch_size:
            return {}
        batch = self.replay.sample(batch_size, self.agent_ids, self.device)
        state, next_state = batch["global_state"], batch["next_global_state"]
        raw = torch.stack([batch["actions"][mg] for mg in self.agent_ids], dim=1)
        with torch.no_grad():
            target_parts = []
            for mg in self.agent_ids:
                action = self.target_actors[mg](batch["next_observations"][mg])
                noise = (torch.randn_like(action) * self.target_policy_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip)
                target_parts.append((action + noise).clamp(-1, 1))
            tgt_qphi1, tgt_qphi2 = self._shapley_values(
                next_state, torch.stack(target_parts, dim=1), self.target_critics)
            tgt_qcoop_min = self._conservative_shapley_sum(tgt_qphi1, tgt_qphi2, self.agent_ids)
            continuation = (1 - batch["done"]) * self.gamma
            y_coop = batch["cooperation_surplus_reward"] + continuation * tgt_qcoop_min
            grand_target = batch["team_reward"] + continuation * (
                batch["next_independent_return_to_go"] + tgt_qcoop_min)
        cur_qphi1, cur_qphi2 = self._shapley_values(state, raw, self.critics)
        qcoop_current = self._conservative_shapley_sum(cur_qphi1, cur_qphi2, self.agent_ids)
        residual = batch["independent_return_to_go"] + qcoop_current - grand_target
        loss_q1 = nn.functional.mse_loss(sum(cur_qphi1.values()), y_coop)
        loss_q2 = nn.functional.mse_loss(sum(cur_qphi2.values()), y_coop)
        loss = loss_q1 + loss_q2
        if not torch.isfinite(loss) or not torch.isfinite(residual).all():
            raise FloatingPointError("Non-finite Shapley critic loss or grand decomposition diagnostic")
        for optimizer in self.critic_optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        critic_grads = {mg: self._gradient_norm(critic) for mg, critic in self.critics.items()}
        if not all(np.isfinite(v) for v in critic_grads.values()):
            raise FloatingPointError("Non-finite Shapley critic gradient")
        for optimizer in self.critic_optimizers.values():
            optimizer.step()
        mean = lambda tensor: float(tensor.detach().mean())
        td = y_coop - qcoop_current
        diagnostics = {
            **{f"shapley_q_{mg}": mean(torch.minimum(cur_qphi1[mg], cur_qphi2[mg])) for mg in self.agent_ids},
            "shapley_q_sum": mean(qcoop_current),
            "cooperation_surplus_q_target": mean(y_coop),
            "efficiency_residual": mean(qcoop_current - y_coop),
            "grand_decomposition_residual": mean(residual),
        }
        if not all(np.isfinite(v) for v in diagnostics.values()):
            raise FloatingPointError("Non-finite Shapley diagnostic")
        metrics = dict(critic_loss=float(loss.detach()), critic1_loss=float(loss_q1.detach()),
            critic2_loss=float(loss_q2.detach()), critic_grad_norm=float(np.mean(list(critic_grads.values()))),
            critic_grad_norms=critic_grads, q1_mean=mean(sum(cur_qphi1.values())),
            q2_mean=mean(sum(cur_qphi2.values())), target_q_mean=mean(y_coop),
            td_error_mean=mean(td), td_error_std=float(td.detach().std(unbiased=False)),
            algorithm_diagnostics=diagnostics, **diagnostics)
        self.update_steps += 1
        if self.update_steps % self.policy_delay == 0:
            losses, gradients = {}, {}
            for active in self.agent_ids:
                critic = self.critics[active]
                critic.requires_grad_(False)
                try:
                    actions = []
                    for mg in self.agent_ids:
                        with torch.set_grad_enabled(mg == active):
                            actions.append(self.actors[mg](batch["observations"][mg]))
                    qphi1, _ = self._agent_shapley_values(state, torch.stack(actions, dim=1), active, critic)
                    actor_loss = -qphi1.mean()
                    if not torch.isfinite(actor_loss):
                        raise FloatingPointError(f"{active}: non-finite Actor loss")
                    self.actor_optimizers[active].zero_grad(set_to_none=True)
                    actor_loss.backward()
                    gradients[active] = self._gradient_norm(self.actors[active])
                    torch.nn.utils.clip_grad_norm_(self.actors[active].parameters(),
                        self.actor_max_grad_norm, error_if_nonfinite=True)
                    self.actor_optimizers[active].step()
                    losses[active] = float(actor_loss.detach())
                finally:
                    critic.requires_grad_(True)
            self._soft_update(self.actors, self.target_actors)
            self._soft_update(self.critics, self.target_critics)
            metrics.update(actor_losses=losses, actor_loss=float(np.mean(list(losses.values()))),
                actor_grad_norms=gradients, actor_grad_norm=float(np.mean(list(gradients.values()))))
        metrics["actor_parameter_counts"] = self.actor_parameter_counts()
        return metrics

    def _checkpoint_metadata(self):
        return dict(algorithm_slug="matd3_shapley_q", shapley_q_version="surplus_q_v2",
            value_definition="cooperation_surplus", action_semantics="single_dispatch_v1",
            projection_signature=self.projection_signature, state_schema=self.state_schema,
            observation_schemas=self.observation_schemas, baseline=self.baseline_config)

    def save(self, path):
        torch.save({**self._checkpoint_metadata(),
            **{name: getattr(self, name).state_dict()
               for name in ("actors", "target_actors", "critics", "target_critics")},
            "update_steps": self.update_steps}, Path(path))

    def load(self, path):
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Shapley checkpoint must contain a mapping")
        for field, expected in self._checkpoint_metadata().items():
            if checkpoint.get(field) != expected:
                raise ValueError(f"Shapley checkpoint {field} does not match surplus_q_v2 configuration")
        states = {}
        for name in ("actors", "target_actors", "critics", "target_critics"):
            state = checkpoint.get(name)
            if not isinstance(state, dict) or any(
                not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()
            ):
                raise ValueError(f"Shapley checkpoint {name} is missing or non-finite")
            try:
                copy.deepcopy(getattr(self, name)).load_state_dict(state, strict=True)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(f"Shapley checkpoint {name} is incompatible: {exc}") from exc
            states[name] = state
        for name, state in states.items():
            getattr(self, name).load_state_dict(state, strict=True)
        self.update_steps = int(checkpoint.get("update_steps", 0))
