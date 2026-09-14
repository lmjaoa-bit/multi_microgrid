from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algorithms.runtime_helpers import diagnostics_due, _actor_diagnostic_payload


def optimized_pnn_update(self, batch_size: int):
    """PNN runtime optimization preserving the original sequential Actor-update semantics."""
    if len(self.replay) < batch_size:
        return {}

    batch = self.replay.sample(batch_size, self.agent_ids, self.device)
    state, next_state = batch["global_state"], batch["next_global_state"]
    raw = torch.stack([batch["actions"][mg] for mg in self.agent_ids], dim=1)
    effective, _ = self.project_actions(state, raw)

    with torch.no_grad():
        target_raw = []
        for mg in self.agent_ids:
            action = self.target_actors[mg](
                batch["next_observations"][mg], collect_diagnostics=False
            )
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

    critic_rows = []
    for mg in self.agent_ids:
        q1, q2 = self.critics[mg](state, effective.flatten(1))
        c1 = nn.functional.mse_loss(q1, target_q[mg])
        c2 = nn.functional.mse_loss(q2, target_q[mg])
        loss = c1 + c2
        optimizer = self.critic_optimizers[mg]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critics[mg].parameters(), float("inf"), error_if_nonfinite=True
        )
        optimizer.step()
        q_min = torch.minimum(q1.detach(), q2.detach())
        td = (target_q[mg] - torch.minimum(q1, q2)).detach()
        critic_rows.append(torch.stack([
            loss.detach(), grad_norm.detach(), q_min.mean(), q_min.std(unbiased=False),
            td.mean(), td.square().mean().sqrt(),
        ]))

    critic_values = torch.stack(critic_rows).detach().cpu().numpy()
    records, q_stats = {}, {}
    for index, mg in enumerate(self.agent_ids):
        loss_v, grad_v, q_mean, q_std, td_mean, td_rmse = map(float, critic_values[index])
        records[mg] = loss_v
        q_stats[mg] = {
            "critic_loss": loss_v,
            "critic_grad_norm": grad_v,
            "q_mean": q_mean,
            "q_std": q_std,
            "td_error_mean": td_mean,
            "td_error_rmse": td_rmse,
        }
    metrics = {
        "critic_loss": float(np.mean(list(records.values()))),
        "critic_losses": records,
        "critic_q_diagnostics": q_stats,
    }
    for stat in ("critic_grad_norm", "q_mean", "q_std", "td_error_mean", "td_error_rmse"):
        metrics[stat] = float(np.mean([q_stats[mg][stat] for mg in self.agent_ids]))

    self.update_steps += 1
    if self.update_steps % self.policy_delay == 0:
        collect = diagnostics_due(self.update_steps, getattr(self, "diagnostics_interval", 100))
        losses = {}
        grads, branches, layers, diagnostic = {}, {}, {}, {}

        for active_index, active in enumerate(self.agent_ids):
            actor = self.actors[active]
            critic = self.critics[active]
            optimizer = self.actor_optimizers[active]
            actor._runtime_actor_max_grad_norm = self.actor_max_grad_norm
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            try:
                active_action = actor(
                    batch["observations"][active], collect_diagnostics=collect
                )
                if collect:
                    active_action.retain_grad()
                joint = []
                for mg in self.agent_ids:
                    if mg == active:
                        joint.append(active_action)
                    else:
                        joint.append(
                            self.actors[mg](
                                batch["observations"][mg], collect_diagnostics=False
                            ).detach()
                        )
                projected, plans = self.project_actions(state, torch.stack(joint, dim=1))
                plan = plans[active_index]
                loss = -critic.q1(state, projected.flatten(1)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if collect:
                    before, branch, layer = self._actor_gradient_diagnostics(active)
                    grads[active], branches[active], layers[active] = before, branch, layer
                    action_grad = (
                        active_action.grad.detach() * batch_size
                        if active_action.grad is not None else torch.zeros_like(active_action)
                    )
                    pre_step = {name: p.detach().clone() for name, p in actor.named_parameters()}
                else:
                    before, branch, pre_step = 0.0, {}, None
                    action_grad = None

                preclip = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), self.actor_max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()

                if collect:
                    step_sq = sum(
                        (p.detach() - pre_step[name]).square().sum()
                        for name, p in actor.named_parameters()
                    )
                    param_sq = sum(p.detach().square().sum() for p in actor.parameters())
                    values = torch.stack([
                        action_grad[:, 0].square().mean().sqrt(),
                        (active_action.detach().abs() >= 0.95).float().mean(),
                        plan["mask"].detach().float().mean(),
                        plan["forced_dispatch"].detach().float().mean(),
                        plan["local_priority_reserve"].detach().abs().max(),
                        step_sq.sqrt(), param_sq.sqrt(), preclip.detach(),
                    ]).detach().cpu().numpy()
                    action_grad_rms, saturation, eligible, forced, override, step_norm, param_norm, preclip_v = map(float, values)
                    diagnostic[active] = _actor_diagnostic_payload(
                        "pnn",
                        before=before,
                        after=min(preclip_v, float(self.actor_max_grad_norm)),
                        branch=branch,
                        action_grad_rms=action_grad_rms,
                        saturation_rate=saturation,
                        offer_eligible_rate=eligible,
                        forced_dispatch_rate=forced,
                        local_override_kw_max=override,
                        step_norm=step_norm,
                        param_norm=param_norm,
                        actor=actor,
                    )

                losses[active] = loss.detach()
                self._soft_update(actor, self.target_actors[active])
                self._soft_update(critic, self.target_critics[active])
            finally:
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)

        loss_values = torch.stack([losses[mg] for mg in self.agent_ids]).detach().cpu().numpy()
        actor_losses = {mg: float(loss_values[i]) for i, mg in enumerate(self.agent_ids)}
        metrics.update(
            actor_loss=float(np.mean(list(actor_losses.values()))),
            actor_losses=actor_losses,
        )
        if collect:
            metrics.update(
                actor_grad_norm=float(np.mean(list(grads.values()))),
                actor_grad_norms=grads,
                actor_branch_grad_norms=branches,
                actor_layer_grad_norms=layers,
                actor_diagnostics=diagnostic,
            )
            prods = self.product_diagnostics()
            metrics["pnn_product_diagnostics"] = prods
            for key in next(iter(prods.values()), {}):
                metrics[f"pnn_{key}"] = float(np.mean([p[key] for p in prods.values()]))

    counts = self.actor_parameter_counts()
    metrics["actor_parameter_counts"] = counts
    metrics["actor_parameter_count"] = float(np.mean(list(counts.values())))
    return metrics
