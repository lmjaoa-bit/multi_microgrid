from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from algorithms.diagnostic_sampling import (
    _actor_metric_payload,
    _critic_metrics,
    _gradient_norm_tensor,
)
from algorithms.matd3_hierarchical_transformer_p2p_value.p2p_auxiliary import (
    compute_p2p_auxiliary_loss,
    extract_p2p_labels,
    p2p_auxiliary_weight,
)
from algorithms.matd3_hierarchical_transformer_predictive_dr.training import (
    map_noisy_predictive_latent,
)


def update_p2p_value(self: Any, batch_size: int) -> dict[str, Any]:
    """Sampled-diagnostic update with P2P value auxiliary training."""
    if len(self.replay) < batch_size:
        return {}

    next_update_step = self.update_steps + 1
    actor_due = next_update_step % self.policy_delay == 0
    diagnostic_due = self.diagnostic_schedule.due(next_update_step)

    batch = self.replay.sample(batch_size, self.agent_ids, self.device)
    state, next_state = batch["global_state"], batch["next_global_state"]
    raw = torch.stack([batch["actions"][mg] for mg in self.agent_ids], dim=1)
    effective, _ = self.project_actions(state, raw)

    with torch.no_grad():
        target_raw = []
        for mg in self.agent_ids:
            next_obs = batch["next_observations"][mg]
            target_actor = self.target_actors[mg]
            target_latent = target_actor.forward_latent(next_obs, collect_diagnostics=False)
            target_noise = torch.randn_like(target_latent).mul(self.target_policy_noise)
            target_raw.append(
                map_noisy_predictive_latent(
                    target_actor,
                    next_obs,
                    target_latent,
                    noise=target_noise,
                    noise_clip=self.target_noise_clip,
                )
            )
        target_effective, _ = self.project_actions(next_state, torch.stack(target_raw, dim=1))
        target_q = {}
        for mg in self.agent_ids:
            q1, q2 = self.target_critics[mg](next_state, target_effective.flatten(1))
            target_q[mg] = batch["team_reward"] + (
                1.0 - batch["done"]
            ) * self.gamma * torch.minimum(q1, q2)

    critic_rows = []
    for mg in self.agent_ids:
        q1, q2 = self.critics[mg](state, effective.flatten(1))
        loss1 = nn.functional.mse_loss(q1, target_q[mg])
        loss2 = nn.functional.mse_loss(q2, target_q[mg])
        loss = loss1 + loss2
        optimizer = self.critic_optimizers[mg]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if diagnostic_due:
            q_min = torch.minimum(q1.detach(), q2.detach())
            td_error = target_q[mg] - q_min
            critic_rows.append(
                torch.stack(
                    [
                        loss.detach(),
                        loss1.detach(),
                        loss2.detach(),
                        _gradient_norm_tensor(self.critics[mg]),
                        q1.detach().mean(),
                        q2.detach().mean(),
                        target_q[mg].mean(),
                        td_error.mean(),
                        td_error.std(unbiased=False),
                        td_error.square().mean().sqrt(),
                        q_min.mean(),
                        q_min.std(unbiased=False),
                        q_min.min(),
                        q_min.max(),
                    ]
                )
            )
        optimizer.step()

    self.update_steps = next_update_step
    actor_loss_tensors = {}
    actor_grad_norms = {}
    actor_branch_norms = {}
    actor_layer_norms = {}
    actor_diagnostics = {}
    p2p_diagnostics_by_agent = {}

    if actor_due:
        current_actions = {}
        representations = {}
        for mg in self.agent_ids:
            obs = batch["observations"][mg]
            actor = self.actors[mg]
            latent, reps = actor.forward_latent_with_representations(
                obs, collect_diagnostics=diagnostic_due
            )
            current_actions[mg] = actor.map_latent_actions(
                obs, latent, collect_diagnostics=diagnostic_due
            )
            representations[mg] = reps

        for active_index, active in enumerate(self.agent_ids):
            actor = self.actors[active]
            critic = self.critics[active]
            optimizer = self.actor_optimizers[active]
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            try:
                active_action = current_actions[active]
                if diagnostic_due:
                    active_action.retain_grad()
                joint = [
                    current_actions[mg] if mg == active else current_actions[mg].detach()
                    for mg in self.agent_ids
                ]
                projected, plans = self.project_actions(state, torch.stack(joint, dim=1))
                policy_loss = -critic.q1(state, projected.flatten(1)).mean()

                labels = extract_p2p_labels(
                    batch["next_observations"][active],
                    self.observation_schemas[active],
                    trade_scale=self.normalizers_by_microgrid[active]["trade"],
                    offer_epsilon_kwh=self.p2p_auxiliary_config["offer_epsilon_kwh"],
                    consistency_tolerance_kwh=self.p2p_auxiliary_config["consistency_tolerance_kwh"],
                )
                behavior_ess = batch["actions"][active][:, 0:1]
                prediction = actor.predict_p2p_value(representations[active], behavior_ess)

                aux = compute_p2p_auxiliary_loss(prediction, labels, self.p2p_auxiliary_config)
                weight = p2p_auxiliary_weight(next_update_step, self.p2p_auxiliary_config)
                total_loss = policy_loss + weight * aux.total

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()

                for param in actor.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise FloatingPointError(f"Non-finite gradient in actor for {active}")

                if diagnostic_due:
                    before, branch, layer = self._actor_gradient_diagnostics(active)
                    action_grad = (
                        active_action.grad.detach() * batch_size
                        if active_action.grad is not None
                        else torch.zeros_like(active_action)
                    )
                    pre_step = {
                        name: parameter.detach().clone()
                        for name, parameter in actor.named_parameters()
                    }
                    actor_grad_norms[active] = before
                    actor_branch_norms[active] = branch
                    actor_layer_norms[active] = layer

                    head_grad = _gradient_norm_tensor(actor.p2p_value_head).item()
                    shared_grad = _gradient_norm_tensor(actor.transformer).item()
                    offered_cnt = float(aux.offered_count.item())
                    offered_rate = offered_cnt / float(batch_size)
                    if offered_cnt > 0.0:
                        mask = labels.offered_mask
                        pos_rate = float(labels.any_fill_target[mask].mean().item())
                        pred_r = prediction.conditional_fill_ratio[mask].detach()
                        tgt_r = labels.fill_ratio_target[mask]
                        mae = float((pred_r - tgt_r).abs().mean().item())
                        pred_f = float(pred_r.mean().item())
                        act_f = float(tgt_r.mean().item())
                    else:
                        pos_rate = 0.0
                        mae = 0.0
                        pred_f = 0.0
                        act_f = 0.0
                    cons_err = float(labels.consistency_residual_kwh.mean().item())

                    p2p_diagnostics_by_agent[active] = {
                        "p2p_auxiliary_weight": float(weight),
                        "p2p_aux_any_fill_loss": float(aux.any_fill.detach().item()),
                        "p2p_aux_fill_ratio_loss": float(aux.fill_ratio.detach().item()),
                        "p2p_aux_total_loss": float(aux.total.detach().item()),
                        "p2p_aux_offered_sample_rate": offered_rate,
                        "p2p_aux_offered_sample_count": offered_cnt,
                        "p2p_aux_positive_fill_rate": pos_rate,
                        "p2p_aux_fill_ratio_mae": mae,
                        "p2p_aux_predicted_fill_mean": pred_f,
                        "p2p_aux_actual_fill_mean": act_f,
                        "p2p_aux_offer_consistency_error_mean": cons_err,
                        "p2p_aux_shared_gradient_norm": shared_grad,
                        "p2p_aux_head_gradient_norm": head_grad,
                    }

                preclip_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    self.actor_max_grad_norm,
                    error_if_nonfinite=False,
                )
                optimizer.step()
                actor_loss_tensors[active] = total_loss.detach()

                if diagnostic_due:
                    payload = _actor_metric_payload(
                        self,
                        actor,
                        active_action,
                        action_grad,
                        plans[active_index],
                        before,
                        branch,
                        preclip_norm,
                        pre_step,
                    )
                    plan = plans[active_index]
                    if "dr_projection_delta" in plan:
                        payload["dr_projection_delta_mean"] = float(
                            plan["dr_projection_delta"].detach().abs().mean().item()
                        )
                    for diag_key, diag_val in p2p_diagnostics_by_agent[active].items():
                        payload[diag_key] = diag_val
                    actor_diagnostics[active] = payload

                self._soft_update(actor, self.target_actors[active])
                self._soft_update(critic, self.target_critics[active])
            finally:
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)

    if not diagnostic_due:
        return {}

    metrics = _critic_metrics(self.agent_ids, critic_rows)
    actor_values = torch.stack(
        [actor_loss_tensors[mg] for mg in self.agent_ids]
    ).detach().cpu().numpy()
    actor_losses = {
        mg: float(actor_values[index]) for index, mg in enumerate(self.agent_ids)
    }
    metrics.update(
        actor_loss=float(np.mean(list(actor_losses.values()))),
        actor_losses=actor_losses,
        actor_grad_norm=float(np.mean(list(actor_grad_norms.values()))),
        actor_grad_norms=actor_grad_norms,
        actor_branch_grad_norms=actor_branch_norms,
        actor_layer_grad_norms=actor_layer_norms,
        actor_diagnostics=actor_diagnostics,
        p2p_auxiliary_diagnostics=p2p_diagnostics_by_agent,
    )

    if p2p_diagnostics_by_agent:
        p2p_keys = [
            "p2p_auxiliary_weight",
            "p2p_aux_any_fill_loss",
            "p2p_aux_fill_ratio_loss",
            "p2p_aux_total_loss",
            "p2p_aux_offered_sample_rate",
            "p2p_aux_offered_sample_count",
            "p2p_aux_positive_fill_rate",
            "p2p_aux_fill_ratio_mae",
            "p2p_aux_predicted_fill_mean",
            "p2p_aux_actual_fill_mean",
            "p2p_aux_offer_consistency_error_mean",
            "p2p_aux_shared_gradient_norm",
            "p2p_aux_head_gradient_norm",
        ]
        for k in p2p_keys:
            metrics[k] = float(np.mean([p2p_diagnostics_by_agent[mg][k] for mg in self.agent_ids]))

    counts = self.actor_parameter_counts()
    metrics["actor_parameter_counts"] = counts
    metrics["actor_parameter_count"] = float(np.mean(list(counts.values())))
    return metrics
