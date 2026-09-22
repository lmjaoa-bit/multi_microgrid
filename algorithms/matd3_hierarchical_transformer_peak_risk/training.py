from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithms.diagnostic_sampling import (
    DiagnosticSchedule,
    _actor_metric_payload,
    _critic_metrics,
    _gradient_norm_tensor,
)
from algorithms.matd3_hierarchical_transformer_peak_risk.peak_risk import (
    assert_finite, system_actual_load_kw, max_return_target, peak_actor_weight,
)


def map_noisy_predictive_latent(
    actor: Any,
    observations: torch.Tensor,
    latent_actions: torch.Tensor,
    *,
    noise_std: float = 0.0,
    noise_clip: float = 0.5,
    noise: torch.Tensor | None = None,
    collect_diagnostics: bool = False,
) -> torch.Tensor:
    """Add clipped noise to latent actions and map through the predictive actor."""
    if noise is None:
        if noise_std > 0.0:
            noise = torch.randn_like(latent_actions).mul(float(noise_std))
        else:
            noise = torch.zeros_like(latent_actions)
    clipped_noise = noise.clamp(-float(noise_clip), float(noise_clip))
    noisy_latent = (latent_actions + clipped_noise).clamp(-1.0, 1.0)
    return actor.map_latent_actions(
        observations, noisy_latent, collect_diagnostics=collect_diagnostics
    )


def select_predictive_actions(
    self: Any,
    observations: Mapping[str, Any],
    noise_std: float = 0.0,
) -> dict[str, np.ndarray]:
    """Select actions with exploration noise applied in latent space before predictive mapping."""
    actions: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for mg in self.agent_ids:
            obs_array = np.asarray(observations[mg], dtype=np.float32)
            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
            actor = self.actors[mg]
            latent = actor.forward_latent(obs_tensor, collect_diagnostics=False)
            if noise_std > 0.0:
                noise = self.rng.normal(0.0, float(noise_std), size=tuple(latent.shape))
                noise_tensor = torch.as_tensor(noise, dtype=latent.dtype, device=latent.device)
                noisy_latent = (latent + noise_tensor).clamp(-1.0, 1.0)
            else:
                noisy_latent = latent.clamp(-1.0, 1.0)
            mapped = actor.map_latent_actions(obs_tensor, noisy_latent, collect_diagnostics=False)
            actions[mg] = mapped.squeeze(0).cpu().numpy().astype(np.float32)
    return actions


def update_peak_risk(self: Any, batch_size: int) -> dict[str, Any]:
    """Fused predictive-DR/peak update with one replay sample and parent projector calls."""
    if len(self.replay) < batch_size:
        return {}

    start_wall = time.perf_counter()
    cfg = self.peak_risk_config
    next_update_step = self.update_steps + 1
    peak_update_due = next_update_step % cfg["critic_update_interval"] == 0
    actor_due = next_update_step % self.policy_delay == 0
    diagnostic_due = self.diagnostic_schedule.due(next_update_step)

    batch = self.replay.sample(batch_size, self.agent_ids, self.device)
    state, next_state = batch["global_state"], batch["next_global_state"]
    raw = torch.stack([batch["actions"][mg] for mg in self.agent_ids], dim=1)
    effective, behavior_plans = self.project_actions(state, raw)
    with torch.no_grad():
        current_peak = system_actual_load_kw(behavior_plans) / self.system_peak_scale_kw

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

    # The parent's target projected actions and sampled replay are reused directly.
    # A cost critic uses the larger twin target; gamma must never enter max-return.
    with torch.no_grad():
        next_peak_1, next_peak_2 = self.target_peak_critic(
            next_state, target_effective.flatten(1))
        target_peak = max_return_target(
            current_peak, next_peak_1, next_peak_2, batch["done"])
    peak_loss = None
    peak_loss1 = None
    peak_loss2 = None
    peak_grad_norm = None
    if peak_update_due:
        prediction1, prediction2 = self.peak_critic(state, effective.flatten(1))
        peak_loss1 = nn.functional.smooth_l1_loss(prediction1, target_peak)
        peak_loss2 = nn.functional.smooth_l1_loss(prediction2, target_peak)
        peak_loss = peak_loss1 + peak_loss2
        assert_finite(peak_loss, "Nonfinite peak critic loss")
        self.peak_optimizer.zero_grad(set_to_none=True)
        peak_loss.backward()
        if diagnostic_due:
            peak_grad_norm = _gradient_norm_tensor(self.peak_critic).detach()
        # error_if_nonfinite=True reads the CUDA total norm back to the CPU.
        # Preserve a per-step nonfinite guard with a stream-ordered async assert.
        peak_total_norm = nn.utils.clip_grad_norm_(
            self.peak_critic.parameters(), cfg["critic_max_grad_norm"],
            error_if_nonfinite=not state.is_cuda)
        if state.is_cuda:
            assert_finite(peak_total_norm, "Nonfinite peak critic gradient norm")
        self.peak_optimizer.step()
        self.peak_optimizer.zero_grad(set_to_none=True)
        self._soft_update(self.peak_critic, self.target_peak_critic)
        self.peak_critic_updates += 1

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
    peak_actor_penalties = []
    peak_actor_grad_norms = []
    applied_peak_weight = (
        peak_actor_weight(next_update_step, cfg["actor_start_update"],
                          cfg["actor_weight_ramp_updates"], cfg["actor_weight_final"])
        if self.peak_critic_updates >= cfg["critic_warmup_updates"] else 0.0
    )

    if actor_due:
        current_actions = {}
        for mg in self.agent_ids:
            obs = batch["observations"][mg]
            actor = self.actors[mg]
            latent = actor.forward_latent(obs, collect_diagnostics=diagnostic_due)
            current_actions[mg] = actor.map_latent_actions(obs, latent, collect_diagnostics=diagnostic_due)

        for active_index, active in enumerate(self.agent_ids):
            actor = self.actors[active]
            critic = self.critics[active]
            optimizer = self.actor_optimizers[active]
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            for parameter in self.peak_critic.parameters():
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
                loss = -critic.q1(state, projected.flatten(1)).mean()
                if applied_peak_weight > 0.0:
                    p1, p2 = self.peak_critic(state, projected.flatten(1))
                    peak_penalty = torch.maximum(p1, p2).mean()
                    loss = loss + applied_peak_weight * peak_penalty
                    if diagnostic_due:
                        (peak_action_grad,) = torch.autograd.grad(
                            peak_penalty, active_action,
                            retain_graph=True, allow_unused=True)
                        if peak_action_grad is not None:
                            peak_actor_grad_norms.append(
                                peak_action_grad.detach().norm(dim=-1).mean())
                    if diagnostic_due:
                        peak_actor_penalties.append(peak_penalty.detach())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

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

                preclip_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    self.actor_max_grad_norm,
                    error_if_nonfinite=False,
                )
                optimizer.step()
                actor_loss_tensors[active] = loss.detach()

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
                    actor_diagnostics[active] = payload

                self._soft_update(actor, self.target_actors[active])
                self._soft_update(critic, self.target_critics[active])
            finally:
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)
                for parameter in self.peak_critic.parameters():
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
    )
    counts = self.actor_parameter_counts()
    metrics["actor_parameter_counts"] = counts
    metrics["actor_parameter_count"] = float(np.mean(list(counts.values())))
    # Scalar synchronization and detailed stats happen only on the parent
    # diagnostic schedule, never on every gradient step.
    with torch.no_grad():
        predicted1, predicted2 = self.peak_critic(state, effective.flatten(1))
        twin_gap = (predicted1 - predicted2).abs().mean()
        predicted_max = torch.maximum(predicted1, predicted2).mean()
        values = torch.stack([
            current_peak.mean(), torch.quantile(current_peak.flatten(), 0.95),
            target_peak.mean(), predicted_max, twin_gap,
            peak_loss.detach() if peak_loss is not None else torch.full_like(twin_gap, float("nan")),
            peak_loss1.detach() if peak_loss1 is not None else torch.full_like(twin_gap, float("nan")),
            peak_loss2.detach() if peak_loss2 is not None else torch.full_like(twin_gap, float("nan")),
            peak_grad_norm if peak_grad_norm is not None else torch.full_like(twin_gap, float("nan")),
            torch.stack(peak_actor_penalties).mean() if peak_actor_penalties else torch.zeros_like(twin_gap),
            torch.stack(peak_actor_grad_norms).mean() if peak_actor_grad_norms else torch.zeros_like(twin_gap),
        ]).detach().cpu().tolist()
    scale = self.system_peak_scale_kw
    metrics["peak"] = {
        "current_system_load_mean_kw": values[0] * scale,
        "current_system_load_p95_kw": values[1] * scale,
        "target_future_max_mean_kw": values[2] * scale,
        "predicted_future_max_mean_kw": values[3] * scale,
        "twin_gap_mean_kw": values[4] * scale,
        "critic_loss": values[5] if peak_loss is not None else None,
        "critic1_loss": values[6] if peak_loss is not None else None,
        "critic2_loss": values[7] if peak_loss is not None else None,
        "critic_grad_norm": values[8] if peak_grad_norm is not None else None,
        "actor_penalty": values[9] if peak_actor_penalties else None,
        "actor_weight": applied_peak_weight,
        "actor_peak_gradient_norm": values[10] if peak_actor_grad_norms else None,
        "update_count": self.peak_critic_updates,
        "nonfinite_count": 0,
    }
    elapsed = time.perf_counter() - start_wall
    metrics["runtime"] = {
        "update_wall_time_ms": elapsed * 1000.0,
        "updates_per_second": 1.0 / max(elapsed, 1e-12),
    }
    return metrics
