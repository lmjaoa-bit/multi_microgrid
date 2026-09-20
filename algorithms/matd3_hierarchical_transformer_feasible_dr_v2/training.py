from __future__ import annotations

from collections.abc import Mapping
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
from algorithms.runtime_helpers import FastReplayBuffer


def map_noisy_adaptive_latent(
    actor: Any,
    observations: torch.Tensor,
    latent_actions: torch.Tensor,
    *,
    noise_std: float = 0.0,
    noise_clip: float = 0.5,
    noise: torch.Tensor | None = None,
    collect_diagnostics: bool = False,
) -> torch.Tensor:
    """Add clipped noise to latent actions and map through the adaptive actor."""
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


def select_adaptive_actions(
    self: Any,
    observations: Mapping[str, Any],
    noise_std: float = 0.0,
) -> dict[str, np.ndarray]:
    """Select actions with exploration noise applied in latent space before adaptive mapping."""
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


def update_adaptive_dr(self: Any, batch_size: int) -> dict[str, Any]:
    """Sampled-diagnostic update with adaptive reference DR actors."""
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
                map_noisy_adaptive_latent(
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
    return metrics
