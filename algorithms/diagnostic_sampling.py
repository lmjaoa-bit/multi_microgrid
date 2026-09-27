from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from numbers import Integral

import numpy as np
import torch
from torch import nn

from algorithms.runtime_helpers import FastReplayBuffer


class DiagnosticSchedule:
    """Schedule diagnostics on the first policy update at/after each boundary."""

    def __init__(self, interval: int, policy_delay: int) -> None:
        if isinstance(interval, bool) or not isinstance(interval, Integral) or interval < 0:
            raise ValueError("logging.diagnostic_interval must be a non-negative integer")
        if (
            isinstance(policy_delay, bool)
            or not isinstance(policy_delay, Integral)
            or policy_delay <= 0
        ):
            raise ValueError("algorithm.policy_delay must be a positive integer")
        self.interval = int(interval)
        self.policy_delay = int(policy_delay)
        self.next_boundary = self.interval

    def due(self, update_step: int) -> bool:
        if self.interval == 0 or update_step % self.policy_delay != 0:
            return False
        if update_step < self.next_boundary:
            return False
        while self.next_boundary <= update_step:
            self.next_boundary += self.interval
        return True


def _actor_forward(actor, observation, *, collect_diagnostics: bool, plain_actor: bool):
    if plain_actor:
        return actor(observation)
    return actor(observation, collect_diagnostics=collect_diagnostics)


def _forward_all(
    actors,
    observations,
    agent_ids,
    *,
    collect_diagnostics: bool,
    plain_actor: bool,
):
    return {
        mg: _actor_forward(
            actors[mg],
            observations[mg],
            collect_diagnostics=collect_diagnostics,
            plain_actor=plain_actor,
        )
        for mg in agent_ids
    }


def _select_actions(self, observations, noise_std: float = 0.0):
    arrays = [np.asarray(observations[mg], dtype=np.float32) for mg in self.agent_ids]
    same_shape = len({array.shape for array in arrays}) == 1
    with torch.no_grad():
        outputs = []
        if same_shape:
            packed = torch.as_tensor(
                np.stack(arrays, axis=0), dtype=torch.float32, device=self.device
            )
            for index, mg in enumerate(self.agent_ids):
                outputs.append(
                    _actor_forward(
                        self.actors[mg],
                        packed[index : index + 1],
                        collect_diagnostics=False,
                        plain_actor=self._diagnostic_plain_actor,
                    )
                )
        else:
            for mg, array in zip(self.agent_ids, arrays):
                observation = torch.as_tensor(
                    array, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                outputs.append(
                    _actor_forward(
                        self.actors[mg],
                        observation,
                        collect_diagnostics=False,
                        plain_actor=self._diagnostic_plain_actor,
                    )
                )
        action_matrix = torch.cat(outputs, dim=0).detach().cpu().numpy()

    actions = {}
    for index, mg in enumerate(self.agent_ids):
        action = action_matrix[index].copy()
        if noise_std > 0.0:
            action += self.rng.normal(0.0, noise_std, size=action.shape)
        actions[mg] = np.clip(action, -1.0, 1.0).astype(np.float32)
    return actions


def _gradient_norm_tensor(module: nn.Module) -> torch.Tensor:
    terms = [
        parameter.grad.detach().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not terms:
        return torch.zeros((), device=next(module.parameters()).device)
    return torch.stack(terms).sum().sqrt()


def _critic_metrics(agent_ids, rows: list[torch.Tensor]) -> dict[str, object]:
    values = torch.stack(rows).detach().cpu().numpy()
    names = (
        "critic_loss",
        "critic1_loss",
        "critic2_loss",
        "critic_grad_norm",
        "q1_mean",
        "q2_mean",
        "target_q_mean",
        "td_error_mean",
        "td_error_std",
        "td_error_rmse",
        "q_mean",
        "q_std",
        "q_min",
        "q_max",
    )
    records = {
        mg: {name: float(value) for name, value in zip(names, values[index])}
        for index, mg in enumerate(agent_ids)
    }
    metrics: dict[str, object] = {}
    for name in names[:-4]:
        metrics[name] = float(np.mean([records[mg][name] for mg in agent_ids]))
    metrics["q1_q2_gap"] = float(
        np.mean(
            [abs(records[mg]["q1_mean"] - records[mg]["q2_mean"]) for mg in agent_ids]
        )
    )
    metrics["critic_losses"] = {
        mg: records[mg]["critic_loss"] for mg in agent_ids
    }
    metrics["critic1_losses"] = {
        mg: records[mg]["critic1_loss"] for mg in agent_ids
    }
    metrics["critic2_losses"] = {
        mg: records[mg]["critic2_loss"] for mg in agent_ids
    }
    metrics["critic_grad_norms"] = {
        mg: records[mg]["critic_grad_norm"] for mg in agent_ids
    }
    metrics["q_stats"] = {
        mg: {
            "mean": records[mg]["q_mean"],
            "std": records[mg]["q_std"],
            "min": records[mg]["q_min"],
            "max": records[mg]["q_max"],
        }
        for mg in agent_ids
    }
    return metrics


def _actor_metric_payload(
    self,
    actor,
    active_action,
    action_grad,
    plan,
    before: float,
    branch,
    preclip_norm: torch.Tensor,
    pre_step,
):
    step_sq = sum(
        (parameter.detach() - pre_step[name]).square().sum()
        for name, parameter in actor.named_parameters()
    )
    param_sq = sum(parameter.detach().square().sum() for parameter in actor.parameters())
    columns = [
        action_grad[:, 0].square().mean().sqrt(),
        (active_action.detach()[:, 0].abs() >= 0.95).float().mean(),
        plan["mask"].detach().float().mean(),
        plan["local_priority_reserve"].detach().abs().max(),
        step_sq.sqrt(),
        param_sq.sqrt(),
        preclip_norm.detach(),
    ]
    if active_action.shape[1] > 1:
        columns.extend(
            [
                action_grad[:, 1].square().mean().sqrt(),
                (active_action.detach()[:, 1].abs() >= 0.95).float().mean(),
                plan["dr_projected"].detach().float().mean(),
            ]
        )
    values = torch.stack(columns).detach().cpu().numpy()
    (
        action_grad_rms,
        saturation_rate,
        offer_eligible_rate,
        local_override_kw_max,
        step_norm,
        param_norm,
        preclip_value,
        *dr_values,
    ) = map(float, values)
    after = min(preclip_value, float(self.actor_max_grad_norm))
    payload = {
        "total_grad_norm": before,
        "post_clip_grad_norm": after,
        "clip_applied": float(before > self.actor_max_grad_norm),
        "dispatch_action_grad_rms": action_grad_rms,
        "dispatch_saturation_rate": saturation_rate,
        "peer_demand_available_rate": offer_eligible_rate,
        "offer_eligible_rate": offer_eligible_rate,
        "local_priority_reserve_kw_max": local_override_kw_max,
        "local_override_kw_max": local_override_kw_max,
        "optimizer_step_norm": step_norm,
        "relative_optimizer_step_norm": step_norm / max(param_norm, 1e-20),
    }
    if dr_values:
        payload.update(
            dr_action_grad_rms=dr_values[0],
            dr_saturation_rate=dr_values[1],
            dr_projection_rate=dr_values[2],
            joint_network_grad_norm=branch.get("joint", 0.0),
        )
    for name, value in branch.items():
        payload[f"{name}_grad_norm"] = value
    for name, value in getattr(actor, "last_diagnostics", {}).items():
        payload[f"actor_{name}"] = float(value)
    return payload


def _sampled_update(self, batch_size: int):
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
        target_actions = _forward_all(
            self.target_actors,
            batch["next_observations"],
            self.agent_ids,
            collect_diagnostics=False,
            plain_actor=self._diagnostic_plain_actor,
        )
        target_raw = []
        for mg in self.agent_ids:
            action = target_actions[mg]
            noise = torch.randn_like(action).mul(self.target_policy_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            target_raw.append((action + noise).clamp(-1.0, 1.0))
        target_effective, _ = self.project_actions(
            next_state, torch.stack(target_raw, dim=1)
        )
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
        current_actions = _forward_all(
            self.actors,
            batch["observations"],
            self.agent_ids,
            collect_diagnostics=diagnostic_due,
            plain_actor=self._diagnostic_plain_actor,
        )
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
                    actor_diagnostics[active] = _actor_metric_payload(
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
    if self._diagnostic_variant == "pnn":
        products = self.product_diagnostics()
        metrics["pnn_product_diagnostics"] = products
        for key in next(iter(products.values()), {}):
            metrics[f"pnn_{key}"] = float(
                np.mean([product[key] for product in products.values()])
            )
    return metrics


def fast_projected_update(self, batch_size: int):
    """Fast sampled-diagnostic update for grouped experimental actors."""
    return _sampled_update(self, batch_size)


def fast_plain_matd3_update(self, batch_size: int):
    """Fast sampled-diagnostic update for the flat MATD3 dispatch actor."""
    return _sampled_update(self, batch_size)


def install_diagnostic_sampling(cls, *, variant: str, plain_actor: bool = False):
    """Install sparse diagnostic collection on a single-dispatch algorithm class."""
    if getattr(cls, "_diagnostic_sampling_installed", False):
        return cls

    original_init = cls.__init__

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        config = args[1] if len(args) > 1 else kwargs.get("config", {})
        logging = config.get("logging", {}) if isinstance(config, Mapping) else {}
        interval = logging.get("diagnostic_interval", 20)
        self.diagnostic_schedule = DiagnosticSchedule(interval, self.policy_delay)
        self._diagnostic_variant = str(variant)
        self._diagnostic_plain_actor = bool(plain_actor)
        self.replay = FastReplayBuffer.from_replay(self.replay)

    cls.__init__ = wrapped_init
    cls.select_actions = _select_actions
    cls.update = fast_plain_matd3_update if plain_actor else fast_projected_update
    cls._diagnostic_sampling_installed = True
    return cls
