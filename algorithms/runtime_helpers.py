from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
from numbers import Integral

import numpy as np
import torch
from torch import nn


def diagnostics_due(update_steps: int, interval: int) -> bool:
    """Return True only on configured diagnostic-sampling updates.

    interval <= 0 disables expensive diagnostics entirely.
    """
    return int(interval) > 0 and int(update_steps) % int(interval) == 0


def forward_policy_actions(
    actors: Mapping[str, nn.Module],
    observations: Mapping[str, torch.Tensor],
    agent_ids: Sequence[str],
    *,
    collect_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Forward every actor exactly once for one policy-update batch."""
    return {
        mg: actors[mg](observations[mg], collect_diagnostics=collect_diagnostics)
        for mg in agent_ids
    }


def select_single_dispatch_actions(
    algorithm,
    observations: Mapping[str, np.ndarray],
    noise_std: float = 0.0,
) -> dict[str, np.ndarray]:
    """Experimental single-dispatch rollout with one GPU->CPU synchronization."""
    tensors: list[torch.Tensor] = []
    with torch.no_grad():
        for mg in algorithm.agent_ids:
            obs = torch.as_tensor(
                observations[mg], dtype=torch.float32, device=algorithm.device
            ).unsqueeze(0)
            tensors.append(algorithm.actors[mg](obs, collect_diagnostics=False))
        action_matrix = torch.cat(tensors, dim=0).detach().cpu().numpy()

    actions: dict[str, np.ndarray] = {}
    for index, mg in enumerate(algorithm.agent_ids):
        action = action_matrix[index].copy()
        if noise_std > 0.0:
            action += algorithm.rng.normal(0.0, noise_std, size=action.shape)
        actions[mg] = np.clip(action, -1.0, 1.0).astype(np.float32)
    return actions


def _actor_diagnostic_payload(
    variant: str,
    *,
    before: float,
    after: float,
    branch: Mapping[str, float],
    action_grad_rms: float,
    saturation_rate: float,
    offer_eligible_rate: float,
    forced_dispatch_rate: float,
    local_override_kw_max: float,
    step_norm: float,
    param_norm: float,
    actor: nn.Module,
) -> dict[str, float]:
    common = {
        "total_grad_norm": before,
        "post_clip_grad_norm": after,
        "clip_applied": float(before > getattr(actor, "_runtime_actor_max_grad_norm", float("inf"))),
        "dispatch_action_grad_rms": action_grad_rms,
        "dispatch_saturation_rate": saturation_rate,
        "offer_eligible_rate": offer_eligible_rate,
        "forced_dispatch_rate": forced_dispatch_rate,
        "local_override_kw_max": local_override_kw_max,
        "optimizer_step_norm": step_norm,
        "relative_optimizer_step_norm": step_norm / max(param_norm, 1e-20),
    }

    if variant == "pnn":
        common.update({
            "local_feature_grad_norm": branch.get("local_features", 0.0),
            "temporal_feature_grad_norm": branch.get("temporal_features", 0.0),
            "market_feature_grad_norm": branch.get("market_features", 0.0),
            "peer_feature_grad_norm": branch.get("peer_features", 0.0),
            "product_norm_grad_norm": branch.get("product_norm", 0.0),
            "dispatch_grad_norm": branch.get("dispatch", 0.0),
        })
        return common

    diag_map = getattr(actor, "last_diagnostics", {})
    common.update({
        "dispatch_token_grad_norm": branch.get("dispatch_token", 0.0),
        "hour_tokenizer_grad_norm": branch.get("hour_tokenizer", 0.0),
        "local_tokenizer_grad_norm": branch.get("local_tokenizer", 0.0),
        "forecast_tokenizer_grad_norm": branch.get("forecast_tokenizer", 0.0),
        "peer_tokenizer_grad_norm": branch.get("peer_tokenizer", 0.0),
        "market_tokenizer_grad_norm": branch.get("market_tokenizer", 0.0),
        "transformer_grad_norm": branch.get("transformer", 0.0),
        "backbone_grad_norm": branch.get("backbone", 0.0),
        "head_grad_norm": branch.get("head", 0.0),
    })
    if variant == "matd3_transformer":
        common.update({
            "interact_norm_grad_norm": branch.get("interact_norm", 0.0),
            "transformer_dispatch_norm": diag_map.get("dispatch_norm", 0.0),
            "transformer_local_residual_norm": diag_map.get("local_norm", 0.0),
            "transformer_interact_norm": diag_map.get("interact_norm", 0.0),
            "dispatch_local_norm_ratio": diag_map.get("norm_ratio", 0.0),
            "forecast_valid_fraction": diag_map.get("forecast_valid_fraction", 0.0),
        })
    elif variant == "pnn_transformer":
        common.update({
            "pnn_encoders_grad_norm": branch.get("pnn_encoders", 0.0),
            "product_norm_grad_norm": branch.get("product_norm", 0.0),
            "transformer_dispatch_norm": diag_map.get("dispatch_norm", 0.0),
            "pnn_product_norm": diag_map.get("product_norm", 0.0),
            "local_residual_norm": diag_map.get("local_norm", 0.0),
            "temporal_product_norm": diag_map.get("temporal_product_norm", 0.0),
            "spatial_product_norm": diag_map.get("spatial_product_norm", 0.0),
            "market_product_norm": diag_map.get("market_product_norm", 0.0),
            "product_to_dispatch_ratio": diag_map.get("product_to_dispatch_ratio", 0.0),
            "local_to_dispatch_ratio": diag_map.get("local_to_dispatch_ratio", 0.0),
            "forecast_valid_fraction": diag_map.get("forecast_valid_fraction", 0.0),
        })
    return common


def optimized_single_dispatch_update(self, batch_size: int):
    """MATD3 update for experimental single-dispatch actors with fewer synchronizations.

    Mathematical updates are unchanged: one critic update per environment step, delayed actor
    updates, TD3 target smoothing, the same differentiable action projection, and the same soft
    target updates. Runtime-only differences are actor-forward caching and sparse diagnostics.
    """
    if len(self.replay) < batch_size:
        return {}

    batch = self.replay.sample(batch_size, self.agent_ids, self.device)
    state, next_state = batch["global_state"], batch["next_global_state"]
    raw = torch.stack([batch["actions"][mg] for mg in self.agent_ids], dim=1)
    effective, _ = self.project_actions(state, raw)

    with torch.no_grad():
        target_actions = forward_policy_actions(
            self.target_actors,
            batch["next_observations"],
            self.agent_ids,
            collect_diagnostics=False,
        )
        target_raw = []
        for mg in self.agent_ids:
            action = target_actions[mg]
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

    critic_rows: list[torch.Tensor] = []
    for mg in self.agent_ids:
        q1, q2 = self.critics[mg](state, effective.flatten(1))
        c1 = nn.functional.mse_loss(q1, target_q[mg])
        c2 = nn.functional.mse_loss(q2, target_q[mg])
        loss = c1 + c2
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"{mg}: nonfinite Critic loss")

        optimizer = self.critic_optimizers[mg]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critics[mg].parameters(),
            max_norm=float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()

        q_min = torch.minimum(q1.detach(), q2.detach())
        td = (target_q[mg] - torch.minimum(q1, q2)).detach()
        critic_rows.append(torch.stack([
            loss.detach(),
            grad_norm.detach(),
            q_min.mean(),
            q_min.std(unbiased=False),
            td.mean(),
            td.square().mean().sqrt(),
        ]))

    critic_values = torch.stack(critic_rows).detach().cpu().numpy()
    records: dict[str, float] = {}
    q_stats: dict[str, dict[str, float]] = {}
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
        collect_diagnostics = diagnostics_due(
            self.update_steps, getattr(self, "diagnostics_interval", 100)
        )
        current_actions = forward_policy_actions(
            self.actors,
            batch["observations"],
            self.agent_ids,
            collect_diagnostics=collect_diagnostics,
        )

        loss_tensors: dict[str, torch.Tensor] = {}
        grads: dict[str, float] = {}
        branches: dict[str, dict[str, float]] = {}
        layers: dict[str, dict[str, float]] = {}
        diagnostic: dict[str, dict[str, float]] = {}

        for active_index, active in enumerate(self.agent_ids):
            actor = self.actors[active]
            critic = self.critics[active]
            optimizer = self.actor_optimizers[active]
            actor._runtime_actor_max_grad_norm = self.actor_max_grad_norm
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            try:
                active_action = current_actions[active]
                if collect_diagnostics:
                    active_action.retain_grad()
                joint = [
                    current_actions[mg] if mg == active else current_actions[mg].detach()
                    for mg in self.agent_ids
                ]
                projected, plans = self.project_actions(state, torch.stack(joint, dim=1))
                plan = plans[active_index]
                q1 = critic.q1(state, projected.flatten(1))
                loss = -q1.mean()
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f"{active}: nonfinite Actor loss")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if collect_diagnostics:
                    before, branch, layer = self._actor_gradient_diagnostics(active)
                    grads[active], branches[active], layers[active] = before, branch, layer
                    action_grad = (
                        active_action.grad.detach() * batch_size
                        if active_action.grad is not None
                        else torch.zeros_like(active_action)
                    )
                    pre_step = {
                        name: param.detach().clone()
                        for name, param in actor.named_parameters()
                    }
                else:
                    before = 0.0
                    branch = {}
                    action_grad = None
                    pre_step = None

                preclip_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    self.actor_max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()

                if collect_diagnostics:
                    step_sq = sum(
                        (param.detach() - pre_step[name]).square().sum()
                        for name, param in actor.named_parameters()
                    )
                    param_sq = sum(param.detach().square().sum() for param in actor.parameters())
                    diagnostic_values = torch.stack([
                        action_grad[:, 0].square().mean().sqrt(),
                        (active_action.detach().abs() >= 0.95).float().mean(),
                        plan["mask"].detach().float().mean(),
                        plan["forced_dispatch"].detach().float().mean(),
                        plan["local_priority_reserve"].detach().abs().max(),
                        step_sq.sqrt(),
                        param_sq.sqrt(),
                        preclip_norm.detach(),
                    ]).detach().cpu().numpy()
                    (
                        action_grad_rms,
                        saturation_rate,
                        offer_eligible_rate,
                        forced_dispatch_rate,
                        local_override_kw_max,
                        step_norm,
                        param_norm,
                        preclip_value,
                    ) = map(float, diagnostic_values)
                    after = min(preclip_value, float(self.actor_max_grad_norm))
                    diagnostic[active] = _actor_diagnostic_payload(
                        getattr(self, "_runtime_variant", ""),
                        before=before,
                        after=after,
                        branch=branch,
                        action_grad_rms=action_grad_rms,
                        saturation_rate=saturation_rate,
                        offer_eligible_rate=offer_eligible_rate,
                        forced_dispatch_rate=forced_dispatch_rate,
                        local_override_kw_max=local_override_kw_max,
                        step_norm=step_norm,
                        param_norm=param_norm,
                        actor=actor,
                    )

                loss_tensors[active] = loss.detach()
                self._soft_update(actor, self.target_actors[active])
                self._soft_update(critic, self.target_critics[active])
            finally:
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)

        actor_loss_values = torch.stack([
            loss_tensors[mg] for mg in self.agent_ids
        ]).detach().cpu().numpy()
        actor_losses = {
            mg: float(actor_loss_values[index])
            for index, mg in enumerate(self.agent_ids)
        }
        metrics.update(
            actor_loss=float(np.mean(list(actor_losses.values()))),
            actor_losses=actor_losses,
        )
        if collect_diagnostics:
            metrics.update(
                actor_grad_norm=float(np.mean(list(grads.values()))),
                actor_grad_norms=grads,
                actor_branch_grad_norms=branches,
                actor_layer_grad_norms=layers,
                actor_diagnostics=diagnostic,
            )
            if getattr(self, "_runtime_variant", "") == "pnn":
                prods = self.product_diagnostics()
                metrics["pnn_product_diagnostics"] = prods
                for key in next(iter(prods.values()), {}):
                    metrics[f"pnn_{key}"] = float(np.mean([p[key] for p in prods.values()]))

    counts = self.actor_parameter_counts()
    metrics["actor_parameter_counts"] = counts
    metrics["actor_parameter_count"] = float(np.mean(list(counts.values())))
    return metrics


def install_single_dispatch_runtime_optimizations(cls, *, variant: str):
    """Install runtime-only optimizations on one experimental algorithm class."""
    if getattr(cls, "_runtime_optimizations_installed", False):
        return cls

    original_init = cls.__init__

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        config = args[1] if len(args) > 1 else kwargs.get("config", {})
        settings = config.get("algorithm", config) if isinstance(config, Mapping) else {}
        raw_interval = settings.get("diagnostics_interval", 100)
        if isinstance(raw_interval, bool) or not isinstance(raw_interval, Integral) or raw_interval < 0:
            raise ValueError("algorithm.diagnostics_interval must be a non-negative integer")
        self.diagnostics_interval = int(raw_interval)
        self._runtime_variant = str(variant)

    cls.__init__ = wrapped_init
    cls.select_actions = select_single_dispatch_actions
    cls.update = optimized_single_dispatch_update
    cls._runtime_optimizations_installed = True
    return cls
