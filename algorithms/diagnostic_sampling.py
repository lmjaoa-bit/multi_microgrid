from __future__ import annotations

import torch
from torch import nn


def _finite_scalar(value: torch.Tensor, label: str) -> None:
    # Keep a cheap correctness guard on every update. The expensive diagnostic
    # reductions (Q distributions, per-branch norms, optimizer-step norms) are
    # what diagnostic sampling removes from ordinary steps.
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite {label}")


def fast_projected_update(algorithm, batch_size: int) -> dict:
    """One normal training update without CPU-facing diagnostic reductions.

    Works for all projector-based variants: MATD3 dispatch, PNN, Transformer,
    and PNN-Transformer. Optimizer, target-noise, delayed-policy and soft-target
    update semantics are unchanged; only monitoring work is omitted.
    """
    if len(algorithm.replay) < batch_size:
        return {}

    batch = algorithm.replay.sample(batch_size, algorithm.agent_ids, algorithm.device)
    state = batch["global_state"]
    next_state = batch["next_global_state"]
    raw = torch.stack([batch["actions"][mg] for mg in algorithm.agent_ids], dim=1)
    effective, _ = algorithm.project_actions(state, raw)

    with torch.no_grad():
        target_raw = []
        for mg in algorithm.agent_ids:
            action = algorithm.target_actors[mg](batch["next_observations"][mg])
            noise = torch.randn_like(action).mul(algorithm.target_policy_noise).clamp(
                -algorithm.target_noise_clip, algorithm.target_noise_clip
            )
            target_raw.append((action + noise).clamp(-1.0, 1.0))
        target_effective, _ = algorithm.project_actions(
            next_state, torch.stack(target_raw, dim=1)
        )
        target_q = {}
        for mg in algorithm.agent_ids:
            q1_target, q2_target = algorithm.target_critics[mg](
                next_state, target_effective.flatten(1)
            )
            target_q[mg] = batch["team_reward"] + (
                1.0 - batch["done"]
            ) * algorithm.gamma * torch.minimum(q1_target, q2_target)

    flat_effective = effective.flatten(1)
    for mg in algorithm.agent_ids:
        q1, q2 = algorithm.critics[mg](state, flat_effective)
        loss = nn.functional.mse_loss(q1, target_q[mg]) + nn.functional.mse_loss(q2, target_q[mg])
        _finite_scalar(loss, f"{mg} Critic loss")
        algorithm.critic_optimizers[mg].zero_grad(set_to_none=True)
        loss.backward()
        algorithm.critic_optimizers[mg].step()

    algorithm.update_steps += 1
    if algorithm.update_steps % algorithm.policy_delay != 0:
        return {}

    with torch.no_grad():
        cached_actions = {
            mg: algorithm.actors[mg](batch["observations"][mg])
            for mg in algorithm.agent_ids
        }

    for active in algorithm.agent_ids:
        actor = algorithm.actors[active]
        critic = algorithm.critics[active]
        optimizer = algorithm.actor_optimizers[active]
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
        try:
            active_action = actor(batch["observations"][active])
            joint = [
                active_action if mg == active else cached_actions[mg]
                for mg in algorithm.agent_ids
            ]
            projected, _ = algorithm.project_actions(state, torch.stack(joint, dim=1))
            loss = -critic.q1(state, projected.flatten(1)).mean()
            _finite_scalar(loss, f"{active} Actor loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            max_norm = getattr(algorithm, "actor_max_grad_norm", None)
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), float(max_norm))
            optimizer.step()
            algorithm._soft_update(actor, algorithm.target_actors[active])
            algorithm._soft_update(critic, algorithm.target_critics[active])
        finally:
            for parameter in critic.parameters():
                parameter.requires_grad_(True)
    return {}


def fast_plain_matd3_update(algorithm, batch_size: int) -> dict:
    """Base MATD3 update without monitoring reductions."""
    if len(algorithm.replay) < batch_size:
        return {}
    batch = algorithm.replay.sample(batch_size, algorithm.agent_ids, algorithm.device)
    observations = batch["observations"]
    next_observations = batch["next_observations"]
    actions = batch["actions"]
    joint_action = torch.cat([actions[mg] for mg in algorithm.agent_ids], dim=-1)

    with torch.no_grad():
        target_parts = []
        for mg in algorithm.agent_ids:
            target = algorithm.target_actors[mg](next_observations[mg])
            noise = torch.randn_like(target).mul(algorithm.target_policy_noise).clamp(
                -algorithm.target_noise_clip, algorithm.target_noise_clip
            )
            target_parts.append((target + noise).clamp(-1.0, 1.0))
        target_joint = torch.cat(target_parts, dim=-1)
        target_q = {}
        for mg in algorithm.agent_ids:
            q1_target, q2_target = algorithm.target_critics[mg](
                batch["next_global_state"], target_joint
            )
            target_q[mg] = batch["team_reward"] + (
                1.0 - batch["done"]
            ) * algorithm.gamma * torch.minimum(q1_target, q2_target)

    for mg in algorithm.agent_ids:
        q1, q2 = algorithm.critics[mg](batch["global_state"], joint_action)
        loss = nn.functional.mse_loss(q1, target_q[mg]) + nn.functional.mse_loss(q2, target_q[mg])
        _finite_scalar(loss, f"{mg} Critic loss")
        algorithm.critic_optimizers[mg].zero_grad(set_to_none=True)
        loss.backward()
        algorithm.critic_optimizers[mg].step()

    algorithm.update_steps += 1
    if algorithm.update_steps % algorithm.policy_delay != 0:
        return {}

    for active in algorithm.agent_ids:
        for parameter in algorithm.critics[active].parameters():
            parameter.requires_grad_(False)
        try:
            parts = []
            for mg in algorithm.agent_ids:
                action = algorithm.actors[mg](observations[mg])
                parts.append(action if mg == active else action.detach())
            loss = -algorithm.critics[active].q1(
                batch["global_state"], torch.cat(parts, dim=-1)
            ).mean()
            _finite_scalar(loss, f"{active} Actor loss")
            algorithm.actor_optimizers[active].zero_grad(set_to_none=True)
            loss.backward()
            algorithm.actor_optimizers[active].step()
            algorithm._soft_update(algorithm.actors[active], algorithm.target_actors[active])
            algorithm._soft_update(algorithm.critics[active], algorithm.target_critics[active])
        finally:
            for parameter in algorithm.critics[active].parameters():
                parameter.requires_grad_(True)
    return {}
