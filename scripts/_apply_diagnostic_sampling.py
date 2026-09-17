from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def replace_once(rel: str, old: str, new: str, marker: str) -> None:
    text = read(rel)
    if old in text:
        write(rel, text.replace(old, new, 1))
        return
    if marker in text:
        return
    raise RuntimeError(f"{rel}: replacement source not found")


write(
    "algorithms/diagnostic_sampling.py",
    '''from __future__ import annotations

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
''',
)

# Base MATD3 accepts the same API. Its optimized path matters for legacy callers
# and also keeps the contract uniform for all descendants.
replace_once(
    "algorithms/matd3/matd3.py",
    'from algorithms.matd3.replay_buffer import ReplayBuffer\n',
    'from algorithms.matd3.replay_buffer import ReplayBuffer\nfrom algorithms.diagnostic_sampling import fast_plain_matd3_update\n',
    "fast_plain_matd3_update",
)
replace_once(
    "algorithms/matd3/matd3.py",
    '    def update(self, batch_size: int) -> dict[str, Any]:\n        if len(self.replay) < batch_size:\n',
    '    def update(self, batch_size: int, collect_diagnostics: bool = True) -> dict[str, Any]:\n        if not collect_diagnostics:\n            return fast_plain_matd3_update(self, batch_size)\n        if len(self.replay) < batch_size:\n',
    "collect_diagnostics: bool = True",
)

for rel in (
    "algorithms/matd3_dispatch/algorithm.py",
    "algorithms/pnn/algorithm.py",
    "algorithms/matd3_transformer/algorithm.py",
    "algorithms/pnn_transformer/algorithm.py",
):
    replace_once(
        rel,
        'from algorithms.matd3 import MATD3\n',
        'from algorithms.matd3 import MATD3\nfrom algorithms.diagnostic_sampling import fast_projected_update\n',
        "fast_projected_update",
    )
    replace_once(
        rel,
        '    def update(self, batch_size):\n        if len(self.replay) < batch_size:\n',
        '    def update(self, batch_size, collect_diagnostics: bool = True):\n        if not collect_diagnostics:\n            return fast_projected_update(self, batch_size)\n        if len(self.replay) < batch_size:\n',
        "collect_diagnostics: bool = True",
    )

# Training pipeline chooses when to request the full monitoring path. Sampling is
# counted in optimizer updates (not environment steps), so warm-up does not skew it.
replace_once(
    "experiments/training_pipeline.py",
    '    snapshot_interval = logging_cfg.get("history_snapshot_interval", 500)\n',
    '    snapshot_interval = logging_cfg.get("history_snapshot_interval", 500)\n    diagnostic_interval = logging_cfg.get("diagnostic_interval", 20)\n',
    "diagnostic_interval = logging_cfg.get",
)
replace_once(
    "experiments/training_pipeline.py",
    '    if incremental_history and (isinstance(snapshot_interval, bool)\n                                or not isinstance(snapshot_interval, int)\n                                or snapshot_interval < 1):\n        raise ValueError("logging.history_snapshot_interval must be a positive integer")\n',
    '    if incremental_history and (isinstance(snapshot_interval, bool)\n                                or not isinstance(snapshot_interval, int)\n                                or snapshot_interval < 1):\n        raise ValueError("logging.history_snapshot_interval must be a positive integer")\n    if (isinstance(diagnostic_interval, bool) or not isinstance(diagnostic_interval, int)\n            or diagnostic_interval < 1):\n        raise ValueError("logging.diagnostic_interval must be a positive integer")\n',
    "logging.diagnostic_interval must be a positive integer",
)
replace_once(
    "experiments/training_pipeline.py",
    '        "runtime": runtime_settings,\n',
    '        "runtime": runtime_settings,\n        "diagnostic_interval": int(diagnostic_interval),\n',
    '"diagnostic_interval": int(diagnostic_interval)',
)
replace_once(
    "experiments/training_pipeline.py",
    '                update_metrics = algorithm.update(batch_size)\n',
    '                collect_diagnostics = (\n                    int(getattr(algorithm, "update_steps", 0)) % int(diagnostic_interval) == 0\n                )\n                update_metrics = algorithm.update(\n                    batch_size, collect_diagnostics=collect_diagnostics\n                )\n',
    "collect_diagnostics=collect_diagnostics",
)

# Strengthen the contract test to include the actual canonical MATD3 plugin class.
replace_once(
    "tests/test_instantaneous_default_and_p0.py",
    'from algorithms.matd3 import MATD3\n',
    'from algorithms.matd3 import MATD3\nfrom algorithms.matd3_dispatch.algorithm import Matd3Dispatch\n',
    "Matd3Dispatch",
)
replace_once(
    "tests/test_instantaneous_default_and_p0.py",
    '    for cls in (MATD3, PNN, Matd3Transformer, PnnTransformer):\n',
    '    for cls in (MATD3, Matd3Dispatch, PNN, Matd3Transformer, PnnTransformer):\n',
    "MATD3, Matd3Dispatch, PNN",
)

print("diagnostic sampling refactor applied")
