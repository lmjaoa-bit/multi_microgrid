from __future__ import annotations

import math
from typing import Any, Mapping


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    result = _number(value, name)
    if not result.is_integer() or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(result)


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = _number(value, name)
    invalid = result < 0.0 if allow_zero else result <= 0.0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def validate_algorithm_config(config: Mapping[str, Any]) -> None:
    """Validate Conditional MATD3 parameters inside its own package boundary."""
    if config.get("name") != "conditional_matd3":
        raise ValueError("algorithm.name must be 'conditional_matd3'.")
    if config.get("actor_update_mode", "synchronous") not in {
        "synchronous",
        "sequential",
    }:
        raise ValueError("algorithm.actor_update_mode must be synchronous or sequential.")
    for key in ("gamma", "tau"):
        value = _number(config.get(key), f"algorithm.{key}")
        if not 0.0 < value <= 1.0:
            raise ValueError(f"algorithm.{key} must lie inside (0, 1].")
    for key, minimum in (
        ("batch_size", 1),
        ("replay_capacity", 1),
        ("warmup_steps", 0),
        ("updates_per_step", 1),
        ("policy_delay", 1),
    ):
        _integer(config.get(key), f"algorithm.{key}", minimum)
    _positive(config.get("actor_lr"), "algorithm.actor_lr")
    _positive(config.get("critic_lr"), "algorithm.critic_lr")
    for key in ("actor_hidden_dims", "critic_hidden_dims"):
        dimensions = config.get(key)
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError(f"algorithm.{key} must be a non-empty list.")
        for index, width in enumerate(dimensions):
            _integer(width, f"algorithm.{key}[{index}]", 1)

    commitment = config.get("commitment", {})
    for key in ("epsilon_start", "epsilon_end"):
        value = _number(commitment.get(key), f"algorithm.commitment.{key}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"algorithm.commitment.{key} must lie inside [0, 1].")
    _integer(
        commitment.get("epsilon_decay_steps"),
        "algorithm.commitment.epsilon_decay_steps",
        0,
    )
    _positive(
        commitment.get("soft_target_temperature"),
        "algorithm.commitment.soft_target_temperature",
    )
    target_min = _number(
        commitment.get("soft_target_min"), "algorithm.commitment.soft_target_min"
    )
    target_max = _number(
        commitment.get("soft_target_max"), "algorithm.commitment.soft_target_max"
    )
    if not 0.0 <= target_min < target_max <= 1.0:
        raise ValueError("Commitment soft-target bounds must satisfy 0 <= min < max <= 1.")
    beta = _number(
        commitment.get("q_scale_ema_beta"),
        "algorithm.commitment.q_scale_ema_beta",
    )
    if not 0.0 <= beta < 1.0:
        raise ValueError("algorithm.commitment.q_scale_ema_beta must lie inside [0, 1).")
    _positive(
        commitment.get("q_scale_epsilon"),
        "algorithm.commitment.q_scale_epsilon",
    )

    exploration = config.get("exploration_noise", {})
    _integer(
        exploration.get("decay_steps"),
        "algorithm.exploration_noise.decay_steps",
        0,
    )
    for phase in ("start", "end"):
        for head in ("battery", "generator_setpoint", "p2p_bid"):
            _positive(
                exploration.get(phase, {}).get(head),
                f"algorithm.exploration_noise.{phase}.{head}",
                allow_zero=True,
            )
    smoothing = config.get("target_smoothing", {})
    for field in ("sigma", "clip"):
        for head in ("battery", "generator_setpoint", "p2p_bid"):
            _positive(
                smoothing.get(field, {}).get(head),
                f"algorithm.target_smoothing.{field}.{head}",
                allow_zero=True,
            )
    for key in ("commitment", "common", "generator"):
        _positive(
            config.get("actor_loss_weights", {}).get(key),
            f"algorithm.actor_loss_weights.{key}",
            allow_zero=True,
        )
