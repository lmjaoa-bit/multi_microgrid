from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from core.reward_v2 import (
    REWARD_V2_VERSION,
    resolve_reward_v2_config,
    stable_contract_hash,
)
from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.evaluate_matd3 import evaluate_policy_on_days
from utils.metrics_collector import distribution_statistics


PolicyFn = Callable[[Mapping[str, np.ndarray]], dict[str, np.ndarray]]

_RAW_TO_RATE = {
    "curtailment_kwh": "curtailment_price_per_kwh",
    "carbon_kg": "carbon_price_per_kg",
    "delta_peak_kw": "peak_price_per_kw",
    "mill_unserved_t": "mill_unserved_price_per_t",
    "comfort_degree_hours": "comfort_price_per_degree_hour",
    "ev_departure_shortfall_kwh": "ev_shortfall_price_per_kwh",
    "unmet_kwh": "unmet_load_cost_per_kwh",
}


def _policy_name(policy: Any) -> str:
    if isinstance(policy, str):
        return policy
    return str(
        getattr(policy, "calibration_policy_name", None)
        or getattr(policy, "__name__", type(policy).__name__)
    )


def build_idle_policy(env: MultiMicrogridEnv) -> PolicyFn:
    """Device-aware low-intervention calibration policy.

    Battery dispatch requests zero throughput.  HVAC requests zero signed power.
    EV requests the lower endpoint (raw=-1 => zero in V2).  The industrial mill
    also requests its lower endpoint; its retained instantaneous silo bound may
    still force the minimum physically admissible production.
    """
    kinds = {
        mg: str(env.cfg_by_id[mg].get("dr", {}).get("type", ""))
        for mg in env.agent_ids
    }

    def policy(_observations: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        actions: dict[str, np.ndarray] = {}
        for mg in env.agent_ids:
            kind = kinds[mg]
            dr_raw = 0.0 if kind == "commercial_hvac" else -1.0
            actions[mg] = np.asarray([0.0, dr_raw], dtype=np.float32)
        return actions

    policy.calibration_policy_name = "device_specific_idle"  # type: ignore[attr-defined]
    return policy


def _ev_urgency_raw(device: Any, t: int) -> float:
    low, high = device.feasible_bounds(t)
    if high <= low + 1e-12:
        return -1.0
    if getattr(device, "cohort_mode", False):
        hour = device._hour(t)
        desired = 0.0
        for session in device.sessions:
            group = session["group_id"]
            if not (
                int(session["arrival_hour"])
                <= hour
                < int(session["departure_hour"])
            ):
                continue
            hours = max(1.0, float(session["departure_hour"] - hour))
            desired += float(device.remaining[group]) / (
                float(device.eta) * hours
            )
    else:
        state = device.state(t)
        hours = max(1.0, float(state.get("ev_hours_to_departure", 0.0)))
        desired = float(state.get("ev_remaining_battery_kwh", 0.0)) / (
            float(device.eta) * hours
        )
    desired = float(np.clip(desired, low, high))
    unit = (desired - low) / max(high - low, 1e-12)
    return float(np.clip(2.0 * unit - 1.0, -1.0, 1.0))


def build_service_priority_policy(env: MultiMicrogridEnv) -> PolicyFn:
    """Current-state service-first rule used only for calibration diagnostics.

    It deliberately does not call HVAC forecast viability or future
    reachability projections.
    """

    def policy(_observations: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        actions: dict[str, np.ndarray] = {}
        t = int(env.t)
        for mg in env.agent_ids:
            device = env.dr_devices.get(mg)
            kind = str(env.cfg_by_id[mg].get("dr", {}).get("type", ""))
            dr_raw = -1.0
            if kind == "industrial_mill":
                # Request the current instantaneous feasible minimum. The
                # environment retains the physical silo bound.
                low, high = device.feasible_bounds(t)
                if high <= low + 1e-12:
                    dr_raw = -1.0
                else:
                    dr_raw = -1.0
            elif kind == "commercial_hvac":
                low_c, high_c = device._comfort_bounds(t)
                target = 0.5 * (float(low_c) + float(high_c))
                error = target - float(device.t_air)
                if error >= 0.0:
                    desired = min(float(device.heat_max_kw), 0.5 * abs(error) * float(device.heat_max_kw))
                    dr_raw = desired / max(float(device.heat_max_kw), 1e-12)
                else:
                    desired = min(float(device.cool_max_kw), 0.5 * abs(error) * float(device.cool_max_kw))
                    dr_raw = -desired / max(float(device.cool_max_kw), 1e-12)
                dr_raw = float(np.clip(dr_raw, -1.0, 1.0))
            elif kind == "ev_charging":
                dr_raw = _ev_urgency_raw(device, t)
            actions[mg] = np.asarray([0.0, dr_raw], dtype=np.float32)
        return actions

    policy.calibration_policy_name = "current_state_service_priority"  # type: ignore[attr-defined]
    return policy


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    return distribution_statistics([float(value) for value in values])


def _configured_rates(resolved: Mapping[str, Any]) -> dict[str, float]:
    keys = (
        "curtailment_price_per_kwh",
        "carbon_price_per_kg",
        "peak_price_per_kw",
        "mill_unserved_price_per_t",
        "comfort_price_per_degree_hour",
        "ev_shortfall_price_per_kwh",
        "unmet_load_cost_per_kwh",
    )
    return {key: float(resolved[key]) for key in keys}


def collect_calibration(
    config: Mapping[str, Any],
    *,
    split: str,
    days: Sequence[str],
    policy: Any,
    seed: int,
) -> dict[str, Any]:
    """Collect Reward V2 raw-component evidence on train/validation days only."""
    if split not in {"train", "val"}:
        raise ValueError("Reward calibration accepts only train or val split")
    if not days:
        raise ValueError("Reward calibration requires at least one complete day")
    resolved = resolve_reward_v2_config(config)
    if resolved is None:
        raise ValueError(
            f"Reward calibration requires {REWARD_V2_VERSION}"
        )

    env = MultiMicrogridEnv(config, split=split)
    if isinstance(policy, str):
        if policy == "idle":
            policy_fn = build_idle_policy(env)
        elif policy in {"service", "service_priority"}:
            policy_fn = build_service_priority_policy(env)
        else:
            raise ValueError(f"Unknown calibration policy: {policy!r}")
    elif callable(policy):
        policy_fn = policy
    else:
        raise TypeError("policy must be callable or one of: idle, service_priority")

    summary, daily = evaluate_policy_on_days(
        env,
        list(days),
        policy_fn,
        seed=int(seed),
        deterministic_environment=True,
        forecast_noise=False,
        capture_hourly=False,
    )
    raw_keys = (
        "grid_import_cost",
        "battery_degradation_cost",
        "curtailment_kwh",
        "carbon_kg",
        "delta_peak_kw",
        "mill_unserved_t",
        "comfort_degree_hours",
        "ev_departure_shortfall_kwh",
        "unmet_kwh",
    )
    raw_distributions: dict[str, Any] = {}
    for key in raw_keys:
        values = [
            float((item.get("reward_v2_raw_components") or {}).get(key, 0.0))
            for item in daily
        ]
        raw_distributions[key] = _distribution(values)

    generalized = [
        float(item["generalized_system_cost"])
        for item in daily
        if item.get("generalized_system_cost") is not None
    ]
    configured = _configured_rates(resolved)

    # This is intentionally a diagnostic scale reference, not an automatic
    # coefficient selector.  It helps expose which raw terms are rare/large
    # without converting "equal numerical magnitude" into a design rule.
    grid_reference = max(
        1e-12,
        float(raw_distributions["grid_import_cost"].get("median") or 0.0),
    )
    scale_reference: dict[str, float | None] = {}
    for raw_key, rate_key in _RAW_TO_RATE.items():
        median = raw_distributions[raw_key].get("median")
        scale_reference[rate_key] = (
            None
            if median is None or abs(float(median)) <= 1e-12
            else float(grid_reference / abs(float(median)))
        )

    config_for_hash = deepcopy(dict(config))
    config_for_hash.pop("_project_root", None)
    report = {
        "schema_version": "reward_calibration_v1",
        "reward_semantics_version": resolved["semantics_version"],
        "calibration_status": resolved["calibration_status"],
        "split": split,
        "days": list(days),
        "day_count": len(days),
        "seed": int(seed),
        "policy": _policy_name(policy_fn),
        "config_hash": stable_contract_hash(config_for_hash),
        "raw_component_distributions": raw_distributions,
        "generalized_system_cost_distribution": _distribution(generalized),
        "weight_candidates": {
            "configured_design_candidate": {
                "coefficients": configured,
                "reason": (
                    "Explicit engineering/economic coefficients supplied by the "
                    "configuration; must be justified by marginal trade-offs."
                ),
            },
            "numerical_scale_reference_not_for_locking": {
                "coefficients": scale_reference,
                "reason": (
                    "Median-magnitude reference only. It must not be used to "
                    "auto-equalize objectives or lock Reward V2."
                ),
            },
        },
        "reward_scale": float(resolved["reward_scale"]),
        "scale_diagnostic": {
            "observed_generalized_cost_p95": (
                _distribution(generalized).get("p95")
            ),
            "unit_p95_reference_scale": max(
                1.0,
                float(_distribution(generalized).get("p95") or 0.0),
            ),
            "reason": (
                "Numerical stability reference only; reward_scale does not "
                "encode optimization preference."
            ),
        },
        "selection_note": (
            "Choose/lock coefficients from train/validation marginal trade-offs; "
            "never from formal test performance and never solely by equalizing "
            "component magnitudes."
        ),
        "evaluation_summary": {
            "team_reward_mean": summary.get("team_reward_mean"),
            "generalized_system_cost_mean": summary.get(
                "generalized_system_cost_mean"
            ),
            "team_profit_mean": summary.get("team_profit_mean"),
        },
    }
    return report
