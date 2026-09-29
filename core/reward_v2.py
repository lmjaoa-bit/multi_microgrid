from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REWARD_V2_VERSION = "generalized_system_cost_v2"
OBSERVATION_V2_VERSION = "reward_v2_public_state_v1"
DISPATCH_V2_VERSION = "renewable_load_first_v2_no_grid_export"
DR_PROJECTION_V2_VERSION = "instantaneous_physical_only_v2"
HVAC_V2_VERSION = "soft_comfort_penalty_v2"
EV_V2_VERSION = "no_deadline_min_projection_v2"
REWARD_DISCOUNT_V2_VERSION = "finite_day_undiscounted_v1"
COMFORT_METRIC_V2_VERSION = "boundary_sampled_degree_hours"
CARBON_INFO_V2_VERSION = "current_observed_no_future_v1"
SETTLEMENT_VALUE_V2_VERSION = "accounting_profit_v2"

_RAW_KEYS = (
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


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _resolve_alias(
    *,
    new_value: Any,
    old_value: Any,
    new_name: str,
    old_name: str,
    required: bool = True,
) -> float | None:
    if new_value is None and old_value is None:
        if required:
            raise ValueError(f"{new_name} is required")
        return None
    new = None if new_value is None else _finite_nonnegative(new_value, new_name)
    old = None if old_value is None else _finite_nonnegative(old_value, old_name)
    if new is not None and old is not None and not math.isclose(
        new, old, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"Conflicting reward aliases: {new_name}={new} and {old_name}={old}"
        )
    return new if new is not None else old


def _ev_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    for microgrid in config.get("microgrids", []):
        dr = microgrid.get("dr", {})
        if str(dr.get("type", "")) == "ev_charging":
            return dr
    return {}


def resolve_reward_v2_config(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a validated Reward V2 contract without mutating the input."""
    reward = config.get("reward")
    if reward is None:
        return None
    if not isinstance(reward, Mapping):
        raise ValueError("reward must be a mapping")
    version = reward.get("semantics_version")
    if version is None:
        return None
    if str(version) != REWARD_V2_VERSION:
        raise ValueError(
            f"Unsupported reward.semantics_version {version!r}; "
            f"expected {REWARD_V2_VERSION!r}"
        )

    status = str(reward.get("calibration_status", ""))
    if status not in {"diagnostic", "locked"}:
        raise ValueError(
            "reward.calibration_status must be diagnostic or locked for Reward V2"
        )
    calibration_report_sha256 = reward.get("calibration_report_sha256")
    if status == "locked":
        value = str(calibration_report_sha256 or "").lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(
                "locked Reward V2 requires reward.calibration_report_sha256 "
                "as a 64-character lowercase SHA256"
            )
        calibration_report_sha256 = value

    time_cfg = config.get("time", {})
    if (
        float(time_cfg.get("delta_t", 0.0)) != 1.0
        or int(time_cfg.get("episode_length", 0)) != 24
        or str(time_cfg.get("episode_sampling", "")) != "natural_day"
    ):
        raise ValueError(
            "Reward V2 requires complete 24-hour natural days with delta_t=1.0"
        )

    cost = config.get("cost", {})
    if not isinstance(cost, Mapping):
        raise ValueError("cost must be a mapping")
    ev = _ev_config(config)

    curtail = _resolve_alias(
        new_value=cost.get("curtailment_price_per_kwh"),
        old_value=cost.get("renewable_curtailment_cost_per_kwh"),
        new_name="cost.curtailment_price_per_kwh",
        old_name="cost.renewable_curtailment_cost_per_kwh",
    )
    ev_shortfall = _resolve_alias(
        new_value=cost.get("ev_shortfall_price_per_kwh"),
        old_value=ev.get("departure_penalty_per_kwh"),
        new_name="cost.ev_shortfall_price_per_kwh",
        old_name="mg3.dr.departure_penalty_per_kwh",
    )

    resolved = {
        "semantics_version": REWARD_V2_VERSION,
        "calibration_status": status,
        "calibration_report_sha256": calibration_report_sha256,
        "curtailment_price_per_kwh": float(curtail),
        "carbon_price_per_kg": _finite_nonnegative(
            cost.get("carbon_price_per_kg"), "cost.carbon_price_per_kg"
        ),
        "peak_price_per_kw": _finite_nonnegative(
            cost.get("peak_price_per_kw"), "cost.peak_price_per_kw"
        ),
        "mill_unserved_price_per_t": _finite_nonnegative(
            cost.get("mill_unserved_price_per_t"),
            "cost.mill_unserved_price_per_t",
        ),
        "comfort_price_per_degree_hour": _finite_nonnegative(
            cost.get("comfort_price_per_degree_hour"),
            "cost.comfort_price_per_degree_hour",
        ),
        "ev_shortfall_price_per_kwh": float(ev_shortfall),
        "unmet_load_cost_per_kwh": _finite_nonnegative(
            cost.get("unmet_load_cost_per_kwh"),
            "cost.unmet_load_cost_per_kwh",
        ),
        "reward_scale": _positive(cost.get("reward_scale"), "cost.reward_scale"),
        "dispatch_semantics": DISPATCH_V2_VERSION,
        "dr_projection_semantics": DR_PROJECTION_V2_VERSION,
        "hvac_control_semantics": HVAC_V2_VERSION,
        "ev_control_semantics": EV_V2_VERSION,
        "observation_semantics_version": OBSERVATION_V2_VERSION,
        "reward_discount_semantics": REWARD_DISCOUNT_V2_VERSION,
        "comfort_metric_semantics": COMFORT_METRIC_V2_VERSION,
        "carbon_information_semantics": CARBON_INFO_V2_VERSION,
        "settlement_value_semantics": SETTLEMENT_VALUE_V2_VERSION,
    }
    fixed = reward.get("fixed_carbon_factor_kg_per_kwh")
    if fixed is not None:
        resolved["fixed_carbon_factor_kg_per_kwh"] = _finite_nonnegative(
            fixed, "reward.fixed_carbon_factor_kg_per_kwh"
        )
    return resolved


def resolve_carbon_factors(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    """Resolve the public current grid carbon factor for every row."""
    column = "grid_carbon_factor_kg_per_kwh"
    if column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"{column} must contain finite non-negative values")
        return values
    reward = config.get("reward", {})
    fixed = reward.get("fixed_carbon_factor_kg_per_kwh") if isinstance(reward, Mapping) else None
    if fixed is None:
        raise ValueError(
            f"{column} is required unless reward.fixed_carbon_factor_kg_per_kwh is explicit"
        )
    factor = _finite_nonnegative(
        fixed, "reward.fixed_carbon_factor_kg_per_kwh"
    )
    return np.full(len(frame), factor, dtype=np.float64)


def calculate_reward_v2(
    *,
    raw: Mapping[str, float],
    rates: Mapping[str, float],
) -> dict[str, float]:
    """Pure Reward V2 cost calculation."""
    missing = [name for name in _RAW_KEYS if name not in raw]
    if missing:
        raise ValueError(f"Reward V2 raw components missing: {missing}")
    values = {
        name: _finite_nonnegative(raw[name], f"raw.{name}") for name in _RAW_KEYS
    }
    scale = _positive(rates.get("reward_scale"), "reward_scale")
    costs = {
        "cost_grid": values["grid_import_cost"],
        "cost_battery": values["battery_degradation_cost"],
        "cost_curtail": values["curtailment_kwh"]
        * _finite_nonnegative(
            rates.get("curtailment_price_per_kwh"),
            "curtailment_price_per_kwh",
        ),
        "cost_carbon": values["carbon_kg"]
        * _finite_nonnegative(rates.get("carbon_price_per_kg"), "carbon_price_per_kg"),
        "cost_peak": values["delta_peak_kw"]
        * _finite_nonnegative(rates.get("peak_price_per_kw"), "peak_price_per_kw"),
        "cost_mill": values["mill_unserved_t"]
        * _finite_nonnegative(
            rates.get("mill_unserved_price_per_t"), "mill_unserved_price_per_t"
        ),
        "cost_comfort": values["comfort_degree_hours"]
        * _finite_nonnegative(
            rates.get("comfort_price_per_degree_hour"),
            "comfort_price_per_degree_hour",
        ),
        "cost_ev": values["ev_departure_shortfall_kwh"]
        * _finite_nonnegative(
            rates.get("ev_shortfall_price_per_kwh"),
            "ev_shortfall_price_per_kwh",
        ),
        "cost_unmet": values["unmet_kwh"]
        * _finite_nonnegative(
            rates.get("unmet_load_cost_per_kwh"), "unmet_load_cost_per_kwh"
        ),
    }
    total = float(sum(costs.values()))
    return {
        **{key: float(value) for key, value in costs.items()},
        "generalized_system_cost": total,
        "team_reward": -total / scale,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_contract_hash(contract: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _jsonable(contract), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_reward_v2_contract(
    config: Mapping[str, Any],
    observation_schemas: Mapping[str, Sequence[str]],
    state_schema: Sequence[str],
    normalizers: Mapping[str, Any],
    *,
    algorithm_slug: str | None = None,
    actor_forecast_context_version: str | None = None,
    peer_orders: Mapping[str, Sequence[str]] | None = None,
    mask_semantics: str | None = None,
    gamma: float | None = None,
    action_semantics: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_reward_v2_config(config)
    if resolved is None:
        raise ValueError("Reward V2 contract requested for a legacy configuration")
    contract = {
        "reward": deepcopy(resolved),
        "action_design": str(config.get("action_design", "")),
        "dispatch_priority_mode": str(
            config.get("dispatch", {}).get("priority_mode", "")
        ),
        "observation_schemas": {
            str(agent): list(schema)
            for agent, schema in sorted(observation_schemas.items())
        },
        "state_schema": list(state_schema),
        "normalizers": _jsonable(normalizers),
        "algorithm_slug": algorithm_slug,
        "actor_forecast_context_version": actor_forecast_context_version,
        "peer_orders": {
            str(agent): list(order)
            for agent, order in sorted((peer_orders or {}).items())
        },
        "mask_semantics": mask_semantics,
        "gamma": None if gamma is None else float(gamma),
        "action_semantics": action_semantics,
    }
    contract["signature"] = stable_contract_hash(contract)
    return contract
