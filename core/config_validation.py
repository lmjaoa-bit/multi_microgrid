from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate only the cooperative-v1 schema; legacy market keys are rejected."""
    required = {"data", "time", "p2p", "terminal_soc", "cost", "microgrids"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if config["p2p"].get("mode") != "cooperative_energy_pool":
        raise ValueError("p2p.mode must be cooperative_energy_pool")
    rejected = {"market", "pricing", "bid", "ask", "clearing"} & set(config["p2p"])
    if rejected:
        raise ValueError(f"Unsupported P2P configuration keys: {sorted(rejected)}")
    p2p = config["p2p"]
    _number(p2p.get("pair_capacity_kw"), "p2p.pair_capacity_kw", 1e-9)
    if not isinstance(p2p.get("allow_grid_resale"), bool):
        raise ValueError("p2p.allow_grid_resale must be boolean")
    sharing = config.get("ess_sharing", {})
    if sharing.get("control_mode", "single_dispatch") != "single_dispatch":
        raise ValueError("Only single_dispatch control is supported")
    if sharing.get("observation_mode", "cross_microgrid") not in {"local", "cross_microgrid"}:
        raise ValueError("ess_sharing.observation_mode must be local or cross_microgrid")
    if not isinstance(sharing.get("ess_p2p_enabled", True), bool):
        raise ValueError("ess_sharing.ess_p2p_enabled must be boolean")
    if sharing.get("offer_projection", "peer_demand_cap") != "peer_demand_cap":
        raise ValueError("single_dispatch requires peer_demand_cap projection")
    terminal = config["terminal_soc"]
    if terminal.get("mode") not in {"none", "penalty", "hard"}:
        raise ValueError("terminal_soc.mode must be none, penalty or hard")
    _number(terminal.get("tolerance"), "terminal_soc.tolerance", 0.0)
    observation = config.get("observation", {})
    horizon = observation.get("future_horizon", 4)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("observation.future_horizon must be a positive integer")
    for key in ("noisy_forecast", "evaluation_forecast_noise", "mask_forecast_beyond_episode"):
        if key in observation and not isinstance(observation[key], bool):
            raise ValueError(f"observation.{key} must be boolean")
    correlation = _number(
        observation.get("forecast_temporal_correlation", 0.0),
        "observation.forecast_temporal_correlation",
        0.0,
    )
    if correlation >= 1.0:
        raise ValueError("observation.forecast_temporal_correlation must be < 1")
    sigma = observation.get("forecast_sigma_1h", {})
    maximum = observation.get("forecast_error_max", {})
    if not isinstance(sigma, Mapping):
        raise ValueError("observation.forecast_sigma_1h must be a mapping")
    if not isinstance(maximum, Mapping):
        raise ValueError("observation.forecast_error_max must be a mapping")
    for name in ("load", "ghi", "wind_speed", "price"):
        _number(sigma.get(name, 0.0), f"observation.forecast_sigma_1h.{name}", 0.0)
        if name in maximum:
            _number(maximum[name], f"observation.forecast_error_max.{name}", 0.0)
    ids: list[str] = []
    for index, mg in enumerate(config["microgrids"]):
        mg_id = str(mg.get("id", ""))
        if not mg_id:
            raise ValueError(f"microgrids[{index}].id is required")
        ids.append(mg_id)
        shape = mg.get("load_profile", {}).get("hourly_shape", [])
        if len(shape) != 24:
            raise ValueError(f"{mg_id}: hourly_shape must contain 24 values")
        shift_hours = _number(
            mg.get("renewables", {}).get("weather_time_shift_hours", 0.0),
            f"{mg_id}.renewables.weather_time_shift_hours",
            0.0,
        )
        delta_t = _number(config["time"].get("delta_t", 1.0), "time.delta_t", 1e-9)
        if not np.isclose(round(shift_hours / delta_t) * delta_t, shift_hours, atol=1e-9):
            raise ValueError(
                f"{mg_id}: weather_time_shift_hours must align with time.delta_t"
            )
        battery = mg.get("battery", {})
        for key in ("capacity_kwh", "max_charge_kw", "max_discharge_kw", "eta_charge", "eta_discharge"):
            _number(battery.get(key), f"{mg_id}.battery.{key}", 1e-9)
        soc_min = _number(battery.get("soc_min"), f"{mg_id}.battery.soc_min", 0.0)
        soc_max = _number(battery.get("soc_max"), f"{mg_id}.battery.soc_max", 0.0)
        initial = _number(battery.get("initial_soc"), f"{mg_id}.battery.initial_soc", 0.0)
        if not 0.0 <= soc_min < soc_max <= 1.0 or not soc_min <= initial <= soc_max:
            raise ValueError(f"{mg_id}: invalid SOC bounds or initial SOC")
        for key in ("import_limit_kw", "export_limit_kw"):
            _number(mg.get("grid", {}).get(key), f"{mg_id}.grid.{key}", 0.0)
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError("Cooperative v1 requires exactly three unique microgrids")
