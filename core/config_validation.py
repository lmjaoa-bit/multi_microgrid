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
    if sharing.get("control_mode", "residual_auto") not in {
        "residual_auto", "fixed_full_offer", "explicit_offer",
    }:
        raise ValueError("ess_sharing.control_mode must be residual_auto, fixed_full_offer, or explicit_offer")
    if sharing.get("observation_mode", "local") not in {"local", "cross_microgrid"}:
        raise ValueError("ess_sharing.observation_mode must be local or cross_microgrid")
    if not isinstance(sharing.get("ess_p2p_enabled", True), bool):
        raise ValueError("ess_sharing.ess_p2p_enabled must be boolean")
    terminal = config["terminal_soc"]
    if terminal.get("mode") not in {"penalty", "hard"}:
        raise ValueError("terminal_soc.mode must be penalty or hard")
    _number(terminal.get("tolerance"), "terminal_soc.tolerance", 0.0)
    ids: list[str] = []
    for index, mg in enumerate(config["microgrids"]):
        mg_id = str(mg.get("id", ""))
        if not mg_id:
            raise ValueError(f"microgrids[{index}].id is required")
        ids.append(mg_id)
        shape = mg.get("load_profile", {}).get("hourly_shape", [])
        if len(shape) != 24:
            raise ValueError(f"{mg_id}: hourly_shape must contain 24 values")
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
