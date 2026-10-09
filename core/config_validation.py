from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from core.reward_v2 import resolve_reward_v2_config
from components.hvac_thermal import thermal_discretization


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain [lower, upper]")
    low = _number(value[0], f"{name}[0]")
    high = _number(value[1], f"{name}[1]")
    if low >= high:
        raise ValueError(f"{name} lower bound must be below upper bound")
    return low, high


def _validate_dr(mg_id: str, dr: Mapping[str, Any]) -> None:
    kind = str(dr.get("type", ""))
    if kind == "industrial_mill":
        for key in ("rated_power_kw", "rated_production_tph", "downstream_consumption_tph",
                    "silo_capacity_t", "silo_min_t", "silo_max_t", "silo_initial_t"):
            _number(dr.get(key), f"{mg_id}.dr.{key}", 0.0)
        _number(dr.get("terminal_tolerance_t"), f"{mg_id}.dr.terminal_tolerance_t", 0.0)
        minimum = float(dr["silo_min_t"])
        maximum = float(dr["silo_max_t"])
        initial = float(dr["silo_initial_t"])
        capacity = float(dr["silo_capacity_t"])
        if not 0.0 <= minimum < initial < maximum <= capacity:
            raise ValueError(f"{mg_id}: invalid raw-meal silo bounds")
        if min(float(dr["rated_power_kw"]), float(dr["rated_production_tph"]),
               float(dr["downstream_consumption_tph"])) <= 0.0:
            raise ValueError(f"{mg_id}: mill power and production rates must be positive")
        return

    if kind == "commercial_hvac":
        thermal_discretization(dr)
        for key in (
            "r_out_air_c_per_kw", "r_air_mass_c_per_kw", "c_air_kwh_per_c",
            "c_mass_kwh_per_c", "heating_max_kw", "cooling_max_kw",
            "cop_heat", "cop_cool", "terminal_air_tolerance_c",
            "terminal_mass_tolerance_c",
        ):
            _number(dr.get(key), f"{mg_id}.dr.{key}", 1e-12)
        _number(dr.get("solar_gain_kw_per_wm2", 0.025),
                f"{mg_id}.dr.solar_gain_kw_per_wm2", 0.0)
        for key in ("solar_air_fraction", "internal_gain_occupied_fraction",
                    "internal_gain_unoccupied_fraction"):
            value = _number(dr.get(key), f"{mg_id}.dr.{key}", 0.0)
            if value > 1.0:
                raise ValueError(f"{mg_id}.dr.{key} must be <= 1")
        _pair(dr.get("occupied_comfort_c"), f"{mg_id}.dr.occupied_comfort_c")
        _pair(dr.get("unoccupied_comfort_c"), f"{mg_id}.dr.unoccupied_comfort_c")
        hours = dr.get("occupied_hours", [])
        if not isinstance(hours, (list, tuple)) or any(
            isinstance(hour, bool) or int(hour) != hour or not 0 <= int(hour) <= 23
            for hour in hours
        ):
            raise ValueError(f"{mg_id}.dr.occupied_hours must contain hours 0..23")
        for key in ("temperature_column", "ghi_column"):
            if not str(dr.get(key, "")).strip():
                raise ValueError(f"{mg_id}.dr.{key} is required")
        if abs(float(dr["heating_max_kw"]) - float(dr["cooling_max_kw"])) > 1e-9:
            raise ValueError(
                f"{mg_id}: first-version differentiable HVAC action requires symmetric electric limits"
            )
        return

    if kind == "ev_charging":
        for key in ("fleet_capacity_kwh", "max_charge_kw", "eta_charge"):
            _number(dr.get(key), f"{mg_id}.dr.{key}", 1e-12)
        initial = _number(dr.get("initial_soc"), f"{mg_id}.dr.initial_soc", 0.0)
        target = _number(dr.get("target_soc"), f"{mg_id}.dr.target_soc", 0.0)
        maximum = _number(dr.get("max_soc"), f"{mg_id}.dr.max_soc", 0.0)
        if not 0.0 <= initial < target <= maximum <= 1.0:
            raise ValueError(f"{mg_id}: invalid EV SOC bounds")
        if float(dr["eta_charge"]) > 1.0:
            raise ValueError(f"{mg_id}.dr.eta_charge must be <= 1")
        cohorts = dr.get("cohorts")
        if cohorts:
            if not isinstance(cohorts, (list, tuple)) or len(cohorts) != 3:
                raise ValueError(f"{mg_id}.dr.cohorts must contain exactly three groups")
            expected_ids = {"G1", "G2", "G3"}
            actual_ids = {str(group.get("group_id", "")) for group in cohorts}
            if actual_ids != expected_ids:
                raise ValueError(f"{mg_id}.dr.cohorts must use group ids G1, G2, G3")
            station_rating = 0.0
            fleet_capacity = 0.0
            for group in cohorts:
                gid = str(group["group_id"])
                count = int(_number(group.get("count"), f"{mg_id}.dr.{gid}.count", 1.0))
                if count <= 0 or float(count) != float(group.get("count")):
                    raise ValueError(f"{mg_id}.dr.{gid}.count must be a positive integer")
                cap = _number(group.get("capacity_per_vehicle_kwh"),
                              f"{mg_id}.dr.{gid}.capacity_per_vehicle_kwh", 1e-12)
                power = _number(group.get("max_charge_per_vehicle_kw"),
                                f"{mg_id}.dr.{gid}.max_charge_per_vehicle_kw", 1e-12)
                need = _pair(group.get("need_per_vehicle_kwh"),
                             f"{mg_id}.dr.{gid}.need_per_vehicle_kwh")
                if need[0] < 0.0 or need[1] > target * cap + 1e-9:
                    raise ValueError(f"{mg_id}.dr.{gid}: invalid EV energy-need range")
                arrivals = group.get("arrival_hours", [])
                arrival_p = group.get("arrival_probabilities", [])
                if not arrivals or len(arrivals) != len(arrival_p):
                    raise ValueError(f"{mg_id}.dr.{gid}: arrival distribution is invalid")
                if any(not 0 <= int(hour) <= 23 for hour in arrivals):
                    raise ValueError(f"{mg_id}.dr.{gid}: arrival hours must be within 0..23")
                if not np.isclose(sum(float(x) for x in arrival_p), 1.0, atol=1e-9):
                    raise ValueError(f"{mg_id}.dr.{gid}: arrival probabilities must sum to one")
                if "departure_hours" in group:
                    departures = group.get("departure_hours", [])
                    probabilities = group.get("departure_probabilities", [])
                    if not departures or len(departures) != len(probabilities):
                        raise ValueError(f"{mg_id}.dr.{gid}: departure distribution is invalid")
                    if any(not 1 <= int(hour) <= 24 for hour in departures):
                        raise ValueError(f"{mg_id}.dr.{gid}: departure hours must be within 1..24")
                    if not np.isclose(sum(float(x) for x in probabilities), 1.0, atol=1e-9):
                        raise ValueError(f"{mg_id}.dr.{gid}: departure probabilities must sum to one")
                else:
                    dwell = group.get("dwell_hours", [])
                    probabilities = group.get("dwell_probabilities", [])
                    if not dwell or len(dwell) != len(probabilities):
                        raise ValueError(f"{mg_id}.dr.{gid}: dwell distribution is invalid")
                    if any(int(hour) <= 0 for hour in dwell):
                        raise ValueError(f"{mg_id}.dr.{gid}: dwell hours must be positive")
                    if not np.isclose(sum(float(x) for x in probabilities), 1.0, atol=1e-9):
                        raise ValueError(f"{mg_id}.dr.{gid}: dwell probabilities must sum to one")
                fleet_capacity += count * cap
                station_rating += count * power
            if not np.isclose(fleet_capacity, float(dr["fleet_capacity_kwh"]), atol=1e-9):
                raise ValueError(f"{mg_id}: fleet_capacity_kwh must equal the sum of cohort capacities")
            if float(dr["max_charge_kw"]) + 1e-9 < station_rating:
                raise ValueError(f"{mg_id}: station max_charge_kw must cover cohort ratings")
            _number(dr.get("departure_penalty_per_kwh"),
                    f"{mg_id}.dr.departure_penalty_per_kwh", 0.0)
            _number(dr.get("scenario_seed", 42), f"{mg_id}.dr.scenario_seed", 0.0)
            return

        start = int(_number(dr.get("available_start_hour"), f"{mg_id}.dr.available_start_hour", 0.0))
        departure = int(_number(dr.get("departure_hour"), f"{mg_id}.dr.departure_hour", 0.0))
        if not 0 <= start < departure <= 24:
            raise ValueError(f"{mg_id}: EV availability window must satisfy 0 <= start < departure <= 24")
        return

    raise ValueError(f"{mg_id}: unsupported dr.type {kind!r}")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the cooperative environment and optional heterogeneous DR layer."""
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
    if p2p.get("pair_capacity_kw") is not None:
        _number(p2p["pair_capacity_kw"], "p2p.pair_capacity_kw", 1e-9)
    if not isinstance(p2p.get("allow_ess_to_ess", False), bool):
        raise ValueError("p2p.allow_ess_to_ess must be boolean")
    if float(p2p.get("loss_rate", 0.0)) != 0.0:
        raise ValueError("The energy pool currently requires loss_rate=0")
    if not isinstance(p2p.get("allow_grid_resale"), bool):
        raise ValueError("p2p.allow_grid_resale must be boolean")

    data_cfg = config.get("data", {})
    if "allow_temperature_fallback" in data_cfg and not isinstance(
        data_cfg["allow_temperature_fallback"], bool
    ):
        raise ValueError("data.allow_temperature_fallback must be boolean")

    sharing = config.get("ess_sharing", {})
    if sharing.get("control_mode", "single_dispatch") != "single_dispatch":
        raise ValueError("Only single_dispatch ESS control is supported")
    if sharing.get("observation_mode", "cross_microgrid") not in {
        "local", "cross_microgrid", "peer_energy_summary"
    }:
        raise ValueError(
            "ess_sharing.observation_mode must be local, cross_microgrid, or peer_energy_summary"
        )
    if not isinstance(sharing.get("ess_p2p_enabled", True), bool):
        raise ValueError("ess_sharing.ess_p2p_enabled must be boolean")
    if sharing.get("offer_projection", "peer_demand_cap") != "peer_demand_cap":
        raise ValueError("single_dispatch requires peer_demand_cap projection")

    dispatch_mode = str(
        config.get("dispatch", {}).get("priority_mode", "legacy_local_first")
    )
    if dispatch_mode not in {"legacy_local_first", "renewable_load_first"}:
        raise ValueError(
            "dispatch.priority_mode must be legacy_local_first or renewable_load_first"
        )
    if dispatch_mode == "renewable_load_first":
        if config["p2p"].get("pair_capacity_kw") is not None:
            raise ValueError(
                "renewable_load_first currently requires p2p.pair_capacity_kw=null"
            )
        if bool(config["p2p"].get("allow_ess_to_ess", False)):
            raise ValueError(
                "renewable_load_first currently requires p2p.allow_ess_to_ess=false"
            )
        if bool(config["p2p"].get("allow_grid_resale", False)):
            raise ValueError(
                "renewable_load_first currently requires p2p.allow_grid_resale=false"
            )
        if any("tariff" in mg for mg in config.get("microgrids", [])):
            raise ValueError(
                "renewable_load_first currently requires one shared external tariff; "
                "per-microgrid tariff overrides are unsupported"
            )

    action_design = str(config.get("action_design", "single_dispatch"))
    if action_design not in {"single_dispatch", "single_dispatch_dr"}:
        raise ValueError("action_design must be single_dispatch or single_dispatch_dr")

    terminal = config["terminal_soc"]
    if terminal.get("mode") not in {"none", "penalty", "hard"}:
        raise ValueError("terminal_soc.mode must be none, penalty or hard")
    _number(terminal.get("tolerance"), "terminal_soc.tolerance", 0.0)
    if action_design == "single_dispatch_dr" and terminal.get("mode") == "hard":
        raise ValueError("single_dispatch_dr supports none or penalty terminal semantics")

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
    for name in ("load", "ghi", "wind_speed", "price", "temperature"):
        _number(sigma.get(name, 0.0), f"observation.forecast_sigma_1h.{name}", 0.0)
        if name in maximum:
            _number(maximum[name], f"observation.forecast_error_max.{name}", 0.0)

    delta_t = _number(config["time"].get("delta_t", 1.0), "time.delta_t", 1e-9)
    ids: list[str] = []
    dr_types: dict[str, str] = {}
    for index, mg in enumerate(config["microgrids"]):
        mg_id = str(mg.get("id", ""))
        if not mg_id:
            raise ValueError(f"microgrids[{index}].id is required")
        ids.append(mg_id)
        shape = mg.get("load_profile", {}).get("hourly_shape", [])
        if len(shape) != 24:
            raise ValueError(f"{mg_id}: hourly_shape must contain 24 values")
        _number(mg.get("load_profile", {}).get("target_mean_kw"),
                f"{mg_id}.load_profile.target_mean_kw", 1e-9)
        shift_hours = _number(
            mg.get("renewables", {}).get("weather_time_shift_hours", 0.0),
            f"{mg_id}.renewables.weather_time_shift_hours",
            0.0,
        )
        if not np.isclose(round(shift_hours / delta_t) * delta_t, shift_hours, atol=1e-9):
            raise ValueError(f"{mg_id}: weather_time_shift_hours must align with time.delta_t")
        battery = mg.get("battery", {})
        for key in ("capacity_kwh", "max_charge_kw", "max_discharge_kw", "eta_charge", "eta_discharge"):
            _number(battery.get(key), f"{mg_id}.battery.{key}", 1e-9)
        soc_min = _number(battery.get("soc_min"), f"{mg_id}.battery.soc_min", 0.0)
        soc_max = _number(battery.get("soc_max"), f"{mg_id}.battery.soc_max", 0.0)
        initial = _number(battery.get("initial_soc"), f"{mg_id}.battery.initial_soc", 0.0)
        if not 0.0 <= soc_min < soc_max <= 1.0 or not soc_min <= initial <= soc_max:
            raise ValueError(f"{mg_id}: invalid SOC bounds or initial SOC")
        for key in ("import_limit_kw", "export_limit_kw"):
            if key == "import_limit_kw" and mg.get("grid", {}).get(key) is None:
                continue
            _number(mg.get("grid", {}).get(key), f"{mg_id}.grid.{key}", 0.0)
        if (
            dispatch_mode == "renewable_load_first"
            and mg.get("grid", {}).get("import_limit_kw") is not None
        ):
            raise ValueError(
                f"{mg_id}: renewable_load_first currently requires unlimited grid import"
            )

        if action_design == "single_dispatch_dr":
            dr = mg.get("dr")
            if not isinstance(dr, Mapping):
                raise ValueError(f"{mg_id}: dr configuration is required")
            _validate_dr(mg_id, dr)
            dr_types[mg_id] = str(dr["type"])

    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError("Cooperative v1 requires exactly three unique microgrids")
    if action_design == "single_dispatch_dr":
        required_types = {
            "mg1": "industrial_mill",
            "mg2": "commercial_hvac",
            "mg3": "ev_charging",
        }
        if dr_types != required_types:
            raise ValueError(
                "single_dispatch_dr requires mg1=industrial_mill, "
                "mg2=commercial_hvac, mg3=ev_charging"
            )

    reward_v2 = resolve_reward_v2_config(config)
    if reward_v2 is not None:
        if dispatch_mode != "renewable_load_first":
            raise ValueError("Reward V2 requires dispatch.priority_mode=renewable_load_first")
        if terminal.get("mode") != "none":
            raise ValueError("Reward V2 requires terminal_soc.mode=none")
        for mg in config["microgrids"]:
            if float(mg.get("grid", {}).get("export_limit_kw", 0.0)) != 0.0:
                raise ValueError(
                    f"{mg['id']}: Reward V2 requires grid.export_limit_kw=0"
                )
        hvac = next(
            mg["dr"] for mg in config["microgrids"]
            if mg.get("dr", {}).get("type") == "commercial_hvac"
        )
        if str(hvac.get("control_semantics", "")) != "soft_comfort":
            raise ValueError(
                "Reward V2 requires mg2.dr.control_semantics=soft_comfort"
            )
        ev = next(
            mg["dr"] for mg in config["microgrids"]
            if mg.get("dr", {}).get("type") == "ev_charging"
        )
        if str(ev.get("control_semantics", "")) != "no_deadline_minimum":
            raise ValueError(
                "Reward V2 requires mg3.dr.control_semantics=no_deadline_minimum"
            )
