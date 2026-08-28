from __future__ import annotations

import math
from typing import Any, Mapping


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _nonnegative(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return number


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    number = _finite(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer.")
    integer = int(number)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return integer


def _require_fixed(mapping: Mapping[str, Any], key: str, expected: Any, prefix: str) -> None:
    actual = mapping.get(key, expected)
    if actual != expected:
        raise ValueError(
            f"{prefix}.{key}={actual!r} is not supported; the current research baseline "
            f"requires {expected!r}."
        )


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the supported heterogeneous three-microgrid research configuration."""

    for section in ("data", "time", "p2p", "microgrids", "weather_to_power"):
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    data = config["data"]
    for key in ("train_path", "val_path", "test_path"):
        if not data.get(key):
            raise ValueError(f"data.{key} must be configured.")

    env = config.get("env", {})
    _require_fixed(env, "name", "MultiMicrogridEnv", "env")
    _integer(env.get("seed", 42), "env.seed")

    time_cfg = config["time"]
    delta_t = _positive(time_cfg.get("delta_t", 1.0), "time.delta_t")
    if not math.isclose(delta_t, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("time.delta_t must remain exactly 1.0 hour for this research baseline.")
    episode_length = _integer(time_cfg.get("episode_length", 24), "time.episode_length", 1)
    if episode_length != 24:
        raise ValueError("time.episode_length must remain 24 for this research baseline.")
    _require_fixed(time_cfg, "episode_sampling", "natural_day", "time")

    p2p = config["p2p"]
    _require_fixed(p2p, "market", "sealed_bid_double_auction", "p2p")
    _require_fixed(p2p, "pricing", "midpoint", "p2p")
    _require_fixed(p2p, "tie_breaking", "proportional", "p2p")
    _require_fixed(p2p, "loss_rate", 0.0, "p2p")
    _require_fixed(p2p, "transaction_fee_per_kwh", 0.0, "p2p")
    _require_fixed(p2p, "grid_priority", "after_p2p", "p2p")
    _nonnegative(p2p.get("min_trade_power_kw", 1.0), "p2p.min_trade_power_kw")
    if not isinstance(p2p.get("enabled", True), bool):
        raise ValueError("p2p.enabled must be boolean.")

    price = config.get("price", {})
    sell_ratio = _finite(price.get("sell_ratio", 0.5), "price.sell_ratio")
    if not 0.0 <= sell_ratio <= 1.0:
        raise ValueError("price.sell_ratio must lie inside [0, 1].")
    if not isinstance(price.get("override_buy_price_with_tou", False), bool):
        raise ValueError("price.override_buy_price_with_tou must be boolean.")
    _require_fixed(price, "override_buy_price_with_tou", False, "price")

    weather = config["weather_to_power"]
    _positive(weather.get("pv", {}).get("irradiance_reference_w_per_m2"), "weather_to_power.pv.irradiance_reference_w_per_m2")
    wind = weather.get("wind", {})
    cut_in = _nonnegative(wind.get("cut_in_mps"), "weather_to_power.wind.cut_in_mps")
    rated = _positive(wind.get("rated_mps"), "weather_to_power.wind.rated_mps")
    cut_out = _positive(wind.get("cut_out_mps"), "weather_to_power.wind.cut_out_mps")
    if not cut_in < rated < cut_out:
        raise ValueError("Wind speeds must satisfy cut_in < rated < cut_out.")

    observation = config.get("observation", {})
    _integer(observation.get("future_horizon", 4), "observation.future_horizon", 0)
    if not isinstance(observation.get("noisy_forecast", True), bool):
        raise ValueError("observation.noisy_forecast must be boolean.")
    for key in ("load", "ghi", "wind_speed", "price"):
        _nonnegative(
            observation.get("forecast_sigma_1h", {}).get(key, 0.0),
            f"observation.forecast_sigma_1h.{key}",
        )

    cost = config.get("cost", {})
    for key in (
        "carbon_price_per_kg",
        "unmet_load_cost_per_kwh",
        "controllable_dump_cost_per_kwh",
    ):
        _nonnegative(cost.get(key, 0.0), f"cost.{key}")
    _positive(cost.get("reward_scale", 100.0), "cost.reward_scale")

    microgrids = config["microgrids"]
    if not isinstance(microgrids, list) or len(microgrids) != 3:
        raise ValueError("This research baseline requires exactly 3 microgrids.")
    ids = [str(mg.get("id", "")) for mg in microgrids]
    if any(not mg_id for mg_id in ids):
        raise ValueError("Every microgrid must have a non-empty id.")
    if len(set(ids)) != len(ids):
        raise ValueError("Microgrid ids must be unique.")

    for mg in microgrids:
        mg_id = str(mg["id"])
        profile = mg.get("load_profile", {})
        _positive(profile.get("target_mean_kw"), f"microgrids.{mg_id}.load_profile.target_mean_kw")
        _positive(profile.get("weekend_multiplier", 1.0), f"microgrids.{mg_id}.load_profile.weekend_multiplier")
        shape = profile.get("hourly_shape", [])
        if not isinstance(shape, list) or len(shape) != 24:
            raise ValueError(f"microgrids.{mg_id}.load_profile.hourly_shape must contain 24 values.")
        for index, value in enumerate(shape):
            _positive(value, f"microgrids.{mg_id}.load_profile.hourly_shape[{index}]")
        clip = profile.get("variability_clip", [0.65, 1.45])
        if not isinstance(clip, list) or len(clip) != 2:
            raise ValueError(f"microgrids.{mg_id}.load_profile.variability_clip must contain two values.")
        low = _positive(clip[0], f"microgrids.{mg_id}.load_profile.variability_clip[0]")
        high = _positive(clip[1], f"microgrids.{mg_id}.load_profile.variability_clip[1]")
        if low > high:
            raise ValueError(f"microgrids.{mg_id}.load_profile variability lower bound exceeds upper bound.")

        renewables = mg.get("renewables", {})
        _nonnegative(renewables.get("pv_capacity_kw"), f"microgrids.{mg_id}.renewables.pv_capacity_kw")
        _nonnegative(renewables.get("wind_capacity_kw"), f"microgrids.{mg_id}.renewables.wind_capacity_kw")

        battery = mg.get("battery", {})
        _positive(battery.get("capacity_kwh"), f"microgrids.{mg_id}.battery.capacity_kwh")
        _positive(battery.get("max_charge_kw"), f"microgrids.{mg_id}.battery.max_charge_kw")
        _positive(battery.get("max_discharge_kw"), f"microgrids.{mg_id}.battery.max_discharge_kw")
        for key in ("eta_charge", "eta_discharge"):
            efficiency = _finite(battery.get(key), f"microgrids.{mg_id}.battery.{key}")
            if not 0.0 < efficiency <= 1.0:
                raise ValueError(f"microgrids.{mg_id}.battery.{key} must lie inside (0, 1].")
        soc_min = _finite(battery.get("soc_min"), f"microgrids.{mg_id}.battery.soc_min")
        soc_max = _finite(battery.get("soc_max"), f"microgrids.{mg_id}.battery.soc_max")
        initial_soc = _finite(battery.get("initial_soc"), f"microgrids.{mg_id}.battery.initial_soc")
        if not 0.0 <= soc_min <= initial_soc <= soc_max <= 1.0:
            raise ValueError(
                f"microgrids.{mg_id} battery SOC values must satisfy "
                "0 <= soc_min <= initial_soc <= soc_max <= 1."
            )
        _nonnegative(
            battery.get("degradation_cost_per_kwh", 0.0),
            f"microgrids.{mg_id}.battery.degradation_cost_per_kwh",
        )

        generator = mg.get("generator", {})
        max_power = _positive(generator.get("max_power_kw"), f"microgrids.{mg_id}.generator.max_power_kw")
        min_power = _positive(generator.get("min_power_kw"), f"microgrids.{mg_id}.generator.min_power_kw")
        if min_power > max_power:
            raise ValueError(f"microgrids.{mg_id} generator min power exceeds max power.")
        for key in (
            "fuel_price_per_kwh_hhv",
            "fuel_emission_factor_kg_per_kwh_hhv",
            "om_cost_per_kwh",
            "startup_cost",
        ):
            _nonnegative(generator.get(key, 0.0), f"microgrids.{mg_id}.generator.{key}")
        curve = generator.get("part_load_efficiency_curve", [])
        if not curve:
            raise ValueError(f"microgrids.{mg_id} generator efficiency curve cannot be empty.")
        fractions = []
        for index, point in enumerate(curve):
            fraction = _finite(
                point.get("load_fraction"),
                f"microgrids.{mg_id}.generator.curve[{index}].load_fraction",
            )
            efficiency = _finite(
                point.get("efficiency"),
                f"microgrids.{mg_id}.generator.curve[{index}].efficiency",
            )
            if not 0.0 <= fraction <= 1.0 or not 0.0 < efficiency <= 1.0:
                raise ValueError(f"microgrids.{mg_id} has an invalid generator efficiency point.")
            fractions.append(fraction)
        if len(set(fractions)) != len(fractions):
            raise ValueError(f"microgrids.{mg_id} generator load fractions must be unique.")

        grid = mg.get("grid", {})
        _nonnegative(grid.get("import_limit_kw"), f"microgrids.{mg_id}.grid.import_limit_kw")
        export_limit = _nonnegative(
            grid.get("export_limit_kw"), f"microgrids.{mg_id}.grid.export_limit_kw"
        )
        if not math.isclose(export_limit, 15.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"microgrids.{mg_id}.grid.export_limit_kw must remain 15.0 in this baseline."
            )
