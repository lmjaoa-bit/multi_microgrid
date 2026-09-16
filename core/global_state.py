from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.tariffs import has_heterogeneous_tariffs, prices_from_row


class GlobalStateBuilder:
    """Centralized critic state containing all current physical states."""

    def __init__(self, config: Mapping[str, Any], profiles: Any, forecasts: Any, delta_t: float) -> None:
        self.agent_ids = [str(m["id"]) for m in config["microgrids"]]
        self.profiles = profiles
        self.config = config
        self.cfg_by_id = {str(m["id"]): m for m in config["microgrids"]}
        self.norm = {mg: profiles.normalizers(mg, delta_t) for mg in self.agent_ids}
        self.price_norm = profiles.price_max
        self.forecasts = forecasts
        self.future_h = int(config.get("observation", {}).get("future_horizon", 4))
        self.include_forecast_validity = bool(
            config.get("observation", {}).get("mask_forecast_beyond_episode", False)
        )
        self.control_mode = str(config.get("ess_sharing", {}).get("control_mode", "single_dispatch"))
        self.action_design = str(config.get("action_design", "single_dispatch"))
        self.dr_enabled = self.action_design == "single_dispatch_dr"
        self.constraint_projection_mode = str(
            config.get("constraint_projection", {}).get("mode", "strong")
        )
        self.heterogeneous_tariffs = has_heterogeneous_tariffs(config)
        self.episode_length = int(config.get("time", {}).get("episode_length", 24))

    def _dr_schema(self, mg: str) -> list[str]:
        kind = str(self.cfg_by_id[mg].get("dr", {}).get("type", ""))
        if kind == "industrial_mill":
            return [
                f"{mg}_silo_ratio", f"{mg}_silo_target_gap",
                f"{mg}_time_to_terminal", f"{mg}_dr_min", f"{mg}_dr_max",
            ]
        if kind == "commercial_hvac":
            return [
                f"{mg}_t_air", f"{mg}_t_mass", f"{mg}_t_out",
                f"{mg}_comfort_lower_margin", f"{mg}_comfort_upper_margin",
                f"{mg}_occupied", f"{mg}_is_weekend",
                f"{mg}_dr_min", f"{mg}_dr_max",
            ]
        if kind == "ev_charging":
            return [
                f"{mg}_ev_soc", f"{mg}_ev_soc_gap", f"{mg}_ev_remaining_energy",
                f"{mg}_ev_hours_to_departure", f"{mg}_ev_available",
                f"{mg}_deadline_pressure", f"{mg}_dr_min", f"{mg}_dr_max",
            ]
        raise ValueError(f"{mg}: unsupported DR type {kind!r}")

    def schema(self) -> list[str]:
        names: list[str] = []
        for mg in self.agent_ids:
            # In the DR environment `load` is the uncontrollable/base load. The
            # joint action projector adds effective flexible load before ESS
            # local-first projection.
            names.extend([f"{mg}_load", f"{mg}_pv", f"{mg}_wind", f"{mg}_soc",
                          f"{mg}_previous_p2p_net"])
            if self.heterogeneous_tariffs:
                names.extend([f"{mg}_grid_buy_price", f"{mg}_grid_sell_price"])
            if self.dr_enabled:
                names.extend(self._dr_schema(mg))
        names.extend(["hour_sin", "hour_cos"])
        if not self.heterogeneous_tariffs:
            names.extend(["grid_buy_price", "grid_sell_price"])
        names.append("previous_system_p2p")
        if self.control_mode == "single_dispatch":
            for mg in self.agent_ids:
                names.extend([f"{mg}_available_charge", f"{mg}_available_discharge"])
            for mg in self.agent_ids:
                names.extend([
                    f"{mg}_previous_ess_p2p_offer",
                    f"{mg}_previous_ess_p2p_accepted",
                    f"{mg}_previous_ess_p2p_unmatched",
                ])
            for k in range(1, self.future_h + 1):
                for mg in self.agent_ids:
                    names.extend([f"{mg}_net_load_t+{k}", f"{mg}_price_t+{k}"])
                    if self.dr_enabled and str(self.cfg_by_id[mg].get("dr", {}).get("type")) == "commercial_hvac":
                        names.append(f"{mg}_temperature_t+{k}")
                if self.include_forecast_validity:
                    names.append(f"forecast_valid_t+{k}")
            names.extend([
                "previous_system_buy_demand", "previous_system_sell_offer",
                "previous_clearing_ratio",
            ])
        if self.control_mode == "single_dispatch":
            for mg in self.agent_ids:
                names.extend(f"{mg}_feasible_{name}" for name in (
                    "charge_min", "charge_max", "discharge_min", "discharge_max"))
        return names

    @property
    def dim(self) -> int:
        return len(self.schema())

    def _dr_values(self, mg: str, state: Mapping[str, float]) -> list[float]:
        dr = self.cfg_by_id[mg]["dr"]
        kind = str(dr["type"])
        if kind == "industrial_mill":
            capacity = max(float(dr["silo_capacity_t"]), 1e-9)
            return [
                float(state["silo_ratio"]),
                float(state["silo_target_gap_t"]) / capacity,
                float(state["time_to_terminal_h"]) / max(float(self.episode_length), 1.0),
                float(state["mill_feasible_min"]),
                float(state["mill_feasible_max"]),
            ]
        if kind == "commercial_hvac":
            scale = max(float(dr["heating_max_kw"]), float(dr["cooling_max_kw"]), 1e-9)
            temp_scale = 40.0
            margin_scale = 15.0
            return [
                float(state["t_air_c"]) / temp_scale,
                float(state["t_mass_c"]) / temp_scale,
                float(state["t_out_c"]) / temp_scale,
                float(state["comfort_lower_margin_c"]) / margin_scale,
                float(state["comfort_upper_margin_c"]) / margin_scale,
                float(state["occupied"]),
                float(state["is_weekend"]),
                float(state["hvac_feasible_min_kw"]) / scale,
                float(state["hvac_feasible_max_kw"]) / scale,
            ]
        if kind == "ev_charging":
            capacity = max(float(dr["fleet_capacity_kwh"]), 1e-9)
            power = max(float(dr["max_charge_kw"]), 1e-9)
            return [
                float(state["ev_soc"]),
                float(state["ev_soc_gap"]),
                float(state["ev_remaining_battery_kwh"]) / capacity,
                float(state["ev_hours_to_departure"]) / max(float(self.episode_length), 1.0),
                float(state["ev_available"]),
                float(state["deadline_pressure"]),
                float(state["ev_feasible_min_kw"]) / power,
                float(state["ev_feasible_max_kw"]) / power,
            ]
        raise ValueError(f"{mg}: unsupported DR type {kind!r}")

    def build(self, row: pd.Series, t: int, cores: Mapping[str, Any], history: Mapping[str, Any],
              exogenous_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
              dr_state_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
              dtype: Any = np.float32) -> np.ndarray:
        values: list[float] = []
        system_norm = max(1.0, sum(self.norm[x]["trade"] for x in self.agent_ids))
        for mg in self.agent_ids:
            ex = (dict(exogenous_by_microgrid[mg]) if exogenous_by_microgrid is not None
                  else cores[mg].exogenous(row))
            n = self.norm[mg]
            previous = history["by_microgrid"][mg]
            values.extend([ex["load_kw"] / n["load"], ex["pv_kw"] / n["pv"],
                           ex["wind_kw"] / n["wind"], float(cores[mg].battery.soc),
                           previous["p2p_net_kwh"] / n["trade"]])
            if self.heterogeneous_tariffs:
                buy_price, sell_price = prices_from_row(row, mg)
                values.extend([buy_price / n["price"], sell_price / n["price"]])
            if self.dr_enabled:
                if dr_state_by_microgrid is None or mg not in dr_state_by_microgrid:
                    raise RuntimeError("DR state is required for the centralized DR environment")
                values.extend(self._dr_values(mg, dr_state_by_microgrid[mg]))
        hour = int(pd.Timestamp(row["timestamp"]).hour)
        values.extend([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])
        if not self.heterogeneous_tariffs:
            values.extend([
                float(row["buy_price_per_kwh"]) / self.price_norm,
                float(row["sell_price_per_kwh"]) / self.price_norm,
            ])
        values.append(history["total_p2p_kwh"] / system_norm)
        if self.control_mode == "single_dispatch":
            for mg in self.agent_ids:
                battery = cores[mg].battery
                values.extend([
                    battery.available_charge_kw() / max(1.0, battery.max_charge_kw),
                    battery.available_discharge_kw() / max(1.0, battery.max_discharge_kw),
                ])
            for mg in self.agent_ids:
                previous = history["by_microgrid"][mg]
                values.extend([
                    previous["ess_p2p_offer_kwh"] / self.norm[mg]["trade"],
                    previous["ess_p2p_accepted_kwh"] / self.norm[mg]["trade"],
                    previous["ess_p2p_unmatched_kwh"] / self.norm[mg]["trade"],
                ])
            for k in range(1, self.future_h + 1):
                for mg in self.agent_ids:
                    forecast = self.forecasts.combined(mg, t, k)
                    net_load = forecast["load_kw"] - forecast["pv_kw"] - forecast["wind_kw"]
                    values.extend([net_load / self.norm[mg]["power"],
                                   forecast["grid_price_per_kwh"] / self.norm[mg]["price"]])
                    if self.dr_enabled and str(self.cfg_by_id[mg].get("dr", {}).get("type")) == "commercial_hvac":
                        values.append(float(forecast["temperature_c"]) / 40.0)
                if self.include_forecast_validity:
                    values.append(self.forecasts.valid(t, k))
            values.extend([
                history["total_buy_demand_kwh"] / system_norm,
                history["total_sell_offer_kwh"] / system_norm,
                float(history["clearing_ratio"]),
            ])
        if self.control_mode == "single_dispatch":
            enforce_terminal = (
                self.constraint_projection_mode == "strong"
                and self.config["terminal_soc"]["mode"] != "none"
            )
            for mg in self.agent_ids:
                battery = cores[mg].battery
                floor, ceiling = battery.reachable_soc_envelope(
                    t, self.episode_length,
                    tolerance=float(self.config["terminal_soc"]["tolerance"]),
                    enforce_terminal=enforce_terminal)
                neutral = battery.map_action(0.0, floor, ceiling)
                values.extend([
                    neutral["charge_requested_kw"] / battery.max_charge_kw,
                    -neutral["feasible_min_kw"] / battery.max_charge_kw,
                    neutral["discharge_requested_kw"] / battery.max_discharge_kw,
                    neutral["feasible_max_kw"] / battery.max_discharge_kw,
                ])
        result = np.asarray(values, dtype=dtype)
        if result.shape != (self.dim,) or not np.isfinite(result).all():
            raise RuntimeError("Invalid centralized state")
        return result
