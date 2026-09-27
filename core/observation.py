from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from components.ev_sessions import cohort_schema, cohort_values
from core.forecast import SharedForecastProvider
from core.profiles import HeterogeneousProfileModel
from core.tariffs import prices_from_row


class ObservationBuilder:
    """Decentralized Actor observation with local DR state and peer energy summaries."""

    def __init__(self, config: Mapping[str, Any], profiles: HeterogeneousProfileModel,
                 forecasts: SharedForecastProvider, delta_t: float) -> None:
        self.config = config
        self.agent_ids = [str(m["id"]) for m in config["microgrids"]]
        self.cfg_by_id = {str(m["id"]): m for m in config["microgrids"]}
        self.future_h = int(config.get("observation", {}).get("future_horizon", 4))
        self.include_forecast_validity = bool(
            config.get("observation", {}).get("mask_forecast_beyond_episode", False)
        )
        sharing_cfg = config.get("ess_sharing", {})
        self.control_mode = str(sharing_cfg.get("control_mode", "single_dispatch"))
        self.observation_mode = str(sharing_cfg.get("observation_mode", "cross_microgrid"))
        self.action_design = str(config.get("action_design", "single_dispatch"))
        self.dr_enabled = self.action_design == "single_dispatch_dr"
        self.profiles = profiles
        self.forecasts = forecasts
        self.delta_t = float(delta_t)
        self.norm = {mg: profiles.normalizers(mg, delta_t) for mg in self.agent_ids}
        self.initial_soc = {
            str(mg["id"]): float(mg["battery"]["initial_soc"])
            for mg in config["microgrids"]
        }
        self.episode_length = int(config.get("time", {}).get("episode_length", 24))

    def _dr_schema(self, mg_id: str) -> list[str]:
        kind = str(self.cfg_by_id[mg_id].get("dr", {}).get("type", ""))
        if kind == "industrial_mill":
            return [
                "silo_ratio", "silo_target_gap", "time_remaining",
                "required_avg_production", "production_pressure",
                "mill_feasible_min", "mill_feasible_max",
            ]
        if kind == "commercial_hvac":
            return [
                "t_air", "t_mass", "t_out", "comfort_lower_margin",
                "comfort_upper_margin", "occupied", "is_weekend",
                "terminal_air_gap", "terminal_mass_gap", "time_remaining",
                "thermal_recovery_pressure",
                "hvac_feasible_min", "hvac_feasible_max",
            ]
        if kind == "ev_charging":
            dr = self.cfg_by_id[mg_id].get("dr", {})
            if dr.get("cohorts"):
                return cohort_schema(dr)
            return [
                "ev_soc", "ev_soc_gap", "remaining_energy",
                "hours_to_departure", "ev_available",
                "ev_feasible_min", "ev_feasible_max", "deadline_pressure",
            ]
        raise ValueError(f"{mg_id}: unsupported DR type {kind!r}")

    def schema(self, mg_id: str) -> list[str]:
        # In the DR environment `load` means uncontrollable/base load. Future
        # flexible load is never forecast because it is a policy decision.
        names = ["hour_sin", "hour_cos", "load", "pv", "wind", "grid_buy_price",
                 "grid_sell_price", "soc"]
        for k in range(1, self.future_h + 1):
            names.extend([f"load_t+{k}", f"pv_t+{k}", f"wind_t+{k}", f"price_t+{k}"])
            if self.dr_enabled and str(self.cfg_by_id[mg_id].get("dr", {}).get("type")) == "commercial_hvac":
                names.append(f"temperature_t+{k}")
            if self.include_forecast_validity:
                names.append(f"forecast_valid_t+{k}")
        names.extend(["previous_p2p_net", "previous_grid_net", "previous_system_p2p"])
        if self.control_mode == "single_dispatch":
            names.extend([
                "available_charge_ratio", "available_discharge_ratio", "terminal_soc_gap",
                "time_to_terminal", "required_avg_power_to_target", "terminal_pressure",
                "previous_ess_p2p_accepted",
            ])
            if self.observation_mode == "cross_microgrid":
                for peer in self.agent_ids:
                    if peer != mg_id:
                        names.extend([f"{peer}_net_load", f"{peer}_soc"])
                for k in range(1, self.future_h + 1):
                    for peer in self.agent_ids:
                        if peer != mg_id:
                            names.append(f"{peer}_net_load_t+{k}")
            elif self.observation_mode == "peer_energy_summary":
                # Public coordination summary: expose what a peer can contribute/needs,
                # not its raw private physical state (for example peer SOC or DR state).
                for peer in self.agent_ids:
                    if peer == mg_id:
                        continue
                    names.extend([
                        f"{peer}_energy_balance",
                        f"{peer}_deficit",
                        f"{peer}_surplus",
                        f"{peer}_charge_flex",
                        f"{peer}_discharge_flex",
                        f"{peer}_previous_p2p_net",
                    ])
                for k in range(1, self.future_h + 1):
                    for peer in self.agent_ids:
                        if peer == mg_id:
                            continue
                        names.extend([
                            f"{peer}_energy_balance_t+{k}",
                            f"{peer}_deficit_t+{k}",
                            f"{peer}_surplus_t+{k}",
                        ])
        if self.control_mode == "single_dispatch":
            names.extend(f"feasible_{name}_ratio" for name in (
                "charge_min", "charge_max", "discharge_min", "discharge_max"))
        if self.dr_enabled:
            names.extend(self._dr_schema(mg_id))
        return names

    def obs_dim(self, mg_id: str) -> int:
        return len(self.schema(mg_id))

    def _dr_values(self, mg_id: str, state: Mapping[str, float]) -> list[float]:
        dr = self.cfg_by_id[mg_id]["dr"]
        kind = str(dr["type"])
        if kind == "industrial_mill":
            capacity = max(float(dr["silo_capacity_t"]), 1e-9)
            production = max(float(dr["rated_production_tph"]), 1e-9)
            return [
                float(state["silo_ratio"]),
                float(state["silo_target_gap_t"]) / capacity,
                float(state["time_remaining_h"]) / max(
                    float(self.episode_length) * self.delta_t, 1.0
                ),
                float(state["required_avg_production_tph"]) / production,
                float(state["production_pressure"]),
                float(state["mill_feasible_min"]),
                float(state["mill_feasible_max"]),
            ]
        if kind == "commercial_hvac":
            scale = max(float(dr["heating_max_kw"]), float(dr["cooling_max_kw"]), 1e-9)
            temp_scale = 40.0
            return [
                float(state["t_air_c"]) / 40.0,
                float(state["t_mass_c"]) / 40.0,
                float(state["t_out_c"]) / 40.0,
                float(state["comfort_lower_margin_c"]) / 15.0,
                float(state["comfort_upper_margin_c"]) / 15.0,
                float(state["occupied"]),
                float(state["is_weekend"]),
                float(state["terminal_air_gap_c"]) / temp_scale,
                float(state["terminal_mass_gap_c"]) / temp_scale,
                float(state["time_remaining_h"]) / max(
                    float(self.episode_length) * self.delta_t, 1.0
                ),
                float(state["thermal_recovery_pressure"]),
                float(state["hvac_feasible_min_kw"]) / scale,
                float(state["hvac_feasible_max_kw"]) / scale,
            ]
        if kind == "ev_charging":
            if dr.get("cohorts"):
                return cohort_values(dr, state)
            capacity = max(float(dr["fleet_capacity_kwh"]), 1e-9)
            power = max(float(dr["max_charge_kw"]), 1e-9)
            return [
                float(state["ev_soc"]),
                float(state["ev_soc_gap"]),
                float(state["ev_remaining_battery_kwh"]) / capacity,
                float(state["ev_hours_to_departure"]) / max(
                    float(self.episode_length) * self.delta_t, 1.0
                ),
                float(state["ev_available"]),
                float(state["ev_feasible_min_kw"]) / power,
                float(state["ev_feasible_max_kw"]) / power,
                float(state["deadline_pressure"]),
            ]
        raise ValueError(f"{mg_id}: unsupported DR type {kind!r}")

    def build(self, mg_id: str, row: pd.Series, t: int, battery_soc: float,
              history: Mapping[str, Any],
              exogenous: Mapping[str, float] | None = None,
              soc_by_microgrid: Mapping[str, float] | None = None,
              battery_flex_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
              dr_state_by_microgrid: Mapping[str, Mapping[str, float]] | None = None) -> np.ndarray:
        n = self.norm[mg_id]
        ex = dict(exogenous) if exogenous is not None else self.profiles.exogenous(mg_id, row)
        hour = int(pd.Timestamp(row["timestamp"]).hour)
        buy_price, sell_price = prices_from_row(row, mg_id)
        system_norm = max(1.0, sum(self.norm[x]["trade"] for x in self.agent_ids))
        values = [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            ex["load_kw"] / n["load"],
            ex["pv_kw"] / n["pv"],
            ex["wind_kw"] / n["wind"],
            buy_price / n["price"],
            sell_price / n["price"],
            float(battery_soc),
        ]
        for k in range(1, self.future_h + 1):
            f = self.forecasts.combined(mg_id, t, k)
            values.extend([
                f["load_kw"] / n["load"],
                f["pv_kw"] / n["pv"],
                f["wind_kw"] / n["wind"],
                f["grid_price_per_kwh"] / n["price"],
            ])
            if self.dr_enabled and str(self.cfg_by_id[mg_id].get("dr", {}).get("type")) == "commercial_hvac":
                values.append(float(f["temperature_c"]) / 40.0)
            if self.include_forecast_validity:
                values.append(self.forecasts.valid(t, k))
        previous = history["by_microgrid"][mg_id]
        values.extend([
            previous["p2p_net_kwh"] / n["trade"],
            previous["grid_net_kwh"] / n["trade"],
            history["total_p2p_kwh"] / system_norm,
        ])
        if self.control_mode == "single_dispatch":
            flex = (battery_flex_by_microgrid or {}).get(mg_id, {})
            values.extend([
                float(flex.get("available_charge_ratio", 0.0)),
                float(flex.get("available_discharge_ratio", 0.0)),
                float(flex.get("terminal_soc_gap", float(battery_soc) - self.initial_soc[mg_id])),
                float(flex.get("time_to_terminal_h", 0.0)) / max(
                    float(self.episode_length) * self.delta_t, 1.0
                ),
                float(flex.get("required_avg_power_to_target", 0.0)),
                float(flex.get("terminal_pressure", 0.0)),
                previous["ess_p2p_accepted_kwh"] / n["trade"],
            ])
            if self.observation_mode == "cross_microgrid":
                peer_soc = soc_by_microgrid or {}
                for peer in self.agent_ids:
                    if peer == mg_id:
                        continue
                    peer_ex = self.profiles.exogenous(peer, row)
                    values.extend([
                        # Legacy cross-microgrid mode retained for reproducibility.
                        (peer_ex["load_kw"] - peer_ex["renewable_kw"]) / self.norm[peer]["power"],
                        float(peer_soc.get(peer, self.initial_soc[peer])),
                    ])
                for k in range(1, self.future_h + 1):
                    for peer in self.agent_ids:
                        if peer == mg_id:
                            continue
                        forecast = self.forecasts.combined(peer, t, k)
                        peer_net = forecast["load_kw"] - forecast["pv_kw"] - forecast["wind_kw"]
                        values.append(peer_net / self.norm[peer]["power"])
            elif self.observation_mode == "peer_energy_summary":
                peer_flex = battery_flex_by_microgrid or {}
                for peer in self.agent_ids:
                    if peer == mg_id:
                        continue
                    peer_ex = self.profiles.exogenous(peer, row)
                    scale = max(float(self.norm[peer]["power"]), 1e-9)
                    balance_kw = float(peer_ex["renewable_kw"] - peer_ex["load_kw"])
                    peer_previous = history["by_microgrid"][peer]
                    flex_peer = peer_flex.get(peer, {})
                    values.extend([
                        balance_kw / scale,
                        max(-balance_kw, 0.0) / scale,
                        max(balance_kw, 0.0) / scale,
                        float(flex_peer.get("available_charge_ratio", 0.0)),
                        float(flex_peer.get("available_discharge_ratio", 0.0)),
                        float(peer_previous["p2p_net_kwh"]) / self.norm[peer]["trade"],
                    ])
                for k in range(1, self.future_h + 1):
                    for peer in self.agent_ids:
                        if peer == mg_id:
                            continue
                        forecast = self.forecasts.combined(peer, t, k)
                        scale = max(float(self.norm[peer]["power"]), 1e-9)
                        renewable_kw = float(forecast["pv_kw"] + forecast["wind_kw"])
                        balance_kw = renewable_kw - float(forecast["load_kw"])
                        values.extend([
                            balance_kw / scale,
                            max(-balance_kw, 0.0) / scale,
                            max(balance_kw, 0.0) / scale,
                        ])
        if self.control_mode == "single_dispatch":
            flex = (battery_flex_by_microgrid or {}).get(mg_id, {})
            values.extend(float(flex[f"feasible_{name}_ratio"]) for name in (
                "charge_min", "charge_max", "discharge_min", "discharge_max"))
        if self.dr_enabled:
            if dr_state_by_microgrid is None or mg_id not in dr_state_by_microgrid:
                raise RuntimeError(f"{mg_id}: DR state is required for DR observation")
            values.extend(self._dr_values(mg_id, dr_state_by_microgrid[mg_id]))
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.obs_dim(mg_id),) or not np.isfinite(result).all():
            raise RuntimeError(f"{mg_id}: invalid observation")
        return result
