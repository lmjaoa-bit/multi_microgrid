from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from components.hvac_viability import (
    HVACForecastSnapshot,
    HVACViabilityResult,
    solve_hvac_viability,
)
from components.controllable_load import (
    clipped_raw,
    delivered_fraction as clamp_delivered_fraction,
)


class CommercialHVACDR:
    """Reversible heat-pump DR backed by a 2R2C building thermal model."""

    def __init__(self, config: Mapping[str, Any], *, delta_t: float, episode_length: int) -> None:
        self.config = dict(config)
        self.delta_t = float(delta_t)
        self.episode_length = int(episode_length)
        self.r_oa = float(config["r_out_air_c_per_kw"])
        self.r_am = float(config["r_air_mass_c_per_kw"])
        self.c_air = float(config["c_air_kwh_per_c"])
        self.c_mass = float(config["c_mass_kwh_per_c"])
        self.heat_max_kw = float(config["heating_max_kw"])
        self.cool_max_kw = float(config["cooling_max_kw"])
        self.cop_heat = float(config["cop_heat"])
        self.cop_cool = float(config["cop_cool"])
        self.temperature_column = str(config["temperature_column"])
        self.ghi_column = str(config["ghi_column"])
        self.occupied_hours = {int(x) for x in config.get("occupied_hours", range(7, 19))}
        self.occupied_comfort = tuple(float(x) for x in config["occupied_comfort_c"])
        self.unoccupied_comfort = tuple(float(x) for x in config["unoccupied_comfort_c"])
        self.terminal_air_tolerance = float(config["terminal_air_tolerance_c"])
        self.terminal_mass_tolerance = float(config["terminal_mass_tolerance_c"])
        self.terminal_mode = str(config.get("terminal_mode", "none"))
        self.solar_gain = float(config.get("solar_gain_kw_per_wm2", 0.025))
        self.solar_air_fraction = float(config.get("solar_air_fraction", 0.20))
        self.internal_occ_fraction = float(config.get("internal_gain_occupied_fraction", 0.50))
        self.internal_unocc_fraction = float(config.get("internal_gain_unoccupied_fraction", 0.20))
        if min(
            self.delta_t,
            self.r_oa,
            self.r_am,
            self.c_air,
            self.c_mass,
            self.heat_max_kw,
            self.cool_max_kw,
            self.cop_heat,
            self.cop_cool,
        ) <= 0.0:
            raise ValueError("HVAC thermal and device parameters must be positive")
        self._frame: pd.DataFrame | None = None
        self._base_load_kw: np.ndarray | None = None
        self.t_air = 23.0
        self.t_mass = 23.0
        self.initial_t_air = 23.0
        self.initial_t_mass = 23.0
        self.comfort_infeasible = False
        self.terminal_thermal_infeasible = False
        self.last_projection = False
        self._bounds_cache_key: tuple[int, float, float] | None = None
        self._bounds_cache_value: tuple[float, float] | None = None
        self.forecast_viability = bool(config.get("forecast_viability", False))
        self.control_semantics = str(config.get("control_semantics", "legacy_comfort_projection"))
        if self.control_semantics not in {"legacy_comfort_projection", "soft_comfort"}:
            raise ValueError(f"Unsupported HVAC control_semantics: {self.control_semantics!r}")
        if self.control_semantics == "soft_comfort":
            self.forecast_viability = False
        self._viability_result: HVACViabilityResult | None = None
        self._viability_t: int | None = None

    @property
    def q_min_kw(self) -> float:
        return -self.cool_max_kw * self.cop_cool

    @property
    def q_max_kw(self) -> float:
        return self.heat_max_kw * self.cop_heat

    def _invalidate_bounds_cache(self) -> None:
        self._bounds_cache_key = None
        self._bounds_cache_value = None
        self._viability_result = None
        self._viability_t = None

    def reset(self, episode_frame: pd.DataFrame, base_load_kw: np.ndarray) -> None:
        if len(episode_frame) < self.episode_length:
            raise ValueError("HVAC reset requires a complete natural-day episode")
        if self.temperature_column not in episode_frame or self.ghi_column not in episode_frame:
            raise ValueError("HVAC episode is missing local temperature or GHI")
        self._frame = episode_frame.reset_index(drop=True)
        self._base_load_kw = np.asarray(base_load_kw, dtype=np.float64)
        if self._base_load_kw.shape[0] < self.episode_length:
            raise ValueError("HVAC reset requires one base-load value per episode hour")
        t_out_0 = float(self._frame.iloc[0][self.temperature_column])
        self.initial_t_air = float(np.clip(23.0 + 0.20 * (t_out_0 - 23.0), 18.0, 28.0))
        self.initial_t_mass = float(np.clip(23.0 + 0.10 * (t_out_0 - 23.0), 18.0, 28.0))
        self.t_air = self.initial_t_air
        self.t_mass = self.initial_t_mass
        self.comfort_infeasible = False
        self.terminal_thermal_infeasible = False
        self.last_projection = False
        self._invalidate_bounds_cache()

    def _timestamp(self, t: int) -> pd.Timestamp:
        if self._frame is None:
            raise RuntimeError("HVAC must be reset before use")
        return pd.Timestamp(self._frame.iloc[int(t)]["timestamp"])

    def _occupied(self, t: int) -> bool:
        ts = self._timestamp(t)
        return ts.dayofweek < 5 and ts.hour in self.occupied_hours

    def _comfort_bounds(self, t: int) -> tuple[float, float]:
        return self.occupied_comfort if self._occupied(t) else self.unoccupied_comfort

    def _state_comfort_bounds(self, state_index: int) -> tuple[float, float]:
        """Comfort band applying to the thermal state at a clock-time boundary.

        Action ``t`` evolves the state from timestamp ``t`` to ``t+1``. Internal
        gains therefore use occupancy during interval ``t``, while the resulting
        air temperature must already satisfy the band in force at timestamp
        ``t+1``. The 24:00 terminal boundary is treated as unoccupied.
        """
        index = int(state_index)
        if 0 <= index < self.episode_length:
            return self._comfort_bounds(index)
        return self.unoccupied_comfort

    def _exogenous(self, t: int) -> tuple[float, float, float]:
        if self._frame is None or self._base_load_kw is None:
            raise RuntimeError("HVAC must be reset before use")
        row = self._frame.iloc[int(t)]
        t_out = float(row[self.temperature_column])
        ghi = max(0.0, float(row[self.ghi_column]))
        q_solar = self.solar_gain * ghi
        q_internal = float(self._base_load_kw[int(t)]) * (
            self.internal_occ_fraction if self._occupied(t) else self.internal_unocc_fraction
        )
        return t_out, q_solar, q_internal

    def _next_state(self, t_air: float, t_mass: float, q_hvac_kw: float, t: int) -> tuple[float, float]:
        t_out, q_solar, q_internal = self._exogenous(t)
        q_air = self.solar_air_fraction * q_solar
        q_mass = (1.0 - self.solar_air_fraction) * q_solar
        air_rate = (
            (t_out - t_air) / self.r_oa
            + (t_mass - t_air) / self.r_am
            + q_internal
            + q_air
            + q_hvac_kw
        ) / self.c_air
        mass_rate = (
            (t_air - t_mass) / self.r_am
            + q_mass
        ) / self.c_mass
        return (
            float(t_air + self.delta_t * air_rate),
            float(t_mass + self.delta_t * mass_rate),
        )

    def _q_to_signed_electric(self, q_hvac_kw: float) -> float:
        if q_hvac_kw >= 0.0:
            return float(q_hvac_kw / self.cop_heat)
        return float(q_hvac_kw / self.cop_cool)

    def _signed_electric_to_q(self, power_kw: float) -> float:
        return float(power_kw * (self.cop_heat if power_kw >= 0.0 else self.cop_cool))

    def _one_step_bounds_q(self, t: int) -> tuple[float, float]:
        base_air, _ = self._next_state(self.t_air, self.t_mass, 0.0, t)
        comfort_low, comfort_high = self._state_comfort_bounds(int(t) + 1)
        q_required_low = (comfort_low - base_air) * self.c_air / self.delta_t
        q_required_high = (comfort_high - base_air) * self.c_air / self.delta_t
        low = max(self.q_min_kw, q_required_low)
        high = min(self.q_max_kw, q_required_high)
        if low <= high + 1e-9:
            return float(low), float(high)
        self.comfort_infeasible = True
        if q_required_low > self.q_max_kw:
            return self.q_max_kw, self.q_max_kw
        if q_required_high < self.q_min_kw:
            return self.q_min_kw, self.q_min_kw
        closest = float(np.clip(0.5 * (low + high), self.q_min_kw, self.q_max_kw))
        return closest, closest


    def prepare_viability(
        self, t: int, snapshot: HVACForecastSnapshot
    ) -> HVACViabilityResult:
        if self.control_semantics == "soft_comfort":
            raise RuntimeError("HVAC soft_comfort semantics do not use forecast viability")
        if not self.forecast_viability:
            raise RuntimeError("HVAC forecast viability is not enabled")
        if self._viability_t == int(t) and self._viability_result is not None:
            if self._viability_result.snapshot_id == snapshot.snapshot_id:
                return self._viability_result
        result = solve_hvac_viability(
            self.t_air,
            self.t_mass,
            snapshot,
            config=self.config,
            delta_t=self.delta_t,
        )
        self._viability_t = int(t)
        self._viability_result = result
        self._bounds_cache_key = (
            int(t), round(float(self.t_air), 10), round(float(self.t_mass), 10)
        )
        self._bounds_cache_value = (
            float(result.min_signed_power_kw),
            float(result.max_signed_power_kw),
        )
        if not result.feasible:
            self.comfort_infeasible = True
        return result

    def feasible_bounds(self, t: int) -> tuple[float, float]:
        if not 0 <= int(t) < self.episode_length:
            return (0.0, 0.0)
        if self.control_semantics == "soft_comfort":
            return (-float(self.cool_max_kw), float(self.heat_max_kw))
        key = (int(t), round(float(self.t_air), 10), round(float(self.t_mass), 10))
        if self._bounds_cache_key == key and self._bounds_cache_value is not None:
            return self._bounds_cache_value

        if self.forecast_viability:
            if self._viability_t != int(t) or self._viability_result is None:
                raise RuntimeError(
                    "HVAC viability bounds were not prepared from the current forecast snapshot"
                )
            return (
                float(self._viability_result.min_signed_power_kw),
                float(self._viability_result.max_signed_power_kw),
            )

        q_low, q_high = self._one_step_bounds_q(t)
        result = (
            float(np.clip(
                self._q_to_signed_electric(q_low),
                -self.cool_max_kw,
                self.heat_max_kw,
            )),
            float(np.clip(
                self._q_to_signed_electric(q_high),
                -self.cool_max_kw,
                self.heat_max_kw,
            )),
        )
        self._bounds_cache_key = key
        self._bounds_cache_value = result
        return result

    def plan(self, raw_action: float, t: int) -> dict[str, Any]:
        raw = clipped_raw(raw_action)
        proposed = raw * (self.heat_max_kw if raw >= 0.0 else self.cool_max_kw)
        low, high = self.feasible_bounds(t)
        effective = float(np.clip(proposed, low, high))
        q_hvac = self._signed_electric_to_q(effective)
        next_air, next_mass = self._next_state(self.t_air, self.t_mass, q_hvac, t)
        projected = abs(effective - proposed) > 1e-9
        self.last_projection = projected
        return {
            "kind": "commercial_hvac",
            "t": int(t),
            "raw_action": raw,
            "requested_signed_power_kw": proposed,
            "effective_signed_power_kw": effective,
            "effective_normalized": effective / (
                self.heat_max_kw if effective >= 0.0 else self.cool_max_kw
            ),
            "flexible_load_kw": abs(effective),
            "thermal_power_kw": q_hvac,
            "heating_power_kw": max(effective, 0.0),
            "cooling_power_kw": max(-effective, 0.0),
            "t_air_before_c": self.t_air,
            "t_mass_before_c": self.t_mass,
            "t_air_after_c": next_air,
            "t_mass_after_c": next_mass,
            "feasible_min_kw": low,
            "feasible_max_kw": high,
            "projection_delta_kw": effective - proposed,
            "projected": projected,
            "comfort_infeasible": self.comfort_infeasible,
            "terminal_thermal_infeasible": self.terminal_thermal_infeasible,
        }

    def commit(
        self,
        plan: Mapping[str, Any],
        delivered_fraction: float = 1.0,
    ) -> dict[str, float]:
        served = clamp_delivered_fraction(delivered_fraction)
        signed_power = float(plan["effective_signed_power_kw"]) * served
        q_hvac = self._signed_electric_to_q(signed_power)
        t = int(plan["t"])
        self.t_air, self.t_mass = self._next_state(
            self.t_air, self.t_mass, q_hvac, t
        )
        self._invalidate_bounds_cache()
        if served < 1.0 - 1e-10:
            self.terminal_thermal_infeasible = True
        return {
            "t_air_c": self.t_air,
            "t_mass_c": self.t_mass,
            "served_hvac_kw": abs(signed_power),
            "delivered_fraction": served,
        }

    def state(self, t: int) -> dict[str, float]:
        low, high = self.feasible_bounds(t)
        remaining_hours = max(self.episode_length - int(t), 0) * self.delta_t
        if self.terminal_mode == "none":
            terminal_air_gap = 0.0
            terminal_mass_gap = 0.0
            required_electric_kw = 0.0
            recovery_pressure = 0.0
        else:
            terminal_air_gap = float(self.initial_t_air - self.t_air)
            terminal_mass_gap = float(self.initial_t_mass - self.t_mass)
            if remaining_hours > 0.0:
                required_thermal_kw = (
                    terminal_air_gap * self.c_air
                    + terminal_mass_gap * self.c_mass
                ) / remaining_hours
                if required_thermal_kw >= 0.0:
                    required_electric_kw = required_thermal_kw / self.cop_heat
                    recovery_pressure = abs(required_electric_kw) / self.heat_max_kw
                else:
                    required_electric_kw = required_thermal_kw / self.cop_cool
                    recovery_pressure = abs(required_electric_kw) / self.cool_max_kw
            else:
                required_electric_kw = 0.0
                recovery_pressure = 0.0
        if 0 <= int(t) < self.episode_length:
            t_out, _, _ = self._exogenous(t)
            comfort_low, comfort_high = self._comfort_bounds(t)
            occupied = float(self._occupied(t))
            weekend = float(self._timestamp(t).dayofweek >= 5)
        else:
            t_out = (
                float(self._frame.iloc[-1][self.temperature_column])
                if self._frame is not None else 0.0
            )
            comfort_low, comfort_high = self.unoccupied_comfort
            occupied = 0.0
            weekend = 0.0
        return {
            "t_air_c": float(self.t_air),
            "t_mass_c": float(self.t_mass),
            "t_out_c": float(t_out),
            "comfort_lower_margin_c": float(self.t_air - comfort_low),
            "comfort_upper_margin_c": float(comfort_high - self.t_air),
            "occupied": occupied,
            "is_weekend": weekend,
            "hvac_feasible_min_kw": low,
            "hvac_feasible_max_kw": high,
            "terminal_air_gap_c": terminal_air_gap,
            "terminal_mass_gap_c": terminal_mass_gap,
            "time_remaining_h": float(remaining_hours),
            "required_avg_recovery_power_kw": float(required_electric_kw),
            "thermal_recovery_pressure": float(recovery_pressure),
            "comfort_infeasible": float(self.comfort_infeasible),
            "terminal_thermal_infeasible": float(self.terminal_thermal_infeasible),
            "hvac_viability_feasible": (
                float(self._viability_result.feasible)
                if self._viability_result is not None else 1.0
            ),
            "hvac_viability_current_min_slack_c": (
                float(self._viability_result.current_min_slack_c)
                if self._viability_result is not None else 0.0
            ),
            "hvac_viability_future_min_slack_sum_c": (
                float(self._viability_result.future_min_slack_sum_c)
                if self._viability_result is not None else 0.0
            ),
            "hvac_viability_status": (
                "soft_comfort"
                if self.control_semantics == "soft_comfort"
                else self._viability_result.status
                if self._viability_result is not None else "legacy_one_step"
            ),
            "hvac_forecast_snapshot_id": (
                self._viability_result.snapshot_id
                if self._viability_result is not None else ""
            ),
        }
