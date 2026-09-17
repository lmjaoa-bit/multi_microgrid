from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from components.controllable_load import (
    delivered_fraction as clamp_delivered_fraction,
    raw_to_unit_interval,
)


class EVChargingDR:
    """Aggregate EV charging with instantaneous availability and SOC limits."""

    def __init__(self, config: Mapping[str, Any], *, delta_t: float, episode_length: int) -> None:
        self.config = dict(config)
        self.delta_t = float(delta_t)
        self.episode_length = int(episode_length)
        self.capacity_kwh = float(config["fleet_capacity_kwh"])
        self.initial_soc = float(config["initial_soc"])
        self.target_soc = float(config["target_soc"])
        self.max_soc = float(config["max_soc"])
        self.max_charge_kw = float(config["max_charge_kw"])
        self.eta = float(config["eta_charge"])
        self.available_start_hour = int(config.get("available_start_hour", 0))
        self.departure_hour = int(config["departure_hour"])
        if not 0.0 <= self.initial_soc < self.target_soc <= self.max_soc <= 1.0:
            raise ValueError("Invalid EV SOC configuration")
        if min(self.capacity_kwh, self.max_charge_kw, self.eta, self.delta_t) <= 0.0:
            raise ValueError("EV capacity, power, efficiency, and timestep must be positive")
        if not 0 <= self.available_start_hour < self.departure_hour <= 24:
            raise ValueError("Invalid EV availability window")
        self.soc = self.initial_soc
        self._frame: pd.DataFrame | None = None
        self._base_load_kw: np.ndarray | None = None
        self.last_projection = False
        self.target_missed = False

    def reset(self, episode_frame: pd.DataFrame, base_load_kw: np.ndarray) -> None:
        if len(episode_frame) < self.episode_length:
            raise ValueError("EV reset requires a complete natural-day episode")
        self._frame = episode_frame.reset_index(drop=True)
        self._base_load_kw = np.asarray(base_load_kw, dtype=np.float64)
        if self._base_load_kw.shape[0] < self.episode_length:
            raise ValueError("EV reset requires one base-load value per episode hour")
        self.soc = self.initial_soc
        self.last_projection = False
        self.target_missed = False

    def _hour(self, t: int) -> int:
        if self._frame is not None and 0 <= int(t) < len(self._frame) and "timestamp" in self._frame:
            return int(pd.Timestamp(self._frame.iloc[int(t)]["timestamp"]).hour)
        return int(t) % 24

    def _remaining_battery_kwh(self) -> float:
        return max(0.0, (self.target_soc - self.soc) * self.capacity_kwh)

    def _remaining_steps(self, t: int) -> int:
        hour = self._hour(t)
        if hour < self.available_start_hour or hour >= self.departure_hour:
            return 0
        return self.departure_hour - hour

    def feasible_bounds(self, t: int) -> tuple[float, float]:
        if self._remaining_steps(t) <= 0:
            return (0.0, 0.0)
        headroom_kwh = max(0.0, (self.max_soc - self.soc) * self.capacity_kwh)
        if headroom_kwh <= 1e-10:
            return (0.0, 0.0)
        maximum = min(self.max_charge_kw, headroom_kwh / (self.eta * self.delta_t))
        return (0.0, float(maximum))

    def plan(self, raw_action: float, t: int) -> dict[str, Any]:
        proposed = raw_to_unit_interval(raw_action) * self.max_charge_kw
        low, high = self.feasible_bounds(t)
        effective = float(np.clip(proposed, low, high))
        projected = abs(effective - proposed) > 1e-9
        self.last_projection = projected
        next_soc = min(
            self.max_soc,
            self.soc + self.eta * effective * self.delta_t / self.capacity_kwh,
        )
        return {
            "kind": "ev_charging",
            "raw_action": float(np.clip(raw_action, -1.0, 1.0)),
            "requested_power_kw": proposed,
            "effective_power_kw": effective,
            "effective_normalized": effective / self.max_charge_kw,
            "flexible_load_kw": effective,
            "soc_before": self.soc,
            "soc_after": next_soc,
            "feasible_min_kw": low,
            "feasible_max_kw": high,
            "projection_delta_kw": effective - proposed,
            "projected": projected,
            "target_missed": self.target_missed,
        }

    def commit(
        self,
        plan: Mapping[str, Any],
        delivered_fraction: float = 1.0,
    ) -> dict[str, float]:
        served = clamp_delivered_fraction(delivered_fraction)
        power = float(plan["effective_power_kw"]) * served
        self.soc = min(
            self.max_soc,
            self.soc + self.eta * power * self.delta_t / self.capacity_kwh,
        )
        return {
            "ev_soc": self.soc,
            "served_charge_kw": power,
            "delivered_fraction": served,
        }

    def state(self, t: int) -> dict[str, float]:
        low, high = self.feasible_bounds(t)
        remaining = self._remaining_battery_kwh()
        steps = self._remaining_steps(t)
        hours = steps * self.delta_t
        pressure = (
            remaining / (self.eta * self.max_charge_kw * hours)
            if steps > 0 and remaining > 0.0
            else 0.0
        )
        available = float(steps > 0)
        return {
            "ev_soc": float(self.soc),
            "ev_soc_gap": float(max(self.target_soc - self.soc, 0.0)),
            "ev_remaining_battery_kwh": float(remaining),
            "ev_hours_to_departure": float(hours),
            "ev_available": available,
            "ev_feasible_min_kw": low,
            "ev_feasible_max_kw": high,
            "deadline_pressure": float(pressure),
            "target_missed": float(self.target_missed),
        }
