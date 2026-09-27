from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from components.controllable_load import (
    delivered_fraction as clamp_delivered_fraction,
    raw_to_unit_interval,
)


class IndustrialMillDR:
    """Raw-meal mill constrained by its instantaneous silo inventory."""

    def __init__(self, config: Mapping[str, Any], *, delta_t: float, episode_length: int) -> None:
        self.config = dict(config)
        self.delta_t = float(delta_t)
        self.episode_length = int(episode_length)
        self.rated_power_kw = float(config["rated_power_kw"])
        self.production_tph = float(config["rated_production_tph"])
        self.downstream_tph = float(config["downstream_consumption_tph"])
        self.silo_capacity_t = float(config["silo_capacity_t"])
        self.silo_min_t = float(config["silo_min_t"])
        self.silo_max_t = float(config["silo_max_t"])
        self.initial_silo_t = float(config["silo_initial_t"])
        self.terminal_tolerance_t = float(config["terminal_tolerance_t"])
        self.terminal_mode = str(config.get("terminal_mode", "none"))
        if not 0.0 <= self.silo_min_t < self.initial_silo_t < self.silo_max_t <= self.silo_capacity_t:
            raise ValueError("Invalid raw-meal silo bounds")
        if min(self.rated_power_kw, self.production_tph, self.delta_t) <= 0.0:
            raise ValueError("Mill power, production, and timestep must be positive")
        self.silo_t = self.initial_silo_t
        self._frame: pd.DataFrame | None = None
        self._base_load_kw: np.ndarray | None = None
        self.last_projection = False
        self.terminal_infeasible = False

    def reset(self, episode_frame: pd.DataFrame, base_load_kw: np.ndarray) -> None:
        if len(episode_frame) < self.episode_length:
            raise ValueError("Mill reset requires a complete natural-day episode")
        self._frame = episode_frame.reset_index(drop=True)
        self._base_load_kw = np.asarray(base_load_kw, dtype=np.float64)
        if self._base_load_kw.shape[0] < self.episode_length:
            raise ValueError("Mill reset requires one base-load value per episode hour")
        self.silo_t = self.initial_silo_t
        self.last_projection = False
        self.terminal_infeasible = False

    def feasible_bounds(self, t: int) -> tuple[float, float]:
        if not 0 <= int(t) < self.episode_length:
            return (0.0, 0.0)
        production = self.production_tph * self.delta_t
        consumption = self.downstream_tph * self.delta_t
        low = (self.silo_min_t - self.silo_t + consumption) / production
        high = (self.silo_max_t - self.silo_t + consumption) / production
        low = max(0.0, low)
        high = min(1.0, high)
        if low > high + 1e-9:
            self.terminal_infeasible = True
            midpoint = float(np.clip((low + high) * 0.5, 0.0, 1.0))
            return (midpoint, midpoint)
        return (float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0)))

    def plan(self, raw_action: float, t: int) -> dict[str, Any]:
        requested = raw_to_unit_interval(raw_action)
        low, high = self.feasible_bounds(t)
        effective = float(np.clip(requested, low, high))
        next_silo = self.silo_t + (
            self.production_tph * effective - self.downstream_tph
        ) * self.delta_t
        projected = abs(effective - requested) > 1e-10
        self.last_projection = projected
        return {
            "kind": "industrial_mill",
            "raw_action": float(np.clip(raw_action, -1.0, 1.0)),
            "requested_duty": requested,
            "effective_duty": effective,
            "effective_normalized": effective,
            "flexible_load_kw": self.rated_power_kw * effective,
            "production_tph": self.production_tph * effective,
            "downstream_consumption_tph": self.downstream_tph,
            "silo_before_t": self.silo_t,
            "silo_after_t": next_silo,
            "feasible_min": low,
            "feasible_max": high,
            "projection_delta": effective - requested,
            "projected": projected,
            "terminal_infeasible": self.terminal_infeasible,
        }

    def commit(
        self,
        plan: Mapping[str, Any],
        delivered_fraction: float = 1.0,
    ) -> dict[str, float]:
        served = clamp_delivered_fraction(delivered_fraction)
        effective = float(plan["effective_duty"]) * served
        self.silo_t += (
            self.production_tph * effective - self.downstream_tph
        ) * self.delta_t
        if self.silo_t < self.silo_min_t - 1e-7 or self.silo_t > self.silo_max_t + 1e-7:
            self.terminal_infeasible = True
        self.silo_t = float(np.clip(self.silo_t, self.silo_min_t, self.silo_max_t))
        return {
            "silo_level_t": self.silo_t,
            "served_duty": effective,
            "delivered_fraction": served,
        }

    def state(self, t: int) -> dict[str, float]:
        low, high = self.feasible_bounds(t)
        remaining_hours = max(self.episode_length - int(t), 0) * self.delta_t
        if self.terminal_mode == "none":
            target_gap = 0.0
            required_production = self.downstream_tph
            production_pressure = required_production / self.production_tph
        else:
            target_gap = self.silo_t - self.initial_silo_t
            if remaining_hours > 0.0:
                required_production = (
                    self.downstream_tph
                    + (self.initial_silo_t - self.silo_t) / remaining_hours
                )
                production_pressure = required_production / self.production_tph
            else:
                required_production = 0.0
                production_pressure = 0.0
        return {
            "silo_level_t": float(self.silo_t),
            "silo_ratio": float(self.silo_t / self.silo_capacity_t),
            "silo_target_gap_t": float(target_gap),
            "mill_feasible_min": low,
            "mill_feasible_max": high,
            "time_to_terminal_h": float(remaining_hours),
            "time_remaining_h": float(remaining_hours),
            "required_avg_production_tph": float(required_production),
            "production_pressure": float(production_pressure),
            "terminal_infeasible": float(self.terminal_infeasible),
        }
