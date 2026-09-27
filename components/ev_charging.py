from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from components.ev_sessions import generate_sessions, GROUP_FIELDS

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
        self.cohort_mode = bool(config.get('cohorts'))
        self.sessions: list[dict[str, Any]] = []
        self.departure_events: list[dict[str, Any]] = []
        self.first_infeasible_hour: int | None = None
        if self.cohort_mode:
            config = dict(config)
            config['fleet_capacity_kwh'] = sum(float(g['count']) * float(g['capacity_per_vehicle_kwh']) for g in config['cohorts'])
            config.setdefault('initial_soc', 0.5)
            config.setdefault('max_soc', config['target_soc'])
            config.setdefault('departure_hour', 24)
            if self.delta_t != 1.0:
                raise ValueError('EV cohorts require hourly natural-day timesteps')
            rated = sum(float(g['count']) * float(g['max_charge_per_vehicle_kw']) for g in config['cohorts'])
            if float(config['max_charge_kw']) < rated - 1e-8:
                raise ValueError('EV cohort station limit must cover all group ratings')
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
        self.departure_events = []
        self.first_infeasible_hour = None
        if self.cohort_mode:
            date = str(pd.Timestamp(self._frame.iloc[0]['timestamp']).date())
            self.sessions = generate_sessions(self.config, date)
            self.remaining = {s['group_id']: float(s['demand_kwh']) for s in self.sessions}
            self.delivered_kwh = {s['group_id']: 0.0 for s in self.sessions}
            self._departed: set[str] = set()
            self._update_soc()

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
        if self.cohort_mode:
            bounds = self._group_bounds(t)
            return sum(b[0] for b in bounds.values()), min(self.max_charge_kw, sum(b[1] for b in bounds.values()))
        if self._remaining_steps(t) <= 0:
            return (0.0, 0.0)
        headroom_kwh = max(0.0, (self.max_soc - self.soc) * self.capacity_kwh)
        if headroom_kwh <= 1e-10:
            return (0.0, 0.0)
        maximum = min(self.max_charge_kw, headroom_kwh / (self.eta * self.delta_t))
        return (0.0, float(maximum))

    def plan(self, raw_action: float, t: int) -> dict[str, Any]:
        if self.cohort_mode:
            return self._cohort_plan(raw_action, t)
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
        if self.cohort_mode:
            return self._cohort_commit(plan, served)
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
        if self.cohort_mode:
            return self._cohort_state(t)
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

    def _group_bounds(self, t: int) -> dict[str, tuple[float, float]]:
        hour = int(t) if int(t) >= self.episode_length else self._hour(t)
        result = {}
        for s in self.sessions:
            group = s['group_id']
            low = high = 0.0
            if s['arrival_hour'] <= hour < s['departure_hour']:
                required = self.remaining[group] / (self.eta * self.delta_t)
                high = min(s['max_charge_kw'], required)
                raw_low = max(0.0, required - s['max_charge_kw'] * (s['departure_hour'] - hour - 1))
                if raw_low > high + 1e-8 and self.first_infeasible_hour is None:
                    self.first_infeasible_hour = hour
                low = min(raw_low, high)
            result[group] = (float(low), float(high))
        return result

    def _cohort_plan(self, raw_action: float, t: int) -> dict[str, Any]:
        bounds = self._group_bounds(t)
        low, high = self.feasible_bounds(t)
        power = low + raw_to_unit_interval(raw_action) * (high - low)
        allocation = {g: b[0] for g, b in bounds.items()}
        extra = power - low
        hour = int(t) if int(t) >= self.episode_length else self._hour(t)
        ordered = sorted(self.sessions, key=lambda s: (
            s['departure_hour'] - hour - self.remaining[s['group_id']] / (self.eta * s['max_charge_kw']),
            s['departure_hour'], s['group_id']))
        for s in ordered:
            group = s['group_id']
            amount = min(extra, bounds[group][1] - allocation[group])
            allocation[group] += amount
            extra = max(0.0, extra - amount)
        self.last_projection = False
        unit_intent = raw_to_unit_interval(raw_action)
        forced = abs(high - low) <= 1e-10
        return dict(
            kind='ev_charging',
            raw_action=float(np.clip(raw_action, -1, 1)),
            mapping_unit_intent=float(unit_intent),
            mapping_low_kw=float(low),
            mapping_high_kw=float(high),
            mapped_power_kw=float(power),
            mapping_span_kw=float(high - low),
            mapping_forced=bool(forced),
            requested_power_kw=power,
            effective_power_kw=power,
            effective_normalized=power / self.max_charge_kw,
            flexible_load_kw=power,
            soc_before=self.soc,
            soc_after=min(
                self.target_soc,
                self.soc + self.eta * power * self.delta_t / self.capacity_kwh,
            ),
            feasible_min_kw=low,
            feasible_max_kw=high,
            safety_correction_kw=0.0,
            safety_projected=False,
            projection_delta_kw=0.0,
            projected=False,
            target_missed=self.target_missed,
            hour=hour,
            group_power_kw=allocation,
        )

    def _update_soc(self) -> None:
        self.soc = self.target_soc - sum(self.remaining.values()) / self.capacity_kwh

    def _cohort_commit(self, plan: Mapping[str, Any], served: float) -> dict[str, float]:
        shortfall = 0.0
        for s in self.sessions:
            group = s['group_id']
            energy = self.eta * float(plan['group_power_kw'][group]) * served * self.delta_t
            self.remaining[group] = max(0.0, self.remaining[group] - energy)
            self.delivered_kwh[group] += energy
            if int(plan['hour']) + 1 >= s['departure_hour'] and group not in self._departed:
                deficit = self.remaining[group]
                self._departed.add(group)
                shortfall += deficit
                self.departure_events.append(dict(group_id=group, departure_hour=s['departure_hour'],
                    demand_kwh=s['demand_kwh'], delivered_battery_kwh=self.delivered_kwh[group], shortfall_kwh=deficit))
        self._update_soc()
        total = sum(e['shortfall_kwh'] for e in self.departure_events)
        self.target_missed = total > 1e-8
        planned_power = float(plan['effective_power_kw'])
        served_power = planned_power * served
        delivery_shortfall_kw = max(0.0, planned_power - served_power)
        result = dict(
            ev_soc=self.soc,
            planned_charge_kw=planned_power,
            served_charge_kw=served_power,
            delivered_fraction=served,
            delivery_shortfall_kw=delivery_shortfall_kw,
            delivery_shortfall_kwh=delivery_shortfall_kw * self.delta_t,
            ev_departure_shortfall_kwh=shortfall,
            ev_departure_shortfall_total_kwh=total,
            ev_departure_count=float(len(self.departure_events)),
            ev_completed_group_count=float(
                sum(e['shortfall_kwh'] <= 1e-8 for e in self.departure_events)
            ),
        )
        for group, power in plan['group_power_kw'].items():
            result[f'ev_{group}_requested_kw'] = float(power)
            result[f'ev_{group}_delivered_kw'] = float(power) * served
            result[f'ev_{group}_remaining_kwh'] = self.remaining[group]
        return result

    def _cohort_state(self, t: int) -> dict[str, float]:
        hour = int(t) if int(t) >= self.episode_length else self._hour(t)
        bounds = self._group_bounds(t)
        low, high = self.feasible_bounds(t)
        state = {}
        connected = []
        for s in self.sessions:
            group = s['group_id']
            revealed = hour >= s['arrival_hour']
            active = s['arrival_hour'] <= hour < s['departure_hour']
            values = [float(revealed), float(active), float(hour >= s['departure_hour']),
                      self.remaining[group] if active else 0.0,
                      float(s['departure_hour'] - hour) if active else 0.0,
                      s['max_charge_kw'] if active else 0.0, *bounds[group]]
            state.update({f'ev_{group}_{field}': float(value) for field, value in zip(GROUP_FIELDS, values)})
            if active:
                connected.append(s)
        remaining = sum(self.remaining[s['group_id']] for s in connected)
        capacity = sum(s['capacity_kwh'] for s in connected)
        hours = min((s['departure_hour'] - hour for s in connected), default=0)
        gap = remaining / capacity if capacity else 0.0
        state.update(ev_soc=self.target_soc - gap if capacity else 0.0, ev_soc_gap=gap,
                     ev_remaining_battery_kwh=remaining, ev_hours_to_departure=float(hours),
                     ev_available=float(bool(connected)), ev_feasible_min_kw=low, ev_feasible_max_kw=high,
                     deadline_pressure=max((self.remaining[s['group_id']] / (self.eta * s['max_charge_kw'] * (s['departure_hour'] - hour)) for s in connected), default=0.0),
                     target_missed=float(self.target_missed))
        return state
