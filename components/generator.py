from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class GeneratorModel:
    max_power_kw: float
    min_power_kw: float
    fuel_price_per_kwh_hhv: float
    fuel_emission_factor_kg_per_kwh_hhv: float
    om_cost_per_kwh: float
    startup_cost: float
    efficiency_curve: List[Tuple[float, float]]
    delta_t: float = 1.0
    last_power_kw: float = 0.0

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], delta_t: float) -> "GeneratorModel":
        curve = [
            (float(p["load_fraction"]), float(p["efficiency"]))
            for p in cfg["part_load_efficiency_curve"]
        ]
        curve.sort()
        return cls(
            max_power_kw=float(cfg["max_power_kw"]),
            min_power_kw=float(cfg["min_power_kw"]),
            fuel_price_per_kwh_hhv=float(cfg["fuel_price_per_kwh_hhv"]),
            fuel_emission_factor_kg_per_kwh_hhv=float(cfg["fuel_emission_factor_kg_per_kwh_hhv"]),
            om_cost_per_kwh=float(cfg.get("om_cost_per_kwh", 0.0)),
            startup_cost=float(cfg.get("startup_cost", 0.0)),
            efficiency_curve=curve,
            delta_t=float(delta_t),
        )

    def reset(self) -> None:
        self.last_power_kw = 0.0

    def setpoint_from_action(self, action: float) -> float:
        a = max(-1.0, min(1.0, float(action)))
        return self.min_power_kw + 0.5 * (a + 1.0) * (self.max_power_kw - self.min_power_kw)

    def efficiency(self, power_kw: float) -> float:
        if power_kw <= 0.0:
            return 0.0
        x = min(1.0, max(0.0, power_kw / self.max_power_kw))
        if x <= self.efficiency_curve[0][0]:
            return self.efficiency_curve[0][1]
        for (x0, y0), (x1, y1) in zip(self.efficiency_curve[:-1], self.efficiency_curve[1:]):
            if x <= x1:
                w = (x - x0) / max(x1 - x0, 1e-12)
                return y0 + w * (y1 - y0)
        return self.efficiency_curve[-1][1]

    def plan(self, commitment_action: float, setpoint_action: float) -> Dict[str, float | bool]:
        on = float(commitment_action) > 0.0
        power_kw = self.setpoint_from_action(setpoint_action) if on else 0.0
        return {"on": bool(on), "power_kw": float(power_kw)}

    def commit(self, power_kw: float) -> Dict[str, float | bool]:
        power_kw = 0.0 if power_kw <= 0.0 else min(self.max_power_kw, max(self.min_power_kw, float(power_kw)))
        was_on = self.last_power_kw >= self.min_power_kw - 1e-9
        is_on = power_kw >= self.min_power_kw - 1e-9
        started = bool(is_on and not was_on)
        energy_kwh = power_kw * self.delta_t
        eta = self.efficiency(power_kw)
        fuel_kwh_hhv = energy_kwh / eta if eta > 1e-12 else 0.0
        fuel_cost = fuel_kwh_hhv * self.fuel_price_per_kwh_hhv
        om_cost = energy_kwh * self.om_cost_per_kwh
        start_cost = self.startup_cost if started else 0.0
        carbon_kg = fuel_kwh_hhv * self.fuel_emission_factor_kg_per_kwh_hhv
        self.last_power_kw = power_kw
        return {
            "is_on": bool(is_on),
            "started": started,
            "power_kw": float(power_kw),
            "energy_kwh": float(energy_kwh),
            "efficiency_hhv": float(eta),
            "fuel_energy_kwh_hhv": float(fuel_kwh_hhv),
            "fuel_cost": float(fuel_cost),
            "om_cost": float(om_cost),
            "startup_cost": float(start_cost),
            "cost": float(fuel_cost + om_cost + start_cost),
            "carbon_kg": float(carbon_kg),
        }
