from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BatteryModel:
    """Battery model with instantaneous SOC and power safety only.

    Positive mapped power is discharge and negative mapped power is charge.
    Cross-time terminal reachability is deliberately not projected into actions;
    terminal requirements are learning signals exposed by the environment.
    """

    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_min: float = 0.10
    soc_max: float = 0.90
    initial_soc: float = 0.10
    degradation_cost_per_kwh: float = 0.01
    delta_t: float = 1.0
    soc: float | None = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], delta_t: float) -> "BatteryModel":
        return cls(
            capacity_kwh=float(cfg["capacity_kwh"]),
            max_charge_kw=float(cfg["max_charge_kw"]),
            max_discharge_kw=float(cfg["max_discharge_kw"]),
            eta_charge=float(cfg.get("eta_charge", 0.95)),
            eta_discharge=float(cfg.get("eta_discharge", 0.95)),
            soc_min=float(cfg.get("soc_min", 0.10)),
            soc_max=float(cfg.get("soc_max", 0.90)),
            initial_soc=float(cfg.get("initial_soc", 0.10)),
            degradation_cost_per_kwh=float(cfg.get("degradation_cost_per_kwh", 0.01)),
            delta_t=float(delta_t),
        )

    def reset(self) -> None:
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("initial_soc must lie inside [soc_min, soc_max]")
        self.soc = float(self.initial_soc)

    def available_charge_kw(self) -> float:
        if self.soc is None:
            self.reset()
        assert self.soc is not None
        headroom_kwh = max(0.0, (self.soc_max - float(self.soc)) * self.capacity_kwh)
        return min(self.max_charge_kw, headroom_kwh / (self.eta_charge * self.delta_t))

    def available_discharge_kw(self) -> float:
        if self.soc is None:
            self.reset()
        assert self.soc is not None
        stored_kwh = max(0.0, (float(self.soc) - self.soc_min) * self.capacity_kwh)
        return min(self.max_discharge_kw, stored_kwh * self.eta_discharge / self.delta_t)

    def map_action(self, action: float) -> Dict[str, float]:
        """Map [-1, 1] to the current physically feasible ESS power range."""
        if self.soc is None:
            self.reset()
        assert self.soc is not None

        a = max(-1.0, min(1.0, float(action)))
        feasible_ch_max = float(self.available_charge_kw())
        feasible_dis_max = float(self.available_discharge_kw())

        if a > 0.0:
            discharge_kw = float(a * feasible_dis_max)
            charge_kw = 0.0
            power_kw = discharge_kw
        elif a < 0.0:
            charge_kw = float((-a) * feasible_ch_max)
            discharge_kw = 0.0
            power_kw = -charge_kw
        else:
            charge_kw = 0.0
            discharge_kw = 0.0
            power_kw = 0.0

        return {
            "raw_action": a,
            "power_kw": float(power_kw),
            "charge_requested_kw": float(charge_kw),
            "discharge_requested_kw": float(discharge_kw),
            "feasible_min_kw": -float(feasible_ch_max),
            "feasible_max_kw": float(feasible_dis_max),
        }

    def commit(self, charge_kw: float, discharge_kw: float) -> Dict[str, float]:
        if self.soc is None:
            self.reset()
        assert self.soc is not None
        charge_kw = min(max(0.0, float(charge_kw)), self.available_charge_kw())
        discharge_kw = min(max(0.0, float(discharge_kw)), self.available_discharge_kw())
        if charge_kw > 1e-9 and discharge_kw > 1e-9:
            raise ValueError("Battery cannot charge and discharge simultaneously.")

        old_soc = float(self.soc)
        self.soc = old_soc + (
            self.eta_charge * charge_kw * self.delta_t / self.capacity_kwh
            - discharge_kw * self.delta_t / (self.eta_discharge * self.capacity_kwh)
        )
        self.soc = min(self.soc_max, max(self.soc_min, float(self.soc)))
        throughput_kwh = (charge_kw + discharge_kw) * self.delta_t
        return {
            "old_soc": old_soc,
            "soc": float(self.soc),
            "charge_kw": charge_kw,
            "discharge_kw": discharge_kw,
            "throughput_kwh": float(throughput_kwh),
            "degradation_cost": float(throughput_kwh * self.degradation_cost_per_kwh),
        }
