from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BatteryModel:
    """Battery model used by every microgrid.

    Action sign after mapping: positive power is discharge, negative power is charge.
    The policy action itself is always in [-1, 1] and is mapped to the feasible
    SOC interval with terminal reachability projection.
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

    def reachable_soc_envelope(
        self,
        t: int,
        episode_length: int = 24,
        target_soc: float | None = None,
        tolerance: float = 0.03,
    ) -> tuple[float, float]:
        """Compute the feasible SOC interval at step t+1 that guarantees terminal target reachability."""
        target = float(self.initial_soc if target_soc is None else target_soc)
        tol = max(0.0, float(tolerance))
        steps_remaining_after_next = max(0, int(episode_length) - 1 - int(t))

        # Max SOC that can be gained by charging in remaining steps after t+1
        max_gain_soc = (
            steps_remaining_after_next
            * self.eta_charge
            * self.max_charge_kw
            * self.delta_t
            / self.capacity_kwh
        )
        soc_min_reach = max(self.soc_min, target - tol - max_gain_soc)

        # Max SOC that can be lost by discharging in remaining steps after t+1
        max_loss_soc = (
            steps_remaining_after_next
            * (self.max_discharge_kw * self.delta_t / self.eta_discharge)
            / self.capacity_kwh
        )
        soc_max_reach = min(self.soc_max, target + tol + max_loss_soc)

        return float(soc_min_reach), float(soc_max_reach)

    def map_action(
        self,
        action: float,
        soc_min_reach: float | None = None,
        soc_max_reach: float | None = None,
    ) -> Dict[str, float]:
        """Map an action into feasible charge/discharge with terminal reachability."""
        if self.soc is None:
            self.reset()
        assert self.soc is not None

        a = max(-1.0, min(1.0, float(action)))
        floor = self.soc_min if soc_min_reach is None else max(self.soc_min, float(soc_min_reach))
        ceiling = self.soc_max if soc_max_reach is None else min(self.soc_max, float(soc_max_reach))

        # Check required minimum charge/discharge if currently outside reachability envelope
        if float(self.soc) < floor:
            needed_gain_kwh = (floor - float(self.soc)) * self.capacity_kwh
            feasible_ch_min = float(needed_gain_kwh / (self.eta_charge * self.delta_t))
        else:
            feasible_ch_min = 0.0

        if float(self.soc) > ceiling:
            needed_loss_kwh = (float(self.soc) - ceiling) * self.capacity_kwh
            feasible_dis_min = float(needed_loss_kwh * self.eta_discharge / self.delta_t)
        else:
            feasible_dis_min = 0.0

        # Maximum available charge up to ceiling
        headroom_kwh = max(0.0, (ceiling - float(self.soc)) * self.capacity_kwh)
        feasible_ch_max = float(min(self.max_charge_kw, headroom_kwh / (self.eta_charge * self.delta_t)))
        feasible_ch_min = float(min(feasible_ch_min, self.max_charge_kw))

        # Maximum available discharge respecting power, energy, and reachability.
        usable_reach_stored_kwh = max(0.0, (float(self.soc) - floor) * self.capacity_kwh)
        reach_dispatch_max_kw = usable_reach_stored_kwh * self.eta_discharge / self.delta_t

        feasible_dis_max = float(min(self.max_discharge_kw, reach_dispatch_max_kw))
        feasible_dis_min = float(min(feasible_dis_min, self.max_discharge_kw))

        if feasible_ch_min > 0.0:
            # Must charge to guarantee reaching terminal SOC
            feasible_dis_max = 0.0
            if a <= 0.0:
                charge_kw = feasible_ch_min + (-a) * max(0.0, feasible_ch_max - feasible_ch_min)
            else:
                charge_kw = feasible_ch_min
            discharge_kw = 0.0
            power_kw = -float(charge_kw)
        elif feasible_dis_min > 0.0:
            # Must discharge to guarantee reaching terminal SOC
            feasible_ch_max = 0.0
            if a >= 0.0:
                discharge_kw = feasible_dis_min + a * max(0.0, feasible_dis_max - feasible_dis_min)
            else:
                discharge_kw = feasible_dis_min
            charge_kw = 0.0
            power_kw = float(discharge_kw)
        else:
            # Normal operating range
            if a > 0.0:
                power_kw = a * feasible_dis_max
                charge_kw = 0.0
                discharge_kw = float(power_kw)
            elif a < 0.0:
                charge_kw = (-a) * feasible_ch_max
                discharge_kw = 0.0
                power_kw = -float(charge_kw)
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
