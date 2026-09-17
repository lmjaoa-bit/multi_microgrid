from __future__ import annotations

from typing import Any


def ess_terminal_guidance(
    battery: Any,
    remaining_hours: float,
    *,
    enabled: bool = True,
) -> dict[str, float]:
    """Describe cyclic-SOC recovery pressure without changing the action bounds."""
    if battery.soc is None:
        battery.reset()
    if not enabled:
        return {
            "terminal_soc_gap": 0.0,
            "time_to_terminal_h": 0.0,
            "required_avg_power_kw": 0.0,
            "required_avg_power_to_target": 0.0,
            "terminal_pressure": 0.0,
        }
    gap_soc = float(battery.soc) - float(battery.initial_soc)
    hours = max(float(remaining_hours), 0.0)
    if hours == 0.0 or abs(gap_soc) <= 1e-15:
        required_kw = 0.0
        required_ratio = 0.0
        pressure = 0.0
    elif gap_soc < 0.0:
        required_kw = (
            gap_soc * float(battery.capacity_kwh)
            / (float(battery.eta_charge) * hours)
        )
        required_ratio = required_kw / max(float(battery.max_charge_kw), 1e-12)
        pressure = abs(required_ratio)
    else:
        required_kw = (
            gap_soc
            * float(battery.capacity_kwh)
            * float(battery.eta_discharge)
            / hours
        )
        required_ratio = required_kw / max(float(battery.max_discharge_kw), 1e-12)
        pressure = abs(required_ratio)
    return {
        "terminal_soc_gap": gap_soc,
        "time_to_terminal_h": hours,
        "required_avg_power_kw": float(required_kw),
        "required_avg_power_to_target": float(required_ratio),
        "terminal_pressure": float(pressure),
    }
