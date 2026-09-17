from __future__ import annotations

from typing import Any, Mapping, Sequence


def terminal_constraint_breakdown(env: Any, mg: str) -> dict[str, float]:
    battery = env.cores[mg].battery
    tolerance = float(env.config["terminal_soc"]["tolerance"])
    ess_equivalent = max(
        0.0,
        abs(float(battery.soc) - float(battery.initial_soc)) - tolerance,
    ) * float(battery.capacity_kwh)

    mill_equivalent = 0.0
    hvac_equivalent = 0.0
    ev_equivalent = 0.0
    device = env.dr_devices.get(mg)
    kind = str(getattr(device, "config", {}).get("type", "")) if device is not None else ""
    if kind == "industrial_mill":
        deviation_t = max(
            0.0,
            abs(float(device.silo_t) - float(device.initial_silo_t))
            - float(device.terminal_tolerance_t),
        )
        mill_equivalent = (
            deviation_t / max(float(device.production_tph), 1e-12)
            * float(device.rated_power_kw)
        )
    elif kind == "commercial_hvac":
        air_deviation = max(
            0.0,
            abs(float(device.t_air) - float(device.initial_t_air))
            - float(device.terminal_air_tolerance),
        )
        mass_deviation = max(
            0.0,
            abs(float(device.t_mass) - float(device.initial_t_mass))
            - float(device.terminal_mass_tolerance),
        )
        air_cop = float(device.cop_cool) if float(device.t_air) > float(device.initial_t_air) else float(device.cop_heat)
        mass_cop = float(device.cop_cool) if float(device.t_mass) > float(device.initial_t_mass) else float(device.cop_heat)
        hvac_equivalent = (
            air_deviation * float(device.c_air) / max(air_cop, 1e-12)
            + mass_deviation * float(device.c_mass) / max(mass_cop, 1e-12)
        )
    elif kind == "ev_charging":
        ev_equivalent = max(0.0, float(device.target_soc) - float(device.soc)) * float(device.capacity_kwh)

    equivalent = ess_equivalent + mill_equivalent + hvac_equivalent + ev_equivalent
    rate = float(env.config["terminal_soc"]["penalty_per_kwh"])
    return {
        "ess_equivalent_kwh": float(ess_equivalent),
        "mill_equivalent_kwh": float(mill_equivalent),
        "hvac_equivalent_kwh": float(hvac_equivalent),
        "ev_equivalent_kwh": float(ev_equivalent),
        "total_equivalent_kwh": float(equivalent),
        "penalty_rate_per_equivalent_kwh": rate,
        "total_penalty": float(equivalent * rate),
    }


def apply_terminal_constraint_penalty(
    env: Any,
    observations: Mapping[str, Any],
    rewards: Mapping[str, float],
    done: bool,
    info: dict[str, Any],
) -> tuple[Mapping[str, Any], dict[str, float], bool, dict[str, Any]]:
    """Add terminal learning cost without constraining any earlier action."""
    info["constraint_projection_mode"] = "instantaneous"
    for mg in env.agent_ids:
        info["costs"][mg]["terminal_constraint_penalty"] = 0.0

    zero = {
        "ess_equivalent_kwh": 0.0,
        "mill_equivalent_kwh": 0.0,
        "hvac_equivalent_kwh": 0.0,
        "ev_equivalent_kwh": 0.0,
        "total_equivalent_kwh": 0.0,
        "penalty_rate_per_equivalent_kwh": float(env.config["terminal_soc"]["penalty_per_kwh"]),
        "total_penalty": 0.0,
    }
    if not done:
        info["terminal_constraint_penalty"] = {mg: dict(zero) for mg in env.agent_ids}
        return observations, dict(rewards), done, info

    breakdown = {mg: terminal_constraint_breakdown(env, mg) for mg in env.agent_ids}
    terminal_penalties: dict[str, float] = {}
    for mg in env.agent_ids:
        penalty = float(breakdown[mg]["total_penalty"])
        terminal_penalties[mg] = penalty
        info["costs"][mg]["terminal_constraint_penalty"] = penalty
        info["costs"][mg]["terminal_soc_penalty"] = penalty
        info["costs"][mg]["physical_profit"] -= penalty
        info["costs"][mg]["individual_accounting_profit"] -= penalty

    team_profit = float(sum(info["costs"][mg]["physical_profit"] for mg in env.agent_ids))
    team_reward = team_profit / float(env.accounting.reward_scale)
    new_rewards = {mg: float(team_reward) for mg in env.agent_ids}
    info["team_profit"] = team_profit
    info["team_reward"] = float(team_reward)
    info["terminal_soc_penalty"] = terminal_penalties
    info["terminal_constraint_penalty"] = breakdown
    info["terminal_constraint_penalty_total"] = float(sum(terminal_penalties.values()))
    return observations, new_rewards, done, info
