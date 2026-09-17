from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def replace_once(rel: str, old: str, new: str, *, marker: str | None = None) -> None:
    text = read(rel)
    if old in text:
        text = text.replace(old, new, 1)
        write(rel, text)
        return
    if marker is not None and marker in text:
        return
    raise RuntimeError(f"{rel}: expected replacement source not found")


def regex_once(rel: str, pattern: str, repl: str, *, marker: str | None = None, flags: int = 0) -> None:
    text = read(rel)
    updated, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count:
        write(rel, updated)
        return
    if marker is not None and marker in text:
        return
    raise RuntimeError(f"{rel}: regex source not found: {pattern[:80]!r}")


# ---------------------------------------------------------------------------
# Canonical config: no strong/instantaneous switch. The one supported runtime
# semantics is instantaneous physical safety plus terminal learning signals.
# ---------------------------------------------------------------------------
regex_once(
    "configs/cooperative.yaml",
    r"\n# Projection ablation switch\.[\s\S]*?constraint_projection:\n  mode: strong\n",
    "\n# Only instantaneous physical safety is projected. Cross-time terminal\n# requirements are exposed as P0 state features and day-end learning penalties.\n",
    marker="Only instantaneous physical safety is projected",
)

for rel in (
    "configs/algorithms/matd3_dispatch.yaml",
    "configs/algorithms/pnn.yaml",
    "configs/algorithms/matd3_transformer.yaml",
    "configs/algorithms/pnn_transformer.yaml",
):
    text = read(rel)
    if "diagnostic_interval:" not in text:
        text = text.replace("logging:\n", "logging:\n  diagnostic_interval: 20\n", 1)
        write(rel, text)

# Keep Transformer forecast_horizon unchanged in this refactor.
for rel in ("configs/algorithms/matd3_transformer.yaml", "configs/algorithms/pnn_transformer.yaml"):
    text = read(rel)
    if "forecast_horizon: 4" not in text:
        raise RuntimeError(f"{rel}: this refactor must not change Transformer forecast_horizon")

# Projection selector is obsolete.
regex_once(
    "core/config_validation.py",
    r"\n    projection = config\.get\(\"constraint_projection\", \{\}\)[\s\S]*?raise ValueError\(\"constraint_projection\.mode must be strong or instantaneous\"\)\n",
    "\n",
    marker="Only instantaneous physical projection is supported",
)
replace_once(
    "core/config_validation.py",
    '    if action_design == "single_dispatch_dr" and terminal.get("mode") != "hard":\n        raise ValueError("single_dispatch_dr baseline requires hard cyclic terminal SOC")\n',
    '    if action_design == "single_dispatch_dr" and terminal.get("mode") != "hard":\n        raise ValueError("single_dispatch_dr requires terminal targets for end-of-day learning penalties")\n',
    marker="end-of-day learning penalties",
)

# Contract still records the fixed semantics for reproducibility, but there is
# no user-selectable projection mode anymore.
replace_once(
    "experiments/profiles.py",
    '    projection_mode = str(env_config.get("constraint_projection", {}).get("mode", "strong"))\n',
    '    projection_mode = "instantaneous"\n',
    marker='projection_mode = "instantaneous"',
)
replace_once(
    "experiments/profiles.py",
    '                "env", "p2p", "terminal_soc", "constraint_projection",\n                "cost", "time", "data", "price",\n',
    '                "env", "p2p", "terminal_soc",\n                "cost", "time", "data", "price",\n',
    marker='"env", "p2p", "terminal_soc",\n                "cost"',
)

# ---------------------------------------------------------------------------
# Industrial mill: retain one-step silo safety only. Terminal information is
# state information, never a future action bound.
# ---------------------------------------------------------------------------
replace_once(
    "components/industrial_mill.py",
    '    """Raw-meal mill with an intermediate silo and hard daily reachability."""\n',
    '    """Raw-meal mill with instantaneous silo safety and terminal P0 signals."""\n',
    marker="instantaneous silo safety",
)
replace_once(
    "components/industrial_mill.py",
    '        self._reachable = self._build_reachable_intervals()\n',
    '',
    marker="instantaneous silo safety",
)
regex_once(
    "components/industrial_mill.py",
    r"\n    def _build_reachable_intervals\(self\)[\s\S]*?\n    def reset\(",
    "\n    def reset(",
    marker="def reset(",
)
regex_once(
    "components/industrial_mill.py",
    r"    def feasible_bounds\(self, t: int\) -> tuple\[float, float\]:\n[\s\S]*?\n    def plan\(",
    '''    def feasible_bounds(self, t: int) -> tuple[float, float]:
        """One-step physical silo bounds; no terminal reachable-set projection."""
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

    def plan(''',
    marker="One-step physical silo bounds",
)
replace_once(
    "components/industrial_mill.py",
    '''    def state(self, t: int) -> dict[str, float]:
        low, high = self.feasible_bounds(t)
        target_gap = self.silo_t - self.initial_silo_t
        return {
            "silo_level_t": float(self.silo_t),
            "silo_ratio": float(self.silo_t / self.silo_capacity_t),
            "silo_target_gap_t": float(target_gap),
            "mill_feasible_min": low,
            "mill_feasible_max": high,
            "time_to_terminal_h": float(max(self.episode_length - int(t), 0)),
            "terminal_infeasible": float(self.terminal_infeasible),
        }
''',
    '''    def state(self, t: int) -> dict[str, float]:
        low, high = self.feasible_bounds(t)
        target_gap = self.silo_t - self.initial_silo_t
        remaining_steps = max(self.episode_length - int(t), 0)
        remaining_hours = remaining_steps * self.delta_t
        if remaining_hours > 0.0:
            required_avg_duty = (
                self.initial_silo_t - self.silo_t
                + self.downstream_tph * remaining_hours
            ) / max(self.production_tph * remaining_hours, 1e-12)
            terminal_pressure = max(
                0.0, abs(target_gap) - self.terminal_tolerance_t
            ) / max(self.production_tph * remaining_hours, 1e-12)
        else:
            required_avg_duty = 0.0
            terminal_pressure = max(0.0, abs(target_gap) - self.terminal_tolerance_t)
        return {
            "silo_level_t": float(self.silo_t),
            "silo_ratio": float(self.silo_t / self.silo_capacity_t),
            "silo_target_gap_t": float(target_gap),
            "mill_feasible_min": low,
            "mill_feasible_max": high,
            "time_to_terminal_h": float(remaining_hours),
            "mill_required_avg_duty": float(required_avg_duty),
            "mill_terminal_pressure": float(terminal_pressure),
            "terminal_infeasible": float(self.terminal_infeasible),
        }
''',
    marker='"mill_required_avg_duty"',
)

# ---------------------------------------------------------------------------
# EV: availability/SOC/power safety only. Deadline urgency remains observable.
# ---------------------------------------------------------------------------
replace_once(
    "components/ev_charging.py",
    '    """Aggregate residential EV charging with a hard departure-energy target."""\n',
    '    """Aggregate EV charging with instantaneous safety and deadline P0 signals."""\n',
    marker="deadline P0 signals",
)
regex_once(
    "components/ev_charging.py",
    r"    def feasible_bounds\(self, t: int\) -> tuple\[float, float\]:\n[\s\S]*?\n    def plan\(",
    '''    def feasible_bounds(self, t: int) -> tuple[float, float]:
        """Instantaneous availability/SOC/power bounds; no forced deadline charging."""
        if self._remaining_steps(t) <= 0:
            return (0.0, 0.0)
        headroom_kwh = max(0.0, (self.max_soc - self.soc) * self.capacity_kwh)
        if headroom_kwh <= 1e-10:
            return (0.0, 0.0)
        maximum = min(
            self.max_charge_kw,
            headroom_kwh / (self.eta * self.delta_t),
        )
        return (0.0, float(maximum))

    def plan(''',
    marker="no forced deadline charging",
)
replace_once(
    "components/ev_charging.py",
    '    def state(self, t: int) -> dict[str, float]:\n        low, high = self.feasible_bounds(t)\n        remaining = self._remaining_battery_kwh()\n',
    '    def state(self, t: int) -> dict[str, float]:\n        low, high = self.feasible_bounds(t)\n        if self._hour(t) >= self.departure_hour and self.soc < self.target_soc - 1e-10:\n            self.target_missed = True\n        remaining = self._remaining_battery_kwh()\n',
    marker="self._hour(t) >= self.departure_hour",
)

# ---------------------------------------------------------------------------
# HVAC: remove the horizon LP from action feasibility; expose terminal thermal
# recovery pressure as P0 instead.
# ---------------------------------------------------------------------------
replace_once(
    "components/commercial_hvac.py",
    '    """Reversible heat-pump DR backed by a 2R2C building thermal model."""\n',
    '    """2R2C HVAC with one-step comfort safety and terminal thermal P0 signals."""\n',
    marker="terminal thermal P0 signals",
)
regex_once(
    "components/commercial_hvac.py",
    r"\n    def _solve_terminal_q\(self, t: int, \*, maximize: bool\) -> float \| None:[\s\S]*?\n    def feasible_bounds\(",
    "\n    def feasible_bounds(",
    marker="One-step comfort/power bounds",
)
regex_once(
    "components/commercial_hvac.py",
    r"    def feasible_bounds\(self, t: int\) -> tuple\[float, float\]:\n[\s\S]*?\n    def plan\(",
    '''    def feasible_bounds(self, t: int) -> tuple[float, float]:
        """One-step comfort/power bounds; terminal state never clips the action."""
        if not 0 <= int(t) < self.episode_length:
            return (0.0, 0.0)
        key = (int(t), round(float(self.t_air), 10), round(float(self.t_mass), 10))
        if self._bounds_cache_key == key and self._bounds_cache_value is not None:
            return self._bounds_cache_value
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

    def plan(''',
    marker="terminal state never clips the action",
)
replace_once(
    "components/commercial_hvac.py",
    '''        return {
            "t_air_c": float(self.t_air),
            "t_mass_c": float(self.t_mass),
            "t_out_c": float(t_out),
            "comfort_lower_margin_c": float(self.t_air - comfort_low),
            "comfort_upper_margin_c": float(comfort_high - self.t_air),
            "occupied": occupied,
            "is_weekend": weekend,
            "hvac_feasible_min_kw": low,
            "hvac_feasible_max_kw": high,
            "comfort_infeasible": float(self.comfort_infeasible),
            "terminal_thermal_infeasible": float(self.terminal_thermal_infeasible),
        }
''',
    '''        air_gap = float(self.t_air - self.initial_t_air)
        mass_gap = float(self.t_mass - self.initial_t_mass)
        remaining_hours = max(self.episode_length - int(t), 0) * self.delta_t
        air_excess = max(0.0, abs(air_gap) - self.terminal_air_tolerance)
        mass_excess = max(0.0, abs(mass_gap) - self.terminal_mass_tolerance)
        air_cop = self.cop_cool if air_gap > 0.0 else self.cop_heat
        mass_cop = self.cop_cool if mass_gap > 0.0 else self.cop_heat
        recovery_kwh = (
            air_excess * self.c_air / max(air_cop, 1e-12)
            + mass_excess * self.c_mass / max(mass_cop, 1e-12)
        )
        max_recovery_kwh = max(self.heat_max_kw, self.cool_max_kw) * max(remaining_hours, self.delta_t)
        thermal_pressure = recovery_kwh / max(max_recovery_kwh, 1e-12)
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
            "terminal_air_gap_c": air_gap,
            "terminal_mass_gap_c": mass_gap,
            "hvac_time_to_terminal_h": float(remaining_hours),
            "thermal_recovery_pressure": float(thermal_pressure),
            "comfort_infeasible": float(self.comfort_infeasible),
            "terminal_thermal_infeasible": float(self.terminal_thermal_infeasible),
        }
''',
    marker='"thermal_recovery_pressure"',
)

# ---------------------------------------------------------------------------
# P0 Actor observation.
# ---------------------------------------------------------------------------
replace_once(
    "core/observation.py",
    '                "silo_ratio", "silo_target_gap", "mill_feasible_min",\n                "mill_feasible_max", "time_to_terminal",\n',
    '                "silo_ratio", "silo_target_gap", "mill_feasible_min",\n                "mill_feasible_max", "time_to_terminal",\n                "mill_required_avg_duty", "mill_terminal_pressure",\n',
    marker='"mill_required_avg_duty"',
)
replace_once(
    "core/observation.py",
    '                "hvac_feasible_min", "hvac_feasible_max",\n',
    '                "hvac_feasible_min", "hvac_feasible_max",\n                "terminal_air_gap", "terminal_mass_gap",\n                "hvac_time_to_terminal", "thermal_recovery_pressure",\n',
    marker='"terminal_air_gap"',
)
replace_once(
    "core/observation.py",
    '                "available_charge_ratio", "available_discharge_ratio", "terminal_soc_gap",\n',
    '                "available_charge_ratio", "available_discharge_ratio", "terminal_soc_gap",\n                "ess_time_to_terminal", "ess_required_avg_power", "ess_terminal_pressure",\n',
    marker='"ess_time_to_terminal"',
)
replace_once(
    "core/observation.py",
    '''                float(state["mill_feasible_max"]),
                float(state["time_to_terminal_h"]) / max(float(self.episode_length), 1.0),
            ]
''',
    '''                float(state["mill_feasible_max"]),
                float(state["time_to_terminal_h"]) / max(float(self.episode_length), 1.0),
                float(state["mill_required_avg_duty"]),
                float(state["mill_terminal_pressure"]),
            ]
''',
    marker='float(state["mill_required_avg_duty"])',
)
replace_once(
    "core/observation.py",
    '''                float(state["hvac_feasible_min_kw"]) / scale,
                float(state["hvac_feasible_max_kw"]) / scale,
            ]
''',
    '''                float(state["hvac_feasible_min_kw"]) / scale,
                float(state["hvac_feasible_max_kw"]) / scale,
                float(state["terminal_air_gap_c"]) / 15.0,
                float(state["terminal_mass_gap_c"]) / 15.0,
                float(state["hvac_time_to_terminal_h"]) / max(float(self.episode_length), 1.0),
                float(state["thermal_recovery_pressure"]),
            ]
''',
    marker='float(state["terminal_air_gap_c"])',
)
replace_once(
    "core/observation.py",
    '''                (float(battery_soc) - self.initial_soc[mg_id]
                 if self.config["terminal_soc"]["mode"] != "none" else 0.0),
                previous["ess_p2p_offer_kwh"] / n["trade"],
''',
    '''                float(battery_soc) - self.initial_soc[mg_id],
                float(flex.get("ess_time_to_terminal", 0.0)),
                float(flex.get("ess_required_avg_power", 0.0)),
                float(flex.get("ess_terminal_pressure", 0.0)),
                previous["ess_p2p_offer_kwh"] / n["trade"],
''',
    marker='flex.get("ess_time_to_terminal"',
)

# ---------------------------------------------------------------------------
# P0 centralized critic state.
# ---------------------------------------------------------------------------
replace_once(
    "core/global_state.py",
    '                f"{mg}_time_to_terminal", f"{mg}_dr_min", f"{mg}_dr_max",\n',
    '                f"{mg}_time_to_terminal", f"{mg}_mill_required_avg_duty",\n                f"{mg}_mill_terminal_pressure", f"{mg}_dr_min", f"{mg}_dr_max",\n',
    marker='f"{mg}_mill_required_avg_duty"',
)
replace_once(
    "core/global_state.py",
    '                f"{mg}_dr_min", f"{mg}_dr_max",\n            ]\n        if kind == "ev_charging":\n',
    '                f"{mg}_dr_min", f"{mg}_dr_max",\n                f"{mg}_terminal_air_gap", f"{mg}_terminal_mass_gap",\n                f"{mg}_hvac_time_to_terminal", f"{mg}_thermal_recovery_pressure",\n            ]\n        if kind == "ev_charging":\n',
    marker='f"{mg}_terminal_air_gap"',
)
replace_once(
    "core/global_state.py",
    '            names.extend([f"{mg}_load", f"{mg}_pv", f"{mg}_wind", f"{mg}_soc",\n                          f"{mg}_previous_p2p_net"])\n',
    '            names.extend([f"{mg}_load", f"{mg}_pv", f"{mg}_wind", f"{mg}_soc",\n                          f"{mg}_previous_p2p_net", f"{mg}_terminal_soc_gap",\n                          f"{mg}_ess_time_to_terminal", f"{mg}_ess_required_avg_power",\n                          f"{mg}_ess_terminal_pressure"])\n',
    marker='f"{mg}_ess_time_to_terminal"',
)
replace_once(
    "core/global_state.py",
    '''                float(state["time_to_terminal_h"]) / max(float(self.episode_length), 1.0),
                float(state["mill_feasible_min"]),
                float(state["mill_feasible_max"]),
            ]
''',
    '''                float(state["time_to_terminal_h"]) / max(float(self.episode_length), 1.0),
                float(state["mill_required_avg_duty"]),
                float(state["mill_terminal_pressure"]),
                float(state["mill_feasible_min"]),
                float(state["mill_feasible_max"]),
            ]
''',
    marker='float(state["mill_required_avg_duty"])',
)
replace_once(
    "core/global_state.py",
    '''                float(state["hvac_feasible_min_kw"]) / scale,
                float(state["hvac_feasible_max_kw"]) / scale,
            ]
''',
    '''                float(state["hvac_feasible_min_kw"]) / scale,
                float(state["hvac_feasible_max_kw"]) / scale,
                float(state["terminal_air_gap_c"]) / 15.0,
                float(state["terminal_mass_gap_c"]) / 15.0,
                float(state["hvac_time_to_terminal_h"]) / max(float(self.episode_length), 1.0),
                float(state["thermal_recovery_pressure"]),
            ]
''',
    marker='float(state["terminal_air_gap_c"])',
)
replace_once(
    "core/global_state.py",
    '''            values.extend([ex["load_kw"] / n["load"], ex["pv_kw"] / n["pv"],
                           ex["wind_kw"] / n["wind"], float(cores[mg].battery.soc),
                           previous["p2p_net_kwh"] / n["trade"]])
''',
    '''            battery = cores[mg].battery
            remaining_hours = max(self.episode_length - int(t), 0) * float(battery.delta_t)
            soc_gap = float(battery.soc) - float(battery.initial_soc)
            if remaining_hours > 0.0:
                if soc_gap >= 0.0:
                    required_kw = soc_gap * float(battery.capacity_kwh) * float(battery.eta_discharge) / remaining_hours
                    required_ratio = required_kw / max(float(battery.max_discharge_kw), 1e-12)
                else:
                    required_kw = -(-soc_gap) * float(battery.capacity_kwh) / max(float(battery.eta_charge) * remaining_hours, 1e-12)
                    required_ratio = required_kw / max(float(battery.max_charge_kw), 1e-12)
            else:
                required_ratio = 0.0
            values.extend([ex["load_kw"] / n["load"], ex["pv_kw"] / n["pv"],
                           ex["wind_kw"] / n["wind"], float(battery.soc),
                           previous["p2p_net_kwh"] / n["trade"], soc_gap,
                           remaining_hours / max(float(self.episode_length), 1.0),
                           required_ratio, abs(required_ratio)])
''',
    marker="required_ratio = required_kw",
)
replace_once(
    "core/global_state.py",
    '                    tolerance=float(self.config["terminal_soc"]["tolerance"]),\n                    enforce_terminal=self.config["terminal_soc"]["mode"] != "none")\n',
    '                    tolerance=float(self.config["terminal_soc"]["tolerance"]),\n                    enforce_terminal=False)\n',
    marker="enforce_terminal=False)",
)

# ---------------------------------------------------------------------------
# Terminal-constraint learning signal helper. It does not alter feasible actions.
# ---------------------------------------------------------------------------
write(
    "core/terminal_constraints.py",
    '''from __future__ import annotations

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
''',
)

# ---------------------------------------------------------------------------
# Canonical environment: instantaneous ESS channels and terminal penalty helper.
# ---------------------------------------------------------------------------
replace_once(
    "envs/multi_microgrid_env.py",
    'from core.tariffs import prices_from_row\n',
    'from core.tariffs import prices_from_row\nfrom core.terminal_constraints import apply_terminal_constraint_penalty\n',
    marker="apply_terminal_constraint_penalty",
)
replace_once(
    "envs/multi_microgrid_env.py",
    '''                floor, ceiling = battery.reachable_soc_envelope(
                    index,
                    self.episode_length,
                    tolerance=float(self.config["terminal_soc"]["tolerance"]),
                    enforce_terminal=self.config["terminal_soc"]["mode"] != "none",
                )
                neutral = battery.map_action(0.0, floor, ceiling)
                result[mg].update({
''',
    '''                floor, ceiling = battery.reachable_soc_envelope(
                    index,
                    self.episode_length,
                    tolerance=float(self.config["terminal_soc"]["tolerance"]),
                    enforce_terminal=False,
                )
                neutral = battery.map_action(0.0, floor, ceiling)
                remaining_hours = max(self.episode_length - int(index), 0) * self.delta_t
                soc_gap = float(battery.soc) - float(battery.initial_soc)
                if remaining_hours > 0.0:
                    if soc_gap >= 0.0:
                        required_kw = soc_gap * battery.capacity_kwh * battery.eta_discharge / remaining_hours
                        required_ratio = required_kw / max(battery.max_discharge_kw, 1e-12)
                    else:
                        required_kw = -(-soc_gap) * battery.capacity_kwh / max(battery.eta_charge * remaining_hours, 1e-12)
                        required_ratio = required_kw / max(battery.max_charge_kw, 1e-12)
                else:
                    required_ratio = 0.0
                result[mg].update({
                    "ess_time_to_terminal": remaining_hours / max(float(self.episode_length), 1.0),
                    "ess_required_avg_power": float(required_ratio),
                    "ess_terminal_pressure": float(abs(required_ratio)),
''',
    marker='"ess_required_avg_power"',
)
replace_once(
    "envs/multi_microgrid_env.py",
    '''    def step(self, actions: Mapping[str, Sequence[float]]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict[str, Any]]:
        if self.dr_enabled:
            return self._step_dr(actions)
        return self._step_legacy(actions)
''',
    '''    def step(self, actions: Mapping[str, Sequence[float]]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict[str, Any]]:
        if self.dr_enabled:
            result = self._step_dr(actions)
            return apply_terminal_constraint_penalty(self, *result)
        return self._step_legacy(actions)
''',
    marker="return apply_terminal_constraint_penalty(self, *result)",
)
replace_once(
    "envs/multi_microgrid_env.py",
    '                enforce_terminal=True,\n',
    '                enforce_terminal=False,\n',
    marker="enforce_terminal=False,",
)

# ---------------------------------------------------------------------------
# Remove the now-obsolete alternate environment/ablation entry points.
# ---------------------------------------------------------------------------
for rel in (
    "configs/cooperative_instantaneous_projection.yaml",
    "envs/instantaneous_projection_env.py",
    "scripts/run_projection_ablation.py",
    "tests/test_projection_ablation.py",
    ".github/workflows/matd3-projection-ablation-short-validation.yml",
):
    path = ROOT / rel
    if path.exists():
        path.unlink()

# Focused CI should no longer reference the deleted ablation test.
workflow = ROOT / ".github/workflows/dr-environment-validation.yml"
if workflow.exists():
    text = workflow.read_text(encoding="utf-8")
    text = text.replace("          tests/test_projection_ablation.py\n", "")
    workflow.write_text(text, encoding="utf-8")

print("instantaneous/P0 refactor applied")
