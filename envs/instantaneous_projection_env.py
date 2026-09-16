from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from components.commercial_hvac import CommercialHVACDR
from components.ev_charging import EVChargingDR
from components.industrial_mill import IndustrialMillDR
from core.microgrid_core import MicrogridCore
from envs.multi_microgrid_env import MultiMicrogridEnv


class InstantaneousMicrogridCore(MicrogridCore):
    """ESS core that keeps current SOC/power safety but drops terminal reachability."""

    def plan_local(
        self,
        row,
        action: Mapping[str, Any],
        t: int = 0,
        episode_length: int = 24,
        tolerance: float = 0.03,
        exogenous: Mapping[str, float] | None = None,
        enforce_terminal: bool = True,
    ) -> dict[str, float]:
        return super().plan_local(
            row,
            action,
            t=t,
            episode_length=episode_length,
            tolerance=tolerance,
            exogenous=exogenous,
            enforce_terminal=False,
        )


class InstantaneousIndustrialMillDR(IndustrialMillDR):
    """Mill projector constrained only by the next silo state."""

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
            return midpoint, midpoint
        return float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0))


class InstantaneousCommercialHVACDR(CommercialHVACDR):
    """HVAC projector constrained only by next-step comfort and power limits."""

    def feasible_bounds(self, t: int) -> tuple[float, float]:
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


class InstantaneousEVChargingDR(EVChargingDR):
    """EV projector with availability/SOC/power limits but no forced deadline charging."""

    def feasible_bounds(self, t: int) -> tuple[float, float]:
        if self._remaining_steps(t) <= 0:
            return (0.0, 0.0)
        headroom_kwh = max(0.0, (self.max_soc - self.soc) * self.capacity_kwh)
        if headroom_kwh <= 1e-10:
            return (0.0, 0.0)
        maximum = min(
            self.max_charge_kw,
            headroom_kwh / (self.eta * self.delta_t),
        )
        return 0.0, float(maximum)

    def state(self, t: int) -> dict[str, float]:
        result = super().state(t)
        if self._hour(t) >= self.departure_hour and self.soc < self.target_soc - 1e-10:
            self.target_missed = True
            result["target_missed"] = 1.0
        return result


class InstantaneousProjectionEnv(MultiMicrogridEnv):
    """Projection ablation that removes only cross-time reachability assistance.

    The canonical ``MultiMicrogridEnv`` remains the strong baseline. This subclass
    retains instantaneous physical bounds and the same action/state interfaces,
    but removes terminal reachability from ESS, mill, HVAC and EV projectors.
    Terminal target violations are converted to equivalent-energy penalties only
    at the end of the day so the policy must learn the long-horizon requirement.
    """

    projection_mode = "instantaneous"

    def __init__(self, config: str | Path | dict[str, Any], split: str = "train") -> None:
        if isinstance(config, (str, Path)):
            path = Path(config).resolve()
            resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
            resolved["_project_root"] = str(path.parent.parent)
        else:
            resolved = deepcopy(config)
        requested_mode = str(
            resolved.get("constraint_projection", {}).get("mode", "strong")
        )
        if requested_mode != "instantaneous":
            raise ValueError(
                "InstantaneousProjectionEnv requires constraint_projection.mode=instantaneous"
            )

        super().__init__(resolved, split=split)
        self.constraint_projection_mode = "instantaneous"
        self._terminal_target_cfg = deepcopy(self.config["terminal_soc"])

        # The parent environment/global-state builder derives ESS reachability from
        # terminal_soc.mode. Disabling that internal hard gate makes their saved
        # feasible-action channels instantaneous while the original target and
        # penalty scale remain preserved in _terminal_target_cfg/config snapshot.
        self.config["terminal_soc"]["mode"] = "none"

        self.cores = {
            str(cfg["id"]): InstantaneousMicrogridCore(
                dict(cfg), self.delta_t, self.profiles
            )
            for cfg in self.config["microgrids"]
        }
        if self.dr_enabled:
            self.dr_devices = {
                str(cfg["id"]): self._build_instantaneous_dr(dict(cfg["dr"]))
                for cfg in self.config["microgrids"]
            }

    def _build_instantaneous_dr(self, config: dict[str, Any]):
        kind = str(config.get("type", ""))
        kwargs = {"delta_t": self.delta_t, "episode_length": self.episode_length}
        if kind == "industrial_mill":
            return InstantaneousIndustrialMillDR(config, **kwargs)
        if kind == "commercial_hvac":
            return InstantaneousCommercialHVACDR(config, **kwargs)
        if kind == "ev_charging":
            return InstantaneousEVChargingDR(config, **kwargs)
        raise ValueError(f"Unsupported controllable load type: {kind!r}")

    def _terminal_constraint_breakdown(self, mg: str) -> dict[str, float]:
        battery = self.cores[mg].battery
        tolerance = float(self._terminal_target_cfg["tolerance"])
        ess_equivalent = max(
            0.0,
            abs(float(battery.soc) - float(battery.initial_soc)) - tolerance,
        ) * float(battery.capacity_kwh)

        mill_equivalent = 0.0
        hvac_equivalent = 0.0
        ev_equivalent = 0.0
        device = self.dr_devices.get(mg)
        if isinstance(device, InstantaneousIndustrialMillDR):
            deviation_t = max(
                0.0,
                abs(float(device.silo_t) - float(device.initial_silo_t))
                - float(device.terminal_tolerance_t),
            )
            mill_equivalent = (
                deviation_t / max(float(device.production_tph), 1e-12)
                * float(device.rated_power_kw)
            )
        elif isinstance(device, InstantaneousCommercialHVACDR):
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
            air_cop = (
                float(device.cop_cool)
                if float(device.t_air) > float(device.initial_t_air)
                else float(device.cop_heat)
            )
            mass_cop = (
                float(device.cop_cool)
                if float(device.t_mass) > float(device.initial_t_mass)
                else float(device.cop_heat)
            )
            hvac_equivalent = (
                air_deviation * float(device.c_air) / max(air_cop, 1e-12)
                + mass_deviation * float(device.c_mass) / max(mass_cop, 1e-12)
            )
        elif isinstance(device, InstantaneousEVChargingDR):
            ev_equivalent = max(
                0.0,
                float(device.target_soc) - float(device.soc),
            ) * float(device.capacity_kwh)

        equivalent = ess_equivalent + mill_equivalent + hvac_equivalent + ev_equivalent
        rate = float(self._terminal_target_cfg["penalty_per_kwh"])
        return {
            "ess_equivalent_kwh": float(ess_equivalent),
            "mill_equivalent_kwh": float(mill_equivalent),
            "hvac_equivalent_kwh": float(hvac_equivalent),
            "ev_equivalent_kwh": float(ev_equivalent),
            "total_equivalent_kwh": float(equivalent),
            "penalty_rate_per_equivalent_kwh": rate,
            "total_penalty": float(equivalent * rate),
        }

    def step(
        self,
        actions: Mapping[str, Sequence[float]],
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict[str, Any]]:
        observations, rewards, done, info = super().step(actions)
        info["constraint_projection_mode"] = self.constraint_projection_mode
        for mg in self.agent_ids:
            info["costs"][mg]["terminal_constraint_penalty"] = 0.0

        if not done:
            info["terminal_constraint_penalty"] = {
                mg: {
                    "ess_equivalent_kwh": 0.0,
                    "mill_equivalent_kwh": 0.0,
                    "hvac_equivalent_kwh": 0.0,
                    "ev_equivalent_kwh": 0.0,
                    "total_equivalent_kwh": 0.0,
                    "penalty_rate_per_equivalent_kwh": float(
                        self._terminal_target_cfg["penalty_per_kwh"]
                    ),
                    "total_penalty": 0.0,
                }
                for mg in self.agent_ids
            }
            return observations, rewards, done, info

        breakdown = {
            mg: self._terminal_constraint_breakdown(mg) for mg in self.agent_ids
        }
        terminal_penalties: dict[str, float] = {}
        for mg in self.agent_ids:
            penalty = float(breakdown[mg]["total_penalty"])
            terminal_penalties[mg] = penalty
            info["costs"][mg]["terminal_constraint_penalty"] = penalty
            # Compatibility alias: existing economic/training summaries already
            # collect terminal_soc_penalty. In this isolated ablation it represents
            # the complete terminal-constraint penalty; the detailed breakdown is
            # retained separately below.
            info["costs"][mg]["terminal_soc_penalty"] = penalty
            info["costs"][mg]["physical_profit"] -= penalty
            info["costs"][mg]["individual_accounting_profit"] -= penalty

        team_profit = float(sum(
            info["costs"][mg]["physical_profit"] for mg in self.agent_ids
        ))
        team_reward = team_profit / float(self.accounting.reward_scale)
        rewards = {mg: float(team_reward) for mg in self.agent_ids}
        info["team_profit"] = team_profit
        info["team_reward"] = float(team_reward)
        info["terminal_soc_penalty"] = terminal_penalties
        info["terminal_constraint_penalty"] = breakdown
        info["terminal_constraint_penalty_total"] = float(sum(terminal_penalties.values()))
        return observations, rewards, done, info
