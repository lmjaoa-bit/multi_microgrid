from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from components.controllable_load import build_controllable_load
from core.microgrid_core import MicrogridCore
from envs.multi_microgrid_env import MultiMicrogridEnv


class InstantaneousMicrogridCore(MicrogridCore):
    """Microgrid core whose ESS mapping never enforces terminal reachability."""

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


class InstantaneousProjectionEnv(MultiMicrogridEnv):
    """Ablation environment with only one-step physical action projection.

    The strong baseline class is deliberately left untouched. This subclass
    removes future/terminal reachability from ESS and heterogeneous DR action
    projection, but preserves instantaneous physical limits, local-first ESS/P2P
    semantics, and the same raw/effective action interfaces.
    """

    def __init__(self, config: str | Path | dict[str, Any], split: str = "train") -> None:
        if isinstance(config, (str, Path)):
            path = Path(config).resolve()
            resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
            resolved["_project_root"] = str(path.parent.parent)
        else:
            resolved = deepcopy(config)
        resolved.setdefault("constraint_projection", {})["mode"] = "instantaneous"
        super().__init__(resolved, split=split)
        self.constraint_projection_mode = "instantaneous"

        # Rebuild only the physical stateful devices that carry reachability
        # semantics. The shared observation/action schemas and projector object
        # stay unchanged, so the Critic still receives the same effective action.
        self.cores = {
            str(cfg["id"]): InstantaneousMicrogridCore(
                dict(cfg), self.delta_t, self.profiles
            )
            for cfg in self.config["microgrids"]
        }
        self.dr_devices = {
            str(cfg["id"]): build_controllable_load(
                cfg["dr"],
                delta_t=self.delta_t,
                episode_length=self.episode_length,
                projection_mode="instantaneous",
            )
            for cfg in self.config["microgrids"]
        } if self.dr_enabled else {}

    @property
    def ctde_spec(self) -> dict[str, Any]:
        result = dict(super().ctde_spec)
        result["constraint_projection_mode"] = self.constraint_projection_mode
        return result

    def _battery_flex(self, index: int) -> dict[str, dict[str, float]]:
        result = {
            mg: {
                "available_charge_ratio": self.cores[mg].battery.available_charge_kw()
                / max(1.0, self.cores[mg].battery.max_charge_kw),
                "available_discharge_ratio": self.cores[mg].battery.available_discharge_kw()
                / max(1.0, self.cores[mg].battery.max_discharge_kw),
            }
            for mg in self.agent_ids
        }
        if self.ess_sharing_control_mode == "single_dispatch":
            for mg in self.agent_ids:
                battery = self.cores[mg].battery
                floor, ceiling = battery.reachable_soc_envelope(
                    index,
                    self.episode_length,
                    tolerance=float(self.config["terminal_soc"]["tolerance"]),
                    enforce_terminal=False,
                )
                neutral = battery.map_action(0.0, floor, ceiling)
                result[mg].update({
                    "feasible_charge_min_ratio": neutral["charge_requested_kw"] / battery.max_charge_kw,
                    "feasible_charge_max_ratio": -neutral["feasible_min_kw"] / battery.max_charge_kw,
                    "feasible_discharge_min_ratio": neutral["discharge_requested_kw"] / battery.max_discharge_kw,
                    "feasible_discharge_max_ratio": neutral["feasible_max_kw"] / battery.max_discharge_kw,
                })
        return result

    def reset(
        self,
        day: str | None = None,
        seed: int | None = None,
        deterministic: bool = False,
        forecast_noise: bool | None = None,
    ):
        observations, info = super().reset(
            day=day,
            seed=seed,
            deterministic=deterministic,
            forecast_noise=forecast_noise,
        )
        info["constraint_projection_mode"] = self.constraint_projection_mode
        return observations, info

    @staticmethod
    def _excess_absolute(value: float, target: float, tolerance: float) -> float:
        return max(0.0, abs(float(value) - float(target)) - max(0.0, float(tolerance)))

    def _empty_terminal_penalty(self) -> dict[str, dict[str, float]]:
        return {
            mg: {
                "ess_equivalent_kwh": 0.0,
                "mill_inventory_violation_t": 0.0,
                "mill_equivalent_kwh": 0.0,
                "hvac_air_violation_c": 0.0,
                "hvac_mass_violation_c": 0.0,
                "hvac_equivalent_kwh": 0.0,
                "ev_soc_shortfall": 0.0,
                "ev_equivalent_kwh": 0.0,
                "dr_equivalent_kwh": 0.0,
                "total_equivalent_kwh": 0.0,
                "total_penalty": 0.0,
            }
            for mg in self.agent_ids
        }

    def _terminal_constraint_penalties(self) -> dict[str, dict[str, float]]:
        details = self._empty_terminal_penalty()
        rate = float(self.config["terminal_soc"]["penalty_per_kwh"])
        soc_tolerance = float(self.config["terminal_soc"]["tolerance"])

        for mg in self.agent_ids:
            battery = self.cores[mg].battery
            soc_excess = self._excess_absolute(
                float(battery.soc), float(battery.initial_soc), soc_tolerance
            )
            ess_equivalent = soc_excess * float(battery.capacity_kwh)
            details[mg]["ess_equivalent_kwh"] = float(ess_equivalent)

            if self.dr_enabled:
                device = self.dr_devices[mg]
                kind = str(self.cfg_by_id[mg]["dr"]["type"])
                if kind == "industrial_mill":
                    violation_t = self._excess_absolute(
                        float(device.silo_t),
                        float(device.initial_silo_t),
                        float(device.terminal_tolerance_t),
                    )
                    equivalent = (
                        violation_t * float(device.rated_power_kw)
                        / max(float(device.production_tph), 1e-12)
                    )
                    details[mg]["mill_inventory_violation_t"] = float(violation_t)
                    details[mg]["mill_equivalent_kwh"] = float(equivalent)
                    details[mg]["dr_equivalent_kwh"] = float(equivalent)
                    device.terminal_infeasible = bool(violation_t > 1e-9)
                elif kind == "commercial_hvac":
                    air_violation = self._excess_absolute(
                        float(device.t_air),
                        float(device.initial_t_air),
                        float(device.terminal_air_tolerance),
                    )
                    mass_violation = self._excess_absolute(
                        float(device.t_mass),
                        float(device.initial_t_mass),
                        float(device.terminal_mass_tolerance),
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
                    equivalent = (
                        air_violation * float(device.c_air) / max(air_cop, 1e-12)
                        + mass_violation * float(device.c_mass) / max(mass_cop, 1e-12)
                    )
                    details[mg]["hvac_air_violation_c"] = float(air_violation)
                    details[mg]["hvac_mass_violation_c"] = float(mass_violation)
                    details[mg]["hvac_equivalent_kwh"] = float(equivalent)
                    details[mg]["dr_equivalent_kwh"] = float(equivalent)
                    device.terminal_thermal_infeasible = bool(
                        air_violation > 1e-9 or mass_violation > 1e-9
                    )
                elif kind == "ev_charging":
                    shortfall = max(
                        0.0, float(device.target_soc) - float(device.soc)
                    )
                    equivalent = (
                        shortfall * float(device.capacity_kwh)
                        / max(float(device.eta), 1e-12)
                    )
                    details[mg]["ev_soc_shortfall"] = float(shortfall)
                    details[mg]["ev_equivalent_kwh"] = float(equivalent)
                    details[mg]["dr_equivalent_kwh"] = float(equivalent)
                    device.target_missed = bool(shortfall > 1e-9)

            total_equivalent = (
                details[mg]["ess_equivalent_kwh"]
                + details[mg]["dr_equivalent_kwh"]
            )
            details[mg]["total_equivalent_kwh"] = float(total_equivalent)
            details[mg]["total_penalty"] = float(total_equivalent * rate)
        return details

    def step(
        self, actions: Mapping[str, Sequence[float]]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict[str, Any]]:
        observations, rewards, done, info = super().step(actions)
        info["constraint_projection_mode"] = self.constraint_projection_mode
        details = self._empty_terminal_penalty()

        if done:
            details = self._terminal_constraint_penalties()
            terminal_penalties = {
                mg: float(details[mg]["total_penalty"]) for mg in self.agent_ids
            }
            for mg in self.agent_ids:
                penalty = terminal_penalties[mg]
                cost = info["costs"][mg]
                cost["terminal_constraint_penalty"] = penalty
                # Backward-compatible aggregate field used by the shared metrics
                # collector and training reward decomposition.
                cost["terminal_soc_penalty"] = penalty
                cost["physical_profit"] = float(cost["physical_profit"]) - penalty
                cost["individual_accounting_profit"] = (
                    float(cost["individual_accounting_profit"]) - penalty
                )

                kind = str(self.cfg_by_id[mg]["dr"]["type"])
                if kind == "industrial_mill":
                    info["dr_plans"][mg]["terminal_infeasible"] = bool(
                        details[mg]["mill_inventory_violation_t"] > 1e-9
                    )
                elif kind == "commercial_hvac":
                    info["dr_plans"][mg]["terminal_thermal_infeasible"] = bool(
                        details[mg]["hvac_air_violation_c"] > 1e-9
                        or details[mg]["hvac_mass_violation_c"] > 1e-9
                    )
                elif kind == "ev_charging":
                    info["dr_plans"][mg]["target_missed"] = bool(
                        details[mg]["ev_soc_shortfall"] > 1e-9
                    )

            team_profit = float(sum(
                info["costs"][mg]["physical_profit"] for mg in self.agent_ids
            ))
            team_reward = team_profit / float(self.accounting.reward_scale)
            rewards = {mg: float(team_reward) for mg in self.agent_ids}
            info["team_profit"] = team_profit
            info["team_reward"] = float(team_reward)
            info["terminal_soc_penalty"] = terminal_penalties
        else:
            for mg in self.agent_ids:
                info["costs"][mg]["terminal_constraint_penalty"] = 0.0

        info["terminal_constraint_penalty"] = details
        return observations, rewards, done, info
