from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from core.action_codec import ActionCodec
from core.config_validation import validate_config
from core.cost_accounting import CostAccounting
from core.data_manager import DataManager
from core.forecast import SharedForecastProvider
from core.global_state import GlobalStateBuilder
from core.grid_settlement import ExternalGrid
from core.microgrid_core import MicrogridCore
from core.observation import ObservationBuilder
from core.profiles import HeterogeneousProfileModel
from core.types import CoordinationHistory
from envs.cooperative_energy_sharing import CooperativeEnergySharing


class MultiMicrogridEnv:
    """Three-agent cooperative RES/ESS/load environment with CTDE state."""

    def __init__(self, config: str | Path | dict[str, Any], split: str = "train") -> None:
        if isinstance(config, (str, Path)):
            path = Path(config).resolve()
            self.config = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.config["_project_root"] = str(path.parent.parent)
        else:
            self.config = dict(config)
            self.config.setdefault("_project_root", str(Path.cwd()))
        validate_config(self.config)
        self.split = split
        self.delta_t = float(self.config["time"].get("delta_t", 1.0))
        self.episode_length = int(self.config["time"].get("episode_length", 24))
        self.seed = int(self.config.get("env", {}).get("seed", 42))
        self.mode = str(self.config.get("env", {}).get("mode", "cooperative"))
        sharing_cfg = self.config.get("ess_sharing", {})
        self.ess_sharing_control_mode = str(sharing_cfg.get("control_mode", "residual_auto"))
        self.ess_sharing_observation_mode = str(sharing_cfg.get("observation_mode", "local"))
        self.agent_ids = [str(mg["id"]) for mg in self.config["microgrids"]]
        self.data = DataManager(self.config, split=split, seed=self.seed)
        normalization = self.data if split == "train" else DataManager(self.config, split="train", seed=self.seed)
        self.profiles = HeterogeneousProfileModel(self.config, normalization.df)
        self.profiles.cache_frame(self.data.df)
        self.forecasts = SharedForecastProvider(self.config, self.profiles, seed=self.seed)
        self.cores = {str(cfg["id"]): MicrogridCore(dict(cfg), self.delta_t, self.profiles)
                      for cfg in self.config["microgrids"]}
        self.grids = {str(cfg["id"]): ExternalGrid(float(cfg["grid"]["import_limit_kw"]),
                                                   float(cfg["grid"]["export_limit_kw"]))
                      for cfg in self.config["microgrids"]}
        p2p_cfg = self.config["p2p"]
        p2p_enabled = bool(p2p_cfg.get("enabled", True)) and self.mode == "cooperative"
        self.allow_grid_resale = bool(p2p_cfg["allow_grid_resale"])
        self.energy_pool = CooperativeEnergySharing(
            self.delta_t, p2p_enabled, float(p2p_cfg["pair_capacity_kw"]),
        )
        cost = self.config["cost"]
        self.accounting = CostAccounting(
            unmet_load_cost_per_kwh=float(cost["unmet_load_cost_per_kwh"]),
            curtailment_cost_per_kwh=float(cost["renewable_curtailment_cost_per_kwh"]),
            reward_scale=float(cost["reward_scale"]),
        )
        self.obs_builder = ObservationBuilder(self.config, self.profiles, self.forecasts, self.delta_t)
        self.global_state_builder = GlobalStateBuilder(self.config, self.profiles, self.forecasts, self.delta_t)
        self.action_codec = ActionCodec(self.ess_sharing_control_mode)
        self.t = 0
        self.episode_df = None
        self.episode_context_df = None
        self._episode_exogenous: dict[str, dict[str, np.ndarray]] | None = None
        self.history: CoordinationHistory | None = None
        self._last_observations: dict[str, np.ndarray] | None = None
        self._last_global_state: np.ndarray | None = None
        self._terminated = False

    @property
    def action_dim(self) -> int:
        return self.action_codec.action_dim

    @property
    def observation_dims(self) -> dict[str, int]:
        return {mg: self.obs_builder.obs_dim(mg) for mg in self.agent_ids}

    @property
    def global_state_dim(self) -> int:
        return self.global_state_builder.dim

    @property
    def ctde_spec(self) -> dict[str, Any]:
        return {"agent_ids": list(self.agent_ids), "action_dim": self.action_dim,
                "observation_dims": self.observation_dims,
                "global_state_dim": self.global_state_dim,
                "global_state_schema": self.global_state_builder.schema()}

    def _empty_history(self) -> CoordinationHistory:
        return {"total_p2p_kwh": 0.0,
                "total_buy_demand_kwh": 0.0,
                "total_sell_offer_kwh": 0.0,
                "clearing_ratio": 0.0,
                "by_microgrid": {mg: {
                    "p2p_net_kwh": 0.0, "grid_net_kwh": 0.0,
                    "ess_p2p_offer_kwh": 0.0,
                    "ess_p2p_accepted_kwh": 0.0,
                    "ess_p2p_unmatched_kwh": 0.0,
                }
                                 for mg in self.agent_ids}}

    def _exogenous_at(self, index: int) -> dict[str, dict[str, float]]:
        if self._episode_exogenous is None:
            raise RuntimeError("Call reset() first")
        bounded = min(max(0, int(index)), len(self.episode_context_df) - 1)
        return {
            mg: {name: float(values[bounded]) for name, values in self._episode_exogenous[mg].items()}
            for mg in self.agent_ids
        }

    def _observations(
        self,
        exogenous_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, np.ndarray]:
        if self.episode_context_df is None or self.history is None:
            raise RuntimeError("Call reset() first")
        index = min(self.t, len(self.episode_context_df) - 1)
        row = self.episode_context_df.iloc[index]
        current = exogenous_by_microgrid or self._exogenous_at(index)
        soc_by_microgrid = {mg: float(self.cores[mg].battery.soc) for mg in self.agent_ids}
        battery_flex = {
            mg: {
                "available_charge_ratio": self.cores[mg].battery.available_charge_kw()
                / max(1.0, self.cores[mg].battery.max_charge_kw),
                "available_discharge_ratio": self.cores[mg].battery.available_discharge_kw()
                / max(1.0, self.cores[mg].battery.max_discharge_kw),
            }
            for mg in self.agent_ids
        }
        return {
            mg: self.obs_builder.build(
                mg, row, index, float(self.cores[mg].battery.soc),
                self.history, exogenous=current[mg],
                soc_by_microgrid=soc_by_microgrid,
                battery_flex_by_microgrid=battery_flex,
            )
            for mg in self.agent_ids
        }

    def _build_global_state(
        self,
        exogenous_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
    ) -> np.ndarray:
        if self.episode_context_df is None or self.history is None:
            raise RuntimeError("Call reset() first")
        index = min(self.t, len(self.episode_context_df) - 1)
        current = exogenous_by_microgrid or self._exogenous_at(index)
        return self.global_state_builder.build(
            self.episode_context_df.iloc[index], index,
            self.cores, self.history,
            exogenous_by_microgrid=current
        )

    def global_state(self) -> np.ndarray:
        if self._last_global_state is None:
            raise RuntimeError("Call reset() first")
        return self._last_global_state.copy()

    def reset(self, day: str | None = None, seed: int | None = None,
              deterministic: bool = False) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.data.reseed(seed)
        context = self.data.get_episode_context(day, self.obs_builder.future_h)
        self.episode_context_df = context
        self.episode_df = context.iloc[:self.episode_length].reset_index(drop=True).copy()
        self._episode_exogenous = {
            mg: self.profiles.exogenous_series(mg, context) for mg in self.agent_ids
        }
        self.forecasts.begin_episode(
            context,
            seed=seed,
            deterministic=deterministic,
            exogenous_by_microgrid=self._episode_exogenous,
        )
        for core in self.cores.values():
            core.reset()
        self.t = 0
        self._terminated = False
        self.history = self._empty_history()
        current = self._exogenous_at(0)
        self._last_observations = self._observations(current)
        self._last_global_state = self._build_global_state(current)
        return {mg: value.copy() for mg, value in self._last_observations.items()}, {
            "global_state": self.global_state(), "day": str(self.episode_df.iloc[0]["timestamp"]),
            "ctde_spec": self.ctde_spec,
        }

    def _soc_potential(self, t: int) -> float:
        """Potential function for terminal SOC shaping."""
        rate = float(self.config["terminal_soc"]["penalty_per_kwh"])
        tolerance = float(self.config["terminal_soc"]["tolerance"])
        progress = (t) / float(self.episode_length)
        potential = 0.0
        for mg in self.agent_ids:
            battery = self.cores[mg].battery
            diff = max(0.0, abs(float(battery.soc) - float(battery.initial_soc)) - tolerance)
            potential -= progress * diff * battery.capacity_kwh * rate
        return potential

    def step(self, actions: Mapping[str, Sequence[float]]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict[str, Any]]:
        if self.episode_df is None or self._terminated:
            raise RuntimeError("Reset the environment before stepping")
        missing = set(self.agent_ids) - set(actions)
        if missing:
            raise KeyError(f"Missing actions for {sorted(missing)}")
        row = self.episode_df.iloc[self.t]
        current = self._exogenous_at(self.t)
        decoded = {mg: self.action_codec.decode(actions[mg]) for mg in self.agent_ids}
        tolerance = float(self.config["terminal_soc"]["tolerance"])

        # 1. Plan local battery dispatch with terminal-SOC reachability constraints.
        plans = {
            mg: self.cores[mg].plan_local(
                row, decoded[mg],
                t=self.t,
                episode_length=self.episode_length,
                tolerance=tolerance,
                exogenous=current[mg],
            )
            for mg in self.agent_ids
        }

        # 2. Build source-aware P2P offers and load/charging demands. Renewable
        # energy serves local load first; a requested battery charge may retain
        # local renewable surplus before the remainder is offered to P2P.
        renewable_offers: dict[str, float] = {}
        ess_offers: dict[str, float] = {}
        load_demands: dict[str, float] = {}
        charge_demands: dict[str, float] = {}
        for mg in self.agent_ids:
            plan = plans[mg]
            load = float(plan["load_kw"])
            renewable = float(plan["renewable_kw"])
            requested_charge = float(plan["battery_charge_kw"])
            requested_discharge = float(plan["battery_discharge_kw"])
            load_after_renewable = max(0.0, load - renewable)
            renewable_surplus = max(0.0, renewable - load)
            local_discharge = min(requested_discharge, load_after_renewable)
            flexible_discharge = max(0.0, requested_discharge - local_discharge)
            renewable_to_battery = min(requested_charge, renewable_surplus)
            renewable_offer = max(0.0, renewable_surplus - renewable_to_battery)
            charge_demand = max(0.0, requested_charge - renewable_to_battery)
            load_demand = max(0.0, load_after_renewable - local_discharge)
            offer_ratio = float(decoded[mg].get("sharing_offer_ratio", 1.0))
            ess_offer = flexible_discharge * offer_ratio
            plans[mg].update({
                "renewable_to_local_load_kw": min(load, renewable),
                "renewable_to_battery_kw": renewable_to_battery,
                "renewable_p2p_offer_kw": renewable_offer,
                "local_load_deficit_before_battery_kw": load_after_renewable,
                "ess_to_local_requested_kw": local_discharge,
                "ess_flexible_discharge_kw": flexible_discharge,
                "sharing_offer_ratio": offer_ratio,
                "ess_p2p_offer_kw": ess_offer,
                "p2p_load_demand_kw": load_demand,
                "p2p_charge_demand_kw": charge_demand,
            })
            renewable_offers[mg] = renewable_offer
            ess_offers[mg] = ess_offer
            load_demands[mg] = load_demand
            charge_demands[mg] = charge_demand

        sharing = self.energy_pool.allocate_structured(
            renewable_offers, ess_offers, load_demands, charge_demands,
            float(row["sell_price_per_kwh"]), float(row["buy_price_per_kwh"]),
        )

        # Residual-auto is the permanent one-dimensional baseline: unmatched
        # flexible discharge is still physically dispatched and may reach the
        # external grid. Offer-controlled variants commit only local plus accepted
        # ESS discharge, so an unmatched offer remains stored in the battery.
        total_buy_demand = sum(load_demands.values()) + sum(charge_demands.values())
        total_sell_offer = sum(renewable_offers.values()) + sum(ess_offers.values())
        for mg in self.agent_ids:
            accepted = float(sharing.by_microgrid[mg]["ess_p2p_sell_kw"])
            local = float(plans[mg]["ess_to_local_requested_kw"])
            requested = float(plans[mg]["battery_discharge_kw"])
            if self.ess_sharing_control_mode == "residual_auto":
                actual_discharge = requested
            else:
                actual_discharge = local + accepted
            plans[mg]["battery_discharge_requested_kw"] = requested
            plans[mg]["battery_discharge_kw"] = actual_discharge
            plans[mg]["ess_to_local_kw"] = local
            plans[mg]["ess_to_p2p_kw"] = accepted
            plans[mg]["ess_p2p_unmatched_kw"] = max(
                0.0, float(plans[mg]["ess_p2p_offer_kw"]) - accepted
            )
            peer_demand = sum(
                load_demands[other] + charge_demands[other]
                for other in self.agent_ids if other != mg
            )
            plans[mg]["sharing_action_masked"] = bool(
                not self.energy_pool.enabled
                or float(plans[mg]["ess_flexible_discharge_kw"]) <= 1e-9
                or peer_demand <= 1e-9
            )

        # 3. External grid settlement
        settlements = {}
        for mg in self.agent_ids:
            residual_net = (
                float(plans[mg]["renewable_kw"])
                + float(plans[mg]["battery_discharge_kw"])
                - float(plans[mg]["load_kw"])
                - float(plans[mg]["battery_charge_kw"])
                + float(sharing.by_microgrid[mg]["p2p_net_kw"])
            )
            settlements[mg] = self.grids[mg].settle(residual_net)
            if abs(settlements[mg]["power_balance_residual_kw"]) > 1e-7:
                raise RuntimeError(f"{mg}: external-grid power balance failed")
            if (not self.allow_grid_resale
                    and settlements[mg]["grid_import_kw"] > 1e-8
                    and sharing.by_microgrid[mg]["p2p_sell_kw"] > 1e-8):
                raise RuntimeError(f"{mg}: grid imports cannot be resold through P2P")

        # 4. Commit battery power
        devices = {mg: self.cores[mg].commit(plans[mg]) for mg in self.agent_ids}

        # 5. Cost accounting
        costs = {}
        for mg in self.agent_ids:
            costs[mg] = self.accounting.calculate(
                float(row["buy_price_per_kwh"]), float(row["sell_price_per_kwh"]), settlements[mg],
                devices[mg]["degradation_cost"], self.delta_t,
            )
            transfer = float(sharing.by_microgrid[mg]["p2p_transfer_payment"])
            costs[mg]["p2p_transfer_payment"] = transfer
            costs[mg]["individual_accounting_profit"] = float(
                costs[mg]["physical_profit"] + transfer
            )

        # 6. Terminal SOC penalty and shaping
        potential_before = self._soc_potential(self.t)
        potential_after = self._soc_potential(self.t + 1)
        shaping_reward = potential_after - potential_before

        terminal_penalties = {mg: 0.0 for mg in self.agent_ids}
        terminal = self.t + 1 >= len(self.episode_df)
        if terminal:
            terminal_cfg = self.config["terminal_soc"]
            rate = float(terminal_cfg["penalty_per_kwh"])
            for mg in self.agent_ids:
                battery = self.cores[mg].battery
                deviation = max(0.0, abs(float(battery.soc) - battery.initial_soc) - tolerance)
                penalty = deviation * battery.capacity_kwh * rate
                terminal_penalties[mg] = penalty
                costs[mg]["terminal_soc_penalty"] = penalty
                costs[mg]["physical_profit"] -= penalty
                costs[mg]["individual_accounting_profit"] -= penalty
        else:
            for mg in self.agent_ids:
                costs[mg]["terminal_soc_penalty"] = 0.0

        team_profit = sum(costs[mg]["physical_profit"] for mg in self.agent_ids)
        team_reward = (team_profit + shaping_reward) / self.accounting.reward_scale
        rewards = {mg: float(team_reward) for mg in self.agent_ids}

        self.history = {
            "total_p2p_kwh": sharing.total_p2p_kwh,
            "total_buy_demand_kwh": total_buy_demand * self.delta_t,
            "total_sell_offer_kwh": total_sell_offer * self.delta_t,
            "clearing_ratio": (
                sharing.total_p2p_kwh / max(total_sell_offer * self.delta_t, 1e-9)
                if total_sell_offer > 1e-9 else 0.0
            ),
            "by_microgrid": {mg: {
                "p2p_net_kwh": sharing.by_microgrid[mg]["p2p_net_kw"] * self.delta_t,
                "grid_net_kwh": (settlements[mg]["grid_import_kw"] - settlements[mg]["grid_export_kw"]) * self.delta_t,
                "ess_p2p_offer_kwh": sharing.by_microgrid[mg]["ess_p2p_offer_kw"] * self.delta_t,
                "ess_p2p_accepted_kwh": sharing.by_microgrid[mg]["ess_p2p_sell_kw"] * self.delta_t,
                "ess_p2p_unmatched_kwh": sharing.by_microgrid[mg]["ess_p2p_unmatched_kw"] * self.delta_t,
            } for mg in self.agent_ids},
        }

        info = {
            "timestamp": str(row["timestamp"]), "mode": self.mode,
            "decoded_actions": decoded, "plans": plans,
            "p2p": {"by_microgrid": sharing.by_microgrid, "pair_flows_kwh": sharing.pair_flows_kwh,
                    "total_p2p_kwh": sharing.total_p2p_kwh,
                    "transfer_price_per_kwh": sharing.transfer_price_per_kwh,
                    "transfer_payment_sum": sum(x["p2p_transfer_payment"] for x in sharing.by_microgrid.values())},
            "settlements": settlements, "devices": devices, "costs": costs,
            "team_profit": float(team_profit), "team_reward": float(team_reward),
            "shaping_reward": float(shaping_reward),
            "terminal_soc_penalty": terminal_penalties,
        }

        self.t += 1
        self._terminated = terminal
        next_exogenous = self._exogenous_at(self.t)
        self._last_observations = self._observations(next_exogenous)
        self._last_global_state = self._build_global_state(next_exogenous)
        info["global_state_after"] = self.global_state()
        return {mg: value.copy() for mg, value in self._last_observations.items()}, rewards, terminal, info
