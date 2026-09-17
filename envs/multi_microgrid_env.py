from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from components.controllable_load import build_controllable_load
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
from core.tariffs import prices_from_row
from core.terminal_constraints import apply_terminal_constraint_penalty
from envs.cooperative_energy_sharing import CooperativeEnergySharing


class MultiMicrogridEnv:
    """Three-agent cooperative RES/ESS/heterogeneous-DR environment with CTDE state."""

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
        self.action_design = str(self.config.get("action_design", "single_dispatch"))
        self.dr_enabled = self.action_design == "single_dispatch_dr"
        sharing_cfg = self.config.get("ess_sharing", {})
        self.ess_sharing_control_mode = str(sharing_cfg.get("control_mode", "single_dispatch"))
        self.ess_sharing_observation_mode = str(sharing_cfg.get("observation_mode", "cross_microgrid"))
        self.ess_p2p_enabled = bool(sharing_cfg.get("ess_p2p_enabled", True))
        self.ess_offer_projection = str(sharing_cfg.get("offer_projection", "peer_demand_cap"))
        self.agent_ids = [str(mg["id"]) for mg in self.config["microgrids"]]
        self.cfg_by_id = {str(mg["id"]): mg for mg in self.config["microgrids"]}

        self.data = DataManager(self.config, split=split, seed=self.seed)
        normalization = self.data if split == "train" else DataManager(
            self.config, split="train", seed=self.seed
        )
        self.profiles = HeterogeneousProfileModel(self.config, normalization.df)
        self.profiles.cache_frame(self.data.df)
        self.forecasts = SharedForecastProvider(self.config, self.profiles, seed=self.seed)
        self.cores = {
            str(cfg["id"]): MicrogridCore(dict(cfg), self.delta_t, self.profiles)
            for cfg in self.config["microgrids"]
        }
        self.dr_devices = {
            str(cfg["id"]): build_controllable_load(
                cfg["dr"], delta_t=self.delta_t, episode_length=self.episode_length
            )
            for cfg in self.config["microgrids"]
        } if self.dr_enabled else {}
        self.grids = {
            str(cfg["id"]): ExternalGrid(
                float(cfg["grid"]["import_limit_kw"]),
                float(cfg["grid"]["export_limit_kw"]),
            )
            for cfg in self.config["microgrids"]
        }
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
        self.obs_builder = ObservationBuilder(
            self.config, self.profiles, self.forecasts, self.delta_t
        )
        self.global_state_builder = GlobalStateBuilder(
            self.config, self.profiles, self.forecasts, self.delta_t
        )
        self.action_codec = ActionCodec(self.action_design)
        if self.dr_enabled:
            from core.joint_action_projection import JointActionProjector
            self.action_projector = JointActionProjector(
                self.config,
                self.global_state_builder.schema(),
                self.global_state_builder.norm,
                self.global_state_builder.price_norm,
            )
        else:
            from core.single_dispatch import SingleDispatchProjector
            self.action_projector = SingleDispatchProjector(
                self.config,
                self.global_state_builder.schema(),
                self.global_state_builder.norm,
                self.global_state_builder.price_norm,
            )
        # Backward-compatible name used by the legacy single-dispatch tests.
        self.dispatch_projector = self.action_projector

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
        return {
            "agent_ids": list(self.agent_ids),
            "action_dim": self.action_dim,
            "observation_dims": self.observation_dims,
            "global_state_dim": self.global_state_dim,
            "global_state_schema": self.global_state_builder.schema(),
            "action_design": self.action_design,
        }

    def _empty_history(self) -> CoordinationHistory:
        return {
            "total_p2p_kwh": 0.0,
            "total_buy_demand_kwh": 0.0,
            "total_sell_offer_kwh": 0.0,
            "clearing_ratio": 0.0,
            "by_microgrid": {
                mg: {
                    "p2p_net_kwh": 0.0,
                    "grid_net_kwh": 0.0,
                    "ess_p2p_offer_kwh": 0.0,
                    "ess_p2p_accepted_kwh": 0.0,
                    "ess_p2p_unmatched_kwh": 0.0,
                }
                for mg in self.agent_ids
            },
        }

    def _exogenous_at(self, index: int) -> dict[str, dict[str, float]]:
        if self._episode_exogenous is None or self.episode_context_df is None:
            raise RuntimeError("Call reset() first")
        bounded = min(max(0, int(index)), len(self.episode_context_df) - 1)
        return {
            mg: {name: float(values[bounded]) for name, values in self._episode_exogenous[mg].items()}
            for mg in self.agent_ids
        }

    def _dr_states(self, index: int) -> dict[str, dict[str, float]]:
        if not self.dr_enabled:
            return {}
        bounded = min(max(0, int(index)), self.episode_length)
        return {mg: self.dr_devices[mg].state(bounded) for mg in self.agent_ids}

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
                    "feasible_charge_min_ratio": neutral["charge_requested_kw"] / battery.max_charge_kw,
                    "feasible_charge_max_ratio": -neutral["feasible_min_kw"] / battery.max_charge_kw,
                    "feasible_discharge_min_ratio": neutral["discharge_requested_kw"] / battery.max_discharge_kw,
                    "feasible_discharge_max_ratio": neutral["feasible_max_kw"] / battery.max_discharge_kw,
                })
        return result

    def _observations(
        self,
        exogenous_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
        dr_state_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, np.ndarray]:
        if self.episode_context_df is None or self.history is None:
            raise RuntimeError("Call reset() first")
        index = min(self.t, len(self.episode_context_df) - 1)
        row = self.episode_context_df.iloc[index]
        current = exogenous_by_microgrid or self._exogenous_at(index)
        dr_states = dr_state_by_microgrid if dr_state_by_microgrid is not None else self._dr_states(index)
        soc_by_microgrid = {mg: float(self.cores[mg].battery.soc) for mg in self.agent_ids}
        battery_flex = self._battery_flex(index)
        return {
            mg: self.obs_builder.build(
                mg,
                row,
                index,
                float(self.cores[mg].battery.soc),
                self.history,
                exogenous=current[mg],
                soc_by_microgrid=soc_by_microgrid,
                battery_flex_by_microgrid=battery_flex,
                dr_state_by_microgrid=dr_states,
            )
            for mg in self.agent_ids
        }

    def _build_global_state(
        self,
        exogenous_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
        dr_state_by_microgrid: Mapping[str, Mapping[str, float]] | None = None,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        if self.episode_context_df is None or self.history is None:
            raise RuntimeError("Call reset() first")
        index = min(self.t, len(self.episode_context_df) - 1)
        current = exogenous_by_microgrid or self._exogenous_at(index)
        dr_states = dr_state_by_microgrid if dr_state_by_microgrid is not None else self._dr_states(index)
        return self.global_state_builder.build(
            self.episode_context_df.iloc[index],
            index,
            self.cores,
            self.history,
            exogenous_by_microgrid=current,
            dr_state_by_microgrid=dr_states,
            dtype=dtype,
        )

    def global_state(self) -> np.ndarray:
        if self._last_global_state is None:
            raise RuntimeError("Call reset() first")
        return self._last_global_state.copy()

    def reset(self, day: str | None = None, seed: int | None = None,
              deterministic: bool = False,
              forecast_noise: bool | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
            forecast_noise=forecast_noise,
            exogenous_by_microgrid=self._episode_exogenous,
        )
        for core in self.cores.values():
            core.reset()
        if self.dr_enabled:
            for mg in self.agent_ids:
                self.dr_devices[mg].reset(
                    self.episode_df,
                    np.asarray(self._episode_exogenous[mg]["load_kw"][:self.episode_length], dtype=np.float64),
                )
        self.t = 0
        self._terminated = False
        self.history = self._empty_history()
        current = self._exogenous_at(0)
        dr_states = self._dr_states(0)
        self._last_observations = self._observations(current, dr_states)
        self._last_global_state = self._build_global_state(current, dr_states)
        return {mg: value.copy() for mg, value in self._last_observations.items()}, {
            "global_state": self.global_state(),
            "day": str(self.episode_df.iloc[0]["timestamp"]),
            "ctde_spec": self.ctde_spec,
            "data_quality_flags": list(self.data.data_quality_flags),
        }

    def _soc_potential(self, t: int) -> float:
        """Legacy potential function; the DR environment does not use reward shaping."""
        if self.config["terminal_soc"]["mode"] == "none":
            return 0.0
        rate = float(self.config["terminal_soc"]["penalty_per_kwh"])
        tolerance = float(self.config["terminal_soc"]["tolerance"])
        progress = t / float(self.episode_length)
        potential = 0.0
        for mg in self.agent_ids:
            battery = self.cores[mg].battery
            diff = max(0.0, abs(float(battery.soc) - float(battery.initial_soc)) - tolerance)
            potential -= progress * diff * battery.capacity_kwh * rate
        return potential

    def _peer_demand_offer_cap(
        self,
        seller: str,
        load_demands: Mapping[str, float],
        charge_demands: Mapping[str, float],
        prices: Mapping[str, tuple[float, float]],
    ) -> float:
        if not self.energy_pool.enabled or not self.ess_p2p_enabled:
            return 0.0
        seller_price = float(prices[seller][1])
        cap = 0.0
        for buyer in self.agent_ids:
            if buyer == seller or float(prices[buyer][0]) + 1e-10 < seller_price:
                continue
            peer_demand = max(0.0, float(load_demands[buyer]) + float(charge_demands[buyer]))
            cap += min(peer_demand, float(self.energy_pool.pair_capacity_kw))
        return float(cap)

    @staticmethod
    def _tensor_value(value: Any) -> float:
        try:
            return float(value.detach().reshape(-1)[0].item())
        except AttributeError:
            return float(value)

    def step(self, actions: Mapping[str, Sequence[float]]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict[str, Any]]:
        if self.dr_enabled:
            result = self._step_dr(actions)
            return apply_terminal_constraint_penalty(self, *result)
        return self._step_legacy(actions)

    def _step_dr(self, actions: Mapping[str, Sequence[float]]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict[str, Any]]:
        if self.episode_df is None or self._terminated:
            raise RuntimeError("Reset the environment before stepping")
        missing = set(self.agent_ids) - set(actions)
        if missing:
            raise KeyError(f"Missing actions for {sorted(missing)}")

        row = self.episode_df.iloc[self.t]
        base_current = self._exogenous_at(self.t)
        prices = {mg: prices_from_row(row, mg) for mg in self.agent_ids}
        decoded = {mg: self.action_codec.decode(actions[mg]) for mg in self.agent_ids}
        dr_states_before = self._dr_states(self.t)
        state_before = self._build_global_state(
            base_current, dr_states_before, dtype=np.float64
        )

        # 1. Heterogeneous controllable loads project first.
        dr_plans = {
            mg: self.dr_devices[mg].plan(float(decoded[mg]["dr"]), self.t)
            for mg in self.agent_ids
        }
        actual_current: dict[str, dict[str, float]] = {}
        for mg in self.agent_ids:
            base = float(base_current[mg]["load_kw"])
            flexible = float(dr_plans[mg]["flexible_load_kw"])
            actual = base + flexible
            actual_current[mg] = {
                **base_current[mg],
                "base_load_kw": base,
                "flexible_load_kw": flexible,
                "actual_load_kw": actual,
                "load_kw": actual,
            }

        # 2. Use the exact differentiable training projector for ESS local-first
        # dispatch after DR. Assert the independent physical DR models agree with
        # the projector's normalized action and electrical load.
        import torch
        raw_array = np.asarray([actions[mg] for mg in self.agent_ids], dtype=np.float64)
        with torch.no_grad():
            effective, projected = self.action_projector(
                torch.as_tensor(state_before, dtype=torch.float64).unsqueeze(0),
                torch.as_tensor(raw_array, dtype=torch.float64).unsqueeze(0),
            )

        tolerance = float(self.config["terminal_soc"]["tolerance"])
        plans: dict[str, dict[str, Any]] = {}
        direct_plans: dict[str, dict[str, float | bool]] = {}
        for i, mg in enumerate(self.agent_ids):
            effective_dr = float(effective[0, i, 2])
            physical_dr = float(dr_plans[mg]["effective_normalized"])
            projected_flexible = self._tensor_value(projected[i]["flexible_load_kw"])
            if abs(effective_dr - physical_dr) > 1e-6:
                raise RuntimeError(f"{mg}: rollout DR projection disagrees with training projector")
            if abs(projected_flexible - float(dr_plans[mg]["flexible_load_kw"])) > 1e-5:
                raise RuntimeError(f"{mg}: DR electrical load disagrees with training projector")

            direct = {
                name: self._tensor_value(projected[i][name])
                for name in (
                    "charge", "local", "mandatory_export", "remaining", "cap", "offer",
                    "requested_ratio", "requested_power", "local_priority_reserve",
                )
            }
            direct["mask"] = bool(projected[i]["mask"].reshape(-1)[0].item())
            direct["forced_dispatch"] = bool(
                projected[i]["forced_dispatch"].reshape(-1)[0].item()
            )
            direct_plans[mg] = direct

            plan = self.cores[mg].plan_local(
                row,
                decoded[mg],
                t=self.t,
                episode_length=self.episode_length,
                tolerance=tolerance,
                exogenous=actual_current[mg],
                enforce_terminal=False,
            )
            requested_dr = self._tensor_value(projected[i]["dr_requested"])
            dr_delta = self._tensor_value(projected[i]["dr_projection_delta"])
            raw_ess_action = float(decoded[mg]["battery"])
            battery_cfg = self.cfg_by_id[mg]["battery"]
            max_charge_kw = float(battery_cfg["max_charge_kw"])
            max_discharge_kw = float(battery_cfg["max_discharge_kw"])
            raw_ess_requested_power_kw = (
                raw_ess_action * max_discharge_kw
                if raw_ess_action >= 0.0
                else raw_ess_action * max_charge_kw
            )
            projected_ess_requested_power_kw = float(direct["requested_power"])
            ess_projection_delta_kw = (
                projected_ess_requested_power_kw - raw_ess_requested_power_kw
            )
            ess_power_scale_kw = max(max_charge_kw, max_discharge_kw, 1e-12)
            plan.update({
                "base_load_kw": float(base_current[mg]["load_kw"]),
                "flexible_load_kw": float(dr_plans[mg]["flexible_load_kw"]),
                "actual_load_kw": float(actual_current[mg]["load_kw"]),
                "load_kw": float(actual_current[mg]["load_kw"]),
                "raw_ess_action": raw_ess_action,
                "raw_dr_action": float(decoded[mg]["dr"]),
                "raw_ess_requested_power_kw": raw_ess_requested_power_kw,
                "projected_ess_requested_power_kw": projected_ess_requested_power_kw,
                "effective_ess_dispatch": float(effective[0, i, 0]),
                "effective_local_action": float(effective[0, i, 0]),
                "effective_trade_action": float(effective[0, i, 1]),
                "effective_dr_action": effective_dr,
                "ess_projection_delta_kw": ess_projection_delta_kw,
                "ess_projection_delta": ess_projection_delta_kw / ess_power_scale_kw,
                "dr_projection_delta": dr_delta,
                "ess_projected": bool(abs(ess_projection_delta_kw) > 1e-7),
                "dr_projected": bool(abs(dr_delta) > 1e-7),
                "dr_requested_normalized": requested_dr,
                "battery_charge_kw": float(direct["charge"]),
                "battery_discharge_kw": float(
                    direct["local"] + direct["mandatory_export"] + direct["offer"]
                ),
                "mandatory_export_kw": float(direct["mandatory_export"]),
                "direct_local_discharge_kw": float(direct["local"]),
                "direct_local_priority_reserve_kw": float(direct["local_priority_reserve"]),
                "direct_remaining_discharge_kw": float(direct["remaining"]),
                "direct_trade_capacity_kw": float(direct["cap"]),
                "dispatch_requested_power_kw": float(direct["requested_power"]),
                "dispatch_forced": bool(direct["forced_dispatch"]),
            })
            plans[mg] = plan

        # 3. Build source-aware P2P offers. Actual load, not base load, is the
        # load receiving renewable/ESS/P2P supply.
        renewable_offers: dict[str, float] = {}
        ess_offers: dict[str, float] = {}
        load_demands: dict[str, float] = {}
        charge_demands: dict[str, float] = {}
        for mg in self.agent_ids:
            plan = plans[mg]
            direct = direct_plans[mg]
            load = float(plan["actual_load_kw"])
            renewable = float(plan["renewable_kw"])
            requested_charge = float(direct["charge"])
            load_after_renewable = max(0.0, load - renewable)
            renewable_surplus = max(0.0, renewable - load)
            local_discharge = float(direct["local"])
            renewable_to_battery = min(requested_charge, renewable_surplus)
            renewable_offer = max(0.0, renewable_surplus - renewable_to_battery)
            charge_demand = max(0.0, requested_charge - renewable_to_battery)
            load_demand = max(0.0, load_after_renewable - local_discharge)
            ess_offer = float(direct["offer"]) if self.ess_p2p_enabled else 0.0
            plan.update({
                "renewable_to_local_load_kw": min(load, renewable),
                "renewable_to_battery_kw": renewable_to_battery,
                "renewable_p2p_offer_kw": renewable_offer,
                "local_load_deficit_before_battery_kw": load_after_renewable,
                "ess_to_local_requested_kw": local_discharge,
                "ess_flexible_discharge_kw": float(direct["remaining"]),
                "requested_sharing_offer_ratio": float(direct["requested_ratio"]),
                "sharing_offer_ratio": (
                    ess_offer / max(float(direct["remaining"]), 1e-12)
                    if float(direct["remaining"]) > 1e-9 else 0.0
                ),
                "ess_p2p_offer_uncapped_kw": self._tensor_value(projected[self.agent_ids.index(mg)]["uncapped_offer"]),
                "ess_p2p_offer_kw": ess_offer,
                "ess_p2p_offer_cap_kw": float(direct["cap"]),
                "ess_offer_projection_active": True,
                "sharing_action_masked": not bool(direct["mask"]),
                "p2p_load_demand_kw": load_demand,
                "p2p_charge_demand_kw": charge_demand,
            })
            renewable_offers[mg] = renewable_offer
            ess_offers[mg] = ess_offer
            load_demands[mg] = load_demand
            charge_demands[mg] = charge_demand

        sharing = self.energy_pool.allocate_structured(
            renewable_offers,
            ess_offers,
            load_demands,
            charge_demands,
            {mg: prices[mg][1] for mg in self.agent_ids},
            {mg: prices[mg][0] for mg in self.agent_ids},
        )

        total_buy_demand = sum(load_demands.values()) + sum(charge_demands.values())
        total_sell_offer = sum(renewable_offers.values()) + sum(ess_offers.values())
        for mg in self.agent_ids:
            accepted = float(sharing.by_microgrid[mg]["ess_p2p_sell_kw"])
            local = float(plans[mg]["ess_to_local_requested_kw"])
            requested = float(plans[mg]["battery_discharge_kw"])
            actual_discharge = local + accepted + float(plans[mg].get("mandatory_export_kw", 0.0))
            plans[mg]["battery_discharge_requested_kw"] = requested
            plans[mg]["battery_discharge_kw"] = actual_discharge
            plans[mg]["ess_to_local_kw"] = local
            plans[mg]["ess_to_p2p_kw"] = accepted
            plans[mg]["ess_p2p_unmatched_kw"] = max(
                0.0, float(plans[mg]["ess_p2p_offer_kw"]) - accepted
            )

        # 4. External grid: remaining actual load is served before optional
        # grid-backed ESS charging. P2P charging already cleared before this step.
        settlements: dict[str, dict[str, float]] = {}
        for mg in self.agent_ids:
            plan = plans[mg]
            peer = sharing.by_microgrid[mg]
            requested_charge = float(plan["battery_charge_kw"])
            renewable_to_battery = float(plan["renewable_to_battery_kw"])
            p2p_to_battery = float(peer["p2p_to_battery_kw"])
            p2p_to_load = float(peer["p2p_to_load_kw"])
            remaining_load = max(0.0, float(plan["p2p_load_demand_kw"]) - p2p_to_load)
            import_limit = float(self.grids[mg].import_limit_kw)
            grid_to_load = min(remaining_load, import_limit)
            remaining_grid_capacity = max(0.0, import_limit - grid_to_load)
            grid_charge_request = max(
                0.0, requested_charge - renewable_to_battery - p2p_to_battery
            )
            grid_to_battery = min(grid_charge_request, remaining_grid_capacity)
            actual_charge = renewable_to_battery + p2p_to_battery + grid_to_battery
            charge_curtailed = max(0.0, requested_charge - actual_charge)
            plan["battery_charge_requested_kw"] = requested_charge
            plan["battery_charge_kw"] = actual_charge
            plan["battery_charge_curtailed_kw"] = charge_curtailed
            plan["p2p_to_load_kw"] = p2p_to_load
            plan["p2p_to_battery_kw"] = p2p_to_battery
            plan["grid_to_load_kw"] = grid_to_load
            plan["grid_to_battery_kw"] = grid_to_battery

            residual_net = (
                float(plan["renewable_kw"])
                + float(plan["battery_discharge_kw"])
                - float(plan["actual_load_kw"])
                - float(plan["battery_charge_kw"])
                + float(peer["p2p_net_kw"])
            )
            settlements[mg] = self.grids[mg].settle(residual_net)
            if abs(settlements[mg]["power_balance_residual_kw"]) > 1e-7:
                raise RuntimeError(f"{mg}: external-grid power balance failed")
            if (not self.allow_grid_resale
                    and settlements[mg]["grid_import_kw"] > 1e-8
                    and peer["p2p_sell_kw"] > 1e-8):
                raise RuntimeError(f"{mg}: grid imports cannot be resold through P2P")

        # 5. Commit ESS and physical DR states. If a rare physical supply limit
        # leaves actual load unmet, device state receives the same served fraction
        # and exposes the resulting infeasibility instead of silently consuming
        # unavailable energy.
        devices = {mg: self.cores[mg].commit(plans[mg]) for mg in self.agent_ids}
        dr_devices_after: dict[str, dict[str, float]] = {}
        for mg in self.agent_ids:
            actual_load = max(float(plans[mg]["actual_load_kw"]), 1e-12)
            unmet = float(settlements[mg]["unmet_load_kw"])
            served_fraction = float(np.clip((actual_load - unmet) / actual_load, 0.0, 1.0))
            plans[mg]["actual_load_served_fraction"] = served_fraction
            plans[mg]["dr_supply_infeasible"] = bool(unmet > 1e-8)
            dr_devices_after[mg] = self.dr_devices[mg].commit(
                dr_plans[mg], delivered_fraction=served_fraction
            )

        # 6. Economics only. Hard device/terminal feasibility is not duplicated
        # by routine DR penalties or SOC shaping.
        costs = {}
        for mg in self.agent_ids:
            costs[mg] = self.accounting.calculate(
                prices[mg][0],
                prices[mg][1],
                settlements[mg],
                devices[mg]["degradation_cost"],
                self.delta_t,
            )
            transfer = float(sharing.by_microgrid[mg]["p2p_transfer_payment"])
            costs[mg]["p2p_transfer_payment"] = transfer
            costs[mg]["individual_accounting_profit"] = float(
                costs[mg]["physical_profit"] + transfer
            )
            costs[mg]["terminal_soc_penalty"] = 0.0

        terminal = self.t + 1 >= len(self.episode_df)
        terminal_soc_infeasible: dict[str, bool] = {}
        for mg in self.agent_ids:
            battery = self.cores[mg].battery
            terminal_soc_infeasible[mg] = bool(
                terminal
                and abs(float(battery.soc) - float(battery.initial_soc))
                > tolerance + 1e-6
            )
        team_profit = sum(costs[mg]["physical_profit"] for mg in self.agent_ids)
        team_reward = team_profit / self.accounting.reward_scale
        rewards = {mg: float(team_reward) for mg in self.agent_ids}

        self.history = {
            "total_p2p_kwh": sharing.total_p2p_kwh,
            "total_buy_demand_kwh": total_buy_demand * self.delta_t,
            "total_sell_offer_kwh": total_sell_offer * self.delta_t,
            "clearing_ratio": (
                sharing.total_p2p_kwh / max(total_sell_offer * self.delta_t, 1e-9)
                if total_sell_offer > 1e-9 else 0.0
            ),
            "by_microgrid": {
                mg: {
                    "p2p_net_kwh": sharing.by_microgrid[mg]["p2p_net_kw"] * self.delta_t,
                    "grid_net_kwh": (
                        settlements[mg]["grid_import_kw"] - settlements[mg]["grid_export_kw"]
                    ) * self.delta_t,
                    "ess_p2p_offer_kwh": sharing.by_microgrid[mg]["ess_p2p_offer_kw"] * self.delta_t,
                    "ess_p2p_accepted_kwh": sharing.by_microgrid[mg]["ess_p2p_sell_kw"] * self.delta_t,
                    "ess_p2p_unmatched_kwh": sharing.by_microgrid[mg]["ess_p2p_unmatched_kw"] * self.delta_t,
                }
                for mg in self.agent_ids
            },
        }

        info = {
            "timestamp": str(row["timestamp"]),
            "mode": self.mode,
            "action_design": self.action_design,
            "decoded_actions": decoded,
            "plans": plans,
            "dr_plans": dr_plans,
            "dr_devices": dr_devices_after,
            "p2p": {
                "by_microgrid": sharing.by_microgrid,
                "pair_flows_kwh": sharing.pair_flows_kwh,
                "total_p2p_kwh": sharing.total_p2p_kwh,
                "transfer_price_per_kwh": sharing.transfer_price_per_kwh,
                "transfer_payment_sum": sum(
                    x["p2p_transfer_payment"] for x in sharing.by_microgrid.values()
                ),
            },
            "settlements": settlements,
            "devices": devices,
            "costs": costs,
            "team_profit": float(team_profit),
            "team_reward": float(team_reward),
            "shaping_reward": 0.0,
            "terminal_soc_penalty": {mg: 0.0 for mg in self.agent_ids},
            "terminal_soc_infeasible": terminal_soc_infeasible,
            "data_quality_flags": list(self.data.data_quality_flags),
        }

        self.t += 1
        self._terminated = terminal
        next_exogenous = self._exogenous_at(self.t)
        next_dr_states = self._dr_states(self.t)
        self._last_observations = self._observations(next_exogenous, next_dr_states)
        self._last_global_state = self._build_global_state(next_exogenous, next_dr_states)
        info["dr_state_after"] = next_dr_states
        info["global_state_after"] = self.global_state()
        return {
            mg: value.copy() for mg, value in self._last_observations.items()
        }, rewards, terminal, info

    def _step_legacy(self, actions: Mapping[str, Sequence[float]]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict[str, Any]]:
        """Pre-DR single-dispatch environment retained for unmigrated algorithms."""
        if self.episode_df is None or self._terminated:
            raise RuntimeError("Reset the environment before stepping")
        missing = set(self.agent_ids) - set(actions)
        if missing:
            raise KeyError(f"Missing actions for {sorted(missing)}")
        row = self.episode_df.iloc[self.t]
        current = self._exogenous_at(self.t)
        prices = {mg: prices_from_row(row, mg) for mg in self.agent_ids}
        decoded = {mg: self.action_codec.decode(actions[mg]) for mg in self.agent_ids}
        tolerance = float(self.config["terminal_soc"]["tolerance"])

        plans = {
            mg: self.cores[mg].plan_local(
                row,
                decoded[mg],
                t=self.t,
                episode_length=self.episode_length,
                tolerance=tolerance,
                enforce_terminal=self.config["terminal_soc"]["mode"] != "none",
                exogenous=current[mg],
            )
            for mg in self.agent_ids
        }

        direct_plans = {}
        import torch
        with torch.no_grad():
            effective, projected = self.action_projector(
                torch.as_tensor(self._build_global_state(current, dtype=np.float64),
                                dtype=torch.float64).unsqueeze(0),
                torch.as_tensor(np.asarray([actions[mg] for mg in self.agent_ids]),
                                dtype=torch.float64).unsqueeze(0),
            )
        for i, mg in enumerate(self.agent_ids):
            direct = {name: float(value[0]) for name, value in projected[i].items()}
            direct_plans[mg] = direct
            plans[mg].update({
                "battery_charge_kw": direct["charge"],
                "battery_discharge_kw": direct["local"] + direct["mandatory_export"] + direct["offer"],
                "mandatory_export_kw": direct["mandatory_export"],
                "direct_local_discharge_kw": direct["local"],
                "direct_local_priority_reserve_kw": direct["local_priority_reserve"],
                "direct_remaining_discharge_kw": direct["remaining"],
                "direct_trade_capacity_kw": direct["cap"],
                "effective_local_action": float(effective[0, i, 0]),
                "effective_trade_action": float(effective[0, i, 1]),
                "dispatch_requested_power_kw": direct["requested_power"],
                "dispatch_forced": bool(direct["forced_dispatch"]),
            })

        renewable_offers: dict[str, float] = {}
        uncapped_ess_offers: dict[str, float] = {}
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
            if mg in direct_plans:
                local_discharge = direct_plans[mg]["local"]
                flexible_discharge = direct_plans[mg]["remaining"]
            renewable_to_battery = min(requested_charge, renewable_surplus)
            renewable_offer = max(0.0, renewable_surplus - renewable_to_battery)
            charge_demand = max(0.0, requested_charge - renewable_to_battery)
            load_demand = max(0.0, load_after_renewable - local_discharge)
            requested_offer_ratio = direct_plans[mg]["requested_ratio"]
            offer_ratio = requested_offer_ratio if self.ess_p2p_enabled else 0.0
            uncapped_ess_offer = flexible_discharge * offer_ratio
            plans[mg].update({
                "renewable_to_local_load_kw": min(load, renewable),
                "renewable_to_battery_kw": renewable_to_battery,
                "renewable_p2p_offer_kw": renewable_offer,
                "local_load_deficit_before_battery_kw": load_after_renewable,
                "ess_to_local_requested_kw": local_discharge,
                "ess_flexible_discharge_kw": flexible_discharge,
                "requested_sharing_offer_ratio": requested_offer_ratio,
                "sharing_offer_ratio": offer_ratio,
                "ess_p2p_offer_uncapped_kw": uncapped_ess_offer,
                "p2p_load_demand_kw": load_demand,
                "p2p_charge_demand_kw": charge_demand,
            })
            renewable_offers[mg] = renewable_offer
            uncapped_ess_offers[mg] = uncapped_ess_offer
            load_demands[mg] = load_demand
            charge_demands[mg] = charge_demand

        projection_active = self.ess_offer_projection == "peer_demand_cap"
        for mg in self.agent_ids:
            flexible = float(plans[mg]["ess_flexible_discharge_kw"])
            eligible_cap = self._peer_demand_offer_cap(mg, load_demands, charge_demands, prices)
            uncapped = float(uncapped_ess_offers[mg])
            direct = direct_plans[mg]
            projected_offer = direct["offer"]
            blocked = not bool(direct["mask"])
            effective_ratio = projected_offer / flexible if flexible > 1e-9 else 0.0
            plans[mg].update({
                "sharing_offer_ratio": float(effective_ratio),
                "ess_p2p_offer_kw": float(projected_offer),
                "ess_p2p_offer_cap_kw": float(direct["cap"] if projection_active else eligible_cap),
                "ess_p2p_offer_projection_reduction_kw": float(max(0.0, uncapped - projected_offer)),
                "ess_offer_projection_active": bool(projection_active),
                "sharing_action_masked": blocked,
            })
            ess_offers[mg] = float(projected_offer)

        sharing = self.energy_pool.allocate_structured(
            renewable_offers, ess_offers, load_demands, charge_demands,
            {mg: prices[mg][1] for mg in self.agent_ids},
            {mg: prices[mg][0] for mg in self.agent_ids},
        )
        total_buy_demand = sum(load_demands.values()) + sum(charge_demands.values())
        total_sell_offer = sum(renewable_offers.values()) + sum(ess_offers.values())
        for mg in self.agent_ids:
            accepted = float(sharing.by_microgrid[mg]["ess_p2p_sell_kw"])
            local = float(plans[mg]["ess_to_local_requested_kw"])
            requested = float(plans[mg]["battery_discharge_kw"])
            actual_discharge = local + accepted + float(plans[mg].get("mandatory_export_kw", 0.0))
            plans[mg]["battery_discharge_requested_kw"] = requested
            plans[mg]["battery_discharge_kw"] = actual_discharge
            plans[mg]["ess_to_local_kw"] = local
            plans[mg]["ess_to_p2p_kw"] = accepted
            plans[mg]["ess_p2p_unmatched_kw"] = max(
                0.0, float(plans[mg]["ess_p2p_offer_kw"]) - accepted
            )

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

        devices = {mg: self.cores[mg].commit(plans[mg]) for mg in self.agent_ids}
        costs = {}
        for mg in self.agent_ids:
            costs[mg] = self.accounting.calculate(
                prices[mg][0], prices[mg][1], settlements[mg],
                devices[mg]["degradation_cost"], self.delta_t,
            )
            transfer = float(sharing.by_microgrid[mg]["p2p_transfer_payment"])
            costs[mg]["p2p_transfer_payment"] = transfer
            costs[mg]["individual_accounting_profit"] = float(
                costs[mg]["physical_profit"] + transfer
            )

        potential_before = self._soc_potential(self.t)
        potential_after = self._soc_potential(self.t + 1)
        shaping_reward = potential_after - potential_before
        terminal_penalties = {mg: 0.0 for mg in self.agent_ids}
        terminal = self.t + 1 >= len(self.episode_df)
        if terminal and self.config["terminal_soc"]["mode"] != "none":
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
            "by_microgrid": {
                mg: {
                    "p2p_net_kwh": sharing.by_microgrid[mg]["p2p_net_kw"] * self.delta_t,
                    "grid_net_kwh": (
                        settlements[mg]["grid_import_kw"] - settlements[mg]["grid_export_kw"]
                    ) * self.delta_t,
                    "ess_p2p_offer_kwh": sharing.by_microgrid[mg]["ess_p2p_offer_kw"] * self.delta_t,
                    "ess_p2p_accepted_kwh": sharing.by_microgrid[mg]["ess_p2p_sell_kw"] * self.delta_t,
                    "ess_p2p_unmatched_kwh": sharing.by_microgrid[mg]["ess_p2p_unmatched_kw"] * self.delta_t,
                }
                for mg in self.agent_ids
            },
        }
        info = {
            "timestamp": str(row["timestamp"]),
            "mode": self.mode,
            "decoded_actions": decoded,
            "plans": plans,
            "p2p": {
                "by_microgrid": sharing.by_microgrid,
                "pair_flows_kwh": sharing.pair_flows_kwh,
                "total_p2p_kwh": sharing.total_p2p_kwh,
                "transfer_price_per_kwh": sharing.transfer_price_per_kwh,
                "transfer_payment_sum": sum(
                    x["p2p_transfer_payment"] for x in sharing.by_microgrid.values()
                ),
            },
            "settlements": settlements,
            "devices": devices,
            "costs": costs,
            "team_profit": float(team_profit),
            "team_reward": float(team_reward),
            "shaping_reward": float(shaping_reward),
            "terminal_soc_penalty": terminal_penalties,
        }
        self.t += 1
        self._terminated = terminal
        next_exogenous = self._exogenous_at(self.t)
        self._last_observations = self._observations(next_exogenous)
        self._last_global_state = self._build_global_state(next_exogenous)
        info["global_state_after"] = self.global_state()
        return {
            mg: value.copy() for mg, value in self._last_observations.items()
        }, rewards, terminal, info
