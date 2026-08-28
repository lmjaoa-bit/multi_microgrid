from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import yaml  # type: ignore[import-untyped]

from core.action_codec import ActionCodec
from core.config_validation import validate_config
from core.cost_accounting import CostAccounting
from core.data_manager import DataManager
from core.forecast import SharedForecastProvider
from core.global_state import GlobalStateBuilder
from core.grid_settlement import GridSettlement
from core.microgrid_core import MicrogridCore
from core.observation import ObservationBuilder
from core.profiles import HeterogeneousProfileModel
from core.types import MarketHistory, MarketHistoryEntry
from market.double_auction import DoubleAuctionMarket, P2POrder


class MultiMicrogridEnv:
    """Standalone three-microgrid P2P environment.

    This class is intentionally independent from the old single-microgrid Gym
    environment. It exposes dictionary observations/actions and a CTDE global
    state, but has no dependency on Gym/Gymnasium or any RL algorithm.
    """

    def __init__(self, config: str | Path | Dict[str, Any], split: str = "train"):
        if isinstance(config, (str, Path)):
            config_path = Path(config).resolve()
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            self.config["_project_root"] = str(config_path.parent.parent)
        else:
            self.config = dict(config)
            self.config.setdefault("_project_root", str(Path.cwd()))

        validate_config(self.config)
        self.delta_t = float(self.config["time"].get("delta_t", 1.0))
        self.episode_length = int(self.config["time"].get("episode_length", 24))
        self.seed = int(self.config.get("env", {}).get("seed", 42))
        self.split = str(split)
        self.data = DataManager(self.config, split=split, seed=self.seed)
        self.normalization_data = (
            self.data
            if self.split == "train"
            else DataManager(self.config, split="train", seed=self.seed)
        )
        self.agent_ids = [str(m["id"]) for m in self.config["microgrids"]]
        if len(self.agent_ids) != 3:
            raise ValueError("This research baseline is configured for exactly 3 microgrids.")

        self.profiles = HeterogeneousProfileModel(self.config, self.normalization_data.df)
        self.forecasts = SharedForecastProvider(self.config, self.profiles, seed=self.seed)
        self.cores = {
            str(cfg["id"]): MicrogridCore(dict(cfg), self.delta_t, self.profiles)
            for cfg in self.config["microgrids"]
        }
        self.grid = {
            str(cfg["id"]): GridSettlement(
                import_limit_kw=float(cfg["grid"]["import_limit_kw"]),
                export_limit_kw=float(cfg["grid"]["export_limit_kw"]),
                delta_t=self.delta_t,
            )
            for cfg in self.config["microgrids"]
        }
        p2p_cfg = self.config["p2p"]
        self.market = DoubleAuctionMarket(
            min_trade_energy_kwh=float(p2p_cfg.get("min_trade_power_kw", 1.0)) * self.delta_t
        )
        cost_cfg = self.config.get("cost", {})
        self.accounting = CostAccounting(
            carbon_price_per_kg=float(cost_cfg.get("carbon_price_per_kg", 0.25)),
            unmet_load_cost_per_kwh=float(cost_cfg.get("unmet_load_cost_per_kwh", 20.0)),
            dump_cost_per_kwh=float(cost_cfg.get("controllable_dump_cost_per_kwh", 1.0)),
            reward_scale=float(cost_cfg.get("reward_scale", 100.0)),
        )
        self.obs_builder = ObservationBuilder(
            self.config, self.profiles, self.forecasts, delta_t=self.delta_t
        )
        self.global_state_builder = GlobalStateBuilder(
            self.config, self.profiles, self.forecasts, delta_t=self.delta_t
        )
        self.action_codec = ActionCodec()
        self.p2p_enabled = bool(p2p_cfg.get("enabled", True))
        self.t = 0
        self.episode_df = None
        self.episode_context_df = None
        self.market_history: MarketHistory | None = None
        self._last_observations: Dict[str, np.ndarray] | None = None
        self._last_global_state: np.ndarray | None = None
        self._terminated = False

    @property
    def action_dim(self) -> int:
        return self.action_codec.action_dim

    @property
    def observation_dims(self) -> Dict[str, int]:
        return {mg: self.obs_builder.obs_dim(mg) for mg in self.agent_ids}

    @property
    def global_state_dim(self) -> int:
        return self.global_state_builder.dim

    @property
    def global_state_schema(self) -> list[str]:
        return self.global_state_builder.schema()

    @property
    def ctde_spec(self) -> Dict[str, Any]:
        return {
            "agent_ids": list(self.agent_ids),
            "action_dim": self.action_dim,
            "observation_dims": self.observation_dims,
            "global_state_dim": self.global_state_dim,
            "global_state_schema": self.global_state_schema,
        }

    def _initial_market_history(self) -> MarketHistory:
        if self.episode_df is None:
            raise RuntimeError("Episode data is not initialized.")
        row = self.episode_df.iloc[0]
        midpoint = 0.5 * (float(row["buy_price_per_kwh"]) + float(row["sell_price_per_kwh"]))
        by_microgrid: Dict[str, MarketHistoryEntry] = {
            mg: {"bid_price": midpoint, "trade_kwh": 0.0, "role": 0.0}
            for mg in self.agent_ids
        }
        return {
            "weighted_average_price": midpoint,
            "total_trade_kwh": 0.0,
            "submitted_buy_kwh": 0.0,
            "submitted_sell_kwh": 0.0,
            "cleared_buy_kwh": 0.0,
            "cleared_sell_kwh": 0.0,
            "by_microgrid": by_microgrid,
        }

    def _observations(self) -> Dict[str, np.ndarray]:
        if self.episode_context_df is None or self.market_history is None:
            raise RuntimeError("Episode context is not initialized.")
        context = self.episode_context_df
        idx = min(self.t, len(context) - 1)
        row = context.iloc[idx]
        obs = {}
        for mg in self.agent_ids:
            core = self.cores[mg]
            if core.battery.soc is None:
                raise RuntimeError(f"{mg}: battery is not initialized.")
            obs[mg] = self.obs_builder.build(
                mg, row, idx,
                battery_soc=float(core.battery.soc),
                generator_power_kw=float(core.generator.last_power_kw),
                market_history=self.market_history,
            )
        return obs

    @staticmethod
    def _copy_observations(observations: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {mg: value.copy() for mg, value in observations.items()}

    def _build_global_state(self) -> np.ndarray:
        if self.episode_context_df is None or self.market_history is None:
            raise RuntimeError("Episode context is not initialized.")
        context = self.episode_context_df
        idx = min(self.t, len(context) - 1)
        row = context.iloc[idx]
        return self.global_state_builder.build(row, idx, self.cores, self.market_history)

    def global_state(self) -> np.ndarray:
        if self._last_global_state is None:
            raise RuntimeError("Call reset() first.")
        return self._last_global_state.copy()

    def reset(
        self,
        day: str | None = None,
        seed: int | None = None,
        deterministic: bool = False,
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.data.reseed(seed)
        context = self.data.get_episode_context(
            day,
            future_horizon=self.obs_builder.future_h,
        )
        episode = context.iloc[: self.episode_length].reset_index(drop=True).copy()
        self.episode_context_df = context
        self.episode_df = episode
        self.forecasts.begin_episode(context, seed=seed, deterministic=deterministic)
        self.t = 0
        self._terminated = False
        for core in self.cores.values():
            core.reset()
        self.market_history = self._initial_market_history()
        self._last_observations = self._observations()
        self._last_global_state = self._build_global_state()
        return self._copy_observations(self._last_observations), {
            "global_state": self.global_state(),
            "day": str(episode.iloc[0]["timestamp"]),
            "ctde_spec": self.ctde_spec,
        }

    def _build_orders(self, plans: Mapping[str, Mapping[str, Any]], decoded: Mapping[str, Mapping[str, Any]]) -> tuple[list[P2POrder], Dict[str, float]]:
        orders: list[P2POrder] = []
        roles: Dict[str, float] = {}
        if not self.p2p_enabled:
            return orders, {mg: 0.0 for mg in self.agent_ids}
        for mg in self.agent_ids:
            net = float(plans[mg]["net_external_kw"])
            if net > 0.0 and self.market.is_quantity_eligible(net * self.delta_t):
                role = "seller"
                roles[mg] = -1.0
                quantity = net * self.delta_t
            elif net < 0.0 and self.market.is_quantity_eligible(-net * self.delta_t):
                role = "buyer"
                roles[mg] = 1.0
                quantity = -net * self.delta_t
            else:
                roles[mg] = 0.0
                continue
            orders.append(P2POrder(mg, role, quantity, float(decoded[mg]["p2p_bid_price"])))
        return orders, roles

    def _update_market_history(self, result: Mapping[str, Any], decoded: Mapping[str, Mapping[str, Any]], roles: Mapping[str, float]) -> None:
        by: Dict[str, MarketHistoryEntry] = {}
        for mg in self.agent_ids:
            x = result["by_microgrid"].get(mg, {})
            by[mg] = {
                "bid_price": float(decoded[mg]["p2p_bid_price"]),
                "trade_kwh": float(x.get("p2p_buy_kwh", 0.0) + x.get("p2p_sell_kwh", 0.0)),
                "role": float(roles.get(mg, 0.0)),
            }
        self.market_history = {
            "weighted_average_price": float(result["weighted_average_price"]),
            "total_trade_kwh": float(result["total_trade_kwh"]),
            "submitted_buy_kwh": float(result["submitted_buy_kwh"]),
            "submitted_sell_kwh": float(result["submitted_sell_kwh"]),
            "cleared_buy_kwh": float(result["cleared_buy_kwh"]),
            "cleared_sell_kwh": float(result["cleared_sell_kwh"]),
            "by_microgrid": by,
        }

    def _validate(self, market_result: Mapping[str, Any], settlements: Mapping[str, Mapping[str, float]]) -> None:
        market_numbers = [
            market_result["total_trade_kwh"],
            market_result["cleared_buy_kwh"],
            market_result["cleared_sell_kwh"],
            market_result["weighted_average_price"],
            market_result["submitted_buy_kwh"],
            market_result["submitted_sell_kwh"],
        ]
        for item in market_result["by_microgrid"].values():
            market_numbers.extend(item.values())
        if not np.isfinite(np.asarray(market_numbers, dtype=np.float64)).all():
            raise RuntimeError("P2P market result contains NaN or infinite values.")
        total_buy = sum(float(x["p2p_buy_kwh"]) for x in market_result["by_microgrid"].values())
        total_sell = sum(float(x["p2p_sell_kwh"]) for x in market_result["by_microgrid"].values())
        if abs(total_buy - total_sell) > 1e-7:
            raise RuntimeError("P2P energy is not conserved.")
        p2p_money = sum(float(x["p2p_net_cost"]) for x in market_result["by_microgrid"].values())
        if abs(p2p_money) > 1e-7:
            raise RuntimeError("P2P money is not conserved.")
        for mg, s in settlements.items():
            if not np.isfinite(np.asarray(list(s.values()), dtype=np.float64)).all():
                raise RuntimeError(f"{mg}: settlement contains NaN or infinite values.")
            if abs(float(s["power_balance_residual_kw"])) > 1e-6:
                raise RuntimeError(f"{mg}: power balance residual {s['power_balance_residual_kw']}")
            if float(s["grid_buy_kw"]) > 1e-8 and float(s["p2p_sell_kw"]) > 1e-8:
                raise RuntimeError(f"{mg}: prohibited grid-buy and P2P-sell combination.")
            if float(s["grid_to_battery_kw"]) > 1e-9:
                raise RuntimeError(f"{mg}: grid-to-battery charging is prohibited.")
            if float(s["p2p_to_battery_kw"]) > 1e-9:
                raise RuntimeError(f"{mg}: P2P-to-battery charging is prohibited.")
            if float(s["battery_grid_sell_kw"]) > 1e-9:
                raise RuntimeError(f"{mg}: battery-to-grid export is prohibited.")

    @staticmethod
    def _validate_finite_runtime_data(label: str, data: Mapping[str, Any]) -> None:
        values: list[float] = []

        def collect(item: Any) -> None:
            if isinstance(item, Mapping):
                for nested in item.values():
                    collect(nested)
            elif isinstance(item, (int, float, np.number, bool)):
                values.append(float(item))

        collect(data)
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise RuntimeError(f"{label} contains NaN or infinite values.")

    def step(self, actions: Mapping[str, Sequence[float]]) -> tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        if self.episode_df is None:
            raise RuntimeError("Call reset() before step().")
        if self._terminated:
            raise RuntimeError("Episode has terminated; call reset() before step() again.")
        missing = set(self.agent_ids) - set(actions)
        if missing:
            raise KeyError(f"Missing actions for {sorted(missing)}")

        global_state_before = self.global_state()
        row = self.episode_df.iloc[self.t]
        sell_price = float(row["sell_price_per_kwh"])
        buy_price = float(row["buy_price_per_kwh"])
        decoded = {mg: self.action_codec.decode(actions[mg], sell_price, buy_price) for mg in self.agent_ids}
        executed_actions = {
            mg: np.asarray(
                [
                    decoded[mg]["battery"],
                    decoded[mg]["generator_commitment"],
                    decoded[mg]["generator_setpoint"],
                    decoded[mg]["p2p_bid_action"],
                ],
                dtype=np.float32,
            )
            for mg in self.agent_ids
        }
        plans = {mg: self.cores[mg].plan_local(row, decoded[mg]) for mg in self.agent_ids}
        orders, roles = self._build_orders(plans, decoded)
        no_trade_price = 0.5 * (sell_price + buy_price)
        market_result = self.market.clear(orders, no_trade_price=no_trade_price)
        # Market returns only active IDs; guarantee all microgrids exist for accounting.
        for mg in self.agent_ids:
            market_result["by_microgrid"].setdefault(mg, {
                "p2p_buy_kwh": 0.0, "p2p_sell_kwh": 0.0, "p2p_buy_cost": 0.0,
                "p2p_sell_revenue": 0.0, "p2p_net_cost": 0.0, "p2p_net_kwh": 0.0,
                "weighted_average_trade_price": 0.0,
            })

        settlements = {
            mg: self.grid[mg].settle(plans[mg], market_result["by_microgrid"][mg])
            for mg in self.agent_ids
        }
        self._validate(market_result, settlements)
        devices = {mg: self.cores[mg].commit(plans[mg], settlements[mg]) for mg in self.agent_ids}
        costs = {
            mg: self.accounting.calculate(row, market_result["by_microgrid"][mg], settlements[mg], devices[mg], self.delta_t)
            for mg in self.agent_ids
        }
        self._validate_finite_runtime_data("Device state", devices)
        self._validate_finite_runtime_data("Cost accounting", costs)
        rewards = {mg: float(costs[mg]["reward"]) for mg in self.agent_ids}
        self._update_market_history(market_result, decoded, roles)

        info = {
            "timestamp": str(row["timestamp"]),
            "executed_actions": executed_actions,
            "plans": plans,
            "orders": [o.__dict__.copy() for o in orders],
            "market": market_result,
            "settlements": settlements,
            "devices": devices,
            "costs": costs,
            "roles": dict(roles),
            "global_state_before": global_state_before,
        }

        self.t += 1
        terminated = self.t >= min(self.episode_length, len(self.episode_df))
        self._terminated = terminated
        self._last_observations = self._observations()
        self._last_global_state = self._build_global_state()
        info["global_state_after"] = self.global_state()
        return self._copy_observations(self._last_observations), rewards, terminated, info
