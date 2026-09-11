from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from algorithms.base import MultiAgentAlgorithm, ReplayBufferProtocol
from optimization.common_constraints import DispatchData, solve_dispatch


class DummyReplayBuffer:
    """Minimal replay buffer stub to satisfy the shared pipeline contract."""

    def __len__(self) -> int:
        return 0

    def add(
        self,
        observations: Mapping[str, np.ndarray],
        global_state: np.ndarray,
        actions: Mapping[str, np.ndarray],
        team_reward: float,
        next_observations: Mapping[str, np.ndarray],
        next_global_state: np.ndarray,
        done: bool,
    ) -> None:
        pass


class MilpMpc:
    """4-hour Rolling Horizon Model Predictive Control (MILP-MPC) dispatch controller.

    At each timestep t, solves a 4-hour cooperative dispatch optimization problem
    using HiGHS, then extracts and maps the first step's battery dispatch decisions
    to the single_dispatch action space [-1, 1].
    """

    @classmethod
    def from_env(cls, env: Any, config: Mapping[str, Any], device: str = "cpu", seed: int | None = None) -> MilpMpc:
        return cls(env, config, device=device, seed=seed)

    def __init__(
        self,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.config = config
        self.device = device
        self.seed = 42 if seed is None else int(seed)
        self.agent_ids = list(env.agent_ids)
        self.action_dim = 1

        # Extract MPC configuration
        alg_cfg = config.get("algorithm", config)
        mpc_cfg = config.get("mpc", alg_cfg.get("mpc", {}))
        self.horizon = int(mpc_cfg.get("forecast_horizon", 4))
        self.forecast_mode = str(mpc_cfg.get("forecast_mode", "perfect"))
        self.solver_output = bool(mpc_cfg.get("solver_output", False))

        self.replay = DummyReplayBuffer()

    def select_actions(
        self,
        observations: Mapping[str, np.ndarray],
        noise_std: float = 0.0,
    ) -> dict[str, np.ndarray]:
        # Determine current timestep t (0..23)
        if hasattr(self.env, "t"):
            t = int(self.env.t)
        else:
            first_obs = next(iter(observations.values()))
            sin_h, cos_h = float(first_obs[0]), float(first_obs[1])
            angle = np.arctan2(sin_h, cos_h)
            t = int(round((angle % (2 * np.pi)) / (2 * np.pi / 24))) % 24

        H = min(self.horizon, 24 - t)
        if H <= 0:
            return {mg: np.zeros(self.action_dim, dtype=np.float32) for mg in self.agent_ids}

        # Collect current initial SOC
        if hasattr(self.env, "cores"):
            initial_soc = {mg: float(self.env.cores[mg].battery.soc) for mg in self.agent_ids}
        else:
            initial_soc = {mg: float(observations[mg][7]) for mg in self.agent_ids}

        # Build exogenous series and prices
        if hasattr(self.env, "episode_context_df") and self.env.episode_context_df is not None:
            df = self.env.episode_context_df
            timestamps = [str(df.iloc[t + i]["timestamp"]) for i in range(H)]
            exog = getattr(self.env, "_episode_exogenous", {})
            loads = {mg: exog[mg]["load_kw"][t : t + H] for mg in self.agent_ids}
            renewables = {mg: exog[mg]["renewable_kw"][t : t + H] for mg in self.agent_ids}

            buy_prices = {
                mg: df[f"{mg}_buy_price_per_kwh" if f"{mg}_buy_price_per_kwh" in df else "buy_price_per_kwh"]
                .iloc[t : t + H].to_numpy(float)
                for mg in self.agent_ids
            }
            sell_prices = {
                mg: df[f"{mg}_sell_price_per_kwh" if f"{mg}_sell_price_per_kwh" in df else "sell_price_per_kwh"]
                .iloc[t : t + H].to_numpy(float)
                for mg in self.agent_ids
            }
        else:
            timestamps = [f"t_{t + i}" for i in range(H)]
            loads = {mg: np.ones(H) * 50.0 for mg in self.agent_ids}
            renewables = {mg: np.ones(H) * 50.0 for mg in self.agent_ids}
            buy_prices = {mg: np.ones(H) * 0.60 for mg in self.agent_ids}
            sell_prices = {mg: np.ones(H) * 0.20 for mg in self.agent_ids}

        data = DispatchData(timestamps, self.agent_ids, loads, renewables, buy_prices, sell_prices)

        # Enforce terminal condition only on the final window that touches hour 24
        apply_term = (t + H == 24)

        try:
            result = solve_dispatch(
                self.env.config,
                data,
                self.agent_ids,
                cooperative=True,
                initial_soc_by_microgrid=initial_soc,
                apply_terminal_condition=apply_term,
                solver_output=self.solver_output,
            )
            traj = result.trajectories
            step0 = traj[traj["timestamp"] == timestamps[0]]

            actions = {}
            for mg in self.agent_ids:
                row = step0[step0["microgrid"] == mg].iloc[0]
                chg = float(row["battery_charge_kw"])
                dis = float(row["battery_discharge_kw"])

                battery = self.env.cores[mg].battery
                soc_min_r, soc_max_r = battery.reachable_soc_envelope(
                    t=t, episode_length=24, target_soc=battery.initial_soc, tolerance=0.03,
                    enforce_terminal=self.env.config["terminal_soc"]["mode"] != "none"
                )
                neutral = battery.map_action(0.0, soc_min_reach=soc_min_r, soc_max_reach=soc_max_r)

                if dis > 1e-3:
                    max_dis = neutral["feasible_max_kw"]
                    a = dis / max(max_dis, 1e-4) if max_dis > 1e-4 else 0.0
                    a = min(1.0, max(0.0, a))
                elif chg > 1e-3:
                    max_chg = -neutral["feasible_min_kw"]
                    a = -(chg / max(max_chg, 1e-4)) if max_chg > 1e-4 else 0.0
                    a = max(-1.0, min(0.0, a))
                else:
                    a = 0.0

                actions[mg] = np.array([a], dtype=np.float32)
            return actions
        except Exception:
            return {mg: np.zeros(self.action_dim, dtype=np.float32) for mg in self.agent_ids}

    def update(self, batch_size: int) -> dict[str, Any]:
        return {}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "algorithm": "milp_mpc",
            "horizon": self.horizon,
            "forecast_mode": self.forecast_mode,
        }
        target.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        pass
