from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from algorithms.milp_ess_sharing_mpc.mpc_solver import solve_receding_horizon


class _NoOpReplay:
    """Minimal pipeline-compatible replay counter for a non-learning controller."""

    def __init__(self, capacity: int = 24) -> None:
        self.capacity = max(1, int(capacity))
        self.count = 0

    def __len__(self) -> int:
        return self.count

    def add(self, *args: Any, **kwargs: Any) -> None:
        self.count = min(self.capacity, self.count + 1)


class MilpEssSharingMpc:
    """Centralized rolling MILP using only Sharing's current data and four forecasts."""

    def __init__(self, env: Any, forecast_horizon: int = 4, replay_capacity: int = 24) -> None:
        if int(env.action_dim) != 2:
            raise ValueError("MILP ESS Sharing MPC requires Sharing's two-dimensional action")
        self.env = env
        self.agent_ids = list(env.agent_ids)
        self.action_dim = int(env.action_dim)
        self.forecast_horizon = int(forecast_horizon)
        if self.forecast_horizon != int(env.obs_builder.future_h):
            raise ValueError(
                "MPC forecast_horizon must equal the environment observation future_horizon"
            )
        self.replay = _NoOpReplay(replay_capacity)
        self.last_status: str | None = None

    @classmethod
    def from_env(
        cls,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
        seed: int | None = None,
    ) -> "MilpEssSharingMpc":
        settings = config.get("mpc", {})
        algorithm = config.get("algorithm", {})
        return cls(
            env,
            forecast_horizon=int(settings.get("forecast_horizon", 4)),
            replay_capacity=int(algorithm.get("replay_capacity", 24)),
        )

    def select_actions(
        self,
        observations: Mapping[str, np.ndarray],
        noise_std: float = 0.0,
    ) -> dict[str, np.ndarray]:
        if set(observations) != set(self.agent_ids):
            raise KeyError("MPC observations must contain every configured microgrid")
        solution = solve_receding_horizon(
            self.env,
            forecast_horizon=self.forecast_horizon,
        )
        self.last_status = solution.status
        actions: dict[str, np.ndarray] = {}
        current = self.env._exogenous_at(self.env.t)
        tolerance = float(self.env.config["terminal_soc"]["tolerance"])
        for mg in self.agent_ids:
            battery = self.env.cores[mg].battery
            floor, ceiling = battery.reachable_soc_envelope(
                t=self.env.t,
                episode_length=self.env.episode_length,
                target_soc=battery.initial_soc,
                tolerance=tolerance,
            )
            battery_action = _invert_battery_action(
                battery,
                float(solution.charge_kw[mg]),
                float(solution.discharge_kw[mg]),
                floor,
                ceiling,
            )
            mapped = battery.map_action(
                battery_action,
                soc_min_reach=floor,
                soc_max_reach=ceiling,
            )
            load = float(current[mg]["load_kw"])
            renewable = float(current[mg]["renewable_kw"])
            actual_charge = float(mapped["charge_requested_kw"])
            actual_discharge = float(mapped["discharge_requested_kw"])
            load_deficit = max(0.0, load - renewable)
            flexible_discharge = max(0.0, actual_discharge - min(actual_discharge, load_deficit))
            renewable_surplus = max(0.0, renewable - load)
            renewable_offer = max(0.0, renewable_surplus - min(actual_charge, renewable_surplus))
            desired_ess_export = max(0.0, float(solution.p2p_export_kw[mg]) - renewable_offer)
            offer_ratio = (
                float(np.clip(desired_ess_export / flexible_discharge, 0.0, 1.0))
                if flexible_discharge > 1e-9 else 0.0
            )
            actions[mg] = np.asarray(
                [battery_action, 2.0 * offer_ratio - 1.0],
                dtype=np.float32,
            )
        return actions

    def update(self, batch_size: int) -> dict[str, Any]:
        return {}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({
                "algorithm": "milp_ess_sharing_mpc",
                "forecast_horizon": self.forecast_horizon,
            }),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("algorithm") != "milp_ess_sharing_mpc":
            raise ValueError("Checkpoint is not a MILP ESS Sharing MPC checkpoint")
        if int(payload.get("forecast_horizon", -1)) != self.forecast_horizon:
            raise ValueError("Checkpoint forecast horizon does not match the configured MPC")


def _invert_battery_action(
    battery: Any,
    target_charge_kw: float,
    target_discharge_kw: float,
    soc_min_reach: float,
    soc_max_reach: float,
) -> float:
    """Invert the environment battery mapping without mutating battery state."""
    zero = battery.map_action(0.0, soc_min_reach, soc_max_reach)
    if target_discharge_kw > 1e-9:
        maximum = battery.map_action(1.0, soc_min_reach, soc_max_reach)
        low = float(zero["discharge_requested_kw"])
        high = float(maximum["discharge_requested_kw"])
        if high <= low + 1e-12:
            return 0.0
        return float(np.clip((target_discharge_kw - low) / (high - low), 0.0, 1.0))
    if target_charge_kw > 1e-9:
        maximum = battery.map_action(-1.0, soc_min_reach, soc_max_reach)
        low = float(zero["charge_requested_kw"])
        high = float(maximum["charge_requested_kw"])
        if high <= low + 1e-12:
            return 0.0
        return float(-np.clip((target_charge_kw - low) / (high - low), 0.0, 1.0))
    return 0.0
