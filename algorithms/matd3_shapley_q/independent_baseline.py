from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from envs.multi_microgrid_env import MultiMicrogridEnv


class IndependentBaselineEvaluator:
    """Fixed no-cooperation counterfactual for MATD3-ShapleyQ."""

    def __init__(
        self,
        env: Any,
        config: Mapping[str, Any],
        device: str = "cpu",
    ) -> None:
        settings = config.get("algorithm", config)
        shapley_settings = settings.get("shapley_q", {})
        baseline = shapley_settings.get("baseline", {})
        self.mode = str(baseline.get("mode", "heuristic"))
        self.checkpoint_path = baseline.get("checkpoint")
        self.gamma = float(settings.get("gamma", 0.99))
        self.device = torch.device(device)

        allowed_modes = {"heuristic", "frozen_policy"}
        if self.mode not in allowed_modes:
            raise ValueError(f"Unsupported independent baseline mode: {self.mode}")
        if self.mode == "frozen_policy" and not self.checkpoint_path:
            raise ValueError("frozen_policy baseline requires an explicit checkpoint")

        self.project_root = Path(env.config["_project_root"])
        independent_config = copy.deepcopy(env.config)
        independent_config["p2p"]["enabled"] = False
        independent_config.setdefault("ess_sharing", {})["ess_p2p_enabled"] = False
        self.ind_env = MultiMicrogridEnv(independent_config, split=env.split)
        self.frozen_actors = None
        self.step_rewards: list[float] = []
        self.step_profits: list[float] = []
        self.discounted_returns: list[float] = []
        self.step_infos: list[dict[str, Any]] = []
        self.total_return = 0.0
        self.total_profit = 0.0

        if self.mode == "frozen_policy":
            self._load_frozen_policy(config, env.seed)

    @staticmethod
    def _physical_config(config: Mapping[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(dict(config))
        result.get("p2p", {}).pop("enabled", None)
        result.get("ess_sharing", {}).pop("ess_p2p_enabled", None)
        return result

    def scenario_signature(self, env: MultiMicrogridEnv) -> dict[str, Any]:
        if env.episode_context_df is None or env._episode_exogenous is None:
            raise RuntimeError("Environment must be reset before reading its scenario")
        return {
            "reset_options": dict(env.last_reset_options),
            "context": env.episode_context_df.to_json(
                orient="split", date_format="iso", double_precision=15
            ),
            "exogenous": {
                mg: {
                    field: tuple(np.asarray(values, dtype=np.float64))
                    for field, values in env._episode_exogenous[mg].items()
                }
                for mg in env.agent_ids
            },
            "local_forecasts": {
                mg: {
                    field: tuple(np.asarray(values, dtype=np.float64).reshape(-1))
                    for field, values in env.forecasts._local_arrays[mg].items()
                }
                for mg in env.agent_ids
            },
            "public_forecasts": {
                field: tuple(np.asarray(values, dtype=np.float64).reshape(-1))
                for field, values in env.forecasts._public_arrays.items()
            },
            "forecast_validity": tuple(
                np.asarray(env.forecasts._valid_array, dtype=np.float64).reshape(-1)
            ),
            "initial_soc": {
                mg: float(env.cores[mg].battery.initial_soc) for mg in env.agent_ids
            },
        }

    def _assert_common_scenario(self, env: MultiMicrogridEnv) -> None:
        if self._physical_config(env.config) != self._physical_config(self.ind_env.config):
            raise ValueError(
                "Independent baseline physical configuration does not match Grand scenario"
            )
        try:
            pd.testing.assert_frame_equal(
                env.episode_context_df, self.ind_env.episode_context_df
            )
        except AssertionError as exc:
            raise ValueError(
                "Independent baseline context does not match Grand scenario"
            ) from exc
        if self.scenario_signature(env) != self.scenario_signature(self.ind_env):
            raise ValueError("Independent baseline scenario does not match Grand scenario")

    def _load_frozen_policy(self, config: Mapping[str, Any], seed: int) -> None:
        checkpoint_path = Path(str(self.checkpoint_path))
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.project_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise ValueError(f"Independent baseline checkpoint does not exist: {checkpoint_path}")
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
        except Exception as exc:
            raise ValueError(f"Unable to load independent baseline checkpoint: {exc}") from exc
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Independent baseline checkpoint must contain a mapping")
        if checkpoint.get("algorithm_slug") != "matd3_dispatch":
            raise ValueError("Independent baseline checkpoint algorithm_slug is incompatible")
        if checkpoint.get("action_semantics") != "single_dispatch_v1":
            raise ValueError("Independent baseline checkpoint action semantics are incompatible")

        source = Matd3Dispatch.from_env(
            self.ind_env, config, device=str(self.device), seed=seed
        )
        if checkpoint.get("projection_signature") != source.projection_signature:
            raise ValueError("Independent baseline checkpoint physical projection is incompatible")
        if checkpoint.get("state_schema") != source.state_schema:
            raise ValueError("Independent baseline checkpoint state schema is incompatible")
        if checkpoint.get("observation_schemas") != source.observation_schemas:
            raise ValueError("Independent baseline checkpoint observation schemas are incompatible")
        actor_state = checkpoint.get("actors")
        if not isinstance(actor_state, Mapping):
            raise ValueError("Independent baseline checkpoint actor state dict is missing")
        try:
            source.actors.load_state_dict(actor_state, strict=True)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Independent baseline checkpoint actor state dict is incompatible: {exc}"
            ) from exc
        self.frozen_actors = source.actors
        self.frozen_actors.requires_grad_(False)
        self.frozen_actors.eval()

    def _heuristic_actions(self) -> dict[str, np.ndarray]:
        midpoint = max(1, self.ind_env.episode_length // 2)
        dispatch = -1.0 if self.ind_env.t < midpoint else 1.0
        return {
            mg: np.asarray([dispatch], dtype=np.float32)
            for mg in self.ind_env.agent_ids
        }

    def _frozen_policy_actions(
        self, observations: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if self.frozen_actors is None:
            raise RuntimeError("Frozen independent baseline policy is unavailable")
        with torch.no_grad():
            return {
                mg: self.frozen_actors[mg](
                    torch.as_tensor(
                        observations[mg], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                )[0].cpu().numpy()
                for mg in self.ind_env.agent_ids
            }

    def reset_episode(self, env: MultiMicrogridEnv) -> None:
        if not hasattr(env, "last_reset_options"):
            raise RuntimeError("Grand environment must be reset before its baseline")
        self.step_rewards = []
        self.step_profits = []
        self.discounted_returns = []
        self.step_infos = []
        observations, _ = self.ind_env.reset(**dict(env.last_reset_options))
        self._assert_common_scenario(env)

        done = False
        while not done:
            actions = (
                self._heuristic_actions()
                if self.mode == "heuristic"
                else self._frozen_policy_actions(observations)
            )
            observations, rewards, done, info = self.ind_env.step(actions)
            self.step_rewards.append(float(rewards[self.ind_env.agent_ids[0]]))
            self.step_profits.append(float(info["team_profit"]))
            self.step_infos.append(info)

        self.total_return = float(sum(self.step_rewards))
        self.total_profit = float(sum(self.step_profits))
        running = 0.0
        self.discounted_returns = [0.0] * len(self.step_rewards)
        for index in range(len(self.step_rewards) - 1, -1, -1):
            running = self.step_rewards[index] + self.gamma * running
            self.discounted_returns[index] = float(running)

    def get_step_data(self, t: int) -> dict[str, float]:
        if not 0 <= t < len(self.step_rewards):
            raise IndexError(f"Independent baseline step {t} is unavailable")
        return {
            "reward": self.step_rewards[t],
            "profit": self.step_profits[t],
            "return_to_go": self.discounted_returns[t],
            "next_return_to_go": (
                self.discounted_returns[t + 1]
                if t + 1 < len(self.discounted_returns)
                else 0.0
            ),
        }

    def episode_metrics(self) -> dict[str, float]:
        return {
            "independent_baseline_return": self.total_return,
            "independent_baseline_profit": self.total_profit,
        }
