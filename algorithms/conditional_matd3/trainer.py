from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from algorithms.conditional_matd3.conditional_matd3 import ConditionalMATD3
from algorithms.conditional_matd3.replay_buffer import JointReplayBuffer
from envs.multi_microgrid_env import MultiMicrogridEnv


@dataclass
class ConditionalMATD3Trainer:
    """Minimal training orchestrator; no CrossGate or equilibrium verification."""

    env: MultiMicrogridEnv
    algorithm: ConditionalMATD3
    replay: JointReplayBuffer
    batch_size: int
    warmup_steps: int = 5000
    updates_per_step: int = 1
    env_steps: int = 0
    episodes: int = 0

    @classmethod
    def from_env(cls, env: MultiMicrogridEnv, device: str = "cpu") -> "ConditionalMATD3Trainer":
        algo_cfg = env.config["algorithm"]
        algorithm = ConditionalMATD3(
            env.agent_ids,
            env.observation_dims,
            env.global_state_dim,
            env.config,
            device=device,
        )
        replay = JointReplayBuffer(
            capacity=int(algo_cfg.get("replay_capacity", 500000)),
            agent_ids=env.agent_ids,
            observation_dims=env.observation_dims,
            state_dim=env.global_state_dim,
            action_dim=env.action_dim,
            seed=int(env.config.get("env", {}).get("seed", 42)),
        )
        return cls(
            env=env,
            algorithm=algorithm,
            replay=replay,
            batch_size=int(algo_cfg.get("batch_size", 256)),
            warmup_steps=int(algo_cfg.get("warmup_steps", 5000)),
            updates_per_step=int(algo_cfg.get("updates_per_step", 1)),
        )

    def run_episode(
        self,
        day: str | None = None,
        training: bool = True,
        deterministic_eval: bool = False,
        seed: int | None = None,
        active_agents: set[str] | None = None,
        environment_deterministic: bool | None = None,
    ) -> Dict[str, Any]:
        if active_agents is not None:
            self.algorithm._resolve_active_agents(active_agents)
        if environment_deterministic is None:
            environment_deterministic = bool(
                deterministic_eval or not training
            )
        obs, reset_info = self.env.reset(
            day=day,
            seed=seed,
            deterministic=environment_deterministic,
        )
        state = reset_info["global_state"]
        done = False
        totals = {
            mg: {
                "reward": 0.0,
                "cost": 0.0,
                "p2p_kwh": 0.0,
                "grid_import_kwh": 0.0,
                "grid_export_kwh": 0.0,
                "carbon_kg": 0.0,
                "carbon_cost": 0.0,
                "generator_cost": 0.0,
                "battery_cost": 0.0,
            }
            for mg in self.env.agent_ids
        }
        market_trade = 0.0
        renewable_curtailment = 0.0
        grid_import = 0.0
        grid_export = 0.0
        update_sums: Dict[str, float] = {}
        update_counts: Dict[str, int] = {}
        critic_update_count = 0
        actor_update_count = 0

        while not done:
            warmup = training and self.env_steps < self.warmup_steps
            action_kwargs: Dict[str, Any] = {}
            if training and active_agents is not None:
                action_kwargs = {
                    "explore_agents": active_agents,
                    "warmup_agents": active_agents,
                }
            actions, _ = self.algorithm.select_actions(
                obs,
                env_step=self.env_steps,
                explore=training,
                warmup=warmup,
                **action_kwargs,
            )
            next_obs, rewards, done, info = self.env.step(actions)
            next_state = info["global_state_after"]
            if training:
                self.replay.add(
                    state,
                    obs,
                    info["executed_actions"],
                    rewards,
                    next_state,
                    next_obs,
                    done,
                )
                self.env_steps += 1
                if self.env_steps >= self.warmup_steps and len(self.replay) >= self.batch_size:
                    for _ in range(max(1, self.updates_per_step)):
                        batch = self.replay.sample(self.batch_size, device=self.algorithm.device)
                        update_logs = (
                            self.algorithm.update(batch)
                            if active_agents is None
                            else self.algorithm.update(batch, active_agents=active_agents)
                        )
                        if not np.isfinite(
                            np.asarray(list(update_logs.values()), dtype=np.float64)
                        ).all():
                            raise RuntimeError(
                                "Conditional MATD3 update produced non-finite loss or Q metrics."
                            )
                        critic_update_count += 1
                        actor_update_count += int(update_logs.get("policy_updated", 0.0) > 0.5)
                        for name, value in update_logs.items():
                            if name == "policy_updated":
                                continue
                            update_sums[name] = update_sums.get(name, 0.0) + float(value)
                            update_counts[name] = update_counts.get(name, 0) + 1

            for mg in self.env.agent_ids:
                totals[mg]["reward"] += float(rewards[mg])
                totals[mg]["cost"] += float(info["costs"][mg]["total_cost"])
                p2p = info["market"]["by_microgrid"][mg]
                totals[mg]["p2p_kwh"] += float(p2p["p2p_buy_kwh"] + p2p["p2p_sell_kwh"])
                settlement = info["settlements"][mg]
                costs = info["costs"][mg]
                totals[mg]["grid_import_kwh"] += (
                    float(settlement["grid_buy_kw"]) * self.env.delta_t
                )
                totals[mg]["grid_export_kwh"] += (
                    float(settlement["grid_sell_kw"]) * self.env.delta_t
                )
                totals[mg]["carbon_kg"] += float(costs["carbon_kg"])
                totals[mg]["carbon_cost"] += float(costs["carbon_cost"])
                totals[mg]["generator_cost"] += float(costs["generator_cost"])
                totals[mg]["battery_cost"] += float(costs["battery_degradation_cost"])
            market_trade += float(info["market"]["total_trade_kwh"])
            renewable_curtailment += sum(
                float(value["renewable_curtailment_kw"]) * self.env.delta_t
                for value in info["settlements"].values()
            )
            grid_import += sum(
                float(value["grid_buy_kw"]) * self.env.delta_t
                for value in info["settlements"].values()
            )
            grid_export += sum(
                float(value["grid_sell_kw"]) * self.env.delta_t
                for value in info["settlements"].values()
            )
            obs, state = next_obs, next_state

        self.episodes += int(training)
        if not all(np.isfinite(value["reward"]) for value in totals.values()):
            raise RuntimeError("Episode produced non-finite returns.")
        episode_update_metrics = {
            name: update_sums[name] / update_counts[name]
            for name in update_sums
        }
        episode_update_metrics["critic_update_count"] = float(critic_update_count)
        episode_update_metrics["actor_update_count"] = float(actor_update_count)
        episode_update_metrics["policy_updated"] = float(actor_update_count > 0)
        return {
            "by_microgrid": totals,
            "market_trade_kwh": market_trade,
            "renewable_curtailment_kwh": renewable_curtailment,
            "grid_import_kwh": grid_import,
            "grid_export_kwh": grid_export,
            "env_steps": self.env_steps,
            "replay_size": len(self.replay),
            "last_update": episode_update_metrics,
        }
