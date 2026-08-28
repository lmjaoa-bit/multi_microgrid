from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from algorithms.conditional_matd3.config_validation import validate_algorithm_config
from algorithms.conditional_matd3.networks import CentralizedCritic, ConditionalActor


@contextmanager
def _freeze(modules: Sequence[nn.Module]):
    previous: list[list[bool]] = []
    for module in modules:
        flags = [parameter.requires_grad for parameter in module.parameters()]
        previous.append(flags)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for module, flags in zip(modules, previous):
            for parameter, flag in zip(module.parameters(), flags):
                parameter.requires_grad_(flag)


def _linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(end)
    fraction = min(1.0, max(0.0, float(step) / float(decay_steps)))
    return float(start + fraction * (end - start))


class ConditionalMATD3:
    """Three-agent CTDE MATD3 with counterfactual generator commitment learning.

    Action convention for the critic/replay/environment is always:
      [battery, hard_commitment, generator_setpoint, p2p_bid]
    where hard commitment is -1 (OFF) or +1 (ON), and generator_setpoint is
    forced to -1 whenever commitment is OFF.
    """

    def __init__(
        self,
        agent_ids: Sequence[str],
        observation_dims: Mapping[str, int],
        state_dim: int,
        config: Mapping[str, Any],
        device: str | torch.device = "cpu",
    ) -> None:
        self.agent_ids = [str(x) for x in agent_ids]
        self.n_agents = len(self.agent_ids)
        if self.n_agents != 3:
            raise ValueError("The current Conditional MATD3 baseline requires exactly 3 agents.")
        self.agent_index = {mg: i for i, mg in enumerate(self.agent_ids)}
        self.state_dim = int(state_dim)
        self.action_dim = 4
        self.joint_action_dim = self.n_agents * self.action_dim
        self.device = torch.device(device)
        algo = config.get("algorithm", config)
        validate_algorithm_config(algo)
        self.gamma = float(algo.get("gamma", 0.99))
        self.tau = float(algo.get("tau", 0.005))
        self.policy_delay = int(algo.get("policy_delay", 2))
        self.actor_update_mode = str(algo.get("actor_update_mode", "synchronous"))
        if self.actor_update_mode not in {"synchronous", "sequential"}:
            raise ValueError("actor_update_mode must be synchronous or sequential.")

        actor_hidden = tuple(int(x) for x in algo.get("actor_hidden_dims", [256, 256, 128]))
        critic_hidden = tuple(int(x) for x in algo.get("critic_hidden_dims", [256, 256, 128]))
        actor_lr = float(algo.get("actor_lr", 3e-4))
        critic_lr = float(algo.get("critic_lr", 3e-4))

        self.actors: Dict[str, ConditionalActor] = {}
        self.target_actors: Dict[str, ConditionalActor] = {}
        self.critics1: Dict[str, CentralizedCritic] = {}
        self.critics2: Dict[str, CentralizedCritic] = {}
        self.target_critics1: Dict[str, CentralizedCritic] = {}
        self.target_critics2: Dict[str, CentralizedCritic] = {}
        self.actor_optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.critic_optimizers: Dict[str, torch.optim.Optimizer] = {}

        for mg in self.agent_ids:
            actor = ConditionalActor(int(observation_dims[mg]), actor_hidden).to(self.device)
            target_actor = copy.deepcopy(actor).to(self.device)
            critic1 = CentralizedCritic(self.state_dim, self.joint_action_dim, critic_hidden).to(self.device)
            critic2 = CentralizedCritic(self.state_dim, self.joint_action_dim, critic_hidden).to(self.device)
            target_critic1 = copy.deepcopy(critic1).to(self.device)
            target_critic2 = copy.deepcopy(critic2).to(self.device)
            self.actors[mg] = actor
            self.target_actors[mg] = target_actor
            self.critics1[mg] = critic1
            self.critics2[mg] = critic2
            self.target_critics1[mg] = target_critic1
            self.target_critics2[mg] = target_critic2
            self.actor_optimizers[mg] = torch.optim.Adam(actor.parameters(), lr=actor_lr)
            self.critic_optimizers[mg] = torch.optim.Adam(
                list(critic1.parameters()) + list(critic2.parameters()), lr=critic_lr
            )

        commitment = algo.get("commitment", {})
        self.commit_epsilon_start = float(commitment.get("epsilon_start", 0.20))
        self.commit_epsilon_end = float(commitment.get("epsilon_end", 0.03))
        self.commit_epsilon_decay = int(commitment.get("epsilon_decay_steps", 100000))
        self.commit_temperature = float(commitment.get("soft_target_temperature", 1.0))
        self.commit_target_min = float(commitment.get("soft_target_min", 0.10))
        self.commit_target_max = float(commitment.get("soft_target_max", 0.90))
        self.q_scale_beta = float(commitment.get("q_scale_ema_beta", 0.99))
        self.q_scale_eps = float(commitment.get("q_scale_epsilon", 1e-3))
        self.q_scale_ema: Dict[str, float | None] = {mg: None for mg in self.agent_ids}

        exploration = algo.get("exploration_noise", {})
        self.exploration_start = {k: float(v) for k, v in exploration.get("start", {}).items()}
        self.exploration_end = {k: float(v) for k, v in exploration.get("end", {}).items()}
        self.exploration_decay = int(exploration.get("decay_steps", 100000))

        smoothing = algo.get("target_smoothing", {})
        self.target_sigma = {k: float(v) for k, v in smoothing.get("sigma", {}).items()}
        self.target_clip = {k: float(v) for k, v in smoothing.get("clip", {}).items()}

        weights = algo.get("actor_loss_weights", {})
        self.loss_weights = {
            "commitment": float(weights.get("commitment", 1.0)),
            "common": float(weights.get("common", 1.0)),
            "generator": float(weights.get("generator", 1.0)),
        }
        self.update_counter = 0
        self.rng = np.random.default_rng(int(config.get("env", {}).get("seed", 42)))

    def commitment_epsilon(self, env_step: int) -> float:
        return _linear_schedule(
            self.commit_epsilon_start,
            self.commit_epsilon_end,
            env_step,
            self.commit_epsilon_decay,
        )

    def exploration_sigma(self, head: str, env_step: int) -> float:
        return _linear_schedule(
            self.exploration_start.get(head, 0.0),
            self.exploration_end.get(head, 0.0),
            env_step,
            self.exploration_decay,
        )

    def _resolve_active_agents(self, agents: set[str] | None) -> list[str]:
        if agents is None:
            return list(self.agent_ids)
        normalized = {str(agent) for agent in agents}
        unknown = normalized - set(self.agent_ids)
        if unknown:
            raise ValueError(f"Unknown agent IDs: {sorted(unknown)}")
        if not normalized:
            raise ValueError("Agent selection cannot be empty.")
        return [mg for mg in self.agent_ids if mg in normalized]

    @torch.no_grad()
    def select_actions(
        self,
        observations: Mapping[str, np.ndarray],
        env_step: int = 0,
        explore: bool = True,
        warmup: bool = False,
        explore_agents: set[str] | None = None,
        warmup_agents: set[str] | None = None,
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        if explore_agents is not None:
            self._resolve_active_agents(explore_agents)
        if warmup_agents is not None:
            self._resolve_active_agents(warmup_agents)
        actions: Dict[str, np.ndarray] = {}
        diagnostics: Dict[str, Dict[str, float]] = {}
        for mg in self.agent_ids:
            agent_explore = explore and (
                explore_agents is None or mg in explore_agents
            )
            agent_warmup = warmup and (
                warmup_agents is None or mg in warmup_agents
            )
            epsilon = self.commitment_epsilon(env_step) if agent_explore else 0.0
            if agent_warmup:
                battery = float(self.rng.uniform(-1.0, 1.0))
                generator = float(self.rng.uniform(-1.0, 1.0))
                bid = float(self.rng.uniform(-1.0, 1.0))
                commitment = 1.0 if self.rng.random() < 0.5 else -1.0
                p_on = 0.5
            else:
                obs = torch.as_tensor(observations[mg], dtype=torch.float32, device=self.device).unsqueeze(0)
                outputs = self.actors[mg](obs)
                battery = float(outputs["battery"].item())
                generator = float(outputs["generator_setpoint"].item())
                bid = float(outputs["p2p_bid"].item())
                p_on = float(torch.sigmoid(outputs["commitment_logit"]).item())
                commitment = 1.0 if p_on >= 0.5 else -1.0
                if agent_explore:
                    battery += float(self.rng.normal(0.0, self.exploration_sigma("battery", env_step)))
                    generator += float(self.rng.normal(0.0, self.exploration_sigma("generator_setpoint", env_step)))
                    bid += float(self.rng.normal(0.0, self.exploration_sigma("p2p_bid", env_step)))
                    if self.rng.random() < epsilon:
                        commitment *= -1.0

            battery = float(np.clip(battery, -1.0, 1.0))
            generator = float(np.clip(generator, -1.0, 1.0))
            bid = float(np.clip(bid, -1.0, 1.0))
            executed_setpoint = generator if commitment > 0.0 else -1.0
            actions[mg] = np.asarray([battery, commitment, executed_setpoint, bid], dtype=np.float32)
            diagnostics[mg] = {
                "p_on": p_on,
                "commitment_epsilon": epsilon,
                "executed_commitment": commitment,
            }
        return actions, diagnostics

    def _target_joint_action(
        self,
        next_obs: torch.Tensor,
        smoothing_agents: set[str] | None = None,
    ) -> torch.Tensor:
        if smoothing_agents is not None:
            self._resolve_active_agents(smoothing_agents)
        actions = []
        for index, mg in enumerate(self.agent_ids):
            outputs = self.target_actors[mg](next_obs[:, index, :])
            smooth = smoothing_agents is None or mg in smoothing_agents
            battery = (
                self._smooth(outputs["battery"], "battery")
                if smooth else outputs["battery"].clamp(-1.0, 1.0)
            )
            generator = (
                self._smooth(outputs["generator_setpoint"], "generator_setpoint")
                if smooth else outputs["generator_setpoint"].clamp(-1.0, 1.0)
            )
            bid = (
                self._smooth(outputs["p2p_bid"], "p2p_bid")
                if smooth else outputs["p2p_bid"].clamp(-1.0, 1.0)
            )
            probability = torch.sigmoid(outputs["commitment_logit"])
            commitment = torch.where(probability >= 0.5, torch.ones_like(probability), -torch.ones_like(probability))
            generator = torch.where(commitment > 0.0, generator, -torch.ones_like(generator))
            actions.append(torch.cat([battery, commitment, generator, bid], dim=-1))
        return torch.cat(actions, dim=-1)

    def _smooth(self, value: torch.Tensor, head: str) -> torch.Tensor:
        sigma = self.target_sigma.get(head, 0.0)
        clip = self.target_clip.get(head, 0.0)
        if sigma <= 0.0:
            return value.clamp(-1.0, 1.0)
        noise = torch.randn_like(value) * sigma
        if clip > 0.0:
            noise = noise.clamp(-clip, clip)
        return (value + noise).clamp(-1.0, 1.0)

    @staticmethod
    def _replace_agent_action(joint: torch.Tensor, agent_index: int, action: torch.Tensor) -> torch.Tensor:
        start = agent_index * 4
        end = start + 4
        return torch.cat([joint[:, :start], action, joint[:, end:]], dim=-1)

    @torch.no_grad()
    def _current_hard_joint_action(self, obs: torch.Tensor) -> torch.Tensor:
        actions = []
        for index, mg in enumerate(self.agent_ids):
            outputs = self.actors[mg](obs[:, index, :])
            actions.append(ConditionalActor.hard_action(outputs))
        return torch.cat(actions, dim=-1)

    def _critic_update(
        self,
        batch: Mapping[str, torch.Tensor],
        active_agents: set[str] | None = None,
    ) -> Dict[str, float]:
        state = batch["state"]
        action = batch["action"].reshape(batch["action"].shape[0], -1)
        reward = batch["reward"]
        next_state = batch["next_state"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        logs: Dict[str, float] = {}
        with torch.no_grad():
            next_joint = self._target_joint_action(
                next_obs,
                smoothing_agents=active_agents,
            )

        for index, mg in enumerate(self.agent_ids):
            if active_agents is not None and mg not in active_agents:
                continue
            with torch.no_grad():
                target_q1 = self.target_critics1[mg](next_state, next_joint)
                target_q2 = self.target_critics2[mg](next_state, next_joint)
                target_q = torch.minimum(target_q1, target_q2)
                y = reward[:, index:index + 1] + self.gamma * (1.0 - done) * target_q
            q1 = self.critics1[mg](state, action)
            q2 = self.critics2[mg](state, action)
            loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            optimizer = self.critic_optimizers[mg]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.critics1[mg].parameters()) + list(self.critics2[mg].parameters()),
                10.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            logs[f"{mg}/critic_loss"] = float(loss.detach().item())
            q1_mean = float(q1.detach().mean().item())
            q2_mean = float(q2.detach().mean().item())
            logs[f"{mg}/q1_mean"] = q1_mean
            logs[f"{mg}/q2_mean"] = q2_mean
            logs[f"{mg}/q_mean"] = float(0.5 * (q1_mean + q2_mean))
            logs[f"{mg}/q_disagreement"] = float(
                torch.mean(torch.abs(q1.detach() - q2.detach())).item()
            )
        return logs

    def _actor_loss(
        self,
        mg: str,
        obs: torch.Tensor,
        state: torch.Tensor,
        frozen_joint: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        index = self.agent_index[mg]
        outputs = self.actors[mg](obs[:, index, :])
        battery = outputs["battery"]
        generator = outputs["generator_setpoint"]
        bid = outputs["p2p_bid"]
        logit = outputs["commitment_logit"]
        p_on = torch.sigmoid(logit)

        off_common = torch.cat([battery, -torch.ones_like(logit), -torch.ones_like(generator), bid], dim=-1)
        on_common = torch.cat([battery, torch.ones_like(logit), generator.detach(), bid], dim=-1)
        joint_off_common = self._replace_agent_action(frozen_joint, index, off_common)
        joint_on_common = self._replace_agent_action(frozen_joint, index, on_common)
        q1_off = self.critics1[mg](state, joint_off_common)
        q2_off = self.critics2[mg](state, joint_off_common)
        q1_on = self.critics1[mg](state, joint_on_common)
        q2_on = self.critics2[mg](state, joint_on_common)
        q_off = 0.5 * (q1_off + q2_off)
        q_on = 0.5 * (q1_on + q2_on)

        delta_q = (q_on.detach() - q_off.detach())
        batch_scale = max(self.q_scale_eps, float(delta_q.abs().mean().item()))
        previous_scale = self.q_scale_ema[mg]
        if previous_scale is None:
            self.q_scale_ema[mg] = batch_scale
        else:
            self.q_scale_ema[mg] = (
                self.q_scale_beta * previous_scale + (1.0 - self.q_scale_beta) * batch_scale
            )
        scale = max(self.q_scale_eps, float(self.q_scale_ema[mg]))
        soft_target = torch.sigmoid(delta_q / max(self.commit_temperature * scale, self.q_scale_eps))
        soft_target = soft_target.clamp(self.commit_target_min, self.commit_target_max)
        commitment_loss = F.binary_cross_entropy_with_logits(logit, soft_target)

        expected_q = (1.0 - p_on.detach()) * q_off + p_on.detach() * q_on
        common_loss = -expected_q.mean()

        gen_action = torch.cat(
            [battery.detach(), torch.ones_like(logit), generator, bid.detach()], dim=-1
        )
        joint_gen = self._replace_agent_action(frozen_joint, index, gen_action)
        q_gen = 0.5 * (
            self.critics1[mg](state, joint_gen) + self.critics2[mg](state, joint_gen)
        )
        generator_loss = -q_gen.mean()

        total = (
            self.loss_weights["commitment"] * commitment_loss
            + self.loss_weights["common"] * common_loss
            + self.loss_weights["generator"] * generator_loss
        )
        logs = {
            f"{mg}/actor_loss": float(total.detach().item()),
            f"{mg}/commitment_loss": float(commitment_loss.detach().item()),
            f"{mg}/common_loss": float(common_loss.detach().item()),
            f"{mg}/generator_loss": float(generator_loss.detach().item()),
            f"{mg}/q_on": float(q_on.detach().mean().item()),
            f"{mg}/q_off": float(q_off.detach().mean().item()),
            f"{mg}/delta_q": float(delta_q.mean().item()),
            f"{mg}/p_on": float(p_on.detach().mean().item()),
            f"{mg}/commit_target": float(soft_target.mean().item()),
            f"{mg}/q_scale": float(scale),
        }
        return total, logs

    def _actor_update_synchronous(
        self,
        batch: Mapping[str, torch.Tensor],
        active_agents: set[str] | None = None,
    ) -> Dict[str, float]:
        selected_agents = self._resolve_active_agents(active_agents)
        obs = batch["obs"]
        state = batch["state"]
        with torch.no_grad():
            frozen_joint = self._current_hard_joint_action(obs)
        critic_modules = [
            module
            for mg in self.agent_ids
            for module in (self.critics1[mg], self.critics2[mg])
        ]
        for mg in selected_agents:
            self.actor_optimizers[mg].zero_grad(set_to_none=True)
        logs: Dict[str, float] = {}
        with _freeze(critic_modules):
            losses = []
            for mg in selected_agents:
                loss, local_logs = self._actor_loss(mg, obs, state, frozen_joint)
                losses.append(loss)
                logs.update(local_logs)
            for loss in losses:
                loss.backward()
            for mg in selected_agents:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actors[mg].parameters(), 10.0, error_if_nonfinite=True
                )
                logs[f"{mg}/actor_grad_norm"] = float(grad_norm)
            for mg in selected_agents:
                self.actor_optimizers[mg].step()
        return logs

    def _actor_update_sequential(
        self,
        batch: Mapping[str, torch.Tensor],
        active_agents: set[str] | None = None,
    ) -> Dict[str, float]:
        selected_agents = self._resolve_active_agents(active_agents)
        obs = batch["obs"]
        state = batch["state"]
        logs: Dict[str, float] = {}
        critic_modules = [
            module
            for mg in self.agent_ids
            for module in (self.critics1[mg], self.critics2[mg])
        ]
        with _freeze(critic_modules):
            for mg in selected_agents:
                with torch.no_grad():
                    frozen_joint = self._current_hard_joint_action(obs)
                optimizer = self.actor_optimizers[mg]
                optimizer.zero_grad(set_to_none=True)
                loss, local_logs = self._actor_loss(mg, obs, state, frozen_joint)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actors[mg].parameters(), 10.0, error_if_nonfinite=True
                )
                optimizer.step()
                logs.update(local_logs)
                logs[f"{mg}/actor_grad_norm"] = float(grad_norm)
        return logs

    def update(
        self,
        batch: Mapping[str, torch.Tensor],
        active_agents: set[str] | None = None,
    ) -> Dict[str, float]:
        if active_agents is not None:
            self._resolve_active_agents(active_agents)
        self.update_counter += 1
        logs = self._critic_update(batch, active_agents=active_agents)
        if self.update_counter % self.policy_delay == 0:
            if self.actor_update_mode == "synchronous":
                logs.update(
                    self._actor_update_synchronous(
                        batch,
                        active_agents=active_agents,
                    )
                )
            else:
                logs.update(
                    self._actor_update_sequential(
                        batch,
                        active_agents=active_agents,
                    )
                )
            self._soft_update_targets(active_agents=active_agents)
            logs["policy_updated"] = 1.0
        else:
            logs["policy_updated"] = 0.0
        return logs

    @torch.no_grad()
    def _soft_update_targets(self, active_agents: set[str] | None = None) -> None:
        for mg in self._resolve_active_agents(active_agents):
            self._soft_update(self.actors[mg], self.target_actors[mg])
            self._soft_update(self.critics1[mg], self.target_critics1[mg])
            self._soft_update(self.critics2[mg], self.target_critics2[mg])

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(source_param.data, alpha=self.tau)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actors": {mg: self.actors[mg].state_dict() for mg in self.agent_ids},
            "target_actors": {mg: self.target_actors[mg].state_dict() for mg in self.agent_ids},
            "critics1": {mg: self.critics1[mg].state_dict() for mg in self.agent_ids},
            "critics2": {mg: self.critics2[mg].state_dict() for mg in self.agent_ids},
            "target_critics1": {mg: self.target_critics1[mg].state_dict() for mg in self.agent_ids},
            "target_critics2": {mg: self.target_critics2[mg].state_dict() for mg in self.agent_ids},
            "actor_optimizers": {mg: self.actor_optimizers[mg].state_dict() for mg in self.agent_ids},
            "critic_optimizers": {mg: self.critic_optimizers[mg].state_dict() for mg in self.agent_ids},
            "q_scale_ema": dict(self.q_scale_ema),
            "update_counter": self.update_counter,
        }

    def get_policy_state_dict(self) -> Dict[str, Any]:
        """Return the lightweight state required for deterministic policy evaluation."""
        return {
            "actors": {mg: self.actors[mg].state_dict() for mg in self.agent_ids},
            "metadata": {
                "algorithm": "conditional_matd3",
                "agent_ids": list(self.agent_ids),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "actor_update_mode": self.actor_update_mode,
            },
        }

    def load_policy_state_dict(
        self,
        state: Mapping[str, Any],
        sync_targets: bool = True,
    ) -> None:
        """Load online actors from a lightweight policy or best-model checkpoint."""
        actors = state.get("actors")
        if not isinstance(actors, Mapping):
            raise ValueError("Policy state must contain an 'actors' mapping.")
        missing = set(self.agent_ids) - set(actors)
        if missing:
            raise ValueError(f"Policy state is missing actors: {sorted(missing)}")
        for mg in self.agent_ids:
            self.actors[mg].load_state_dict(actors[mg])
            if sync_targets:
                self.target_actors[mg].load_state_dict(actors[mg])

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for mg in self.agent_ids:
            self.actors[mg].load_state_dict(state["actors"][mg])
            self.target_actors[mg].load_state_dict(state["target_actors"][mg])
            self.critics1[mg].load_state_dict(state["critics1"][mg])
            self.critics2[mg].load_state_dict(state["critics2"][mg])
            self.target_critics1[mg].load_state_dict(state["target_critics1"][mg])
            self.target_critics2[mg].load_state_dict(state["target_critics2"][mg])
            if "actor_optimizers" in state:
                self.actor_optimizers[mg].load_state_dict(state["actor_optimizers"][mg])
            if "critic_optimizers" in state:
                self.critic_optimizers[mg].load_state_dict(state["critic_optimizers"][mg])
        self.q_scale_ema = {mg: state.get("q_scale_ema", {}).get(mg) for mg in self.agent_ids}
        self.update_counter = int(state.get("update_counter", 0))
