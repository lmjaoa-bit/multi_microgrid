from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from algorithms.matd3.actor import Actor
from algorithms.matd3.critic import TwinCritic
from algorithms.matd3.replay_buffer import ReplayBuffer


class MATD3:
    """Canonical CTDE MATD3: three actors, per-agent centralized twin critics, shared team reward."""

    def __init__(self, agent_ids: list[str], observation_dims: Mapping[str, int],
                 global_state_dim: int, action_dim: int = 2,
                 hidden_dims: tuple[int, ...] = (128, 128), actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 policy_delay: int = 2, target_policy_noise: float = 0.2,
                 target_noise_clip: float = 0.5, replay_capacity: int = 200_000,
                 seed: int = 42, device: str = "cpu") -> None:
        torch.manual_seed(seed)
        self.agent_ids = list(agent_ids)
        self.action_dim = int(action_dim)
        self.gamma, self.tau = float(gamma), float(tau)
        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.device = torch.device(device)

        # Actors
        self.actors = nn.ModuleDict({
            mg: Actor(int(observation_dims[mg]), action_dim, hidden_dims)
            for mg in self.agent_ids
        }).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.actor_optimizers = {
            mg: torch.optim.Adam(self.actors[mg].parameters(), lr=actor_lr)
            for mg in self.agent_ids
        }

        # Per-agent Centralized Twin Critics
        joint_action_dim = action_dim * len(self.agent_ids)
        self.critics = nn.ModuleDict({
            mg: TwinCritic(global_state_dim, joint_action_dim, hidden_dims)
            for mg in self.agent_ids
        }).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device)
        self.critic_optimizers = {
            mg: torch.optim.Adam(self.critics[mg].parameters(), lr=critic_lr)
            for mg in self.agent_ids
        }

        self.replay = ReplayBuffer(replay_capacity, seed)
        self.update_steps = 0
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_env(cls, env: Any, config: Mapping[str, Any], device: str = "cpu",
                 seed: int | None = None) -> "MATD3":
        algorithm = config.get("algorithm", config)
        allowed = {"hidden_dims", "actor_lr", "critic_lr", "gamma", "tau", "policy_delay",
                   "target_policy_noise", "target_noise_clip", "replay_capacity"}
        kwargs = {key: algorithm[key] for key in allowed if key in algorithm}
        if "hidden_dims" in kwargs:
            kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])
        return cls(env.agent_ids, env.observation_dims, env.global_state_dim, env.action_dim,
                   seed=env.seed if seed is None else int(seed), device=device, **kwargs)

    def select_actions(self, observations: Mapping[str, np.ndarray], noise_std: float = 0.0) -> dict[str, np.ndarray]:
        actions = {}
        for mg in self.agent_ids:
            obs = torch.as_tensor(observations[mg], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action = self.actors[mg](obs).cpu().numpy()[0]
            if noise_std > 0.0:
                action += self.rng.normal(0.0, noise_std, size=action.shape)
            actions[mg] = np.clip(action, -1.0, 1.0).astype(np.float32)
        return actions

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.mul_(1.0 - self.tau).add_(self.tau * parameter.data)

    @staticmethod
    def _gradient_norm(module: nn.Module) -> float:
        """Read the L2 gradient norm without changing the optimizer step."""
        squared = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().norm(2).item()) ** 2
        return float(squared ** 0.5)

    def update(self, batch_size: int) -> dict[str, Any]:
        if len(self.replay) < batch_size:
            return {}
        batch = self.replay.sample(batch_size, self.agent_ids, self.device)
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        actions = batch["actions"]
        assert isinstance(observations, dict) and isinstance(next_observations, dict) and isinstance(actions, dict)
        joint_action = torch.cat([actions[mg] for mg in self.agent_ids], dim=-1)

        # 1. Target joint action with target policy smoothing
        with torch.no_grad():
            target_parts = []
            for mg in self.agent_ids:
                target = self.target_actors[mg](next_observations[mg])
                noise = torch.randn_like(target).mul(self.target_policy_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip
                )
                target_parts.append((target + noise).clamp(-1.0, 1.0))
            target_joint = torch.cat(target_parts, dim=-1)

            # Target Q per agent
            target_q = {}
            for mg in self.agent_ids:
                q1_target, q2_target = self.target_critics[mg](batch["next_global_state"], target_joint)
                target_q[mg] = batch["team_reward"] + (1.0 - batch["done"]) * self.gamma * torch.minimum(q1_target, q2_target)

        # 2. Update each agent's centralized twin critic
        critic_losses = {}
        critic1_losses = {}
        critic2_losses = {}
        critic_grad_norms = {}
        q_stats = {}
        q1_means = {}
        q2_means = {}
        target_q_means = {}
        td_error_means = {}
        td_error_stds = {}
        for mg in self.agent_ids:
            q1, q2 = self.critics[mg](batch["global_state"], joint_action)
            loss_c1 = nn.functional.mse_loss(q1, target_q[mg])
            loss_c2 = nn.functional.mse_loss(q2, target_q[mg])
            loss_c = loss_c1 + loss_c2
            self.critic_optimizers[mg].zero_grad()
            loss_c.backward()
            critic_grad_norms[mg] = self._gradient_norm(self.critics[mg])
            self.critic_optimizers[mg].step()
            critic_losses[mg] = float(loss_c.item())
            critic1_losses[mg] = float(loss_c1.item())
            critic2_losses[mg] = float(loss_c2.item())
            q_min = torch.minimum(q1, q2)
            td_error = target_q[mg] - q_min
            q1_means[mg] = float(q1.mean().item())
            q2_means[mg] = float(q2.mean().item())
            target_q_means[mg] = float(target_q[mg].mean().item())
            td_error_means[mg] = float(td_error.mean().item())
            td_error_stds[mg] = float(td_error.std(unbiased=False).item())
            q_stats[mg] = {
                "mean": float(q_min.mean().item()),
                "min": float(q_min.min().item()),
                "max": float(q_min.max().item()),
                "std": float(q_min.std().item()),
            }

        self.update_steps += 1
        metrics: dict[str, Any] = {
            "critic_loss": float(np.mean(list(critic_losses.values()))),
            "critic_losses": critic_losses,
            "critic1_loss": float(np.mean(list(critic1_losses.values()))),
            "critic2_loss": float(np.mean(list(critic2_losses.values()))),
            "critic1_losses": critic1_losses,
            "critic2_losses": critic2_losses,
            "critic_grad_norm": float(np.mean(list(critic_grad_norms.values()))),
            "critic_grad_norms": critic_grad_norms,
            "q1_mean": float(np.mean(list(q1_means.values()))),
            "q2_mean": float(np.mean(list(q2_means.values()))),
            "q1_q2_gap": float(np.mean([abs(q1_means[mg] - q2_means[mg]) for mg in self.agent_ids])),
            "target_q_mean": float(np.mean(list(target_q_means.values()))),
            "td_error_mean": float(np.mean(list(td_error_means.values()))),
            "td_error_std": float(np.mean(list(td_error_stds.values()))),
            "q_stats": q_stats,
        }

        # 3. Delayed policy update
        if self.update_steps % self.policy_delay == 0:
            actor_losses = {}
            actor_grad_norms = {}
            for active in self.agent_ids:
                for parameter in self.critics[active].parameters():
                    parameter.requires_grad_(False)

                parts = []
                for mg in self.agent_ids:
                    action = self.actors[mg](observations[mg])
                    parts.append(action if mg == active else action.detach())
                loss = -self.critics[active].q1(batch["global_state"], torch.cat(parts, dim=-1)).mean()
                self.actor_optimizers[active].zero_grad()
                loss.backward()
                actor_grad_norms[active] = self._gradient_norm(self.actors[active])
                self.actor_optimizers[active].step()
                actor_losses[active] = float(loss.item())

                for parameter in self.critics[active].parameters():
                    parameter.requires_grad_(True)

                self._soft_update(self.actors[active], self.target_actors[active])
                self._soft_update(self.critics[active], self.target_critics[active])

            metrics["actor_loss"] = float(np.mean(list(actor_losses.values())))
            metrics["actor_losses"] = actor_losses
            metrics["actor_grad_norm"] = float(np.mean(list(actor_grad_norms.values())))
            metrics["actor_grad_norms"] = actor_grad_norms

        return metrics

    def save(self, path: str | Path) -> None:
        torch.save({
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "update_steps": self.update_steps,
        }, Path(path))

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.actors.load_state_dict(state["actors"])
        self.target_actors.load_state_dict(state["target_actors"])
        if "critics" in state:
            self.critics.load_state_dict(state["critics"])
            self.target_critics.load_state_dict(state["target_critics"])
        elif "critic" in state:
            # Backward compatibility with single-critic checkpoint
            for mg in self.agent_ids:
                self.critics[mg].load_state_dict(state["critic"])
                self.target_critics[mg].load_state_dict(state["target_critic"])
        self.update_steps = int(state.get("update_steps", 0))
