from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch


class JointReplayBuffer:
    """One replay buffer storing time-aligned transitions for all agents."""

    def __init__(
        self,
        capacity: int,
        agent_ids: Sequence[str],
        observation_dims: Mapping[str, int],
        state_dim: int,
        action_dim: int = 4,
        seed: int = 42,
    ) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.agent_ids = [str(x) for x in agent_ids]
        self.n_agents = len(self.agent_ids)
        dims = {int(observation_dims[x]) for x in self.agent_ids}
        if len(dims) != 1:
            raise ValueError("The current joint replay implementation requires equal observation dimensions.")
        self.obs_dim = dims.pop()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(int(seed))
        self.ptr = 0
        self.size = 0

        self.states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.observations = np.empty((self.capacity, self.n_agents, self.obs_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, self.n_agents, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity, self.n_agents), dtype=np.float32)
        self.next_states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.next_observations = np.empty((self.capacity, self.n_agents, self.obs_dim), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        observations: Mapping[str, np.ndarray],
        actions: Mapping[str, np.ndarray],
        rewards: Mapping[str, float],
        next_state: np.ndarray,
        next_observations: Mapping[str, np.ndarray],
        done: bool,
    ) -> None:
        idx = self.ptr
        if not isinstance(done, (bool, np.bool_, int, np.integer, float, np.floating)):
            raise ValueError("Replay done must be a boolean or numeric 0/1 value.")
        done_value = float(done)
        if not np.isfinite(done_value) or done_value not in (0.0, 1.0):
            raise ValueError("Replay done must be finite and strictly encoded as 0 or 1.")
        obs = np.stack([np.asarray(observations[x], dtype=np.float32) for x in self.agent_ids])
        act = np.stack([np.asarray(actions[x], dtype=np.float32) for x in self.agent_ids])
        rew = np.asarray([float(rewards[x]) for x in self.agent_ids], dtype=np.float32)
        next_obs = np.stack([np.asarray(next_observations[x], dtype=np.float32) for x in self.agent_ids])
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError("Replay state shape does not match configured state_dim.")
        if obs.shape != (self.n_agents, self.obs_dim) or next_obs.shape != (self.n_agents, self.obs_dim):
            raise ValueError("Replay observation shape mismatch.")
        if act.shape != (self.n_agents, self.action_dim):
            raise ValueError("Replay action shape mismatch.")
        commitments = act[:, 1]
        if not np.all(np.isin(commitments, [-1.0, 1.0])):
            raise ValueError("Replay commitment actions must be hard encoded as -1 or +1.")
        off = commitments < 0.0
        if np.any(np.abs(act[off, 2] + 1.0) > 1e-6):
            raise ValueError("Replay OFF actions must store generator_setpoint=-1.")
        if not np.isfinite(
            np.concatenate(
                [state, next_state, obs.ravel(), next_obs.ravel(), act.ravel(), rew, [done_value]]
            )
        ).all():
            raise ValueError("Replay transition contains NaN or infinite values.")

        self.states[idx] = state
        self.observations[idx] = obs
        self.actions[idx] = act
        self.rewards[idx] = rew
        self.next_states[idx] = next_state
        self.next_observations[idx] = next_obs
        self.dones[idx, 0] = done_value
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        batch_size = int(batch_size)
        indices = self.rng.integers(0, self.size, size=batch_size)
        batch = {
            "state": torch.as_tensor(self.states[indices], device=device),
            "obs": torch.as_tensor(self.observations[indices], device=device),
            "action": torch.as_tensor(self.actions[indices], device=device),
            "reward": torch.as_tensor(self.rewards[indices], device=device),
            "next_state": torch.as_tensor(self.next_states[indices], device=device),
            "next_obs": torch.as_tensor(self.next_observations[indices], device=device),
            "done": torch.as_tensor(self.dones[indices], device=device),
        }
        for name, tensor in batch.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Replay sample tensor {name!r} contains NaN or infinite values.")
        return batch
