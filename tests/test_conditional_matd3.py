from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.conditional_matd3 import (
    ConditionalMATD3,
    JointReplayBuffer,
    load_runtime_config,
)
from algorithms.conditional_matd3.conditional_matd3 import _freeze
from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def _build_algo_and_buffer():
    env = MultiMicrogridEnv(load_runtime_config())
    algo = ConditionalMATD3(
        env.agent_ids,
        env.observation_dims,
        env.global_state_dim,
        env.config,
        device="cpu",
    )
    buffer = JointReplayBuffer(
        capacity=128,
        agent_ids=env.agent_ids,
        observation_dims=env.observation_dims,
        state_dim=env.global_state_dim,
        action_dim=env.action_dim,
        seed=7,
    )
    return env, algo, buffer


def test_conditional_actor_executes_only_hard_commitment_actions():
    env, algo, _ = _build_algo_and_buffer()
    obs, _ = env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    actions, diagnostics = algo.select_actions(obs, explore=False)
    for mg in env.agent_ids:
        assert actions[mg].shape == (4,)
        assert actions[mg][1] in (-1.0, 1.0)
        if actions[mg][1] < 0.0:
            assert actions[mg][2] == -1.0
        assert 0.0 <= diagnostics[mg]["p_on"] <= 1.0


def test_joint_replay_stores_time_aligned_hard_actions_and_samples_tensors():
    env, algo, buffer = _build_algo_and_buffer()
    obs, reset_info = env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    state = reset_info["global_state"]
    actions, _ = algo.select_actions(obs, warmup=True)
    next_obs, rewards, done, info = env.step(actions)
    buffer.add(
        state,
        obs,
        info["executed_actions"],
        rewards,
        info["global_state_after"],
        next_obs,
        done,
    )
    batch = buffer.sample(1)
    assert batch["obs"].shape == (1, 3, env.observation_dims["mg1"])
    assert batch["action"].shape == (1, 3, 4)
    commitment = batch["action"][0, :, 1].numpy()
    assert np.all(np.isin(commitment, [-1.0, 1.0]))


def test_joint_replay_rejects_nonfinite_sampled_tensor():
    _, _, buffer = _build_algo_and_buffer()
    state = np.zeros(buffer.state_dim, dtype=np.float32)
    observations = {
        mg: np.zeros(buffer.obs_dim, dtype=np.float32)
        for mg in buffer.agent_ids
    }
    actions = {
        mg: np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        for mg in buffer.agent_ids
    }
    rewards = {mg: 0.0 for mg in buffer.agent_ids}
    buffer.add(state, observations, actions, rewards, state, observations, False)
    buffer.states[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="Replay sample tensor 'state'"):
        buffer.sample(1)


@pytest.mark.parametrize("bad_done", [float("nan"), float("inf"), float("-inf"), 0.5])
def test_joint_replay_rejects_invalid_done_values(bad_done):
    _, _, buffer = _build_algo_and_buffer()
    state = np.zeros(buffer.state_dim, dtype=np.float32)
    observations = {
        mg: np.zeros(buffer.obs_dim, dtype=np.float32)
        for mg in buffer.agent_ids
    }
    actions = {
        mg: np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        for mg in buffer.agent_ids
    }
    rewards = {mg: 0.0 for mg in buffer.agent_ids}
    with pytest.raises(ValueError, match="done must be finite"):
        buffer.add(
            state,
            observations,
            actions,
            rewards,
            state,
            observations,
            bad_done,
        )


def test_joint_replay_circular_overwrite_preserves_latest_transition_order():
    buffer = JointReplayBuffer(
        capacity=3,
        agent_ids=["mg1", "mg2", "mg3"],
        observation_dims={"mg1": 1, "mg2": 1, "mg3": 1},
        state_dim=1,
        seed=9,
    )
    actions = {
        mg: np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        for mg in buffer.agent_ids
    }
    for value in range(5):
        observations = {
            mg: np.array([value], dtype=np.float32)
            for mg in buffer.agent_ids
        }
        rewards = {mg: float(value) for mg in buffer.agent_ids}
        buffer.add(
            np.array([value], dtype=np.float32),
            observations,
            actions,
            rewards,
            np.array([value + 1], dtype=np.float32),
            observations,
            value == 4,
        )

    chronological_indices = (
        np.arange(buffer.size) + buffer.ptr - buffer.size
    ) % buffer.capacity
    assert buffer.size == 3
    assert buffer.ptr == 2
    assert buffer.states[chronological_indices, 0].tolist() == [2.0, 3.0, 4.0]
    assert buffer.next_states[chronological_indices, 0].tolist() == [3.0, 4.0, 5.0]
    assert buffer.dones[chronological_indices, 0].tolist() == [0.0, 0.0, 1.0]


def test_conditional_matd3_can_run_critic_and_synchronous_actor_updates():
    torch.manual_seed(3)
    env, algo, buffer = _build_algo_and_buffer()
    days = env.data.valid_days[:2]
    for day_index, day in enumerate(days):
        obs, reset_info = env.reset(day=str(day.date()), seed=day_index + 1)
        state = reset_info["global_state"]
        done = False
        while not done:
            actions, _ = algo.select_actions(obs, env_step=len(buffer), warmup=True)
            next_obs, rewards, done, info = env.step(actions)
            next_state = info["global_state_after"]
            buffer.add(
                state,
                obs,
                info["executed_actions"],
                rewards,
                next_state,
                next_obs,
                done,
            )
            obs, state = next_obs, next_state
    batch = buffer.sample(16, device="cpu")
    logs1 = algo.update(batch)
    logs2 = algo.update(batch)
    assert logs1["policy_updated"] == 0.0
    assert logs2["policy_updated"] == 1.0
    for value in logs2.values():
        assert np.isfinite(float(value))
    for mg in env.agent_ids:
        assert f"{mg}/delta_q" in logs2
        assert f"{mg}/commit_target" in logs2
        assert 0.0 <= logs2[f"{mg}/commit_target"] <= 1.0


def test_actor_loss_components_respect_head_gradient_boundaries():
    torch.manual_seed(11)
    env, algo, _ = _build_algo_and_buffer()
    batch_size = 32
    obs = torch.randn(batch_size, 3, env.observation_dims["mg1"])
    state = torch.randn(batch_size, env.global_state_dim)
    with torch.no_grad():
        frozen_joint = algo._current_hard_joint_action(obs)

    actor = algo.actors["mg1"]
    critic_modules = [algo.critics1["mg1"], algo.critics2["mg1"]]
    head_names = [
        "battery_head",
        "generator_setpoint_head",
        "commitment_logit_head",
        "p2p_bid_head",
    ]
    expected_active = {
        "commitment": {"commitment_logit_head"},
        "common": {"battery_head", "p2p_bid_head"},
        "generator": {"generator_setpoint_head"},
    }

    for component, active_heads in expected_active.items():
        actor.zero_grad(set_to_none=True)
        algo.loss_weights = {
            name: float(name == component)
            for name in ("commitment", "common", "generator")
        }
        with _freeze(critic_modules):
            loss, _ = algo._actor_loss("mg1", obs, state, frozen_joint)
            loss.backward()
        gradient_by_head = {
            name: sum(
                parameter.grad.abs().sum().item()
                if parameter.grad is not None else 0.0
                for parameter in getattr(actor, name).parameters()
            )
            for name in head_names
        }
        for name, magnitude in gradient_by_head.items():
            if name in active_heads:
                assert magnitude > 0.0
            else:
                assert magnitude == 0.0


def test_algorithm_checkpoint_roundtrip_preserves_training_state():
    env, algo, _ = _build_algo_and_buffer()
    algo.update_counter = 7
    algo.q_scale_ema["mg1"] = 0.42
    saved = algo.state_dict()
    restored = ConditionalMATD3(
        env.agent_ids,
        env.observation_dims,
        env.global_state_dim,
        env.config,
        device="cpu",
    )
    restored.load_state_dict(saved)
    assert restored.update_counter == 7
    assert restored.q_scale_ema["mg1"] == 0.42
    for mg in env.agent_ids:
        for p1, p2 in zip(algo.actors[mg].parameters(), restored.actors[mg].parameters()):
            assert torch.equal(p1, p2)
