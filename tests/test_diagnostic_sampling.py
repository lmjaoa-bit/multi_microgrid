from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from algorithms.diagnostic_sampling import (
    DiagnosticSchedule,
    install_diagnostic_sampling,
)
from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "configs" / "cooperative.yaml"
ALGORITHM_CONFIG = ROOT / "configs" / "algorithms" / "matd3_dispatch.yaml"


@pytest.mark.parametrize(
    ("interval", "policy_delay", "expected"),
    [(20, 2, [20, 40]), (21, 2, [22, 42]), (5, 3, [6, 12])],
)
def test_schedule_only_fires_on_actor_updates(
    interval: int,
    policy_delay: int,
    expected: list[int],
) -> None:
    schedule = DiagnosticSchedule(interval, policy_delay)

    actual = [step for step in range(1, 45) if schedule.due(step)]

    assert actual[: len(expected)] == expected
    assert all(step % policy_delay == 0 for step in actual)


def _algorithm_with_replay(interval: int) -> Matd3Dispatch:
    class SampledMatd3(Matd3Dispatch):
        pass

    install_diagnostic_sampling(
        SampledMatd3,
        variant="matd3_dispatch",
        plain_actor=True,
    )
    env = MultiMicrogridEnv(ENV_CONFIG, split="train")
    config = yaml.safe_load(ALGORITHM_CONFIG.read_text(encoding="utf-8"))
    config = deepcopy(config)
    config.setdefault("logging", {})["diagnostic_interval"] = interval
    config["algorithm"]["replay_capacity"] = 100
    algorithm = SampledMatd3.from_env(env, config, device="cpu", seed=9)

    observations, reset_info = env.reset(seed=9, deterministic=True)
    state = reset_info["global_state"]
    for _ in range(4):
        actions = {
            mg: np.zeros(env.action_dim, dtype=np.float32) for mg in env.agent_ids
        }
        next_observations, rewards, done, info = env.step(actions)
        algorithm.replay.add(
            observations,
            state,
            actions,
            rewards[env.agent_ids[0]],
            next_observations,
            info["global_state_after"],
            done,
        )
        observations = next_observations
        state = info["global_state_after"]
    return algorithm


def test_ordinary_actor_update_skips_diagnostics_and_host_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm = _algorithm_with_replay(interval=20)
    algorithm.update_steps = 1

    def forbidden(*_args, **_kwargs):
        pytest.fail("gradient diagnostics called on ordinary update")

    monkeypatch.setattr(algorithm, "_actor_gradient_diagnostics", forbidden)

    metrics = algorithm.update(batch_size=2)

    assert metrics == {}
    assert algorithm.update_steps == 2


def test_diagnostic_update_contains_actor_critic_q_and_td_metrics() -> None:
    algorithm = _algorithm_with_replay(interval=2)
    algorithm.update_steps = 1
    algorithm.diagnostic_schedule.next_boundary = 2

    metrics = algorithm.update(batch_size=2)

    assert set(metrics) >= {
        "critic_loss",
        "actor_loss",
        "critic_grad_norm",
        "actor_grad_norm",
        "q_stats",
        "td_error_mean",
        "td_error_std",
        "td_error_rmse",
    }
    assert all(
        {"mean", "std", "min", "max"} <= set(row)
        for row in metrics["q_stats"].values()
    )
    assert algorithm.update_steps == 2
