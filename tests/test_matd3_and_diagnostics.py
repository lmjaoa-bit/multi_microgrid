from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.matd3 import MATD3
from diagnostics.scenario_diagnostic import diagnose_scenario
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative_legacy_pre_dr.yaml"


def test_vectorized_profiles_and_forecasts_match_episode_data() -> None:
    env = MultiMicrogridEnv(CONFIG)
    env.reset(seed=11, deterministic=True)
    assert env._episode_exogenous is not None
    assert env.episode_context_df is not None
    for mg in env.agent_ids:
        for index in (0, 5, len(env.episode_context_df) - 1):
            cached = env.profiles.exogenous(mg, env.episode_context_df.iloc[index])
            for name, values in env._episode_exogenous[mg].items():
                assert np.isclose(cached[name], values[index])
        forecast = env.forecasts.combined(mg, 0, 1)
        assert np.isclose(forecast["load_kw"], env._episode_exogenous[mg]["load_kw"][1])
        assert np.isclose(forecast["pv_kw"], env._episode_exogenous[mg]["pv_kw"][1])
        assert np.isclose(forecast["wind_kw"], env._episode_exogenous[mg]["wind_kw"][1])


def test_canonical_matd3_legacy_architecture_and_training_smoke() -> None:
    env = MultiMicrogridEnv(CONFIG)
    algorithm = MATD3(env.agent_ids, env.observation_dims, env.global_state_dim,
                      action_dim=env.action_dim, hidden_dims=(32, 32),
                      replay_capacity=100, seed=7)
    assert len(algorithm.critics) == len(env.agent_ids)
    assert set(algorithm.critics.keys()) == set(env.agent_ids)
    assert len(algorithm.target_critics) == len(env.agent_ids)

    observations, reset_info = env.reset(seed=7, deterministic=True)
    state = reset_info["global_state"]
    metrics = {}
    for _ in range(5):
        actions = algorithm.select_actions(observations, noise_std=0.1)
        next_observations, rewards, _, info = env.step(actions)
        algorithm.replay.add(observations, state, actions, rewards[env.agent_ids[0]],
                             next_observations, info["global_state_after"], False)
        metrics = algorithm.update(batch_size=2)
        observations, state = next_observations, info["global_state_after"]
    assert np.isfinite(metrics["critic_loss"])
    assert "critic_losses" in metrics
    assert set(metrics["critic_losses"].keys()) == set(env.agent_ids)
    assert "q_stats" in metrics
    assert set(metrics["q_stats"].keys()) == set(env.agent_ids)
    assert algorithm.update_steps > 0


def test_scenario_diagnostic_smoke_and_dynamic_complementarity() -> None:
    result = diagnose_scenario(CONFIG, split="train")
    assert set(result["microgrids"]) == {"mg1", "mg2", "mg3"}
    assert result["system"]["total_p2p_potential_kwh"] >= 0.0
    assert result["system"]["spatiotemporal_opportunity_windows"] > 0
    assert result["system"]["spatiotemporal_opportunity_day_ratio"] > 0.5
    assert result["system"]["pair_net_load_correlation"]["mg1-mg3"] < 0.8
    assert result["system"]["p2p_pair_capacity_kw"] == 50.0
    assert not result["system"]["allow_grid_resale"]
    for metrics in result["microgrids"].values():
        assert 0.0 <= metrics["surplus_hour_ratio"] <= 1.0
        assert 0.0 <= metrics["deficit_hour_ratio"] <= 1.0
