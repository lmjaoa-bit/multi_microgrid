import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from algorithms.matd3_shapley_q.independent_baseline import IndependentBaselineEvaluator
from envs.multi_microgrid_env import MultiMicrogridEnv

from algorithms.matd3_shapley_q.coalition import (
    all_coalitions,
    exact_shapley_values,
    normalize_characteristic_values,
)


AGENTS = ("mg1", "mg2", "mg3")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def env():
    return MultiMicrogridEnv(ROOT / "configs/cooperative.yaml")


def algorithm_config_with_baseline(mode, checkpoint=None):
    cfg = yaml.safe_load(
        (ROOT / "configs/algorithms/matd3_shapley_q.yaml").read_text(encoding="utf-8")
    )
    cfg["algorithm"].setdefault("shapley_q", {})["baseline"] = {
        "mode": mode,
        "checkpoint": checkpoint,
        "disable_p2p": True,
        "freeze_parameters": True,
    }
    return cfg


def save_independent_dispatch_checkpoint(env, path):
    independent_env_config = copy.deepcopy(env.config)
    independent_env_config["p2p"]["enabled"] = False
    independent_env_config.setdefault("ess_sharing", {})["ess_p2p_enabled"] = False
    independent_env = MultiMicrogridEnv(independent_env_config, split=env.split)
    source = Matd3Dispatch.from_env(
        independent_env,
        yaml.safe_load(
            (ROOT / "configs/algorithms/matd3_dispatch.yaml").read_text(
                encoding="utf-8"
            )
        ),
        seed=42,
    )
    with torch.no_grad():
        for parameter in source.actors.parameters():
            parameter.fill_(0.125)
    source.save(path)
    return path


def test_baseline_rejects_unsupported_mode(env):
    cfg = algorithm_config_with_baseline(mode="trainable")
    with pytest.raises(ValueError, match="baseline mode"):
        IndependentBaselineEvaluator(env, cfg)


def test_frozen_baseline_requires_checkpoint(env):
    cfg = algorithm_config_with_baseline(mode="frozen_policy", checkpoint=None)
    with pytest.raises(ValueError, match="checkpoint"):
        IndependentBaselineEvaluator(env, cfg)


def test_reset_records_effective_common_scenario_options(env):
    _, info = env.reset(seed=73, deterministic=False, forecast_noise=True)
    assert env.last_reset_options == {
        "day": info["day"],
        "seed": 73,
        "deterministic": False,
        "forecast_noise": True,
    }


def test_common_scenario_matches_all_physical_and_forecast_inputs(env):
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    env.reset(seed=73, deterministic=False, forecast_noise=True)
    evaluator.reset_episode(env)
    ind = evaluator.ind_env
    pd.testing.assert_frame_equal(env.episode_context_df, ind.episode_context_df)
    for mg in env.agent_ids:
        for field in ("load_kw", "pv_kw", "wind_kw"):
            np.testing.assert_allclose(
                env._episode_exogenous[mg][field], ind._episode_exogenous[mg][field]
            )
        for field in env.forecasts._local_arrays[mg]:
            np.testing.assert_allclose(
                env.forecasts._local_arrays[mg][field],
                ind.forecasts._local_arrays[mg][field],
            )
    for field in env.forecasts._public_arrays:
        np.testing.assert_allclose(
            env.forecasts._public_arrays[field], ind.forecasts._public_arrays[field]
        )
    np.testing.assert_allclose(env.forecasts._valid_array, ind.forecasts._valid_array)
    assert evaluator.scenario_signature(env) == evaluator.scenario_signature(ind)


def test_common_scenario_restores_no_seed_forecast_rng_after_prior_resets(env):
    env.reset(seed=None, deterministic=False, forecast_noise=True)
    env.reset(seed=None, deterministic=False, forecast_noise=True)
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    evaluator.reset_episode(env)
    ind = evaluator.ind_env
    pd.testing.assert_frame_equal(env.episode_context_df, ind.episode_context_df)
    for mg in env.agent_ids:
        for field in ("load_kw", "pv_kw", "wind_kw"):
            np.testing.assert_allclose(
                env._episode_exogenous[mg][field], ind._episode_exogenous[mg][field]
            )
        for field in env.forecasts._local_arrays[mg]:
            np.testing.assert_allclose(
                env.forecasts._local_arrays[mg][field],
                ind.forecasts._local_arrays[mg][field],
            )
    for field in env.forecasts._public_arrays:
        np.testing.assert_allclose(
            env.forecasts._public_arrays[field], ind.forecasts._public_arrays[field]
        )
    np.testing.assert_allclose(env.forecasts._valid_array, ind.forecasts._valid_array)


def test_common_scenario_mismatch_fails_fast(env):
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    env.reset(seed=73, deterministic=False, forecast_noise=True)
    env._episode_exogenous["mg1"]["load_kw"][0] += 1.0
    with pytest.raises(ValueError, match="scenario"):
        evaluator.reset_episode(env)


def test_independent_baseline_detects_initial_soc_mismatch_during_reset(
    env, monkeypatch
):
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    env.reset(seed=73, deterministic=True)
    original_reset = evaluator.ind_env.reset

    def reset_with_mismatched_initial_soc(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        evaluator.ind_env.cores["mg1"].battery.initial_soc += 0.01
        return result

    monkeypatch.setattr(evaluator.ind_env, "reset", reset_with_mismatched_initial_soc)
    with pytest.raises(ValueError, match="scenario"):
        evaluator.reset_episode(env)


def test_independent_baseline_rejects_noncooperation_physics_mismatch(env):
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    env.config["microgrids"][0]["battery"]["capacity_kwh"] += 1.0
    env.cores["mg1"].battery.capacity_kwh += 1.0
    env.reset(seed=73, deterministic=True)
    with pytest.raises(ValueError, match="physical configuration"):
        evaluator.reset_episode(env)


@pytest.mark.parametrize("gamma", [float("nan"), -0.01, 1.01])
def test_independent_baseline_rejects_invalid_gamma(env, gamma):
    cfg = algorithm_config_with_baseline("heuristic")
    cfg["algorithm"]["gamma"] = gamma
    with pytest.raises(ValueError, match="gamma"):
        IndependentBaselineEvaluator(env, cfg)


def test_independent_baseline_keeps_local_ess_and_grid_but_disables_p2p(env):
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    env.reset(seed=42, deterministic=True)
    evaluator.reset_episode(env)
    rows = evaluator.step_infos
    assert len(rows) == env.episode_length
    assert all(row["p2p"]["total_p2p_kwh"] == pytest.approx(0.0) for row in rows)
    assert any(
        sum(v["throughput_kwh"] for v in row["devices"].values()) > 0
        for row in rows
    )
    assert any(
        sum(
            v["grid_import_kw"] + v["grid_export_kw"]
            for v in row["settlements"].values()
        )
        > 0
        for row in rows
    )


def test_baseline_step_data_uses_discounted_returns_and_strict_bounds(env):
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    with pytest.raises(IndexError, match="step 0"):
        evaluator.get_step_data(0)
    env.reset(seed=42, deterministic=True)
    evaluator.reset_episode(env)
    expected = 0.0
    for reward in reversed(evaluator.step_rewards):
        expected = reward + evaluator.gamma * expected
    first = evaluator.get_step_data(0)
    last = evaluator.get_step_data(env.episode_length - 1)
    assert first == pytest.approx(
        {
            "reward": evaluator.step_rewards[0],
            "profit": evaluator.step_profits[0],
            "return_to_go": expected,
            "next_return_to_go": evaluator.discounted_returns[1],
        }
    )
    assert last["next_return_to_go"] == pytest.approx(0.0)
    with pytest.raises(IndexError, match="step -1"):
        evaluator.get_step_data(-1)
    with pytest.raises(IndexError, match=f"step {env.episode_length}"):
        evaluator.get_step_data(env.episode_length)


def test_baseline_episode_metrics_match_cached_trajectory_and_reset_clears_infos(env):
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    env.reset(seed=42, deterministic=True)
    evaluator.reset_episode(env)
    first_infos = list(evaluator.step_infos)
    assert evaluator.episode_metrics() == pytest.approx(
        {
            "independent_baseline_return": sum(evaluator.step_rewards),
            "independent_baseline_profit": sum(evaluator.step_profits),
        }
    )
    env.reset(seed=43, deterministic=True)
    evaluator.reset_episode(env)
    assert len(evaluator.step_infos) == env.episode_length
    assert evaluator.step_infos[0] is not first_infos[0]


@pytest.mark.parametrize("field", ["reward", "profit"])
def test_baseline_rollout_rejects_nonfinite_step_values(env, monkeypatch, field):
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    env.reset(seed=42, deterministic=True)
    original_step = evaluator.ind_env.step

    def step_with_nonfinite_value(actions):
        observations, rewards, done, info = original_step(actions)
        if field == "reward":
            rewards = dict(rewards)
            rewards[evaluator.ind_env.agent_ids[0]] = float("nan")
        else:
            info = dict(info)
            info["team_profit"] = float("inf")
        return observations, rewards, done, info

    monkeypatch.setattr(evaluator.ind_env, "step", step_with_nonfinite_value)
    with pytest.raises(ValueError, match=rf"step 0.*{field}"):
        evaluator.reset_episode(env)


@pytest.mark.parametrize(
    ("field", "total_field"),
    [("reward", "total_return"), ("profit", "total_profit")],
)
def test_baseline_rollout_rejects_nonfinite_totals(
    env, monkeypatch, field, total_field
):
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    env.reset(seed=42, deterministic=True)
    original_step = evaluator.ind_env.step

    def step_with_large_finite_value(actions):
        observations, rewards, done, info = original_step(actions)
        if field == "reward":
            rewards = dict(rewards)
            rewards[evaluator.ind_env.agent_ids[0]] = 1e308
        else:
            info = dict(info)
            info["team_profit"] = 1e308
        return observations, rewards, done, info

    monkeypatch.setattr(evaluator.ind_env, "step", step_with_large_finite_value)
    with pytest.raises(ValueError, match=rf"{total_field}.*non-finite"):
        evaluator.reset_episode(env)


def test_baseline_rollout_rejects_nonfinite_discounted_return(env, monkeypatch):
    evaluator = IndependentBaselineEvaluator(
        env, algorithm_config_with_baseline("heuristic")
    )
    env.reset(seed=42, deterministic=True)
    original_step = evaluator.ind_env.step
    rewards_by_step = [1e308, -1e308, -1e308]
    step_index = 0

    def step_with_discount_overflow(actions):
        nonlocal step_index
        observations, rewards, done, info = original_step(actions)
        rewards = dict(rewards)
        rewards[evaluator.ind_env.agent_ids[0]] = (
            rewards_by_step[step_index] if step_index < len(rewards_by_step) else 0.0
        )
        step_index += 1
        return observations, rewards, done, info

    monkeypatch.setattr(evaluator.ind_env, "step", step_with_discount_overflow)
    with pytest.raises(ValueError, match=r"step 1.*return_to_go.*non-finite"):
        evaluator.reset_episode(env)


def test_frozen_baseline_rejects_missing_checkpoint_file(env, tmp_path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="checkpoint.*exist"):
        IndependentBaselineEvaluator(
            env, algorithm_config_with_baseline("frozen_policy", str(missing))
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("algorithm_slug", "crossgate", "algorithm_slug"),
        ("action_semantics", "retired_action", "action semantics"),
        ("projection_signature", "bad-projection", "physical projection"),
        ("state_schema", (), "state schema"),
        ("observation_schemas", {}, "observation schemas"),
        ("actors", {}, "actor state dict"),
    ],
)
def test_frozen_baseline_rejects_incompatible_checkpoint(
    env, tmp_path, field, bad_value, message
):
    checkpoint = save_independent_dispatch_checkpoint(env, tmp_path / "source.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload[field] = bad_value
    broken = tmp_path / f"broken_{field}.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match=message):
        IndependentBaselineEvaluator(
            env, algorithm_config_with_baseline("frozen_policy", str(broken))
        )


def test_frozen_baseline_rejects_nonfinite_actor_tensor(env, tmp_path):
    checkpoint = save_independent_dispatch_checkpoint(env, tmp_path / "source.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tensor_name = next(iter(payload["actors"]))
    payload["actors"][tensor_name].reshape(-1)[0] = float("nan")
    broken = tmp_path / "nonfinite_actor.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match=rf"checkpoint.*{tensor_name}.*non-finite"):
        IndependentBaselineEvaluator(
            env, algorithm_config_with_baseline("frozen_policy", str(broken))
        )


def test_frozen_policy_relative_checkpoint_is_loaded_without_gradients(env, tmp_path):
    for field in ("train_path", "val_path", "test_path"):
        env.config["data"][field] = str(ROOT / env.config["data"][field])
    env.config["_project_root"] = str(tmp_path)
    checkpoint = save_independent_dispatch_checkpoint(env, tmp_path / "independent.pt")
    evaluator = IndependentBaselineEvaluator(
        env,
        algorithm_config_with_baseline("frozen_policy", checkpoint.name),
    )
    env.reset(seed=42)
    evaluator.reset_episode(env)
    assert evaluator.frozen_actors is not None
    assert evaluator.frozen_actors.training is False
    assert all(
        not parameter.requires_grad for parameter in evaluator.frozen_actors.parameters()
    )
    assert all(parameter.grad is None for parameter in evaluator.frozen_actors.parameters())
    assert all(
        torch.allclose(parameter, torch.full_like(parameter, 0.125))
        for parameter in evaluator.frozen_actors.parameters()
    )


def test_default_baseline_configuration_is_fixed_and_noncooperative():
    cfg = yaml.safe_load(
        (ROOT / "configs/algorithms/matd3_shapley_q.yaml").read_text(encoding="utf-8")
    )
    assert cfg["algorithm"]["shapley_q"]["baseline"] == {
        "mode": "heuristic",
        "checkpoint": None,
        "disable_p2p": True,
        "freeze_parameters": True,
    }


def positive_cooperation_values():
    return {
        frozenset(): 80.0,
        frozenset({"mg1"}): 84.0,
        frozenset({"mg2"}): 83.0,
        frozenset({"mg3"}): 82.0,
        frozenset({"mg1", "mg2"}): 91.0,
        frozenset({"mg1", "mg3"}): 89.0,
        frozenset({"mg2", "mg3"}): 88.0,
        frozenset(AGENTS): 100.0,
    }


def test_characteristic_normalization_uses_nonzero_independent_value():
    physical = positive_cooperation_values()
    normalized = normalize_characteristic_values(physical, AGENTS)
    assert physical[frozenset()] != 0.0
    assert normalized[frozenset()] == pytest.approx(0.0)
    assert normalized[frozenset(AGENTS)] == pytest.approx(20.0)


def test_exact_shapley_efficiency_targets_cooperation_surplus():
    values = positive_cooperation_values()
    phi = exact_shapley_values(values, AGENTS)
    assert sum(phi.values()) == pytest.approx(
        values[frozenset(AGENTS)] - values[frozenset()]
    )


def test_exact_shapley_is_zero_when_cooperation_changes_no_value():
    values = {coalition: 80.0 for coalition in all_coalitions(AGENTS)}
    phi = exact_shapley_values(values, AGENTS)
    assert phi == pytest.approx({mg: 0.0 for mg in AGENTS})


def test_exact_shapley_sum_is_positive_for_positive_cooperation():
    values = positive_cooperation_values()
    phi = exact_shapley_values(values, AGENTS)
    assert sum(phi.values()) > 0.0
