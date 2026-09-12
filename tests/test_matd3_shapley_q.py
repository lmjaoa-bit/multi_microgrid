import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from algorithms.matd3_shapley_q.algorithm import Matd3ShapleyQ
from algorithms.matd3_shapley_q.replay_buffer import ShapleyReplayBuffer
from algorithms.matd3_shapley_q.independent_baseline import IndependentBaselineEvaluator
from envs.multi_microgrid_env import MultiMicrogridEnv
from experiments.training_pipeline import (
    _algorithm_episode_metrics,
    _algorithm_replay_extras,
    _diagnostic_means,
    _runtime_hyperparameters,
    _training_diagnostic_summary,
    build_parser,
)
from evaluation.evaluate_matd3 import evaluate

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


def test_replay_surplus_consistency(env):
    alg = Matd3ShapleyQ.from_env(env, algorithm_config_with_baseline("heuristic"))
    obs, reset = env.reset(seed=42)
    actions = {mg: np.array([-0.5], dtype=np.float32) for mg in AGENTS}
    buf = alg.replay
    buf.add(obs, reset["global_state"], actions, 5.0, obs, reset["global_state"], False,
            independent_reward=2.0, cooperation_surplus_reward=3.0,
            independent_return_to_go=12.0, next_independent_return_to_go=7.0)
    batch = buf.sample(1, AGENTS, torch.device("cpu"))
    for field, expected in {"team_reward": 5.0, "independent_reward": 2.0,
                            "cooperation_surplus_reward": 3.0,
                            "independent_return_to_go": 12.0,
                            "next_independent_return_to_go": 7.0}.items():
        assert batch[field].shape == (1, 1)
        assert batch[field].dtype == torch.float32
        assert batch[field].item() == pytest.approx(expected)


def test_conservative_target_takes_twin_min_per_agent_before_sum():
    q1 = {mg: torch.tensor([[v]]) for mg, v in zip(AGENTS, (10., 1., 10.))}
    q2 = {mg: torch.tensor([[v]]) for mg, v in zip(AGENTS, (1., 10., 1.))}
    result = Matd3ShapleyQ._conservative_shapley_sum(q1, q2, AGENTS)
    assert result.item() == pytest.approx(3.0)


@pytest.mark.parametrize("field", ["team_reward", "independent_reward",
    "cooperation_surplus_reward", "independent_return_to_go", "next_independent_return_to_go"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_replay_rejects_nonfinite_rewards_and_references(env, field, bad):
    alg = Matd3ShapleyQ.from_env(env, algorithm_config_with_baseline("heuristic"))
    obs, info = env.reset(seed=42)
    values = dict(team_reward=5., independent_reward=2., cooperation_surplus_reward=3.,
                  independent_return_to_go=12., next_independent_return_to_go=7.)
    values[field] = bad
    with pytest.raises(ValueError, match=field):
        alg.replay.add(obs, info["global_state"], alg.select_actions(obs),
                       next_observations=obs, next_global_state=info["global_state"],
                       done=False, **values)
    assert len(alg.replay) == 0


def test_replay_rejects_inconsistent_surplus_and_copies_inputs(env):
    alg = Matd3ShapleyQ.from_env(env, algorithm_config_with_baseline("heuristic"))
    obs, info = env.reset(seed=42)
    state, actions = info["global_state"], alg.select_actions(obs)
    with pytest.raises(ValueError, match="surplus"):
        alg.replay.add(obs, state, actions, 5., obs, state, False,
                       independent_reward=2., cooperation_surplus_reward=9.,
                       independent_return_to_go=12., next_independent_return_to_go=7.)
    alg.replay.add(obs, state, actions, 5., obs, state, False,
                   independent_reward=2., cooperation_surplus_reward=3.,
                   independent_return_to_go=12., next_independent_return_to_go=7.)
    expected_obs, expected_state = obs["mg1"].copy(), state.copy()
    obs["mg1"][:] = -999
    state[:] = -999
    actions["mg1"][:] = -999
    batch = alg.replay.sample(1, AGENTS, torch.device("cpu"))
    np.testing.assert_array_equal(batch["observations"]["mg1"][0], expected_obs)
    np.testing.assert_array_equal(batch["next_global_state"][0], expected_state)
    assert batch["actions"]["mg1"].item() != -999


def test_coalition_projection_preserves_local_dispatch_and_masks_only_cooperation(env):
    alg = Matd3ShapleyQ.from_env(env, algorithm_config_with_baseline("heuristic"))
    _, reset = env.reset(seed=42)
    state = torch.tensor(reset["global_state"]).unsqueeze(0)
    raw = torch.full((1, 3, 1), -0.5)
    grand, _ = alg.project_actions(state, raw)
    for coalition in all_coalitions(AGENTS):
        projected, plans = alg.coalition_projector(state, raw, coalition)
        torch.testing.assert_close(projected[..., 0], grand[..., 0])
        assert projected[..., 0].abs().sum() > 0
        for i, mg in enumerate(AGENTS):
            assert plans[i]["in_coalition"] == (mg in coalition)
            if mg not in coalition:
                assert projected[:, i, 1].item() == 0.0
    # Exercise an actual seller/buyer opportunity; charging-only inputs cannot
    # detect accidentally leaving a nonmember's offer enabled.
    for mg in AGENTS:
        for field, value in {"load": 0.0 if mg == "mg1" else 1.0,
                             "pv": 1.0 if mg == "mg1" else 0.0, "wind": 0.0,
                             "feasible_discharge_max": 1.0,
                             "feasible_discharge_min": 0.0,
                             "feasible_charge_min": 0.0}.items():
            state[:, alg.state_schema.index(f"{mg}_{field}")] = value
    raw = torch.tensor([[[1.0], [0.0], [0.0]]])
    grand, _ = alg.project_actions(state, raw)
    assert grand[0, 0, 1] > 0
    empty, _ = alg.coalition_projector(state, raw, frozenset())
    singleton, _ = alg.coalition_projector(state, raw, {"mg1"})
    pair, _ = alg.coalition_projector(state, raw, {"mg1", "mg2"})
    assert empty[..., 1].abs().sum() == 0
    assert singleton[..., 1].abs().sum() == 0
    assert pair[0, 0, 1] > 0
    assert pair[0, 2, 1] == 0
    torch.testing.assert_close(empty[..., 0], grand[..., 0])


def test_marginal_critics_enumerate_exact_predecessor_weights(env):
    cfg = small_shapley_config()
    cfg["algorithm"]["hidden_dims"] = [1]
    alg = Matd3ShapleyQ.from_env(env, cfg, seed=42)
    # ReLU(|S|-1) only activates for the two-agent predecessor. Its correct
    # weight is 1/3, not the uniform-enumeration weight 1/4.
    with torch.no_grad():
        for critic in alg.critics.values():
            for p in critic.parameters():
                p.zero_()
            for network, output in [(critic.q1_network, 3.), (critic.q2_network, 6.)]:
                network[0].weight[:, env.global_state_dim:env.global_state_dim + 3] = 1.0
                network[0].bias.fill_(-1.)
                network[-1].weight.fill_(output)
    _, info = env.reset(seed=42)
    q1, q2 = alg._shapley_values(torch.tensor(info["global_state"]).unsqueeze(0),
                               torch.full((1, 3, 1), -0.5), alg.critics)
    for mg in AGENTS:
        assert q1[mg].item() == pytest.approx(1.0)
        assert q2[mg].item() == pytest.approx(2.0)


def small_shapley_config(mode="heuristic", checkpoint=None):
    cfg = algorithm_config_with_baseline(mode, checkpoint)
    cfg["algorithm"].update(hidden_dims=[16, 16], target_policy_noise=0.0)
    return cfg


def constant_twin_critics(critics, q1_values, q2_values):
    with torch.no_grad():
        for mg, v1, v2 in zip(AGENTS, q1_values, q2_values):
            critic = critics[mg]
            for p in critic.parameters():
                p.zero_()
            critic.q1_network[-1].bias.fill_(v1)
            critic.q2_network[-1].bias.fill_(v2)


@pytest.mark.parametrize("done, target, residual", [(False, 4.5, 8.0), (True, 3.0, 13.0)])
def test_update_uses_surplus_bellman_target_and_real_grand_reference(env, done, target, residual):
    cfg = small_shapley_config()
    cfg["algorithm"]["gamma"] = 0.5
    alg = Matd3ShapleyQ.from_env(env, cfg, seed=42)
    constant_twin_critics(alg.critics, (2., 2., 2.), (2., 2., 2.))
    constant_twin_critics(alg.target_critics, (10., 1., 10.), (1., 10., 1.))
    obs, info = env.reset(seed=42)
    alg.replay.add(obs, info["global_state"], alg.select_actions(obs), 5.,
                   obs, info["global_state"], done, independent_reward=2.,
                   cooperation_surplus_reward=3., independent_return_to_go=12.,
                   next_independent_return_to_go=7.)
    metrics = alg.update(1)
    assert metrics["cooperation_surplus_q_target"] == pytest.approx(target)
    assert metrics["critic1_loss"] == pytest.approx((6.0 - target) ** 2)
    assert metrics["critic2_loss"] == pytest.approx((6.0 - target) ** 2)
    assert metrics["grand_decomposition_residual"] == pytest.approx(residual)
    assert metrics["algorithm_diagnostics"]["grand_decomposition_residual"] == pytest.approx(residual)
    assert metrics["shapley_q_sum"] == pytest.approx(6.)


@pytest.mark.parametrize("mode", ["heuristic", "frozen_policy"])
def test_actor_gradients_are_finite_nonzero_and_baseline_is_frozen(env, tmp_path, mode):
    checkpoint = None
    if mode == "frozen_policy":
        checkpoint = str(save_independent_dispatch_checkpoint(env, tmp_path / "baseline.pt"))
    # The frozen source uses the default actor architecture.
    alg = Matd3ShapleyQ.from_env(env, algorithm_config_with_baseline(mode, checkpoint), seed=42)
    alg.policy_delay = 1
    obs, reset = env.reset(seed=42)
    alg.on_episode_start(env, seed=42)
    assert len(alg.actors) == len(alg.critics) == 3
    assert alg.replay.__class__.__module__ == "algorithms.matd3_shapley_q.replay_buffer"
    state = reset["global_state"]
    before = {mg: [p.detach().clone() for p in alg.actors[mg].parameters()] for mg in AGENTS}
    for t in range(8):
        actions = {mg: np.array([-0.5], dtype=np.float32) for mg in AGENTS}
        next_obs, rewards, done, info = env.step(actions)
        assert len(set(rewards.values())) == 1
        baseline = alg.get_independent_step_data(t)
        alg.replay.add(obs, state, actions, rewards["mg1"], next_obs,
            info["global_state_after"], done, independent_reward=baseline["reward"],
            cooperation_surplus_reward=rewards["mg1"] - baseline["reward"],
            independent_return_to_go=baseline["return_to_go"],
            next_independent_return_to_go=baseline["next_return_to_go"])
        obs, state = next_obs, info["global_state_after"]
    metrics = alg.update(8)
    for mg in AGENTS:
        assert np.isfinite(metrics["actor_grad_norms"][mg])
        assert metrics["actor_grad_norms"][mg] > 0
        assert any(not torch.equal(old, p) for old, p in zip(before[mg], alg.actors[mg].parameters()))
    if alg.baseline_evaluator.frozen_actors is not None:
        assert all(p.grad is None and not p.requires_grad
                   for p in alg.baseline_evaluator.frozen_actors.parameters())


@pytest.mark.parametrize("field,bad", [("shapley_q_version", "full_enumeration_v1"),
    ("value_definition", "grand_return"), ("algorithm_slug", "matd3"),
    ("action_semantics", "retired"), ("projection_signature", "wrong"),
    ("state_schema", ()), ("observation_schemas", {}),
    ("baseline", {"mode": "trainable"})])
def test_checkpoint_roundtrip_and_strict_surplus_metadata(env, tmp_path, field, bad):
    cfg = small_shapley_config()
    alg = Matd3ShapleyQ.from_env(env, cfg, seed=42)
    obs, _ = env.reset(seed=42)
    expected = alg.select_actions(obs)
    path = tmp_path / "shapley.pt"
    alg.save(path)
    payload = torch.load(path, weights_only=True)
    assert payload["shapley_q_version"] == "surplus_q_v2"
    assert payload["value_definition"] == "cooperation_surplus"
    assert payload["baseline"]["mode"] == "heuristic"
    restored = Matd3ShapleyQ.from_env(env, cfg, seed=73)
    restored.load(path)
    for mg in AGENTS:
        np.testing.assert_array_equal(restored.select_actions(obs)[mg], expected[mg])
    payload[field] = bad
    torch.save(payload, path)
    with pytest.raises(ValueError):
        restored.load(path)


def assert_algorithm_state_unchanged(algorithm, saved, update_steps):
    assert algorithm.update_steps == update_steps
    for name, state in saved.items():
        for key, value in getattr(algorithm, name).state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)


@pytest.mark.parametrize("bad_steps", ["missing", float("nan"), -2, 1.5, True, "3", None])
def test_checkpoint_invalid_update_steps_is_atomic(env, tmp_path, bad_steps):
    source = Matd3ShapleyQ.from_env(env, small_shapley_config(), seed=42)
    destination = Matd3ShapleyQ.from_env(env, small_shapley_config(), seed=73)
    destination.update_steps = 9
    saved = {name: copy.deepcopy(getattr(destination, name).state_dict())
             for name in ("actors", "target_actors", "critics", "target_critics")}
    path = tmp_path / "invalid_steps.pt"
    source.save(path)
    payload = torch.load(path, weights_only=True)
    if bad_steps == "missing":
        payload.pop("update_steps")
    else:
        payload["update_steps"] = bad_steps
    torch.save(payload, path)
    with pytest.raises(ValueError, match="update_steps"):
        destination.load(path)
    assert_algorithm_state_unchanged(destination, saved, 9)


def test_checkpoint_float64_overflow_is_rejected_before_live_mutation(env, tmp_path):
    source = Matd3ShapleyQ.from_env(env, small_shapley_config(), seed=42)
    destination = Matd3ShapleyQ.from_env(env, small_shapley_config(), seed=73)
    destination.update_steps = 9
    saved = {name: copy.deepcopy(getattr(destination, name).state_dict())
             for name in ("actors", "target_actors", "critics", "target_critics")}
    path = tmp_path / "overflow.pt"
    source.save(path)
    payload = torch.load(path, weights_only=True)
    tensor_name = next(iter(payload["target_critics"]))
    original = payload["target_critics"][tensor_name]
    payload["target_critics"][tensor_name] = torch.full(original.shape, 1e300, dtype=torch.float64)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="target_critics.*finite"):
        destination.load(path)
    assert_algorithm_state_unchanged(destination, saved, 9)


def test_coalition_projector_accepts_batch_masks_and_returns_tensor_membership(env):
    alg = Matd3ShapleyQ.from_env(env, small_shapley_config())
    _, info = env.reset(seed=42)
    state = torch.tensor(info["global_state"]).unsqueeze(0).repeat(3, 1)
    for mg in AGENTS:
        for field, value in {"load": 0.0 if mg == "mg1" else 1.0,
                             "pv": 1.0 if mg == "mg1" else 0.0, "wind": 0.0,
                             "feasible_discharge_max": 1.0,
                             "feasible_discharge_min": 0.0,
                             "feasible_charge_min": 0.0}.items():
            state[:, alg.state_schema.index(f"{mg}_{field}")] = value
    raw = torch.tensor([[[1.0], [-0.5], [-0.5]]]).repeat(3, 1, 1)
    grand, _ = alg.project_actions(state, raw)
    empty, plans = alg.coalition_projector(state, raw, torch.zeros((3, 3)))
    torch.testing.assert_close(empty[..., 0], grand[..., 0])
    assert not empty[..., 1].any()
    for plan in plans:
        assert plan["in_coalition"].dtype == torch.bool
        assert plan["in_coalition"].shape == (3,)
        assert not plan["in_coalition"].any()
    mask = torch.tensor([[False, False, False], [True, False, False], [True, True, False]])
    mixed, plans = alg.coalition_projector(state, raw, mask)
    torch.testing.assert_close(mixed[..., 0], grand[..., 0])
    assert not mixed[:2, :, 1].any()
    assert mixed[2, 0, 1] > 0
    assert not mixed[:, 2, 1].any()
    for i, plan in enumerate(plans):
        torch.testing.assert_close(plan["in_coalition"], mask[:, i])
    _, set_plans = alg.coalition_projector(state, raw, {"mg1", "mg2"})
    for i, plan in enumerate(set_plans):
        torch.testing.assert_close(plan["in_coalition"], torch.full((3,), i < 2))


@pytest.mark.parametrize("mask", [torch.zeros(3), torch.zeros(2, 3), torch.zeros(3, 2),
    torch.full((3, 3), 0.5), torch.full((3, 3), float("nan")),
    torch.full((3, 3), float("inf")), torch.full((3, 3), -1)])
def test_coalition_projector_rejects_invalid_batch_masks(env, mask):
    alg = Matd3ShapleyQ.from_env(env, small_shapley_config())
    _, info = env.reset(seed=42)
    state = torch.tensor(info["global_state"]).unsqueeze(0).repeat(3, 1)
    with pytest.raises(ValueError, match="coalition|mask"):
        alg.coalition_projector(state, torch.zeros((3, 3, 1)), mask)


def valid_replay_transition():
    return dict(observations={mg: np.array([1., 2.]) for mg in AGENTS},
        global_state=np.array([1., 2., 3.]), actions={mg: np.array([0.25]) for mg in AGENTS},
        team_reward=5., next_observations={mg: np.array([3., 4.]) for mg in AGENTS},
        next_global_state=np.array([3., 4., 5.]), done=False,
        independent_reward=2., cooperation_surplus_reward=3.,
        independent_return_to_go=12., next_independent_return_to_go=7.)


@pytest.mark.parametrize("field", ["team_reward", "independent_reward", "cooperation_surplus_reward",
    "independent_return_to_go", "next_independent_return_to_go", "observations", "actions",
    "next_observations", "global_state", "next_global_state"])
@pytest.mark.parametrize("bad", [1e40, float("nan"), float("inf")])
def test_replay_rejects_nonfinite_or_float32_overflow_every_field(field, bad):
    buf = ShapleyReplayBuffer(4)
    data = valid_replay_transition()
    if field in ("observations", "next_observations", "actions"):
        data[field]["mg2"][0] = bad
    elif field in ("global_state", "next_global_state"):
        data[field][0] = bad
    else:
        data[field] = bad
    with pytest.raises(ValueError, match=field):
        buf.add(**data)
    assert len(buf) == 0
    buf.add(**valid_replay_transition())
    batch = buf.sample(1, AGENTS, torch.device("cpu"))
    assert batch["team_reward"].item() == 5.0


@pytest.mark.parametrize("done", ["False", 0, 1, 2, -1, 0.5, float("nan"), None, [False]])
def test_replay_rejects_nonboolean_done(done):
    data = valid_replay_transition()
    data["done"] = done
    with pytest.raises(ValueError, match="done"):
        ShapleyReplayBuffer(4).add(**data)


@pytest.mark.parametrize("field", ["actions", "observations", "next_observations"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "scalar", "matrix", "empty", "wrong_width"])
def test_replay_rejects_agent_or_shape_mismatch_without_poisoning_buffer(field, mutation):
    buf = ShapleyReplayBuffer(4)
    buf.add(**valid_replay_transition())
    data = valid_replay_transition()
    if mutation == "missing":
        data[field].pop("mg3")
    elif mutation == "extra":
        data[field]["mg4"] = data[field]["mg3"].copy()
    else:
        data[field]["mg2"] = {"scalar": np.array(0.5), "matrix": np.ones((1, 2)),
                               "empty": np.array([]), "wrong_width": np.ones(4)}[mutation]
    with pytest.raises(ValueError, match=field):
        buf.add(**data)
    assert len(buf) == 1
    buf.add(**valid_replay_transition())
    assert buf.sample(2, AGENTS, torch.device("cpu"))["actions"]["mg3"].shape == (2, 1)


@pytest.mark.parametrize("field", ["global_state", "next_global_state"])
@pytest.mark.parametrize("bad", [np.array(3.0), np.array([]), np.ones((1, 3)), np.ones(4)])
def test_replay_rejects_state_shape_mismatch(field, bad):
    data = valid_replay_transition()
    data[field] = bad
    with pytest.raises(ValueError, match="state"):
        ShapleyReplayBuffer(4).add(**data)


def test_replay_first_add_rejects_inconsistent_agent_keys():
    data = valid_replay_transition()
    data["actions"].pop("mg3")
    with pytest.raises(ValueError, match="actions"):
        ShapleyReplayBuffer(4).add(**data)


def test_algorithm_replay_extras_uses_fixed_baseline_step():
    class FakeAlgorithm:
        def get_independent_step_data(self, t):
            assert t == 3
            return {"reward": 2.0, "profit": 200.0,
                    "return_to_go": 8.0, "next_return_to_go": 6.0}

    extras = _algorithm_replay_extras(FakeAlgorithm(), 3, team_reward=5.0)
    assert extras == pytest.approx({
        "independent_reward": 2.0,
        "cooperation_surplus_reward": 3.0,
        "independent_return_to_go": 8.0,
        "next_independent_return_to_go": 6.0,
    })
    assert _algorithm_replay_extras(object(), 3, team_reward=5.0) == {}


def test_algorithm_episode_metrics_derives_grand_and_surplus_fields():
    class FakeAlgorithm:
        def episode_metrics(self):
            return {"independent_baseline_return": 8.0,
                    "independent_baseline_profit": 80.0}

    result = _algorithm_episode_metrics(
        FakeAlgorithm(), {"team_reward": 12.0, "team_profit": 125.0}
    )
    assert result == pytest.approx({
        "grand_team_return": 12.0,
        "grand_team_profit": 125.0,
        "independent_baseline_return": 8.0,
        "independent_baseline_profit": 80.0,
        "cooperation_surplus_return": 4.0,
        "cooperation_surplus_profit": 45.0,
    })
    assert _algorithm_episode_metrics(object(), {"team_reward": 1.0, "team_profit": 2.0}) == {}


def test_shapley_diagnostics_and_summary_preserve_missing_as_none():
    diagnostics = _diagnostic_means([{
        "shapley_q_mg1": 1.0, "shapley_q_mg2": 2.0, "shapley_q_mg3": 3.0,
        "shapley_q_sum": 6.0, "cooperation_surplus_q_target": 5.0,
        "efficiency_residual": 1.0, "grand_decomposition_residual": -0.5,
    }])
    assert diagnostics["shapley_q_sum"] == pytest.approx(6.0)
    assert diagnostics["grand_decomposition_residual"] == pytest.approx(-0.5)

    history = [{
        "grand_team_return": 12.0,
        "independent_baseline_return": 8.0,
        "cooperation_surplus_return": 4.0,
        **diagnostics,
    }]
    summary = _training_diagnostic_summary(history)
    assert summary["final_100_grand_return_mean"] == pytest.approx(12.0)
    assert summary["final_100_shapley_q_sum_mean"] == pytest.approx(6.0)
    assert summary["final_100_grand_decomposition_residual_mean"] == pytest.approx(-0.5)
    empty = _training_diagnostic_summary([{}])
    assert all(value is None for value in empty.values())


def test_hidden_smoke_hyperparameter_overrides_are_run_local():
    parser = build_parser()
    args = parser.parse_args(["--warmup-steps", "0", "--batch-size", "8"])
    settings = {"warmup_steps": 1000, "batch_size": 128}
    original = copy.deepcopy(settings)
    assert _runtime_hyperparameters(args, settings) == (0, 8)
    assert settings == original
    assert "--warmup-steps" not in parser.format_help()
    assert "--batch-size" not in parser.format_help()


@pytest.mark.parametrize("argv", [
    ["--warmup-steps", "-1"],
    ["--batch-size", "0"],
])
def test_invalid_smoke_hyperparameter_overrides_fail_fast(argv):
    args = build_parser().parse_args(argv)
    with pytest.raises(ValueError, match="warmup_steps|batch_size"):
        _runtime_hyperparameters(args, {"warmup_steps": 10, "batch_size": 4})


def test_formal_evaluation_exports_matched_independent_baseline(env, tmp_path):
    config = small_shapley_config()
    algorithm_path = tmp_path / "matd3_shapley_q.yaml"
    algorithm_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    algorithm = Matd3ShapleyQ.from_env(env, config, seed=42)
    checkpoint = tmp_path / "best_model.pt"
    algorithm.save(checkpoint)
    output = tmp_path / "formal"

    result = evaluate(
        ROOT / "configs/cooperative.yaml",
        algorithm_path,
        checkpoint,
        split="test",
        run_milp=False,
        output_dir=output,
        test_seed=42,
        max_days=1,
        algorithm_name="matd3_shapley_q",
    )
    row = pd.read_csv(output / "test_daily_metrics.csv").iloc[0]
    assert row["cooperation_surplus_return"] == pytest.approx(
        row["grand_team_return"] - row["independent_baseline_return"]
    )
    assert row["cooperation_surplus_profit"] == pytest.approx(
        row["algorithm_team_profit"] - row["independent_baseline_profit"]
    )
    assert result["independent_baseline_profit_statistics"]["mean"] is not None
    assert result["cooperation_surplus_return_statistics"]["mean"] is not None


TEST_DAY = "2018-01-01"


def make_deterministic_cooperation_config(tmp_path):
    timestamps = pd.date_range(TEST_DAY, periods=24, freq="h")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "load_kw": np.ones(24),
        "pv_kw": np.zeros(24),
        "wind_kw": np.zeros(24),
        "buy_price_per_kwh": np.full(24, 1.20),
        "grid_carbon_factor_kg_per_kwh": np.full(24, 0.5),
        "temperature_c": np.full(24, 20.0),
        "ghi_w_per_m2": np.full(24, 1000.0),
        "wind_speed_mps": np.zeros(24),
        "mg1_ghi_w_per_m2": np.full(24, 1000.0),
        "mg1_wind_speed_mps": np.zeros(24),
        "mg2_ghi_w_per_m2": np.zeros(24),
        "mg2_wind_speed_mps": np.zeros(24),
        "mg3_ghi_w_per_m2": np.zeros(24),
        "mg3_wind_speed_mps": np.zeros(24),
    })
    csv_path = tmp_path / "deterministic.csv"
    frame.to_csv(csv_path, index=False)
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    for key in ("train_path", "val_path", "test_path"):
        cfg["data"][key] = str(csv_path)
    cfg["observation"]["noisy_forecast"] = False
    return cfg


def rollout_fixed_zero_actions(current_env, day=TEST_DAY, seed=17):
    current_env.reset(day=day, seed=seed, deterministic=True)
    rewards, profit, p2p = [], 0.0, 0.0
    done = False
    while not done:
        actions = {mg: np.zeros(1, dtype=np.float32) for mg in current_env.agent_ids}
        _, shared_rewards, done, info = current_env.step(actions)
        rewards.append(float(shared_rewards[current_env.agent_ids[0]]))
        profit += float(info["team_profit"])
        p2p += float(info["p2p"]["total_p2p_kwh"])
    return {"rewards": rewards, "profit": profit, "p2p": p2p}


def test_deterministic_physics_has_positive_cooperation_surplus(tmp_path):
    base = make_deterministic_cooperation_config(tmp_path)
    grand_cfg = copy.deepcopy(base)
    independent_cfg = copy.deepcopy(base)
    grand_cfg["p2p"]["enabled"] = True
    grand_cfg["ess_sharing"]["ess_p2p_enabled"] = True
    independent_cfg["p2p"]["enabled"] = False
    independent_cfg["ess_sharing"]["ess_p2p_enabled"] = False
    grand = rollout_fixed_zero_actions(MultiMicrogridEnv(grand_cfg))
    independent = rollout_fixed_zero_actions(MultiMicrogridEnv(independent_cfg))
    assert grand["p2p"] > 0.0
    assert grand["profit"] > independent["profit"]
    assert sum(grand["rewards"]) - sum(independent["rewards"]) > 0.0


def test_deterministic_no_cooperation_has_zero_shapley_surplus(tmp_path):
    config = make_deterministic_cooperation_config(tmp_path)
    config["p2p"]["enabled"] = False
    config["ess_sharing"]["ess_p2p_enabled"] = False
    first = rollout_fixed_zero_actions(MultiMicrogridEnv(copy.deepcopy(config)), seed=23)
    second = rollout_fixed_zero_actions(MultiMicrogridEnv(copy.deepcopy(config)), seed=23)
    assert first["rewards"] == pytest.approx(second["rewards"])
    assert first["profit"] == pytest.approx(second["profit"])
    values = {coalition: second["profit"] for coalition in all_coalitions(AGENTS)}
    assert exact_shapley_values(values, AGENTS) == pytest.approx(
        {mg: 0.0 for mg in AGENTS}
    )


def test_empty_coalition_projector_preserves_local_dispatch(env):
    algorithm = Matd3ShapleyQ.from_env(env, small_shapley_config())
    _, info = env.reset(seed=42)
    state = torch.as_tensor(info["global_state"]).unsqueeze(0)
    raw = torch.full((1, 3, 1), -0.5)
    effective, plans = algorithm.coalition_projector(
        state, raw, torch.zeros((1, 3))
    )
    assert torch.count_nonzero(effective[..., 0]).item() > 0
    assert torch.count_nonzero(effective[..., 1]).item() == 0
    assert all(not plan["in_coalition"].any() for plan in plans)
