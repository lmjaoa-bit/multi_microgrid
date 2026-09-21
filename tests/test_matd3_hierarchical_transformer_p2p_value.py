from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from algorithms.matd3_hierarchical_transformer_p2p_value.p2p_auxiliary import (
    P2PLabels,
    P2PLosses,
    P2PPrediction,
    P2PRepresentations,
    P2PValueHead,
    compute_p2p_auxiliary_loss,
    extract_p2p_labels,
    p2p_auxiliary_weight,
    validate_p2p_auxiliary_config,
)
from envs.multi_microgrid_env import MultiMicrogridEnv

VALID_AUX: dict[str, Any] = {
    "enabled": True,
    "hidden_dim": 64,
    "loss_weight_max": 0.05,
    "ramp_updates": 5000,
    "any_fill_loss_weight": 1.0,
    "fill_ratio_loss_weight": 1.0,
    "offer_epsilon_kwh": 1.0e-6,
    "consistency_tolerance_kwh": 1.0e-5,
    "detach_behavior_action": True,
}

CONFIG_PATH = Path("configs/algorithms/matd3_hierarchical_transformer_p2p_value.yaml")


@pytest.fixture
def env_and_config():
    env_config = yaml.safe_load(
        Path("configs/cooperative.yaml").read_text(encoding="utf-8")
    )
    algo_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return MultiMicrogridEnv(env_config, split="train"), algo_config


def test_label_extraction_restores_kwh_and_masks_zero_offer():
    schema = [
        "previous_ess_p2p_offer",
        "previous_ess_p2p_accepted",
        "previous_ess_p2p_unmatched",
    ]
    next_obs = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.2, 0.3], [1.0, 1.0, 0.0]])
    labels = extract_p2p_labels(
        next_obs,
        schema,
        trade_scale=10.0,
        offer_epsilon_kwh=1.0e-6,
        consistency_tolerance_kwh=1.0e-5,
    )
    assert labels.offered_mask.squeeze(-1).tolist() == [False, True, True]
    assert torch.equal(labels.offer_kwh.squeeze(-1), torch.tensor([0.0, 5.0, 10.0]))
    assert torch.allclose(
        labels.fill_ratio_target.squeeze(-1), torch.tensor([0.0, 0.4, 1.0])
    )
    assert labels.any_fill_target.squeeze(-1).tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("field", ["offer", "accepted", "unmatched"])
def test_label_extraction_rejects_nonfinite_values(field: str):
    schema = [
        "previous_ess_p2p_offer",
        "previous_ess_p2p_accepted",
        "previous_ess_p2p_unmatched",
    ]
    column = {"offer": 0, "accepted": 1, "unmatched": 2}[field]
    next_obs = torch.tensor([[1.0, 0.5, 0.5]])
    next_obs[0, column] = torch.nan
    with pytest.raises(ValueError, match=field):
        extract_p2p_labels(
            next_obs,
            schema,
            trade_scale=10.0,
            offer_epsilon_kwh=1e-6,
            consistency_tolerance_kwh=1e-5,
        )


def test_label_extraction_rejects_negative_and_accepted_above_offer():
    schema = [
        "previous_ess_p2p_offer",
        "previous_ess_p2p_accepted",
        "previous_ess_p2p_unmatched",
    ]
    for row, match in [([-0.1, 0.0, 0.0], "offer"), ([0.5, 0.6, 0.0], "accepted")]:
        with pytest.raises(ValueError, match=match):
            extract_p2p_labels(
                torch.tensor([row]),
                schema,
                trade_scale=10.0,
                offer_epsilon_kwh=1e-6,
                consistency_tolerance_kwh=1e-5,
            )


def test_consistency_residual_obeys_tolerance_boundary():
    schema = [
        "previous_ess_p2p_offer",
        "previous_ess_p2p_accepted",
        "previous_ess_p2p_unmatched",
    ]
    scale, tol = 10.0, 1e-5
    boundary = torch.tensor([[1.0, 0.4, (6.0 - tol) / scale]])
    labels = extract_p2p_labels(
        boundary,
        schema,
        trade_scale=scale,
        offer_epsilon_kwh=1e-6,
        consistency_tolerance_kwh=tol,
    )
    assert labels.consistency_residual_kwh.item() == pytest.approx(tol, abs=1e-6)
    outside = torch.tensor([[1.0, 0.4, (6.0 - 2 * tol) / scale]])
    with pytest.raises(ValueError, match="consistency"):
        extract_p2p_labels(
            outside,
            schema,
            trade_scale=scale,
            offer_epsilon_kwh=1e-6,
            consistency_tolerance_kwh=tol,
        )


def test_label_extraction_rejects_missing_field_and_invalid_trade_scale():
    with pytest.raises(
        ValueError, match="previous_ess_p2p_accepted.*previous_ess_p2p_unmatched"
    ):
        extract_p2p_labels(
            torch.zeros(1, 1),
            ["previous_ess_p2p_offer"],
            trade_scale=1.0,
            offer_epsilon_kwh=1e-6,
            consistency_tolerance_kwh=1e-5,
        )
    schema = [
        "previous_ess_p2p_offer",
        "previous_ess_p2p_accepted",
        "previous_ess_p2p_unmatched",
    ]
    for scale in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="trade_scale"):
            extract_p2p_labels(
                torch.zeros(1, 3),
                schema,
                trade_scale=scale,
                offer_epsilon_kwh=1e-6,
                consistency_tolerance_kwh=1e-5,
            )


def test_no_offer_loss_is_differentiable_zero():
    logits = torch.randn(4, 1, requires_grad=True)
    ratios = torch.sigmoid(torch.randn(4, 1, requires_grad=True))
    schema = [
        "previous_ess_p2p_offer",
        "previous_ess_p2p_accepted",
        "previous_ess_p2p_unmatched",
    ]
    labels = extract_p2p_labels(
        torch.zeros(4, 3),
        schema,
        trade_scale=10.0,
        offer_epsilon_kwh=1e-6,
        consistency_tolerance_kwh=1e-5,
    )
    losses = compute_p2p_auxiliary_loss(
        P2PPrediction(logits, ratios), labels, VALID_AUX
    )
    losses.total.backward()
    assert losses.total.item() == 0.0
    assert torch.isfinite(logits.grad).all()


def test_auxiliary_weight_linear_ramp():
    assert p2p_auxiliary_weight(0, VALID_AUX) == pytest.approx(0.0)
    assert p2p_auxiliary_weight(2500, VALID_AUX) == pytest.approx(0.025)
    assert p2p_auxiliary_weight(5000, VALID_AUX) == pytest.approx(0.05)
    assert p2p_auxiliary_weight(9000, VALID_AUX) == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("key", "bad_value", "err_match"),
    [
        ("detach_behavior_action", False, "detach_behavior_action"),
        ("loss_weight_max", -0.01, "loss_weight_max"),
        ("any_fill_loss_weight", -1.0, "any_fill_loss_weight"),
        ("fill_ratio_loss_weight", -1.0, "fill_ratio_loss_weight"),
        ("ramp_updates", 0, "ramp_updates"),
        ("ramp_updates", -10, "ramp_updates"),
        ("ramp_updates", True, "ramp_updates"),
        ("hidden_dim", 0, "hidden_dim"),
        ("hidden_dim", -32, "hidden_dim"),
        ("hidden_dim", True, "hidden_dim"),
        ("offer_epsilon_kwh", 0.0, "offer_epsilon_kwh"),
        ("offer_epsilon_kwh", -1e-6, "offer_epsilon_kwh"),
        ("consistency_tolerance_kwh", 0.0, "consistency_tolerance_kwh"),
        ("consistency_tolerance_kwh", -1e-5, "consistency_tolerance_kwh"),
        ("loss_weight_max", float("nan"), "loss_weight_max"),
        ("loss_weight_max", float("inf"), "loss_weight_max"),
    ],
)
def test_validate_p2p_auxiliary_config_rejections(
    key: str, bad_value: Any, err_match: str
):
    cfg = copy.deepcopy(VALID_AUX)
    cfg[key] = bad_value
    with pytest.raises(ValueError, match=err_match):
        validate_p2p_auxiliary_config(cfg, require_enabled=True)


def test_validate_p2p_auxiliary_config_require_enabled():
    cfg = copy.deepcopy(VALID_AUX)
    cfg["enabled"] = False
    with pytest.raises(ValueError, match="enabled"):
        validate_p2p_auxiliary_config(cfg, require_enabled=True)
    res = validate_p2p_auxiliary_config(cfg, require_enabled=False)
    assert res["enabled"] is False


from algorithms.matd3_hierarchical_transformer_predictive_dr.actor import PredictiveDRActor
from algorithms.matd3_hierarchical_transformer_p2p_value.actor import P2PValueActor


def actor_constructor_kwargs(env, config, mg):
    settings = config["algorithm"]
    net = settings["network"]
    normalizers = {
        agent: dict(env.global_state_builder.norm[agent]) for agent in env.agent_ids
    }
    return {
        "schema": env.obs_builder.schema(mg),
        "agent_ids": env.agent_ids,
        "active_agent": mg,
        "mg_config": env.cfg_by_id[mg],
        "predictive_config": settings,
        "normalizers_by_microgrid": normalizers,
        "embed_dim": net["embed_dim"],
        "num_heads": net["num_heads"],
        "num_layers": net["num_layers"],
        "forecast_horizon": net["forecast_horizon"],
        "hidden_dim": net["hidden_dim"],
        "delta_t": env.delta_t,
        "episode_length": env.episode_length,
    }


def make_parent_actor(env_and_config, mg):
    env, config = env_and_config
    return PredictiveDRActor(**actor_constructor_kwargs(env, config, mg))


def make_child_actor(env_and_config, mg, *, enabled):
    env, config = env_and_config
    aux = copy.deepcopy(config["algorithm"]["p2p_auxiliary"])
    aux["enabled"] = enabled
    return P2PValueActor(
        **actor_constructor_kwargs(env, config, mg), p2p_auxiliary_config=aux
    )


def make_child_actor_and_batch(env_and_config, mg, *, batch):
    env, _ = env_and_config
    actor = make_child_actor(env_and_config, mg, enabled=True)
    observations, _ = env.reset(seed=42)
    observation = torch.as_tensor(observations[mg], dtype=torch.float32).unsqueeze(0)
    return actor, observation.repeat(batch, 1)


def module_grad_norm(module):
    squared = [
        parameter.grad.detach().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return 0.0 if not squared else float(torch.stack(squared).sum().sqrt().item())


@pytest.mark.parametrize("mg", ["mg1", "mg2", "mg3"])
@pytest.mark.parametrize("enabled", [False, True])
def test_child_actor_policy_path_exactly_matches_parent(env_and_config, mg, enabled):
    torch.manual_seed(42)
    parent = make_parent_actor(env_and_config, mg)
    torch.manual_seed(42)
    child = make_child_actor(env_and_config, mg, enabled=enabled)
    incompatible = child.load_state_dict(parent.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert all(key.startswith("p2p_value_head.") for key in incompatible.missing_keys)
    env, _ = env_and_config
    observations, _ = env.reset(seed=42)
    obs = torch.as_tensor(observations[mg], dtype=torch.float32).unsqueeze(0).repeat(4, 1)
    assert torch.equal(parent.forward_latent(obs), child.forward_latent(obs))
    assert torch.equal(parent(obs), child(obs))


def test_p2p_head_shapes_and_ratio_bounds(env_and_config):
    actor, obs = make_child_actor_and_batch(env_and_config, "mg2", batch=8)
    latent, reps = actor.forward_latent_with_representations(obs)
    pred = actor.predict_p2p_value(reps, torch.linspace(-1, 1, 8).unsqueeze(-1))
    assert latent.shape == (8, 2)
    assert pred.any_fill_logit.shape == (8, 1)
    assert pred.conditional_fill_ratio.shape == (8, 1)
    assert torch.all((pred.conditional_fill_ratio >= 0) & (pred.conditional_fill_ratio <= 1))


def test_auxiliary_only_gradient_boundaries(env_and_config):
    actor, obs = make_child_actor_and_batch(env_and_config, "mg2", batch=8)
    behavior = torch.zeros(8, 1, requires_grad=True)
    _, reps = actor.forward_latent_with_representations(obs)
    pred = actor.predict_p2p_value(reps, behavior)
    (pred.any_fill_logit.mean() + pred.conditional_fill_ratio.mean()).backward()
    assert module_grad_norm(actor.p2p_value_head) > 0
    assert module_grad_norm(actor.trade_encoder) > 0
    assert module_grad_norm(actor.transformer) > 0
    assert module_grad_norm(actor.ess_head) == 0
    assert module_grad_norm(actor.ess_linear_skip) == 0
    assert module_grad_norm(actor.dr_head) == 0
    assert module_grad_norm(actor.dr_linear_skip) == 0
    assert behavior.grad is None


def test_child_actor_observation_shape_validation(env_and_config):
    actor, _ = make_child_actor_and_batch(env_and_config, "mg2", batch=1)
    with pytest.raises(ValueError, match=r"Expected observation shape \[batch_size,"):
        actor.forward_latent_with_representations(torch.zeros(len(actor.schema)))


def test_child_actor_missing_p2p_schema_fields(env_and_config):
    env, config = env_and_config
    kwargs = actor_constructor_kwargs(env, config, "mg2")
    # Drop one of the p2p fields from schema
    bad_schema = [c for c in kwargs["schema"] if c != "previous_ess_p2p_accepted"]
    kwargs["schema"] = bad_schema
    aux = copy.deepcopy(config["algorithm"]["p2p_auxiliary"])
    with pytest.raises(ValueError, match="previous_ess_p2p_accepted"):
        P2PValueActor(**kwargs, p2p_auxiliary_config=aux)


from algorithms.registry import get_algorithm_spec
from algorithms.matd3_hierarchical_transformer_predictive_dr import (
    Matd3HierarchicalTransformerPredictiveDr,
)
from algorithms.matd3_hierarchical_transformer_p2p_value.algorithm import (
    Matd3HierarchicalTransformerP2PValue,
    ALGORITHM_SLUG,
    PARENT_PREDICTIVE_CONTRACT_VERSION,
    P2P_AUXILIARY_CONTRACT_VERSION,
)


def test_registry_resolves_p2p_value_algorithm():
    spec = get_algorithm_spec("matd3_hierarchical_transformer_p2p_value")
    assert spec.slug == "matd3_hierarchical_transformer_p2p_value"
    assert spec.algorithm_class is Matd3HierarchicalTransformerP2PValue
    assert spec.default_config_path == CONFIG_PATH.resolve()


def test_algorithm_from_env_constructs_p2p_value_actors_and_normalizers(env_and_config):
    env, config = env_and_config
    algo = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)
    assert algo.slug == "matd3_hierarchical_transformer_p2p_value"
    for mg in env.agent_ids:
        assert isinstance(algo.actors[mg], P2PValueActor)
        assert isinstance(algo.target_actors[mg], P2PValueActor)
        norm = algo.normalizers_by_microgrid[mg]
        assert "trade" in norm
        assert norm["trade"] > 0.0


def test_algorithm_select_actions_matches_parent_with_identical_weights(env_and_config):
    env, config = env_and_config
    parent = Matd3HierarchicalTransformerPredictiveDr.from_env(env, config, "cpu", 42)
    child = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)

    for mg in env.agent_ids:
        child.actors[mg].load_state_dict(parent.actors[mg].state_dict(), strict=False)

    obs, _ = env.reset(seed=42)
    parent_actions = parent.select_actions(obs, noise_std=0.0)
    child_actions = child.select_actions(obs, noise_std=0.0)

    for mg in env.agent_ids:
        np.testing.assert_allclose(parent_actions[mg], child_actions[mg], atol=1e-6)


def test_checkpoint_roundtrip_and_exact_equality(env_and_config, tmp_path):
    env, config = env_and_config
    algo = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)
    algo.update_steps = 123
    ckpt_path = tmp_path / "test_ckpt.pt"
    algo.save(ckpt_path)

    loaded_algo = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 999)
    loaded_algo.load(ckpt_path)

    assert loaded_algo.update_steps == 123
    for mg in env.agent_ids:
        for p1, p2 in zip(algo.actors[mg].parameters(), loaded_algo.actors[mg].parameters()):
            assert torch.equal(p1, p2)
        for p1, p2 in zip(algo.critics[mg].parameters(), loaded_algo.critics[mg].parameters()):
            assert torch.equal(p1, p2)


@pytest.mark.parametrize(
    ("field", "bad_value", "err_match"),
    [
        ("algorithm_slug", "matd3_hierarchical_transformer_predictive_dr", "algorithm_slug"),
        ("action_semantics", "wrong", "action_semantics"),
        ("parent_predictive_contract_version", "9.9", "parent_predictive_contract_version"),
        ("p2p_auxiliary_contract_version", "9.9", "p2p_auxiliary_contract_version"),
        ("predictive_hvac_config", {"enabled": False}, "predictive_hvac_config"),
        ("feasible_dr_config", {"enabled": False}, "feasible_dr_config"),
        ("p2p_auxiliary_config", {"enabled": False}, "p2p_auxiliary_config"),
        ("observation_schema_by_agent", {"mg1": ["wrong"]}, "observation_schema_by_agent"),
        ("normalizers_by_microgrid", {"mg1": {"trade": 1.0}}, "normalizers_by_microgrid"),
    ],
)
def test_checkpoint_mutations_rejected(env_and_config, tmp_path, field, bad_value, err_match):
    env, config = env_and_config
    algo = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)
    ckpt_path = tmp_path / "mutate.pt"
    algo.save(ckpt_path)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    payload[field] = bad_value
    torch.save(payload, ckpt_path)

    loaded = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)
    with pytest.raises(ValueError, match=err_match):
        loaded.load(ckpt_path)


def test_parent_checkpoint_rejected_by_child(env_and_config, tmp_path):
    env, config = env_and_config
    parent = Matd3HierarchicalTransformerPredictiveDr.from_env(env, config, "cpu", 42)
    parent_path = tmp_path / "parent_ckpt.pt"
    parent.save(parent_path)

    child = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)
    with pytest.raises(ValueError, match="algorithm_slug"):
        child.load(parent_path)


def test_deterministic_closed_loop_preservation(env_and_config):
    env, config = env_and_config
    parent = Matd3HierarchicalTransformerPredictiveDr.from_env(env, config, "cpu", 42)
    child = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)

    for mg in env.agent_ids:
        child.actors[mg].load_state_dict(parent.actors[mg].state_dict(), strict=False)

    obs, _ = env.reset(seed=42)
    for _ in range(24):
        a_parent = parent.select_actions(obs, noise_std=0.0)
        a_child = child.select_actions(obs, noise_std=0.0)
        for mg in env.agent_ids:
            np.testing.assert_allclose(a_parent[mg], a_child[mg], atol=1e-6)
        obs, _, done, _ = env.step(a_child)
        if done:
            break


def clone_parameters(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def parameters_changed(before, module):
    return any(
        not torch.equal(before[name], value.detach())
        for name, value in module.named_parameters()
    )


def add_replay_rows(algo, env, *, count, outcome_kwh):
    observations, reset_info = env.reset(seed=123)
    state = reset_info["global_state"]
    for row_index in range(count):
        actions = {
            mg: np.zeros(env.action_dim, dtype=np.float32) for mg in env.agent_ids
        }
        next_observations, rewards, done, info = env.step(actions)
        patched = {mg: value.copy() for mg, value in next_observations.items()}
        for mg in env.agent_ids:
            scale = float(env.global_state_builder.norm[mg]["trade"])
            schema = algo.observation_schemas[mg]
            for name, physical in zip(
                (
                    "previous_ess_p2p_offer",
                    "previous_ess_p2p_accepted",
                    "previous_ess_p2p_unmatched",
                ),
                outcome_kwh,
            ):
                patched[mg][schema.index(name)] = physical / scale
        algo.replay.add(
            observations,
            state,
            actions,
            float(rewards[env.agent_ids[0]]),
            patched,
            info["global_state_after"],
            done,
        )
        if done and row_index + 1 < count:
            observations, reset_info = env.reset(seed=124 + row_index)
            state = reset_info["global_state"]
        else:
            observations, state = next_observations, info["global_state_after"]


def test_update_changes_auxiliary_and_shared_parameters_on_actor_step(env_and_config):
    env, config = env_and_config
    config = copy.deepcopy(config)
    config["algorithm"]["policy_delay"] = 1
    config.setdefault("logging", {})["diagnostic_interval"] = 1
    algo = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)
    add_replay_rows(algo, env, count=4, outcome_kwh=(10.0, 4.0, 6.0))
    algo.policy_delay = 1
    before_head = clone_parameters(algo.actors["mg2"].p2p_value_head)
    before_trade = clone_parameters(algo.actors["mg2"].trade_encoder)
    metrics = algo.update(batch_size=4)
    assert parameters_changed(before_head, algo.actors["mg2"].p2p_value_head)
    assert parameters_changed(before_trade, algo.actors["mg2"].trade_encoder)
    assert algo.update_steps == 1
    assert metrics["p2p_aux_offered_sample_count"] == pytest.approx(4.0)


def test_no_offer_training_batch_is_finite(env_and_config):
    env, config = env_and_config
    config = copy.deepcopy(config)
    config["algorithm"]["policy_delay"] = 1
    config.setdefault("logging", {})["diagnostic_interval"] = 1
    algo = Matd3HierarchicalTransformerP2PValue.from_env(env, config, "cpu", 42)
    add_replay_rows(algo, env, count=4, outcome_kwh=(0.0, 0.0, 0.0))
    metrics = algo.update(batch_size=4)
    assert metrics["p2p_aux_total_loss"] == pytest.approx(0.0)
    assert metrics["p2p_aux_offered_sample_count"] == pytest.approx(0.0)
    assert all(torch.isfinite(p).all() for p in algo.actors.parameters())


def test_zero_weight_parent_equivalence(env_and_config):
    env, config = env_and_config
    cfg_parent = copy.deepcopy(config)
    cfg_child = copy.deepcopy(config)
    cfg_child["algorithm"]["p2p_auxiliary"]["loss_weight_max"] = 0.0
    cfg_parent["algorithm"]["policy_delay"] = 1
    cfg_child["algorithm"]["policy_delay"] = 1

    torch.manual_seed(42)
    parent = Matd3HierarchicalTransformerPredictiveDr.from_env(env, cfg_parent, "cpu", 42)
    torch.manual_seed(42)
    child = Matd3HierarchicalTransformerP2PValue.from_env(env, cfg_child, "cpu", 42)

    for mg in env.agent_ids:
        child.actors[mg].load_state_dict(parent.actors[mg].state_dict(), strict=False)
        child.target_actors[mg].load_state_dict(parent.target_actors[mg].state_dict(), strict=False)
        child.critics[mg].load_state_dict(parent.critics[mg].state_dict())
        child.target_critics[mg].load_state_dict(parent.target_critics[mg].state_dict())

    add_replay_rows(parent, env, count=8, outcome_kwh=(10.0, 4.0, 6.0))
    # mirror replay into child
    for i in range(len(parent.replay)):
        entry = parent.replay.storage[i]
        child.replay.add(
            entry.observations,
            entry.global_state,
            entry.actions,
            entry.team_reward,
            entry.next_observations,
            entry.next_global_state,
            entry.done,
        )

    rng_state = torch.random.get_rng_state()
    parent.update(batch_size=4)
    torch.random.set_rng_state(rng_state)
    child.update(batch_size=4)

    for mg in env.agent_ids:
        for name, p_param in parent.actors[mg].named_parameters():
            c_param = dict(child.actors[mg].named_parameters())[name]
            torch.testing.assert_close(c_param, p_param)


def test_diagnostic_non_interference(env_and_config):
    env, config = env_and_config
    c1 = copy.deepcopy(config)
    c2 = copy.deepcopy(config)
    c1["logging"] = {"diagnostic_interval": 0}
    c2["logging"] = {"diagnostic_interval": 1}
    c1["algorithm"]["policy_delay"] = 1
    c2["algorithm"]["policy_delay"] = 1

    torch.manual_seed(42)
    a1 = Matd3HierarchicalTransformerP2PValue.from_env(env, c1, "cpu", 42)
    torch.manual_seed(42)
    a2 = Matd3HierarchicalTransformerP2PValue.from_env(env, c2, "cpu", 42)

    add_replay_rows(a1, env, count=6, outcome_kwh=(10.0, 4.0, 6.0))
    for i in range(len(a1.replay)):
        e = a1.replay.storage[i]
        a2.replay.add(e.observations, e.global_state, e.actions, e.team_reward, e.next_observations, e.next_global_state, e.done)

    rng = torch.random.get_rng_state()
    a1.update(batch_size=4)
    torch.random.set_rng_state(rng)
    a2.update(batch_size=4)

    for left, right in zip(a1.actors.parameters(), a2.actors.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(a1.critics.parameters(), a2.critics.parameters()):
        assert torch.equal(left, right)


def test_critic_equivalence(env_and_config):
    env, config = env_and_config
    cp = copy.deepcopy(config)
    cc = copy.deepcopy(config)
    cc["algorithm"]["p2p_auxiliary"]["loss_weight_max"] = 0.0
    cp["algorithm"]["policy_delay"] = 2
    cc["algorithm"]["policy_delay"] = 2

    torch.manual_seed(42)
    parent = Matd3HierarchicalTransformerPredictiveDr.from_env(env, cp, "cpu", 42)
    torch.manual_seed(42)
    child = Matd3HierarchicalTransformerP2PValue.from_env(env, cc, "cpu", 42)

    for mg in env.agent_ids:
        child.actors[mg].load_state_dict(parent.actors[mg].state_dict(), strict=False)
        child.target_actors[mg].load_state_dict(parent.target_actors[mg].state_dict(), strict=False)
        child.critics[mg].load_state_dict(parent.critics[mg].state_dict())
        child.target_critics[mg].load_state_dict(parent.target_critics[mg].state_dict())

    add_replay_rows(parent, env, count=8, outcome_kwh=(10.0, 4.0, 6.0))
    for i in range(len(parent.replay)):
        e = parent.replay.storage[i]
        child.replay.add(e.observations, e.global_state, e.actions, e.team_reward, e.next_observations, e.next_global_state, e.done)

    rng = torch.random.get_rng_state()
    parent.update(batch_size=4)  # step 1: critic only
    torch.random.set_rng_state(rng)
    child.update(batch_size=4)   # step 1: critic only

    for mg in env.agent_ids:
        for p1, p2 in zip(parent.critics[mg].parameters(), child.critics[mg].parameters()):
            assert torch.equal(p1, p2)
        for p1, p2 in zip(parent.target_critics[mg].parameters(), child.target_critics[mg].parameters()):
            assert torch.equal(p1, p2)


def test_diagnostics_produced_at_cadence(env_and_config):
    env, config = env_and_config
    c = copy.deepcopy(config)
    c["algorithm"]["policy_delay"] = 1
    c["logging"] = {"diagnostic_interval": 1}

    algo = Matd3HierarchicalTransformerP2PValue.from_env(env, c, "cpu", 42)
    add_replay_rows(algo, env, count=6, outcome_kwh=(10.0, 4.0, 6.0))
    metrics = algo.update(batch_size=4)

    required_keys = [
        "p2p_auxiliary_weight",
        "p2p_aux_any_fill_loss",
        "p2p_aux_fill_ratio_loss",
        "p2p_aux_total_loss",
        "p2p_aux_offered_sample_rate",
        "p2p_aux_offered_sample_count",
        "p2p_aux_positive_fill_rate",
        "p2p_aux_fill_ratio_mae",
        "p2p_aux_predicted_fill_mean",
        "p2p_aux_actual_fill_mean",
        "p2p_aux_offer_consistency_error_mean",
        "p2p_aux_shared_gradient_norm",
        "p2p_aux_head_gradient_norm",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing {key} in metrics"
        assert math.isfinite(metrics[key])


def test_nonfinite_prediction_raises_error(env_and_config):
    env, config = env_and_config
    c = copy.deepcopy(config)
    c["algorithm"]["policy_delay"] = 1
    c["logging"] = {"diagnostic_interval": 1}

    algo = Matd3HierarchicalTransformerP2PValue.from_env(env, c, "cpu", 42)
    add_replay_rows(algo, env, count=4, outcome_kwh=(10.0, 4.0, 6.0))

    def nonfinite_prediction(representations, behavior_ess_action):
        batch = behavior_ess_action.shape[0]
        device = behavior_ess_action.device
        return P2PPrediction(
            torch.full((batch, 1), float("nan"), device=device),
            torch.zeros((batch, 1), device=device),
        )

    algo.actors["mg2"].p2p_value_head.forward = nonfinite_prediction
    before = clone_parameters(algo.actors["mg2"])
    with pytest.raises(FloatingPointError):
        algo.update(batch_size=4)
    assert not parameters_changed(before, algo.actors["mg2"])



