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

