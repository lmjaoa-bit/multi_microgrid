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
