from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class P2PLabels:
    offer_kwh: torch.Tensor
    accepted_kwh: torch.Tensor
    unmatched_kwh: torch.Tensor
    offered_mask: torch.Tensor
    any_fill_target: torch.Tensor
    fill_ratio_target: torch.Tensor
    consistency_residual_kwh: torch.Tensor


@dataclass(frozen=True)
class P2PPrediction:
    any_fill_logit: torch.Tensor
    conditional_fill_ratio: torch.Tensor


@dataclass(frozen=True)
class P2PLosses:
    any_fill: torch.Tensor
    fill_ratio: torch.Tensor
    total: torch.Tensor
    offered_count: torch.Tensor


@dataclass(frozen=True)
class P2PRepresentations:
    ess: torch.Tensor
    supply: torch.Tensor
    trade: torch.Tensor
    peer: torch.Tensor


def validate_p2p_auxiliary_config(
    config: Mapping[str, Any],
    *,
    require_enabled: bool = True,
) -> dict[str, Any]:
    """Validate P2P auxiliary configuration."""
    if not isinstance(config, Mapping):
        raise TypeError(f"p2p_auxiliary config must be a mapping, got {type(config)}")

    validated: dict[str, Any] = dict(config)

    enabled = validated.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"p2p_auxiliary.enabled must be a bool, got {enabled}")
    if require_enabled and not enabled:
        raise ValueError("p2p_auxiliary.enabled must be True for this experiment")

    hidden_dim = validated.get("hidden_dim", 64)
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, numbers.Integral) or hidden_dim <= 0:
        raise ValueError(f"p2p_auxiliary.hidden_dim must be a positive integer, got {hidden_dim}")

    loss_weight_max = validated.get("loss_weight_max", 0.05)
    if (
        isinstance(loss_weight_max, bool)
        or not isinstance(loss_weight_max, numbers.Real)
        or not math.isfinite(loss_weight_max)
        or loss_weight_max < 0.0
    ):
        raise ValueError(
            f"p2p_auxiliary.loss_weight_max must be a non-negative finite real, got {loss_weight_max}"
        )

    ramp_updates = validated.get("ramp_updates", 5000)
    if (
        isinstance(ramp_updates, bool)
        or not isinstance(ramp_updates, numbers.Integral)
        or ramp_updates <= 0
    ):
        raise ValueError(
            f"p2p_auxiliary.ramp_updates must be a positive integer, got {ramp_updates}"
        )

    for weight_key in ("any_fill_loss_weight", "fill_ratio_loss_weight"):
        val = validated.get(weight_key, 1.0)
        if (
            isinstance(val, bool)
            or not isinstance(val, numbers.Real)
            or not math.isfinite(val)
            or val < 0.0
        ):
            raise ValueError(
                f"p2p_auxiliary.{weight_key} must be a non-negative finite real, got {val}"
            )

    offer_eps = validated.get("offer_epsilon_kwh", 1.0e-6)
    if (
        isinstance(offer_eps, bool)
        or not isinstance(offer_eps, numbers.Real)
        or not math.isfinite(offer_eps)
        or offer_eps <= 0.0
    ):
        raise ValueError(
            f"p2p_auxiliary.offer_epsilon_kwh must be a positive finite real, got {offer_eps}"
        )

    cons_tol = validated.get("consistency_tolerance_kwh", 1.0e-5)
    if (
        isinstance(cons_tol, bool)
        or not isinstance(cons_tol, numbers.Real)
        or not math.isfinite(cons_tol)
        or cons_tol <= 0.0
    ):
        raise ValueError(
            f"p2p_auxiliary.consistency_tolerance_kwh must be a positive finite real, got {cons_tol}"
        )

    detach_behavior = validated.get("detach_behavior_action", True)
    if not isinstance(detach_behavior, bool) or not detach_behavior:
        raise ValueError(
            f"p2p_auxiliary.detach_behavior_action must be True, got {detach_behavior}"
        )

    return validated


def extract_p2p_labels(
    next_observation: torch.Tensor,
    schema: list[str],
    *,
    trade_scale: float,
    offer_epsilon_kwh: float = 1.0e-6,
    consistency_tolerance_kwh: float = 1.0e-5,
) -> P2PLabels:
    """Extract physical P2P supervision targets from next observations."""
    if not (
        isinstance(trade_scale, numbers.Real)
        and math.isfinite(trade_scale)
        and trade_scale > 0.0
    ):
        raise ValueError(f"trade_scale must be positive and finite, got {trade_scale}")

    required_fields = [
        "previous_ess_p2p_offer",
        "previous_ess_p2p_accepted",
        "previous_ess_p2p_unmatched",
    ]
    missing = [f for f in required_fields if f not in schema]
    if missing:
        raise ValueError(f"Missing required P2P fields in schema: {missing}")

    offer_idx = schema.index("previous_ess_p2p_offer")
    accepted_idx = schema.index("previous_ess_p2p_accepted")
    unmatched_idx = schema.index("previous_ess_p2p_unmatched")

    raw_offer = next_observation[:, offer_idx : offer_idx + 1]
    raw_accepted = next_observation[:, accepted_idx : accepted_idx + 1]
    raw_unmatched = next_observation[:, unmatched_idx : unmatched_idx + 1]

    if not torch.isfinite(raw_offer).all():
        raise ValueError("Non-finite values found in previous_ess_p2p_offer")
    if not torch.isfinite(raw_accepted).all():
        raise ValueError("Non-finite values found in previous_ess_p2p_accepted")
    if not torch.isfinite(raw_unmatched).all():
        raise ValueError("Non-finite values found in previous_ess_p2p_unmatched")

    scale = float(trade_scale)
    offer_kwh = raw_offer * scale
    accepted_kwh = raw_accepted * scale
    unmatched_kwh = raw_unmatched * scale

    tol = float(consistency_tolerance_kwh)
    eps = float(offer_epsilon_kwh)

    if (offer_kwh < -tol).any():
        raise ValueError("Materially negative offer found in previous_ess_p2p_offer")
    if (accepted_kwh < -tol).any():
        raise ValueError("Materially negative accepted energy found in previous_ess_p2p_accepted")
    if (unmatched_kwh < -tol).any():
        raise ValueError("Materially negative unmatched energy found in previous_ess_p2p_unmatched")

    offer_kwh = offer_kwh.clamp_min(0.0)
    accepted_kwh = accepted_kwh.clamp_min(0.0)
    unmatched_kwh = unmatched_kwh.clamp_min(0.0)

    if (accepted_kwh > offer_kwh + tol).any():
        raise ValueError(
            "Materially invalid P2P outcome: accepted energy exceeds offer beyond tolerance"
        )

    accepted_kwh = torch.min(accepted_kwh, offer_kwh)

    residual = offer_kwh - accepted_kwh - unmatched_kwh
    if (residual.abs() > tol + 1e-7).any():
        raise ValueError(
            f"P2P consistency error exceeds tolerance: max residual={residual.abs().max().item()} > {tol}"
        )

    offered = offer_kwh > eps
    any_fill = (accepted_kwh > eps).to(next_observation.dtype)
    ratio = torch.where(
        offered,
        accepted_kwh / offer_kwh.clamp_min(eps),
        torch.zeros_like(offer_kwh),
    ).clamp(0.0, 1.0)

    return P2PLabels(
        offer_kwh=offer_kwh,
        accepted_kwh=accepted_kwh,
        unmatched_kwh=unmatched_kwh,
        offered_mask=offered,
        any_fill_target=any_fill,
        fill_ratio_target=ratio,
        consistency_residual_kwh=residual.abs(),
    )


class P2PValueHead(nn.Module):
    """Dual-head action-conditioned P2P transaction value predictor."""

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(4 * embed_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.any_fill_head = nn.Linear(hidden_dim, 1)
        self.fill_ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        representations: P2PRepresentations,
        behavior_ess_action: torch.Tensor,
    ) -> P2PPrediction:
        behavior = behavior_ess_action.detach()
        features = torch.cat(
            [
                representations.ess,
                representations.supply,
                representations.trade,
                representations.peer,
                behavior,
            ],
            dim=-1,
        )
        hidden = self.trunk(features)
        return P2PPrediction(
            any_fill_logit=self.any_fill_head(hidden),
            conditional_fill_ratio=self.fill_ratio_head(hidden),
        )


def compute_p2p_auxiliary_loss(
    prediction: P2PPrediction,
    labels: P2PLabels,
    config: Mapping[str, Any],
) -> P2PLosses:
    """Compute masked supervised P2P auxiliary loss."""
    if not (
        torch.isfinite(prediction.any_fill_logit).all()
        and torch.isfinite(prediction.conditional_fill_ratio).all()
    ):
        raise FloatingPointError("Non-finite predictions encountered in P2P auxiliary head")

    mask = labels.offered_mask
    offered_count = mask.sum().float()

    if offered_count.item() == 0.0:
        zero = (
            prediction.any_fill_logit.sum() + prediction.conditional_fill_ratio.sum()
        ) * 0.0
        return P2PLosses(
            any_fill=zero,
            fill_ratio=zero,
            total=zero,
            offered_count=offered_count,
        )

    logits_offered = prediction.any_fill_logit[mask]
    target_any_offered = labels.any_fill_target[mask]
    l_any = F.binary_cross_entropy_with_logits(logits_offered, target_any_offered)

    ratios_offered = prediction.conditional_fill_ratio[mask]
    target_ratios_offered = labels.fill_ratio_target[mask]
    l_fill = F.smooth_l1_loss(ratios_offered, target_ratios_offered)

    w_any = float(config.get("any_fill_loss_weight", 1.0))
    w_fill = float(config.get("fill_ratio_loss_weight", 1.0))
    total = w_any * l_any + w_fill * l_fill

    return P2PLosses(
        any_fill=l_any,
        fill_ratio=l_fill,
        total=total,
        offered_count=offered_count,
    )


def p2p_auxiliary_weight(update_step: int, config: Mapping[str, Any]) -> float:
    """Compute linearly ramped auxiliary loss weight."""
    loss_weight_max = float(config.get("loss_weight_max", 0.05))
    ramp_updates = float(config.get("ramp_updates", 5000))
    if ramp_updates <= 0.0:
        return loss_weight_max
    return loss_weight_max * min(float(update_step) / ramp_updates, 1.0)
