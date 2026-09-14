"""Numerical safeguards for MATD3-ShapleyQ training."""
from __future__ import annotations

import math

import torch


def conservative_cooperation_q(q1, q2, agent_ids):
    """Reconstruct both complete cooperation-surplus Qs, then take one twin min."""
    total1 = sum(q1[mg] for mg in agent_ids)
    total2 = sum(q2[mg] for mg in agent_ids)
    use_first = total1 <= total2
    conservative = torch.minimum(total1, total2)
    selected = {mg: torch.where(use_first, q1[mg], q2[mg]) for mg in agent_ids}
    return total1, total2, conservative, selected


def lower_soc_discharge_gradient_proxy(effective_local, raw_action, discharge_max, eps=1e-9):
    """Keep the physical forward value while giving blocked discharge a straight-through gradient."""
    blocked_discharge = (discharge_max <= eps) & (raw_action > 0)
    zero_forward_identity_backward = raw_action - raw_action.detach()
    proxied = effective_local + torch.where(
        blocked_discharge,
        zero_forward_identity_backward,
        torch.zeros_like(zero_forward_identity_backward),
    )
    return proxied, blocked_discharge


def clip_module_grad_norm(module, max_norm):
    """Clip one module's gradients and return pre/post norms plus whether clipping occurred."""
    max_norm = float(max_norm)
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be positive and finite")
    try:
        pre = float(torch.nn.utils.clip_grad_norm_(
            module.parameters(), max_norm, error_if_nonfinite=True
        ))
    except RuntimeError as exc:
        raise FloatingPointError("Non-finite gradient norm") from exc
    post_sq = sum(
        float(parameter.grad.detach().square().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    post = math.sqrt(post_sq)
    if not math.isfinite(pre) or not math.isfinite(post):
        raise FloatingPointError("Non-finite gradient norm")
    return pre, post, bool(pre > max_norm)
