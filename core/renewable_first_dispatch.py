from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def _nonnegative(name: str, value: Tensor) -> Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, agents]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if torch.any(value < -1e-7):
        raise ValueError(f"{name} must be non-negative")
    return value.clamp_min(0.0)


def proportional_cross_flow(supply_kw: Tensor, demand_kw: Tensor) -> Tensor:
    """Lossless proportional all-to-all clearing without self-trade."""
    supply = _nonnegative("supply_kw", supply_kw)
    demand = _nonnegative("demand_kw", demand_kw)
    if supply.shape != demand.shape:
        raise ValueError("supply_kw and demand_kw must have the same shape")
    if torch.any((supply > 1e-7) & (demand > 1e-7)):
        raise ValueError("a node cannot be simultaneous supplier and demander")
    total_supply = supply.sum(dim=1, keepdim=True)
    total_demand = demand.sum(dim=1, keepdim=True)
    denominator = torch.maximum(total_supply, total_demand).clamp_min(
        torch.finfo(supply.dtype).tiny
    )
    flow = supply.unsqueeze(2) * demand.unsqueeze(1) / denominator.unsqueeze(2)
    eye = torch.eye(
        supply.shape[1], dtype=torch.bool, device=supply.device
    ).unsqueeze(0)
    return flow.masked_fill(eye, 0.0)


def clear_renewable_first(
    renewable_kw: Tensor,
    load_kw: Tensor,
    charge_request_kw: Tensor,
    discharge_request_kw: Tensor,
    *,
    p2p_enabled: bool,
) -> Dict[str, Tensor]:
    """Pure differentiable renewable-first physical dispatch.

    Priority: local RES->load; peer RES->load; ESS->local residual load;
    capped ESS->peer residual load; remaining RES->local/peer ESS charging;
    grid->remaining load/charging; final RES surplus->export/curtailment.
    """
    renewable = _nonnegative("renewable_kw", renewable_kw)
    load = _nonnegative("load_kw", load_kw)
    charge_request = _nonnegative("charge_request_kw", charge_request_kw)
    discharge_request = _nonnegative("discharge_request_kw", discharge_request_kw)
    shape = renewable.shape
    if any(x.shape != shape for x in (load, charge_request, discharge_request)):
        raise ValueError("all dispatch inputs must have shape [batch, agents]")
    if torch.any((charge_request > 1e-7) & (discharge_request > 1e-7)):
        raise ValueError("ESS cannot request charge and discharge simultaneously")
    zero_flow = renewable.new_zeros((shape[0], shape[1], shape[1]))

    res_local_load = torch.minimum(renewable, load)
    res_after_local_load = (renewable - res_local_load).clamp_min(0.0)
    load_after_local_res = (load - res_local_load).clamp_min(0.0)

    res_peer_load = (
        proportional_cross_flow(res_after_local_load, load_after_local_res)
        if p2p_enabled else zero_flow
    )
    res_after_load = (
        res_after_local_load - res_peer_load.sum(dim=2)
    ).clamp_min(0.0)
    load_after_renewable = (
        load_after_local_res - res_peer_load.sum(dim=1)
    ).clamp_min(0.0)

    ess_local_load = torch.minimum(discharge_request, load_after_renewable)
    load_after_local_ess = (
        load_after_renewable - ess_local_load
    ).clamp_min(0.0)
    ess_offer_uncapped = (
        discharge_request - ess_local_load
    ).clamp_min(0.0)

    total_residual_load = load_after_local_ess.sum(dim=1, keepdim=True)
    peer_load_cap = (
        total_residual_load - load_after_local_ess
    ).clamp_min(0.0)
    ess_offer_cap = (
        peer_load_cap if p2p_enabled else torch.zeros_like(peer_load_cap)
    )
    ess_offer = torch.minimum(ess_offer_uncapped, ess_offer_cap)
    ess_offer_projection_reduction = (
        ess_offer_uncapped - ess_offer
    ).clamp_min(0.0)
    ess_peer_load = (
        proportional_cross_flow(ess_offer, load_after_local_ess)
        if p2p_enabled else zero_flow
    )
    ess_peer_out = ess_peer_load.sum(dim=2)
    ess_unmatched = (ess_offer - ess_peer_out).clamp_min(0.0)
    load_after_ess = (
        load_after_local_ess - ess_peer_load.sum(dim=1)
    ).clamp_min(0.0)

    res_local_charge = torch.minimum(res_after_load, charge_request)
    res_after_local_charge = (
        res_after_load - res_local_charge
    ).clamp_min(0.0)
    charge_after_local_res = (
        charge_request - res_local_charge
    ).clamp_min(0.0)
    res_peer_charge = (
        proportional_cross_flow(res_after_local_charge, charge_after_local_res)
        if p2p_enabled else zero_flow
    )
    renewable_surplus = (
        res_after_local_charge - res_peer_charge.sum(dim=2)
    ).clamp_min(0.0)
    charge_after_renewable = (
        charge_after_local_res - res_peer_charge.sum(dim=1)
    ).clamp_min(0.0)

    grid_load = load_after_ess
    grid_charge = charge_after_renewable
    actual_charge = (
        res_local_charge + res_peer_charge.sum(dim=1) + grid_charge
    )
    actual_discharge = ess_local_load + ess_peer_out
    return {
        "res_local_load_kw": res_local_load,
        "res_peer_load_kw": res_peer_load,
        "load_after_renewable_kw": load_after_renewable,
        "ess_local_load_kw": ess_local_load,
        "ess_offer_uncapped_kw": ess_offer_uncapped,
        "ess_offer_cap_kw": ess_offer_cap,
        "ess_offer_kw": ess_offer,
        "ess_offer_projection_reduction_kw": ess_offer_projection_reduction,
        "ess_peer_load_kw": ess_peer_load,
        "ess_unmatched_kw": ess_unmatched,
        "res_local_charge_kw": res_local_charge,
        "res_peer_charge_kw": res_peer_charge,
        "grid_load_kw": grid_load,
        "grid_charge_kw": grid_charge,
        "renewable_surplus_for_export_kw": renewable_surplus,
        "actual_charge_kw": actual_charge,
        "actual_discharge_kw": actual_discharge,
    }
