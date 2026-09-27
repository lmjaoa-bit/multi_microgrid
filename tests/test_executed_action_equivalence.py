from __future__ import annotations

import torch

from core.renewable_first_dispatch import clear_renewable_first


def _effective(discharge_kw: float) -> torch.Tensor:
    renewable = torch.zeros((1, 3), dtype=torch.float64)
    load = torch.zeros((1, 3), dtype=torch.float64)
    charge = torch.zeros_like(load)
    discharge = torch.tensor(
        [[discharge_kw, 0.0, 0.0]], dtype=torch.float64
    )
    ledger = clear_renewable_first(
        renewable, load, charge, discharge, p2p_enabled=True
    )
    return torch.stack([
        ledger["ess_local_load_kw"][:, 0] - ledger["actual_charge_kw"][:, 0],
        ledger["ess_peer_load_kw"][:, 0, :].sum(dim=1),
    ], dim=1)


def test_invalid_ess_requests_have_identical_executed_critic_channels():
    e10 = _effective(10.0)
    e20 = _effective(20.0)
    e40 = _effective(40.0)
    assert torch.equal(e10, e20)
    assert torch.equal(e20, e40)
    assert torch.equal(e10, torch.zeros_like(e10))


def test_no_demand_region_has_zero_local_gradient():
    discharge = torch.tensor(
        [[10.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    zeros = torch.zeros((1, 3), dtype=torch.float64)
    ledger = clear_renewable_first(
        zeros, zeros, zeros, discharge, p2p_enabled=True
    )
    executed = (
        ledger["ess_local_load_kw"].sum()
        + ledger["ess_peer_load_kw"].sum()
    )
    grad = torch.autograd.grad(
        executed, discharge, allow_unused=False
    )[0]
    assert torch.equal(grad, torch.zeros_like(grad))


def test_unsaturated_positive_demand_retains_gradient():
    discharge = torch.tensor(
        [[3.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    renewable = torch.zeros((1, 3), dtype=torch.float64)
    load = torch.tensor([[8.0, 0.0, 0.0]], dtype=torch.float64)
    charge = torch.zeros_like(load)
    ledger = clear_renewable_first(
        renewable, load, charge, discharge, p2p_enabled=True
    )
    executed = ledger["actual_discharge_kw"][0, 0]
    grad = torch.autograd.grad(executed, discharge)[0]
    assert grad[0, 0] > 0.0
