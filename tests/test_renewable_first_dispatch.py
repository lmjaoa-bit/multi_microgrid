from __future__ import annotations

import pytest
import torch

from core.renewable_first_dispatch import clear_renewable_first


def _clear(res, load, charge=None, discharge=None, *, p2p=True):
    n = len(res)
    charge = [0.0] * n if charge is None else charge
    discharge = [0.0] * n if discharge is None else discharge
    vals = [
        torch.tensor([x], dtype=torch.float64)
        for x in (res, load, charge, discharge)
    ]
    return clear_renewable_first(*vals, p2p_enabled=p2p)


def test_peer_renewable_precedes_local_ess():
    x = _clear([0, 20, 0], [30, 0, 0], discharge=[25, 0, 0])
    assert x["res_peer_load_kw"][0, 1, 0] == pytest.approx(20)
    assert x["actual_discharge_kw"][0, 0] == pytest.approx(10)
    assert x["ess_offer_projection_reduction_kw"][0, 0] == pytest.approx(15)


def test_multiple_ess_sellers_scaled_without_iteration_priority():
    x = _clear([0, 0, 0], [0, 0, 10], discharge=[20, 20, 0])
    assert x["ess_offer_cap_kw"][0].tolist() == pytest.approx([10, 10, 0])
    assert x["ess_peer_load_kw"][0, 0, 2] == pytest.approx(5)
    assert x["ess_peer_load_kw"][0, 1, 2] == pytest.approx(5)
    assert x["ess_unmatched_kw"][0].tolist() == pytest.approx([5, 5, 0])


def test_renewable_eliminates_peer_demand_before_ess():
    x = _clear([0, 0, 20], [0, 20, 0], discharge=[20, 0, 0])
    assert x["res_peer_load_kw"][0, 2, 1] == pytest.approx(20)
    assert x["actual_discharge_kw"][0, 0] == pytest.approx(0)


def test_load_priority_can_grid_charge_after_peer_res_supply():
    x = _clear([30, 0], [0, 30], charge=[30, 0])
    assert x["res_peer_load_kw"][0, 0, 1] == pytest.approx(30)
    assert x["grid_charge_kw"][0, 0] == pytest.approx(30)


def test_peer_res_charge_only_after_load_stage():
    x = _clear([40, 0], [0, 20], charge=[10, 15])
    assert x["res_peer_load_kw"][0, 0, 1] == pytest.approx(20)
    assert x["res_local_charge_kw"][0, 0] == pytest.approx(10)
    assert x["res_peer_charge_kw"][0, 0, 1] == pytest.approx(10)
    assert x["grid_charge_kw"][0, 1] == pytest.approx(5)


def test_positive_ess_request_without_load_does_not_discharge():
    x = _clear([0, 0], [0, 0], discharge=[20, 0])
    assert x["actual_discharge_kw"].sum() == pytest.approx(0)
    assert x["ess_offer_projection_reduction_kw"][0, 0] == pytest.approx(20)


def test_disabled_p2p_keeps_resources_local():
    x = _clear([20, 0], [0, 20], p2p=False)
    assert x["res_peer_load_kw"].sum() == pytest.approx(0)
    assert x["grid_load_kw"][0, 1] == pytest.approx(20)
    assert x["renewable_surplus_for_export_kw"][0, 0] == pytest.approx(20)


def test_kernel_retains_actor_gradient():
    res = torch.zeros((1, 3), dtype=torch.float64)
    load = torch.tensor([[8.0, 0.0, 10.0]], dtype=torch.float64)
    charge = torch.zeros_like(load)
    discharge = torch.tensor(
        [[4.0, 2.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    x = clear_renewable_first(
        res, load, charge, discharge, p2p_enabled=True
    )
    (x["ess_local_load_kw"].sum() + x["ess_peer_load_kw"].sum()).backward()
    assert discharge.grad is not None
    assert torch.isfinite(discharge.grad).all()
