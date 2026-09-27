from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from core.renewable_first_dispatch import clear_renewable_first
from envs.multi_microgrid_env import MultiMicrogridEnv


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


def _rollout_invalid_request(raw_ess: float):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (root / "configs" / "cooperative.yaml").read_text(encoding="utf-8")
    )
    cfg["_project_root"] = str(root)
    cfg["p2p"]["enabled"] = False
    cfg["observation"]["noisy_forecast"] = False
    cfg["observation"]["evaluation_forecast_noise"] = False
    env = MultiMicrogridEnv(cfg, split="test")
    _, reset_info = env.reset(
        seed=1234, deterministic=True, forecast_noise=False
    )
    for core in env.cores.values():
        core.battery.soc = 0.50

    fixed = {
        mg: {
            "load_kw": 10.0,
            "pv_kw": 1000.0,
            "wind_kw": 0.0,
            "renewable_kw": 1000.0,
        }
        for mg in env.agent_ids
    }
    env._exogenous_at = lambda index: {
        mg: dict(values) for mg, values in fixed.items()
    }
    actions = {
        "mg1": np.asarray([raw_ess, 0.0], dtype=np.float32),
        "mg2": np.asarray([0.0, 0.0], dtype=np.float32),
        "mg3": np.asarray([0.0, 0.0], dtype=np.float32),
    }
    next_obs, rewards, done, info = env.step(actions)
    return {
        "day": reset_info["day"],
        "next_obs": next_obs,
        "rewards": rewards,
        "done": done,
        "global_state": info["global_state_after"],
        "soc": {
            mg: float(env.cores[mg].battery.soc) for mg in env.agent_ids
        },
        "raw_request_kw": float(
            info["plans"]["mg1"]["raw_ess_requested_power_kw"]
        ),
        "actual_discharge_kw": float(
            info["plans"]["mg1"]["battery_discharge_kw"]
        ),
        "team_profit": float(info["team_profit"]),
    }


def test_invalid_raw_requests_do_not_change_physical_next_state():
    runs = [_rollout_invalid_request(value) for value in (0.2, 0.5, 1.0)]
    assert len({round(run["raw_request_kw"], 6) for run in runs}) == 3
    for run in runs:
        assert run["actual_discharge_kw"] == 0.0
    baseline = runs[0]
    for current in runs[1:]:
        assert current["day"] == baseline["day"]
        assert current["done"] == baseline["done"]
        assert current["soc"] == baseline["soc"]
        assert current["team_profit"] == baseline["team_profit"]
        assert current["rewards"] == baseline["rewards"]
        np.testing.assert_allclose(
            current["global_state"], baseline["global_state"],
            atol=1e-7, rtol=0.0,
        )
        for mg in baseline["next_obs"]:
            np.testing.assert_allclose(
                current["next_obs"][mg], baseline["next_obs"][mg],
                atol=1e-7, rtol=0.0,
            )


def test_competing_sellers_keep_agent_specific_executed_channels():
    renewable = torch.zeros((1, 3), dtype=torch.float64)
    load = torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float64)
    charge = torch.zeros_like(load)

    equal = clear_renewable_first(
        renewable,
        load,
        charge,
        torch.tensor([[5.0, 5.0, 0.0]], dtype=torch.float64),
        p2p_enabled=True,
    )
    skewed = clear_renewable_first(
        renewable,
        load,
        charge,
        torch.tensor([[8.0, 2.0, 0.0]], dtype=torch.float64),
        p2p_enabled=True,
    )

    equal_sales = equal["ess_peer_load_kw"].sum(dim=2)
    skewed_sales = skewed["ess_peer_load_kw"].sum(dim=2)
    assert equal_sales.sum().item() == 10.0
    assert skewed_sales.sum().item() == 10.0
    assert equal_sales[0].tolist() == [5.0, 5.0, 0.0]
    assert skewed_sales[0].tolist() == [8.0, 2.0, 0.0]
    assert not torch.equal(equal_sales, skewed_sales)
    # Actual battery discharge follows those agent-specific executed flows,
    # so future SOC/value must remain distinguishable even though coalition
    # total traded energy is identical.
    assert torch.equal(equal["actual_discharge_kw"], equal_sales)
    assert torch.equal(skewed["actual_discharge_kw"], skewed_sales)
