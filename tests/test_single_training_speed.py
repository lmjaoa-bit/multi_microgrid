from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import core.renewable_first_dispatch as dispatch
from components.commercial_hvac import CommercialHVACDR


def test_runtime_benchmark_prefills_learning_threshold_and_captures_context():
    from scripts.benchmark_training_runtime import _prefill_replay
    class Replay:
        def __init__(self):
            self.count = 0
            self.pending = False
        def __len__(self):
            return self.count
        def add(self, *args):
            assert self.pending, "Counterfactual context must precede every transition"
            self.pending = False
            self.count += 1
    class Model:
        replay = Replay()
        def capture_training_context(self, env, obs, state, metadata):
            assert metadata["total_steps"] == len(self.replay)
            self.replay.pending = True
    class Env:
        agent_ids = ["mg"]
        action_dim = 1
        def reset(self, seed):
            return {"mg": np.zeros(1)}, {"global_state": np.zeros(1), "day": "2018-01-01"}
        def step(self, actions):
            return {"mg": np.zeros(1)}, {"mg": 0.0}, True, {"global_state_after": np.zeros(1)}
    model = Model()
    _prefill_replay(Env(), model, batch_size=8, seed=42, learning_starts=17)
    assert len(model.replay) == 17


def test_hvac_scalar_reads_do_not_allocate_mixed_dataframe_rows(monkeypatch):
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    config = deepcopy(cfg["microgrids"][1]["dr"])
    device = CommercialHVACDR(config, delta_t=1.0, episode_length=24)
    dates = pd.date_range("2018-01-02", periods=24, freq="h")
    frame = pd.DataFrame({"timestamp": dates, config["temperature_column"]: np.arange(24.) + 5,
                          config["ghi_column"]: np.arange(24.) * 30})
    base = np.arange(24.) + 40
    device.reset(frame, base)
    def reject_row_access(frame):
        raise AssertionError("Hot HVAC scalar reads must not allocate a complete mixed-type row")
    monkeypatch.setattr(pd.DataFrame, "iloc", property(reject_row_access))
    for hour in [0, 7, 18, 23, -1]:
        assert device._timestamp(hour) == dates[hour]
        outside, solar, internal = device._exogenous(hour)
        clock_hour = dates[hour].hour
        occupied = clock_hour in device.occupied_hours
        fraction = device.internal_occ_fraction if occupied else device.internal_unocc_fraction
        assert outside == (5 + hour if hour >= 0 else 28)
        assert solar == pytest.approx(device.solar_gain * (hour if hour >= 0 else 23) * 30)
        assert internal == pytest.approx(base[hour] * fraction)
    # Replacing the day must refresh all values; no cached data from the old day.
    next_day = frame.copy()
    next_day["timestamp"] = pd.date_range("2018-01-06", periods=24, freq="h")
    next_day[config["temperature_column"]] = -10.0
    device.reset(next_day, base)
    assert device._timestamp(7).dayofweek == 5
    assert device._exogenous(7) == pytest.approx((-10.0, device.solar_gain * 210, base[7] * device.internal_unocc_fraction))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_internal_cross_flow_matches_checked_values_and_gradients(dtype, device):
    generator = torch.Generator().manual_seed(1234)
    signed = (torch.rand((128, 3), generator=generator, dtype=dtype) - .5) * 100
    # Include exactly inactive nodes: important for the projection gradient at zero.
    signed[0] = 0
    signed = signed.to(device)
    inputs = [signed.clamp_min(0).requires_grad_(), (-signed).clamp_min(0).requires_grad_()]
    checked_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    actual = dispatch._proportional_cross_flow_validated(*inputs)
    expected = dispatch.proportional_cross_flow(*checked_inputs)
    assert torch.equal(actual, expected)
    weights = torch.rand(actual.shape, generator=generator, dtype=dtype).to(device)
    actual_grad = torch.autograd.grad((actual * weights).sum(), inputs)
    expected_grad = torch.autograd.grad((expected * weights).sum(), checked_inputs)
    for a, b in zip(actual_grad, expected_grad):
        assert torch.equal(a, b)


def test_internal_dispatch_avoids_rechecking_generated_flows(monkeypatch):
    def reject_generated_revalidation(*args, **kwargs):
        raise AssertionError("Generated nonnegative disjoint flows already follow from checked dispatch inputs")
    monkeypatch.setattr(dispatch, "proportional_cross_flow", reject_generated_revalidation)
    result = dispatch.clear_renewable_first(
        torch.tensor([[50., 0., 20.]]), torch.tensor([[0., 30., 10.]]),
        torch.tensor([[5., 0., 3.]]), torch.tensor([[0., 20., 0.]]), p2p_enabled=True)
    assert torch.isfinite(result["grid_load_kw"]).all()
    assert result["res_peer_load_kw"].sum() == pytest.approx(30.)


@pytest.mark.parametrize("bad", [float("nan"), -1.0])
def test_fast_dispatch_keeps_external_input_validation(bad):
    x = torch.tensor([[bad, 0., 0.]])
    with pytest.raises(ValueError):
        dispatch.clear_renewable_first(x, torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), p2p_enabled=True)
    with pytest.raises(ValueError):
        dispatch.proportional_cross_flow(x, torch.zeros_like(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p2p", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_full_ledger_and_gradients_match_revalidated_dispatch(monkeypatch, dtype, p2p, device):
    generator = torch.Generator().manual_seed(9876)
    renewable = torch.rand((128, 3), generator=generator, dtype=dtype) * 400
    load = torch.rand((128, 3), generator=generator, dtype=dtype) * 100
    ess = (torch.rand((128, 3), generator=generator, dtype=dtype) - .5) * 60
    renewable[0] = load[0] = ess[0] = 0
    values = [x.to(device) for x in [renewable, load, (-ess).clamp_min(0), ess.clamp_min(0)]]
    inputs = [x.clone().requires_grad_() for x in values]
    actual = dispatch.clear_renewable_first(*inputs, p2p_enabled=p2p)
    kernel = dispatch._proportional_cross_flow_kernel
    def checked_original_intermediate(supply, demand):
        supply = dispatch._nonnegative("supply_kw", supply)
        demand = dispatch._nonnegative("demand_kw", demand)
        assert supply.shape == demand.shape
        assert not torch.any((supply > 1e-7) & (demand > 1e-7))
        return kernel(supply, demand)
    monkeypatch.setattr(dispatch, "_proportional_cross_flow_validated", checked_original_intermediate)
    reference_inputs = [x.clone().requires_grad_() for x in values]
    expected = dispatch.clear_renewable_first(*reference_inputs, p2p_enabled=p2p)
    weights = {key: torch.rand(value.shape, generator=generator, dtype=dtype).to(device) for key, value in actual.items()}
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key
    gradients = torch.autograd.grad(sum((value * weights[key]).sum() for key, value in actual.items()), inputs)
    reference_gradients = torch.autograd.grad(sum((value * weights[key]).sum() for key, value in expected.items()), reference_inputs)
    for actual_grad, reference_grad in zip(gradients, reference_gradients):
        assert torch.equal(actual_grad, reference_grad)
