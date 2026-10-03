from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from algorithms.registry import available_algorithm_names, get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
SLUG = "matd3_hierarchical_transformer_q_readout"
SOURCE = "matd3_hierarchical_transformer_q_guided"


@pytest.fixture(scope="module")
def case():
    spec = get_algorithm_spec(SLUG)
    config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    config["algorithm"]["hidden_dims"] = [16]
    config["algorithm"]["network"].update(
        embed_dim=8, num_heads=2, num_layers=1, hidden_dim=16
    )
    config["algorithm"]["learning_starts"] = 0
    environment = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(spec.build_environment_config(environment), split="train")
    model = spec.create(env, config, device="cpu", seed=42)
    observations, _ = env.reset(seed=42, deterministic=True, forecast_noise=False)
    tensors = {mg: torch.tensor(value[None]) for mg, value in observations.items()}
    return env, model, config, tensors


def test_variant_is_registered_with_unchanged_experiment_conditions():
    assert SLUG in available_algorithm_names(), "The independent readout variant is missing"
    new = get_algorithm_spec(SLUG)
    old = get_algorithm_spec(SOURCE)
    environment = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    assert new.build_environment_config(environment) == old.build_environment_config(environment)
    old_config = yaml.safe_load(old.default_config_path.read_text(encoding="utf-8"))
    new_config = yaml.safe_load(new.default_config_path.read_text(encoding="utf-8"))
    old_config["algorithm"]["name"] = SLUG
    assert new_config == old_config


def test_forecasts_distinguish_base_load_at_equal_net_load(case):
    _, model, _, tensors = case
    actor = model.actors["mg2"]
    obs = tensors["mg2"].expand(2, -1).clone()
    norms = model.actor_feature_normalizers["mg2"]
    for row, (load, wind) in enumerate(((100.0, 50.0), (200.0, 150.0))):
        obs[row, actor.schema.index("load_t+1")] = load / norms["load"]
        obs[row, actor.schema.index("wind_t+1")] = wind / norms["wind"]
        obs[row, actor.schema.index("pv_t+1")] = 0.0
        obs[row, actor.schema.index("forecast_valid_t+1")] = 1.0
    features = actor._build_forecast_features(obs)
    assert features.shape == (2, 8, 9), "Base load must survive the net-load merge"
    torch.testing.assert_close(features[:, 0, 0], torch.full((2,), 50.0 / norms["power"]))
    torch.testing.assert_close(features[:, 0, -1], torch.tensor([100.0, 200.0]) / norms["load"])


def test_invalid_forecasts_cannot_affect_actions_including_all_masked_rows(case):
    _, model, _, tensors = case
    actor = model.actors["mg2"]
    assert hasattr(actor, "ess_forecast_readout"), "Device readouts are missing"
    obs = tensors["mg2"].expand(2, -1).clone()
    for k in range(1, 9):
        obs[0, actor.schema.index(f"forecast_valid_t+{k}")] = float(k <= 3)
        obs[1, actor.schema.index(f"forecast_valid_t+{k}")] = 0.0
    corrupt = obs.clone()
    for row, start in ((0, 4), (1, 1)):
        for k in range(start, 9):
            for name in ("load", "pv", "wind", "price", "temperature", "mg1_energy_balance", "mg3_energy_balance"):
                corrupt[row, actor.schema.index(f"{name}_t+{k}")] = float("nan")
    with torch.no_grad():
        expected = actor(obs)
        actual = actor(corrupt)
    assert torch.isfinite(actual).all()
    assert (actual.abs() <= 1).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("channel,branch", [(0, "ess_forecast_readout"), (1, "dr_forecast_readout")])
def test_actions_learn_through_their_device_readout_only(case, channel, branch):
    _, model, _, tensors = case
    actor = model.actors["mg2"]
    assert hasattr(actor, branch), "Device readouts are missing"
    actor.zero_grad(set_to_none=True)
    actor(tensors["mg2"])[:, channel].sum().backward()
    own = getattr(actor, branch)
    other = actor.dr_forecast_readout if channel == 0 else actor.ess_forecast_readout
    assert sum(p.grad.abs().sum().item() for p in own.parameters() if p.grad is not None) > 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in other.parameters())


def test_q_guidance_uses_the_same_ess_logit_and_new_readout(case):
    _, model, _, tensors = case
    actor = model.actors["mg2"]
    assert hasattr(actor, "action_logits"), "Actions and guidance need one shared logit path"
    logits = actor.action_logits(tensors["mg2"])
    guided = model._ess_pre_tanh_logit(actor, tensors["mg2"])
    torch.testing.assert_close(guided, logits[:, 0])
    torch.testing.assert_close(actor(tensors["mg2"]), logits.tanh())
    actor.zero_grad(set_to_none=True)
    guided.sum().backward()
    assert actor.ess_forecast_readout.in_proj_weight.grad.abs().sum() > 0


class ChargeValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, state, effective):
        value = -self.scale * effective[:, 3:4]
        return value, value

    def q1(self, state, effective):
        return self.forward(state, effective)[0]


def test_blocked_ess_guidance_updates_readout_without_direct_dr_gradient(case):
    env, model, _, tensors = case
    actor = model.actors["mg2"]
    assert hasattr(actor, "ess_forecast_readout"), "Device readouts are missing"
    state = torch.tensor(env.global_state()[None], dtype=torch.float32)
    for name, value in {"mg2_feasible_charge_max": 1.0, "mg2_feasible_discharge_max": 0.0,
                        "mg2_soc": 0.1, "mg2_load": 0.0, "mg2_pv": 5.0, "mg2_wind": 0.0}.items():
        state[:, model.projector.index[name]] = value
    old_critic = model.critics["mg2"]
    old_bias = actor.ess_head[-2].bias.detach().clone()
    try:
        model.critics["mg2"] = ChargeValue()
        with torch.no_grad():
            actor.ess_head[-2].bias.fill_(8.0)
        joint = torch.stack([model.actors[mg](tensors[mg]) for mg in model.agent_ids], dim=1)
        loss, metrics = model.actor_auxiliary_loss("mg2", state, joint, tensors["mg2"])
        actor.zero_grad(set_to_none=True)
        loss.backward()
        assert metrics["dispatch_q_guidance_trigger_rate"].item() == 1.0
        assert actor.ess_forecast_readout.in_proj_weight.grad.abs().sum() > 0
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in actor.dr_forecast_readout.parameters())
        dr_gradient = actor.dr_head[-2].weight.grad
        assert dr_gradient is None or torch.count_nonzero(dr_gradient) == 0
    finally:
        model.critics["mg2"] = old_critic
        with torch.no_grad():
            actor.ess_head[-2].bias.copy_(old_bias)


def test_checkpoint_roundtrip_and_old_identity_rejection(case, tmp_path):
    env, model, config, tensors = case
    assert model.SLUG == SLUG, "New checkpoints need an independent identity"
    path = tmp_path / "readout.pt"
    model.save(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["actor_forecast_context_version"] == "peer_balance_base_load_h8_device_readout_v1"
    restored = get_algorithm_spec(SLUG).create(env, config, device="cpu", seed=99)
    restored.load(path)
    for mg in model.agent_ids:
        torch.testing.assert_close(restored.actors[mg](tensors[mg]), model.actors[mg](tensors[mg]))
    before = {name: value.clone() for name, value in restored.actors.state_dict().items()}
    saved["algorithm_slug"] = SOURCE
    torch.save(saved, path)
    with pytest.raises(ValueError, match="different algorithm"):
        restored.load(path)
    for name, value in restored.actors.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_optimizer_and_target_include_new_readout_parameters(case):
    _, model, _, _ = case
    actor = model.actors["mg2"]
    assert hasattr(actor, "ess_forecast_readout"), "Device readouts are missing"
    optimized = {id(p) for group in model.actor_optimizers["mg2"].param_groups for p in group["params"]}
    assert optimized == {id(p) for p in actor.parameters()}
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, model.target_actors["mg2"].state_dict()[name])


def test_gradient_diagnostics_inspect_live_readout_parameters(case):
    _, model, _, tensors = case
    actor = model.actors["mg2"]
    actor.zero_grad(set_to_none=True)
    actor(tensors["mg2"]).sum().backward()
    total, branches, layers = model._actor_gradient_diagnostics("mg2")
    assert total > 0, "Diagnostics must inspect the replacement actor, not the discarded one"
    assert branches.get("ess_forecast_readout", 0) > 0
    assert branches.get("dr_forecast_readout", 0) > 0
    assert all(np.isfinite(value) for value in layers.values())


def test_sampled_update_trains_and_reports_the_new_ess_readout(case):
    env, _, config, tensors = case
    model = get_algorithm_spec(SLUG).create(env, config, device="cpu", seed=43)
    state = torch.tensor(env.global_state()[None], dtype=torch.float32)
    for name, value in {"mg2_feasible_charge_max": 1.0, "mg2_feasible_discharge_max": 0.0,
                        "mg2_soc": 0.1, "mg2_load": 0.0, "mg2_pv": 5.0, "mg2_wind": 0.0}.items():
        state[:, model.projector.index[name]] = value
    critic = ChargeValue()
    model.critics["mg2"] = critic
    model.critic_optimizers["mg2"] = torch.optim.SGD(critic.parameters(), lr=0.0)
    actor = model.actors["mg2"]
    with torch.no_grad():
        actor.ess_head[-2].bias.fill_(8.0)
    obs = {mg: value[0].numpy() for mg, value in tensors.items()}
    actions = model.select_actions(obs, noise_std=0.0)
    for _ in range(4):
        model.replay.add(obs, state[0].numpy(), actions, -1.0, obs, state[0].numpy(), False)
    model.update(batch_size=2)
    model.diagnostic_schedule.next_boundary = 2
    before = actor.ess_forecast_readout.in_proj_weight.detach().clone()
    metrics = model.update(batch_size=2)
    diagnostic = metrics["actor_diagnostics"]["mg2"]
    assert diagnostic["dispatch_q_guidance_trigger_rate"] == 1.0
    assert diagnostic["ess_forecast_readout_grad_norm"] > 0
    assert np.isfinite(diagnostic["dr_forecast_readout_grad_norm"])
    assert diagnostic["total_grad_norm"] > 0
    assert not torch.equal(before, actor.ess_forecast_readout.in_proj_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_readout_handles_valid_and_all_invalid_forecasts_with_finite_gradients(case):
    _, model, _, tensors = case
    actor = deepcopy(model.actors["mg2"]).cuda()
    obs = tensors["mg2"].expand(2, -1).clone().cuda()
    for k in range(1, 9):
        obs[1, actor.schema.index(f"forecast_valid_t+{k}")] = 0.0
    actions = actor(obs)
    assert torch.isfinite(actions).all()
    actions.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
    assert actor.ess_forecast_readout.in_proj_weight.grad.abs().sum() > 0
