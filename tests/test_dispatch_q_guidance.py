from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from algorithms.registry import get_algorithm_spec
from algorithms.matd3.actor import Actor
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


class ChargeValue(nn.Module):
    """Known test objective: more effective charging has greater value."""

    def __init__(self, disagree=False):
        super().__init__()
        self.disagree = disagree

    def q1(self, state, effective):
        return -effective[:, :1]

    def forward(self, state, effective):
        q = self.q1(state, effective)
        return q, -q if self.disagree else q


def setup_case():
    spec = get_algorithm_spec("matd3_dispatch_q_guided")
    env = MultiMicrogridEnv(ROOT / "configs/cooperative.yaml")
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    settings["algorithm"]["learning_starts"] = 0
    settings["logging"]["diagnostic_interval"] = 2
    model = spec.create(env, settings, device="cpu", seed=42)
    obs, info = env.reset(seed=42, deterministic=True)
    with torch.no_grad():
        model.actors["mg1"].network[-2].weight.zero_()
        model.actors["mg1"].network[-2].bias[0] = 8.0
    observations = {mg: torch.tensor(x[None]) for mg, x in obs.items()}
    joint = torch.stack([model.actors[mg](observations[mg]) for mg in env.agent_ids], dim=1)
    return env, model, torch.tensor(info["global_state"][None]), observations, joint


def test_guidance_provides_gradient_when_projected_policy_gradient_is_zero():
    _, model, state, observations, joint = setup_case()
    model.critics["mg1"] = ChargeValue()
    effective, _ = model.project_actions(state, joint)
    original_loss = -model.critics["mg1"].q1(state, effective.flatten(1)).mean()
    bias = model.actors["mg1"].network[-2].bias
    original_gradient = torch.autograd.grad(original_loss, bias, retain_graph=True)[0]
    assert original_gradient[0] == 0
    loss, diagnostics = model.actor_auxiliary_loss("mg1", state, joint, observations["mg1"])
    gradient = torch.autograd.grad(loss, bias)[0]
    assert gradient[0] > 0.01  # Gradient descent moves the saturated logit toward charging.
    assert gradient[1] == 0  # The auxiliary objective has no direct DR-head gradient.
    assert diagnostics["dispatch_q_guidance_trigger_rate"] == 1.0


@pytest.mark.parametrize("reason", ["critics_disagree", "discharge_feasible", "disabled"])
def test_guidance_does_not_force_charging_without_its_conditions(reason):
    _, model, state, observations, joint = setup_case()
    model.critics["mg1"] = ChargeValue(disagree=reason == "critics_disagree")
    if reason == "discharge_feasible":
        state[:, model.projector.index["mg1_feasible_discharge_max"]] = 1.0
    if reason == "disabled":
        model.dispatch_q_guidance_weight = 0.0
    loss, diagnostics = model.actor_auxiliary_loss("mg1", state, joint, observations["mg1"])
    assert loss.item() == 0.0
    assert diagnostics["dispatch_q_guidance_trigger_rate"] == 0.0


def test_guidance_can_recover_blocked_charging_at_full_battery():
    _, model, state, observations, _ = setup_case()
    with torch.no_grad():
        model.actors["mg1"].network[-2].bias[0] = -8.0
    joint = torch.stack([model.actors[mg](observations[mg]) for mg in model.agent_ids], dim=1)
    state[:, model.projector.index["mg1_feasible_charge_max"]] = 0.0
    state[:, model.projector.index["mg1_feasible_discharge_max"]] = 1.0
    state[:, model.projector.index["mg1_soc"]] = 0.9
    for mg in model.agent_ids:
        state[:, model.projector.index[f"{mg}_pv"]] = 0.0
        state[:, model.projector.index[f"{mg}_wind"]] = 0.0

    class DischargeValue(ChargeValue):
        def q1(self, state, effective):
            return effective[:, :2].sum(dim=1, keepdim=True)

    model.critics["mg1"] = DischargeValue()
    loss, diagnostics = model.actor_auxiliary_loss("mg1", state, joint, observations["mg1"])
    grad = torch.autograd.grad(loss, model.actors["mg1"].network[-2].bias)[0]
    assert grad[0] < -0.01
    assert grad[1] == 0
    assert diagnostics["dispatch_q_guidance_trigger_rate"] == 1.0


def test_mixed_batch_only_updates_approved_rows_and_uses_full_batch_scale():
    _, model, state, _, _ = setup_case()
    actor = Actor(3, 2, hidden_dims=())
    with torch.no_grad():
        actor.network[0].weight.zero_()
        actor.network[0].weight[0] = torch.tensor([8.0, 6.0, 4.0])
        actor.network[0].bias.zero_()
    model.actors["mg1"] = actor
    observations = torch.eye(3)
    joint = torch.stack([actor(observations), torch.zeros(3, 2), torch.zeros(3, 2)], dim=1)
    state = state.repeat(3, 1)
    time_column = model.projector.index["mg1_time_to_terminal"]
    state[:, time_column] = torch.tensor([0.2, 0.8, 0.2])
    state[2, model.projector.index["mg1_feasible_discharge_max"]] = 1.0

    class MixedValue(ChargeValue):
        def forward(self, state, effective):
            q = self.q1(state, effective)
            return q, torch.where(state[:, time_column:time_column + 1] > 0.5, -q, q)

    model.critics["mg1"] = MixedValue()
    loss, diagnostics = model.actor_auxiliary_loss("mg1", state, joint, observations)
    gradient = torch.autograd.grad(loss, actor.network[0].weight)[0]
    expected_error = 8.0 + math.atanh(0.9)
    assert loss.item() == pytest.approx(0.01 * expected_error ** 2 / 3)
    assert gradient[0, 0].item() == pytest.approx(0.02 * expected_error / 3)
    assert torch.count_nonzero(gradient[:, 1:]) == 0
    assert torch.count_nonzero(gradient[1]) == 0
    assert diagnostics["dispatch_q_guidance_trigger_rate"].item() == pytest.approx(1 / 3)


def test_each_critic_gain_must_exceed_margin_despite_different_offsets():
    _, model, state, observations, joint = setup_case()

    class SmallSecondGain(ChargeValue):
        def forward(self, state, effective):
            q = self.q1(state, effective)
            return q - 100.0, 0.005 * q + 100.0

    model.critics["mg1"] = SmallSecondGain()
    loss, diagnostics = model.actor_auxiliary_loss("mg1", state, joint, observations["mg1"])
    assert loss.item() == 0.0
    assert diagnostics["dispatch_q_guidance_trigger_rate"] == 0.0


def test_shared_update_records_auxiliary_diagnostics_and_checkpoint_identity(tmp_path):
    env, model, _, _, _ = setup_case()
    obs, info = env.reset(seed=43, deterministic=True)
    state = info["global_state"]
    for _ in range(4):
        actions = {mg: np.zeros(2, dtype=np.float32) for mg in env.agent_ids}
        nxt, rewards, done, info = env.step(actions)
        model.replay.add(obs, state, actions, rewards["mg1"], nxt, info["global_state_after"], done)
        obs, state = nxt, info["global_state_after"]
    model.update(2)
    metrics = model.update(2)
    assert "dispatch_q_guidance_trigger_rate" in metrics["actor_diagnostics"]["mg1"]
    assert np.isfinite(metrics["actor_loss"])
    path = tmp_path / "model.pt"
    model.save(path)
    assert torch.load(path, weights_only=True)["algorithm_slug"] == "matd3_dispatch_q_guided"
    model.load(path)
    assert model.update_steps == 2
