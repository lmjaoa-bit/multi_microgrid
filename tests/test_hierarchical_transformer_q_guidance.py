from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
SLUG = "matd3_hierarchical_transformer_q_guided"


class ChargeValue(nn.Module):
    def __init__(self, action_index: int, disagree: bool = False):
        super().__init__()
        self.action_index = action_index
        self.disagree = disagree
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, state, effective):
        q1 = -self.scale * effective[:, self.action_index:self.action_index + 1]
        return q1, -q1 if self.disagree else q1

    def q1(self, state, effective):
        return self.forward(state, effective)[0]


def setup_case(*, guidance_weight: float = 0.01, test_critic: bool = True):
    spec = get_algorithm_spec(SLUG)
    environment = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    environment = spec.build_environment_config(environment)
    env = MultiMicrogridEnv(environment, split="train")
    config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    config["algorithm"]["hidden_dims"] = [16]
    config["algorithm"]["network"].update(
        embed_dim=8, num_heads=2, num_layers=1, hidden_dim=16
    )
    config["algorithm"]["learning_starts"] = 0
    config["algorithm"]["dispatch_q_guidance_weight"] = guidance_weight
    config["logging"]["diagnostic_interval"] = 2
    model = spec.create(env, config, device="cpu", seed=42)
    observations, _ = env.reset(seed=42, deterministic=True)
    state = torch.tensor(env.global_state()[None], dtype=torch.float32)
    mg_index = env.agent_ids.index("mg2")
    for key, value in {
        "mg2_feasible_charge_max": 1.0,
        "mg2_feasible_discharge_max": 0.0,
        "mg2_soc": 0.1,
        "mg2_load": 0.0,
        "mg2_pv": 5.0,
        "mg2_wind": 0.0,
    }.items():
        state[:, model.projector.index[key]] = value

    actor = model.actors["mg2"]
    with torch.no_grad():
        actor.ess_head[-2].weight.zero_()
        actor.ess_head[-2].bias.fill_(8.0)
    tensors = {
        mg: torch.as_tensor(value[None], dtype=torch.float32)
        for mg, value in observations.items()
    }
    joint_raw = torch.stack([model.actors[mg](tensors[mg]) for mg in model.agent_ids], dim=1)
    if test_critic:
        model.critics["mg2"] = ChargeValue(action_index=mg_index * 3)
    return env, model, state, tensors, joint_raw


def test_q_guidance_reaches_saturated_ess_logit_when_projection_blocks_discharge():
    _, model, state, observations, joint = setup_case()
    mg_index = model.agent_ids.index("mg2")
    current, _ = model.project_actions(state, joint)
    charge_probe = joint.detach().clone()
    charge_probe[:, mg_index, 0] = -0.9
    charged, plans = model.project_actions(state, charge_probe)
    assert plans[mg_index]["actual_charge_kw"].item() > 0

    bias = model.actors["mg2"].ess_head[-2].bias
    original_loss = -model.critics["mg2"].q1(state, current.flatten(1)).mean()
    original_gradient = torch.autograd.grad(original_loss, bias, retain_graph=True)[0]
    assert original_gradient[0].item() == pytest.approx(0.0, abs=1e-8)

    loss, metrics = model.actor_auxiliary_loss("mg2", state, joint, observations["mg2"])
    gradient, dr_head_gradient = torch.autograd.grad(
        loss,
        (bias, model.actors["mg2"].dr_head[-2].bias),
        allow_unused=True,
    )
    assert gradient[0].item() > 0.01
    assert dr_head_gradient is None
    assert metrics["dispatch_q_guidance_trigger_rate"].item() == pytest.approx(1.0)
    assert metrics["dispatch_q_guidance_q_gain"].item() > 0.01


@pytest.mark.parametrize("condition", ["critics_disagree", "direction_feasible", "disabled"])
def test_q_guidance_does_not_override_critic_or_physical_gate(condition):
    _, model, state, observations, joint = setup_case(
        guidance_weight=0.0 if condition == "disabled" else 0.01
    )
    mg_index = model.agent_ids.index("mg2")
    if condition == "critics_disagree":
        model.critics["mg2"] = ChargeValue(action_index=mg_index * 3, disagree=True)
    elif condition == "direction_feasible":
        state[:, model.projector.index["mg2_feasible_discharge_max"]] = 1.0

    loss, metrics = model.actor_auxiliary_loss("mg2", state, joint, observations["mg2"])
    assert loss.item() == 0.0
    assert metrics["dispatch_q_guidance_trigger_rate"].item() == 0.0


def test_dispatch_random_replacement_changes_only_ess_during_noisy_training():
    _, model, _, tensors, _ = setup_case()
    model.dispatch_random_action_prob = 1.0

    class FixedRng:
        def random(self):
            return 0.0

        def normal(self, mean, std, size):
            return np.zeros(size, dtype=np.float32)

        def uniform(self, low, high):
            return -0.5

    model.rng = FixedRng()
    inputs = {mg: value[0].numpy() for mg, value in tensors.items()}
    deterministic = model.select_actions(inputs, noise_std=0.0)
    exploratory = model.select_actions(inputs, noise_std=0.15)
    for mg in model.agent_ids:
        assert exploratory[mg][0] == pytest.approx(-0.5)
        assert exploratory[mg][1] == pytest.approx(deterministic[mg][1])


def test_sampled_update_emits_q_guidance_diagnostics():
    env, model, state, observations, _ = setup_case(test_critic=False)
    mg_index = model.agent_ids.index("mg2")
    fake_critic = ChargeValue(action_index=mg_index * 3)
    model.critics["mg2"] = fake_critic
    model.critic_optimizers["mg2"] = torch.optim.SGD(
        fake_critic.parameters(), lr=0.0
    )
    observation_np = {mg: value[0].numpy() for mg, value in observations.items()}
    state_np = state[0].numpy()
    actions = model.select_actions(observation_np, noise_std=0.0)
    for _ in range(4):
        model.replay.add(
            observation_np,
            state_np,
            actions,
            -1.0,
            observation_np,
            state_np,
            False,
        )

    model.update(batch_size=2)
    model.diagnostic_schedule.next_boundary = 2
    metrics = model.update(batch_size=2)
    assert "actor_diagnostics" in metrics
    diagnostic = metrics["actor_diagnostics"]["mg2"]
    assert diagnostic["dispatch_q_guidance_trigger_rate"] == pytest.approx(1.0)
    assert diagnostic["dispatch_q_guidance_loss"] > 0.0
    assert np.isfinite(metrics["actor_loss"])


def test_checkpoint_uses_experimental_identity_and_round_trips(tmp_path):
    env, model, *_ = setup_case(test_critic=False)
    assert model.reward_v2_contract["algorithm_slug"] == SLUG
    path = tmp_path / "model.pt"
    model.save(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["algorithm_slug"] == SLUG
    _, restored, *_ = setup_case(test_critic=False)
    restored.load(path)
    assert restored.update_steps == model.update_steps
    assert torch.allclose(
        restored.actors["mg2"].ess_head[-2].bias,
        model.actors["mg2"].ess_head[-2].bias,
    )


def test_q_guided_config_keeps_small_bounded_defaults():
    config = yaml.safe_load(
        (ROOT / "configs/algorithms/matd3_hierarchical_transformer_q_guided.yaml")
        .read_text(encoding="utf-8")
    )
    settings = config["algorithm"]
    assert settings["name"] == SLUG
    assert settings["learning_starts"] == 1000
    assert settings["dispatch_random_action_prob"] == pytest.approx(0.10)
    assert settings["dispatch_q_guidance_weight"] == pytest.approx(0.01)
    assert settings["dispatch_q_guidance_margin"] == pytest.approx(0.01)
