from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]
SLUG = "matd3_mlp_transformer_critic_q_guided"


def test_independent_algorithm_is_available():
    assert (ROOT / "algorithms" / SLUG / "plugin.py").is_file()
    spec = get_algorithm_spec(SLUG)
    assert yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))["algorithm"]["name"] == SLUG


@pytest.fixture
def case():
    spec = get_algorithm_spec(SLUG)
    cfg = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    cfg["algorithm"].update(learning_starts=8, replay_capacity=128)
    cfg["algorithm"]["network"].update(embed_dim=8, num_heads=2, num_layers=1, hidden_dim=16)
    cfg["algorithm"]["critic_network"].update(embed_dim=8, num_heads=2, num_layers=1, feedforward_dim=16)
    cfg["logging"]["diagnostic_interval"] = 2
    env_cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(spec.build_environment_config(env_cfg), split="train")
    model = spec.create(env, cfg, device="cpu", seed=42)
    obs, _ = env.reset(seed=42, deterministic=True)
    return env, model, cfg, obs


def test_mlp_has_same_semantic_h8_inputs_and_parameter_budget_as_t(case):
    from algorithms.matd3_hierarchical_transformer_q_readout.actor import DeviceForecastReadoutActor

    env, model, cfg, obs = case
    for mg in env.agent_ids:
        reference = DeviceForecastReadoutActor(
            model.observation_schemas[mg], env.agent_ids, mg,
            mg_config=env.cfg_by_id[mg], power_normalizers=env.obs_builder.norm[mg],
            **{k: cfg["algorithm"]["network"][k] for k in
               ("embed_dim", "num_heads", "num_layers", "hidden_dim", "forecast_horizon", "include_peer_forecast", "peer_forecast_features")},
        )
        inputs = torch.as_tensor(obs[mg][None])
        captured = {}
        hooks = [getattr(reference, name).register_forward_pre_hook(
            lambda module, args, key=name: captured.update({key: args[0].detach().flatten(1)})
        ) for name in ("ess_encoder", "dr_encoder", "supply_encoder", "trade_encoder", "peer_encoder", "forecast_encoder")]
        reference._build_tokens(inputs)
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(model.actors[mg].features(inputs), torch.cat(list(captured.values()), -1))
        assert not any(isinstance(m, nn.TransformerEncoder) for m in model.actors[mg].modules())
        target = sum(p.numel() for p in reference.parameters())
        assert abs(model.actors[mg].parameter_count - target) / target < .03


def test_actor_ignores_forecasts_beyond_h8_and_invalid_h8_payload(case):
    _, model, _, obs = case
    for mg, actor in model.actors.items():
        x = torch.as_tensor(obs[mg][None]).clone()
        changed = x.clone()
        for i, name in enumerate(actor.schema):
            if "_t+" in name and int(name.rsplit("_t+", 1)[1]) > 8:
                changed[:, i] = 1234.
        torch.testing.assert_close(actor(x), actor(changed), rtol=0, atol=0)
        for k in range(1, 9):
            x[:, actor.schema.index(f"forecast_valid_t+{k}")] = 0.
        changed = x.clone()
        for i, name in enumerate(actor.schema):
            if "_t+" in name and not name.startswith("forecast_valid"):
                changed[:, i] = float("nan")
        torch.testing.assert_close(actor(x), actor(changed), rtol=0, atol=0)


def test_transformer_twin_q_preserves_h24_and_has_independent_weights(case):
    env, model, _, _ = case
    critic = model.critics["mg2"]
    q1, q2 = critic.q1_network, critic.q2_network
    covered = q1.current_indices.tolist() + q1.forecast_indices.flatten().tolist()
    assert sorted(covered) == list(range(env.global_state_dim))
    assert q1.forecast_indices.shape[0] == 24
    assert not ({p.data_ptr() for p in q1.parameters()} & {p.data_ptr() for p in q2.parameters()})
    state = torch.as_tensor(env.global_state()[None])
    actions = torch.zeros(1, 9)
    first, second = critic(state, actions)
    assert first.shape == second.shape == (1, 1)
    torch.testing.assert_close(first, critic.q1(state, actions))
    with torch.no_grad():
        next(q1.parameters()).add_(1.)
    torch.testing.assert_close(second, critic(state, actions)[1])


def test_critic_dr_action_gradient_matches_finite_difference(case):
    env, model, _, _ = case
    critic = model.critics["mg2"].double()
    state = torch.as_tensor(env.global_state()[None], dtype=torch.float64)
    actions = torch.full((1, 9), .1, dtype=torch.float64, requires_grad=True)
    value = critic.q1(state, actions)
    grad = torch.autograd.grad(value.sum(), actions)[0]
    assert torch.isfinite(grad).all() and abs(grad[0, 5]) > 1e-9
    plus, minus = actions.detach().clone(), actions.detach().clone()
    plus[0, 5] += 1e-5
    minus[0, 5] -= 1e-5
    finite = (critic.q1(state, plus) - critic.q1(state, minus)) / 2e-5
    torch.testing.assert_close(grad[0, 5], finite.squeeze(), rtol=1e-4, atol=1e-8)


def test_critic_is_finite_at_day_end_and_ignores_masked_forecast_payload(case):
    env, model, _, _ = case
    x = torch.as_tensor(env.global_state()[None]).clone()
    for k in range(1, 25):
        x[:, model.state_schema.index(f"forecast_valid_t+{k}")] = 0.
    changed = x.clone()
    for i, name in enumerate(model.state_schema):
        if "_t+" in name and not name.startswith("forecast_valid"):
            changed[:, i] = float("nan")
    action = torch.zeros(1, 9, requires_grad=True)
    critic = model.critics["mg2"]
    for left, right in zip(critic(x, action), critic(changed, action)):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    grad = torch.autograd.grad(critic.q1(changed, action).sum(), action)[0]
    assert torch.isfinite(grad).all()


def test_update_trains_both_networks_and_target_networks(case):
    env, model, _, obs = case
    state = env.global_state()
    actions = model.select_actions(obs, noise_std=0.)
    for _ in range(8):
        model.replay.add(obs, state, actions, -1., obs, state, False)
    before = {name: copy.deepcopy(getattr(model, name).state_dict()) for name in
              ("actors", "critics", "target_actors", "target_critics")}
    model.update(batch_size=4)
    metrics = model.update(batch_size=4)
    assert model.update_steps == 2 and np.isfinite(metrics["actor_loss"])
    for name, old in before.items():
        assert any(not torch.equal(old[k], v) for k, v in getattr(model, name).state_dict().items())
    for mg in env.agent_ids:
        assert np.isfinite(metrics["actor_diagnostics"][mg]["dr_action_grad_rms"])
        assert "dispatch_q_guidance_trigger_rate" in metrics["actor_diagnostics"][mg]


def test_learning_start_gate_prevents_early_updates(case):
    env, model, _, obs = case
    model.replay.add(obs, env.global_state(), model.select_actions(obs), -1., obs, env.global_state(), False)
    assert model.update(1) == {} and model.update_steps == 0


def test_omitted_peer_flag_uses_h8_peer_defaults_without_mutating_caller(case):
    env, _, cfg, obs = case
    del cfg["algorithm"]["network"]["include_peer_forecast"]
    before = copy.deepcopy(cfg)
    model = get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=42)
    assert cfg == before
    for mg, action in model.select_actions(obs).items():
        assert np.isfinite(action).all()
        assert model.actors[mg].features.all_peer_forecast_indices.numel() == 16


def test_deterministic_evaluation_has_no_random_ess_replacement(case):
    _, model, _, obs = case
    model.dispatch_random_action_prob = 1.

    class FixedRng:
        def random(self):
            return 0.

        def normal(self, mean, std, size):
            return np.zeros(size, dtype=np.float32)

        def uniform(self, low, high):
            return -.5

    model.rng = FixedRng()
    deterministic = model.select_actions(obs, noise_std=0.)
    exploratory = model.select_actions(obs, noise_std=.15)
    repeated = model.select_actions(obs, noise_std=0.)
    for mg in model.agent_ids:
        np.testing.assert_array_equal(deterministic[mg], repeated[mg])
        assert exploratory[mg][0] == -.5
        assert exploratory[mg][1] == deterministic[mg][1]


class ChargeValue(nn.Module):
    def __init__(self, disagree=False):
        super().__init__()
        self.disagree = disagree

    def forward(self, state, action):
        q = -action[:, 3:4]
        return q, -q if self.disagree else q


@pytest.mark.parametrize("condition", ["accept", "disagree", "disabled"])
def test_ess_guidance_recovers_saturated_mlp_logit_without_direct_dr_gradient(case, condition):
    env, model, _, obs = case
    state = torch.as_tensor(env.global_state()[None]).clone()
    for key, value in {"mg2_feasible_charge_max": 1., "mg2_feasible_discharge_max": 0.,
                       "mg2_soc": .1, "mg2_load": 0., "mg2_pv": 5., "mg2_wind": 0.}.items():
        state[:, model.projector.index[key]] = value
    actor = model.actors["mg2"]
    with torch.no_grad():
        actor.network[-1].weight.zero_()
        actor.network[-1].bias[0] = 8.
    tensors = {mg: torch.as_tensor(v[None]) for mg, v in obs.items()}
    joint = torch.stack([model.actors[mg](tensors[mg]) for mg in model.agent_ids], 1)
    model.critics["mg2"] = ChargeValue(disagree=condition == "disagree")
    if condition == "disabled":
        model.dispatch_q_guidance_weight = 0.
    loss, metrics = model.actor_auxiliary_loss("mg2", state, joint, tensors["mg2"])
    if condition == "accept":
        grad = torch.autograd.grad(loss, actor.network[-1].bias)[0]
        assert grad[0] > .01 and grad[1] == 0.
        assert metrics["dispatch_q_guidance_trigger_rate"] == 1.
    else:
        assert loss == 0. and metrics["dispatch_q_guidance_trigger_rate"] == 0.


def test_checkpoint_roundtrip_and_wrong_structure_rejected_before_mutation(case, tmp_path):
    env, model, cfg, obs = case
    path = tmp_path / "model.pt"
    model.save(path)
    other = get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=43)
    other.load(path)
    for mg in env.agent_ids:
        np.testing.assert_array_equal(model.select_actions(obs)[mg], other.select_actions(obs)[mg])
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["critic_network_config"]["num_heads"] = 1
    torch.save(checkpoint, path)
    before = copy.deepcopy(other.actors.state_dict())
    with pytest.raises(ValueError, match="structure|network"):
        other.load(path)
    for k, v in before.items():
        torch.testing.assert_close(v, other.actors.state_dict()[k], rtol=0, atol=0)


def test_corrupt_later_state_dict_is_rejected_before_any_weights_change(case, tmp_path):
    env, model, cfg, _ = case
    path = tmp_path / "broken.pt"
    model.save(path)
    checkpoint = torch.load(path, weights_only=False)
    key = next(k for k, v in checkpoint["target_critics"].items() if v.ndim == 2)
    checkpoint["target_critics"][key] = torch.zeros(1, 1)
    torch.save(checkpoint, path)
    other = get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=43)
    before = copy.deepcopy(other.actors.state_dict())
    with pytest.raises(ValueError, match="shape|state"):
        other.load(path)
    for k, v in before.items():
        torch.testing.assert_close(v, other.actors.state_dict()[k], rtol=0, atol=0)


def test_checkpoint_cannot_replace_fixed_feature_indices(case, tmp_path):
    _, model, _, _ = case
    path = tmp_path / "fixed.pt"
    model.save(path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["actors"]["mg2.features.forecast_load_indices"][0] += 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="fixed state"):
        model.load(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_can_train_and_load_cpu_checkpoint_with_formal_network_sizes(case, tmp_path):
    env, _, _, obs = case
    spec = get_algorithm_spec(SLUG)
    cfg = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    cfg["algorithm"].update(learning_starts=8, replay_capacity=128)
    cfg["logging"]["diagnostic_interval"] = 2
    cpu_model = spec.create(env, cfg, device="cpu", seed=42)
    path = tmp_path / "cpu.pt"
    cpu_model.save(path)
    model = spec.create(env, cfg, device="cuda", seed=43)
    model.load(path)
    state = env.global_state()
    for _ in range(8):
        model.replay.add(obs, state, model.select_actions(obs), -1., obs, state, False)
    model.update(4)
    metrics = model.update(4)
    assert np.isfinite(metrics["actor_loss"]) and np.isfinite(metrics["critic_loss"])
    for mg in model.agent_ids:
        assert model.actors[mg].features.forecast_load_indices.device.type == "cuda"
        assert metrics["actor_diagnostics"][mg]["dr_action_grad_rms"] >= 0.


@pytest.mark.parametrize("section, field, value", [("network", "forecast_horizon", 24),
                                                  ("critic_network", "forecast_horizon", 8)])
def test_wrong_forecast_window_is_rejected(case, section, field, value):
    env, _, cfg, _ = case
    cfg["algorithm"][section][field] = value
    with pytest.raises(ValueError, match="horizon|H8|H24"):
        get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=42)
