from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from algorithms.registry import available_algorithm_names, get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
SLUG = "matd3_mlp_transformer_critic_h24_q_guided"
H8_SLUG = "matd3_mlp_transformer_critic_q_guided"


def test_h24_variant_can_be_selected_with_its_own_configuration():
    assert SLUG in available_algorithm_names()
    spec = get_algorithm_spec(SLUG)
    cfg = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    assert cfg["algorithm"]["name"] == SLUG
    env_cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(spec.build_environment_config(env_cfg), split="train")
    model = spec.create(env, cfg, device="cpu", seed=42)
    obs, _ = env.reset(seed=42, deterministic=True)
    assert all(np.isfinite(a).all() for a in model.select_actions(obs).values())
    assert all(a.forecast_horizon == 24 for a in model.actors.values())


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


def test_h24_mlp_preserves_semantic_features_and_comparable_parameter_budget(case):
    from algorithms.matd3_hierarchical_transformer_q_readout.actor import DeviceForecastReadoutActor

    env, model, cfg, obs = case
    for mg in env.agent_ids:
        reference = DeviceForecastReadoutActor(
            model.observation_schemas[mg], env.agent_ids, mg,
            mg_config=env.cfg_by_id[mg], power_normalizers=env.obs_builder.norm[mg],
            **{k: cfg["algorithm"]["network"][k] for k in
               ("embed_dim", "num_heads", "num_layers", "hidden_dim", "forecast_horizon",
                "include_peer_forecast", "peer_forecast_features")},
        )
        x = torch.as_tensor(obs[mg][None])
        captured = {}
        hooks = [getattr(reference, name).register_forward_pre_hook(
            lambda module, args, key=name: captured.update({key: args[0].detach().flatten(1)})
        ) for name in ("ess_encoder", "dr_encoder", "supply_encoder", "trade_encoder",
                     "peer_encoder", "forecast_encoder")]
        reference._build_tokens(x)
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(model.actors[mg].features(x), torch.cat(list(captured.values()), -1))
        assert not any(isinstance(m, nn.TransformerEncoder) for m in model.actors[mg].modules())
        target = sum(p.numel() for p in reference.parameters())
        assert abs(model.actors[mg].parameter_count - target) / target < .03


@pytest.mark.parametrize("lead", [9, 24])
@pytest.mark.parametrize("channel", ["base_load", "peer_balance", "temperature"])
def test_forecasts_beyond_h8_reach_h24_actor_actions(case, lead, channel):
    _, model, _, obs = case
    actor = model.actors["mg2"]
    x = torch.as_tensor(obs["mg2"][None]).clone()
    x[:, actor.schema.index(f"forecast_valid_t+{lead}")] = 1.
    x.requires_grad_()
    gradients = torch.autograd.grad(actor.action_logits(x).sum(), x)[0]
    if channel == "base_load":
        # Change load and PV equally in physical units: net load stays fixed,
        # so the derivative must come from the separately retained base load.
        scales = actor.features.power_normalizer_tensor
        value = (gradients[0, actor.schema.index(f"load_t+{lead}")]
                 + gradients[0, actor.schema.index(f"pv_t+{lead}")] * scales[0] / scales[1])
    else:
        key = f"mg1_energy_balance_t+{lead}" if channel == "peer_balance" else f"temperature_t+{lead}"
        value = gradients[0, actor.schema.index(key)]
    assert torch.isfinite(value) and abs(value) > 1e-8


def test_h24_actor_and_critic_ignore_invalid_future_payload_at_day_end(case):
    env, model, _, obs = case
    for mg, actor in model.actors.items():
        x = torch.as_tensor(obs[mg][None]).clone()
        for lead in range(1, 25):
            x[:, actor.schema.index(f"forecast_valid_t+{lead}")] = 0.
        changed = x.clone()
        for index, name in enumerate(actor.schema):
            if "_t+" in name and not name.startswith("forecast_valid"):
                changed[:, index] = float("nan")
        torch.testing.assert_close(actor(x), actor(changed), rtol=0, atol=0)
        assert torch.isfinite(actor(changed)).all()
    state = torch.as_tensor(env.global_state()[None]).clone()
    for lead in range(1, 25):
        state[:, model.state_schema.index(f"forecast_valid_t+{lead}")] = 0.
    changed = state.clone()
    for index, name in enumerate(model.state_schema):
        if "_t+" in name and not name.startswith("forecast_valid"):
            changed[:, index] = float("nan")
    actions = torch.zeros(1, 9, requires_grad=True)
    for left, right in zip(model.critics["mg2"](state, actions), model.critics["mg2"](changed, actions)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert torch.isfinite(right).all()
    assert torch.isfinite(torch.autograd.grad(right.sum(), actions)[0]).all()


def test_missing_horizon_and_peer_flag_default_to_h24_without_mutating_config(case):
    env, _, cfg, obs = case
    del cfg["algorithm"]["network"]["forecast_horizon"]
    del cfg["algorithm"]["network"]["include_peer_forecast"]
    before = copy.deepcopy(cfg)
    model = get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=42)
    assert cfg == before
    assert all(a.forecast_horizon == 24 for a in model.actors.values())
    assert all(np.isfinite(a).all() for a in model.select_actions(obs).values())


@pytest.mark.parametrize("horizon", [8, True, 24.0])
def test_invalid_actor_horizon_is_rejected(case, horizon):
    env, _, cfg, _ = case
    cfg["algorithm"]["network"]["forecast_horizon"] = horizon
    with pytest.raises(ValueError, match="horizon|H24"):
        get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=42)


def test_h24_policy_updates_actor_critic_and_target_networks(case):
    env, model, _, obs = case
    state = env.global_state()
    for _ in range(8):
        model.replay.add(obs, state, model.select_actions(obs), -1., obs, state, False)
    before = {name: copy.deepcopy(getattr(model, name).state_dict()) for name in
              ("actors", "critics", "target_actors", "target_critics")}
    model.update(4)
    metrics = model.update(4)
    assert model.update_steps == 2 and np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    for name, old in before.items():
        assert any(not torch.equal(old[k], value) for k, value in getattr(model, name).state_dict().items())
    for mg in env.agent_ids:
        assert np.isfinite(metrics["actor_diagnostics"][mg]["dr_action_grad_rms"])
        assert "dispatch_q_guidance_trigger_rate" in metrics["actor_diagnostics"][mg]


@pytest.mark.parametrize("condition", ["accept", "disagree", "disabled"])
def test_h24_ess_guidance_requires_agreement_and_preserves_dr_logit(case, condition):
    class ChargeValue(nn.Module):
        def forward(self, state, action):
            q = -action[:, 3:4]
            return q, -q if condition == "disagree" else q

    env, model, _, obs = case
    state = torch.as_tensor(env.global_state()[None]).clone()
    for key, value in {"mg2_feasible_charge_max": 1., "mg2_feasible_discharge_max": 0.,
                       "mg2_soc": .1, "mg2_load": 0., "mg2_pv": 5., "mg2_wind": 0.}.items():
        state[:, model.projector.index[key]] = value
    actor = model.actors["mg2"]
    with torch.no_grad():
        actor.network[-1].weight.zero_()
        actor.network[-1].bias[0] = 8.
    tensors = {mg: torch.as_tensor(value[None]) for mg, value in obs.items()}
    joint = torch.stack([model.actors[mg](tensors[mg]) for mg in model.agent_ids], 1)
    model.critics["mg2"] = ChargeValue()
    if condition == "disabled":
        model.dispatch_q_guidance_weight = 0.
    loss, metrics = model.actor_auxiliary_loss("mg2", state, joint, tensors["mg2"])
    if condition == "accept":
        gradient = torch.autograd.grad(loss, actor.network[-1].bias)[0]
        assert gradient[0] > .01 and gradient[1] == 0.
        assert metrics["dispatch_q_guidance_trigger_rate"] == 1.
    else:
        assert loss == 0. and metrics["dispatch_q_guidance_trigger_rate"] == 0.


@pytest.mark.parametrize("source_slug,target_slug", [(SLUG, H8_SLUG), (H8_SLUG, SLUG)])
def test_h8_and_h24_checkpoints_are_rejected_before_changing_weights(case, tmp_path, source_slug, target_slug):
    env, model, cfg, obs = case
    old_cfg = copy.deepcopy(cfg)
    old_cfg["algorithm"]["name"] = H8_SLUG
    old_cfg["algorithm"]["network"]["forecast_horizon"] = 8
    old = get_algorithm_spec(H8_SLUG).create(env, old_cfg, device="cpu", seed=42)
    models = {SLUG: model, H8_SLUG: old}
    source, target = models[source_slug], models[target_slug]
    path = tmp_path / "model.pt"
    source.save(path)
    before = target.select_actions(obs)
    with pytest.raises(ValueError, match="structure|contract"):
        target.load(path)
    for mg, action in target.select_actions(obs).items():
        np.testing.assert_array_equal(before[mg], action)


def test_h24_checkpoint_roundtrip_and_corrupt_fixed_indices(case, tmp_path):
    env, model, cfg, obs = case
    path = tmp_path / "h24.pt"
    model.save(path)
    other = get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=43)
    other.load(path)
    for mg in env.agent_ids:
        np.testing.assert_array_equal(model.select_actions(obs)[mg], other.select_actions(obs)[mg])
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["actors"]["mg2.features.forecast_load_indices"][-1] -= 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="fixed state"):
        other.load(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_h24_formal_sizes_train_on_cuda_from_cpu_checkpoint(case, tmp_path):
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
    assert all(a.features.forecast_load_indices.device.type == "cuda" for a in model.actors.values())
