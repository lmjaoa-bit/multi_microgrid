from __future__ import annotations

import copy
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]
SLUG = "matd3_mlp_dueling_critic_h24_q_guided"
H24 = "matd3_mlp_transformer_critic_h24_q_guided"
DIAGNOSTICS = {
    "critic_v_mean", "critic_adv_std", "critic_adv_abs_mean", "critic_adv_share",
    "critic_dqda_dispatch_rms", "critic_dqda_dr_rms", "critic_v_adv_corr",
}


def schema():
    channels = [f"future_{i}" for i in range(7)] + ["forecast_valid"]
    return [f"current_{i}" for i in range(103)] + [
        f"{channel}_t+{lead}" for lead in range(1, 25) for channel in channels
    ]


def critic(**advantage):
    from algorithms.matd3_mlp_dueling_critic_h24_q_guided.critic import DuelingTwinCritic
    torch.manual_seed(42)
    return DuelingTwinCritic(schema(), value=dict(embed_dim=8, num_heads=2, num_layers=1,
                                                feedforward_dim=16), advantage=advantage)


def inputs(batch=4):
    state = torch.randn(batch, 295)
    state[:, 110::8] = 1.
    return state, torch.randn(batch, 9).tanh()


def config():
    cfg = yaml.safe_load(get_algorithm_spec(SLUG).default_config_path.read_text(encoding="utf-8"))
    cfg["algorithm"].update(learning_starts=8, replay_capacity=128)
    cfg["algorithm"]["network"].update(embed_dim=8, num_heads=2, num_layers=1, hidden_dim=16)
    cfg["algorithm"]["critic_network"]["value"].update(embed_dim=8, num_heads=2, num_layers=1,
                                                       feedforward_dim=16)
    cfg["algorithm"]["critic_network"]["advantage"]["hidden_dim"] = 16
    cfg["logging"]["diagnostic_interval"] = 2
    return cfg


@pytest.fixture
def case():
    cfg = config()
    spec = get_algorithm_spec(SLUG)
    env_cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(spec.build_environment_config(env_cfg), split="train")
    model = spec.create(env, cfg, device="cpu", seed=42)
    obs, _ = env.reset(seed=42, deterministic=True)
    return env, model, cfg, obs


def fill_replay(env, model, obs):
    state = env.global_state()
    for _ in range(8):
        model.replay.add(obs, state, model.select_actions(obs), -1., obs, state, False)


def test_plugin_and_formal_configuration_preserve_h24_settings():
    spec = get_algorithm_spec(SLUG)
    assert spec.slug == SLUG and spec.display_name == "Dueling Critic T-Q H24"
    cfg = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    baseline = yaml.safe_load(get_algorithm_spec(H24).default_config_path.read_text(encoding="utf-8"))
    assert cfg["algorithm"].pop("name") == SLUG
    baseline["algorithm"].pop("name")
    cfg["algorithm"].pop("critic_network")
    baseline["algorithm"].pop("critic_network")
    assert cfg == baseline


def test_plugin_loads_with_h24_comparator_imports_blocked():
    script = '''
import importlib.abc, sys
class NoH24(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("algorithms.matd3_mlp_transformer_critic_h24_q_guided"):
            raise ImportError("H24 comparator is unavailable")
sys.meta_path.insert(0, NoH24())
from algorithms.registry import get_algorithm_spec
spec = get_algorithm_spec("matd3_mlp_dueling_critic_h24_q_guided")
assert spec.algorithm_class.SLUG == spec.slug
'''
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)


def test_shapes_zero_reference_and_independent_twins():
    network = critic()
    state, action = inputs()
    q1, q2 = network(state, action)
    assert q1.shape == q2.shape == network.q1(state, action).shape == (4, 1)
    parts = network.decompose(state, torch.zeros_like(action))
    assert set(parts) == {"v1", "adv1", "q1", "v2", "adv2", "q2"}
    for i in (1, 2):
        torch.testing.assert_close(parts[f"q{i}"], parts[f"v{i}"], rtol=0, atol=1e-6)
        torch.testing.assert_close(parts[f"adv{i}"], torch.zeros_like(parts[f"adv{i}"]), rtol=0, atol=1e-6)
    assert not ({id(p) for p in network.q1_network.parameters()} &
                {id(p) for p in network.q2_network.parameters()})


@pytest.mark.parametrize("bad_state,bad_action", [(torch.zeros(295), torch.zeros(1, 9)),
    (torch.zeros(2, 294), torch.zeros(2, 9)), (torch.zeros(2, 295), torch.zeros(2, 8)),
    (torch.zeros(2, 295), torch.zeros(1, 9))])
def test_rejects_invalid_input_shapes(bad_state, bad_action):
    with pytest.raises(ValueError):
        critic()(bad_state, bad_action)


def test_action_gradient_is_finite_nonzero_and_value_ignores_action():
    network = critic()
    state, action = inputs()
    action.requires_grad_()
    parts = network.decompose(state, action)
    grad = torch.autograd.grad(parts["q1"].sum(), action, retain_graph=True)[0]
    adv_grad = torch.autograd.grad(parts["adv1"].sum(), action)[0]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    torch.testing.assert_close(grad, adv_grad, rtol=0, atol=0)
    changed = network.decompose(state, -action.detach())
    for key in ("v1", "v2"):
        torch.testing.assert_close(parts[key], changed[key], rtol=0, atol=0)


@pytest.mark.parametrize("detach", [True, False])
def test_advantage_context_gradient_switch(detach):
    network = critic(detach_value_context=detach)
    state, action = inputs()
    network.decompose(state, action)["adv1"].sum().backward()
    gradients = [p.grad for p in network.q1_network.transformer.parameters()]
    if detach:
        assert all(g is None or torch.count_nonzero(g) == 0 for g in gradients)
    else:
        assert any(g is not None and torch.count_nonzero(g) > 0 for g in gradients)


@pytest.mark.parametrize("context,reinjection", [(False, True), (True, False), (False, False)])
def test_advantage_ablation_switches(context, reinjection):
    network = critic(use_value_context=context, action_reinjection=reinjection)
    state, action = inputs()
    action.requires_grad_()
    gradient = torch.autograd.grad(network.q1(state, action).sum(), action)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("payload", [float("nan"), float("inf"), -1e6])
def test_invalid_forecast_payload_is_ignored(payload):
    network = critic()
    state, action = inputs()
    state[:, 110::8] = 0.
    changed = state.clone()
    changed[:, 103:] = payload
    changed[:, 110::8] = 0.
    changed.requires_grad_()
    for expected, actual in zip(network(state, action), network(changed, action)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.autograd.grad(actual.sum(), changed)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient[:, 103:]) == 0
    changed = state.clone()
    changed[:, 103:] = float("nan")
    assert all(torch.isfinite(q).all() for q in network(changed, action))


@pytest.mark.parametrize("mutation", ["duplicate", "missing_lead", "reordered", "missing_valid"])
def test_schema_contract_is_validated(mutation):
    from algorithms.matd3_mlp_dueling_critic_h24_q_guided.critic import DuelingTwinCritic
    names = schema()
    if mutation == "duplicate":
        names[1] = names[0]
    elif mutation == "missing_lead":
        names = names[:-8]
    elif mutation == "reordered":
        names[111], names[112] = names[112], names[111]
    else:
        names = [n.replace("forecast_valid", "no_valid") for n in names]
    with pytest.raises(ValueError):
        DuelingTwinCritic(names)


@pytest.mark.parametrize("settings", [
    {"unknown": {}}, {"value": {"unknown": 1}}, {"advantage": {"unknown": True}},
    *[{"value": {key: True}} for key in
      ("forecast_horizon", "embed_dim", "num_heads", "num_layers", "feedforward_dim")],
    {"value": {"embed_dim": 10, "num_heads": 4}}, {"value": {"num_heads": 0}},
    {"value": {"forecast_horizon": 8}}, {"advantage": {"hidden_dim": True}},
    {"advantage": {"hidden_dim": 0}}, {"advantage": {"reference_action": "none"}},
    *[{"advantage": {key: 1}} for key in
      ("use_value_context", "detach_value_context", "action_reinjection")],
    {"value": None}, {"advantage": []}, None, [],
])
def test_nested_configuration_rejects_invalid_settings(case, settings):
    env, _, cfg, _ = case
    cfg["algorithm"]["critic_network"] = settings
    with pytest.raises(ValueError):
        get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=42)


def test_actor_equals_h24_and_terminal_forecasts_are_safe(case):
    env, model, cfg, obs = case
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg["algorithm"]["name"] = H24
    baseline_cfg["algorithm"]["critic_network"] = cfg["algorithm"]["critic_network"]["value"]
    baseline = get_algorithm_spec(H24).create(env, baseline_cfg, device="cpu", seed=42)
    for mg, actor in model.actors.items():
        original = baseline.actors[mg]
        assert actor.parameter_count == original.parameter_count
        # Actor construction, features and RNG sequence remain unchanged.
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
        actor.load_state_dict(original.state_dict())
        x = torch.as_tensor(obs[mg][None])
        torch.testing.assert_close(actor(x), original(x), rtol=0, atol=0)
    for _ in range(23):
        obs, _, _, _ = env.step(model.select_actions(obs))
    state = torch.as_tensor(env.global_state()[None])
    for mg in env.agent_ids:
        assert torch.isfinite(model.actors[mg](torch.as_tensor(obs[mg][None]))).all()
        assert all(torch.isfinite(q).all() for q in model.critics[mg](state, torch.zeros(1, 9)))


def test_checkpoint_roundtrip_and_nested_mismatch_is_atomic(case, tmp_path):
    env, model, cfg, obs = case
    path = tmp_path / "dueling.pt"
    model.save(path)
    other = get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=43)
    other.load(path)
    state = torch.as_tensor(env.global_state()[None])
    action = torch.randn(1, 9)
    for mg in env.agent_ids:
        for left, right in zip(model.critics[mg](state, action), other.critics[mg](state, action)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    before = {
        name: copy.deepcopy(getattr(other, name).state_dict()) for name in
        ("actors", "critics", "target_actors", "target_critics")}
    saved = torch.load(path, weights_only=True)
    saved["critic_network_config"]["advantage"]["action_reinjection"] = False
    torch.save(saved, path)
    with pytest.raises(ValueError, match="critic_network_config"):
        other.load(path)
    for name, values in before.items():
        for key, value in getattr(other, name).state_dict().items():
            torch.testing.assert_close(value, values[key], rtol=0, atol=0)


def test_diagnostics_use_shared_schedule_and_csv_writer(case, tmp_path):
    from experiments.training_pipeline import _diagnostic_means, _append_actor_diagnostics
    env, model, _, obs = case
    fill_replay(env, model, obs)
    rows = []
    for step in range(1, 9):
        result = model.update(4)
        if step % 2:
            assert result == {}
        else:
            rows.append(result)
            for values in result["actor_diagnostics"].values():
                assert DIAGNOSTICS <= values.keys()
                assert all(np.isfinite(values[key]) for key in DIAGNOSTICS)
    path = tmp_path / "actor_diagnostics.csv"
    _append_actor_diagnostics(path, 1, _diagnostic_means(rows))
    with path.open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle))
    assert values and all(f"{key}_mean" in values[0] for key in DIAGNOSTICS)


@pytest.mark.parametrize("interval", [1, 3, 5])
def test_sparse_schedule_diagnostics_match_shared_updates(case, interval):
    env, _, cfg, obs = case
    cfg["logging"]["diagnostic_interval"] = interval
    model = get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=42)
    fill_replay(env, model, obs)
    for step in range(1, 13):
        result = model.update(4)
        if result:
            assert model.update_steps == step
            assert all(DIAGNOSTICS <= row.keys() for row in result["actor_diagnostics"].values())


def test_singleton_diagnostics_do_not_create_or_change_parameter_gradients(case):
    env, model, _, obs = case
    state = torch.as_tensor(env.global_state()[None])
    tensors = {mg: torch.as_tensor(value[None]) for mg, value in obs.items()}
    joint = torch.stack([model.actors[mg](tensors[mg]) for mg in model.agent_ids], 1)
    model.dispatch_q_guidance_weight = 0.
    model.update_steps = 2
    assert model.diagnostic_schedule.due(2)
    parameters = list(model.actors.parameters()) + list(model.critics.parameters())
    for i, p in enumerate(parameters):
        p.grad = torch.ones_like(p) if i % 2 else None
    before = [p.grad.clone() if p.grad is not None else None for p in parameters]
    for active in model.agent_ids:
        loss, metrics = model.actor_auxiliary_loss(active, state, joint, tensors[active])
        assert loss == 0. and DIAGNOSTICS <= metrics.keys()
        assert all(v.ndim == 0 and torch.isfinite(v) and not v.requires_grad for v in metrics.values())
    for p, old in zip(parameters, before):
        if old is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, old, rtol=0, atol=0)


def test_sampling_diagnostics_preserves_exact_update_trajectory(case):
    env, _, cfg, obs = case
    snapshots = []
    for interval in (100000, 1):
        cfg["logging"]["diagnostic_interval"] = interval
        model = get_algorithm_spec(SLUG).create(env, cfg, device="cpu", seed=42)
        fill_replay(env, model, obs)
        torch.manual_seed(1234)
        for _ in range(6):
            model.update(4)
        snapshots.append({name: copy.deepcopy(getattr(model, name).state_dict()) for name in
                          ("actors", "critics", "target_actors", "target_critics")})
    for name, old in snapshots[0].items():
        for key, value in snapshots[1][name].items():
            torch.testing.assert_close(value, old[key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_formal_networks_update_on_cuda_after_cpu_checkpoint(case, tmp_path):
    env, _, _, obs = case
    spec = get_algorithm_spec(SLUG)
    cfg = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    cfg["algorithm"].update(learning_starts=8, replay_capacity=128)
    cfg["logging"]["diagnostic_interval"] = 2
    cpu = spec.create(env, cfg, device="cpu", seed=42)
    path = tmp_path / "cpu.pt"
    cpu.save(path)
    gpu = spec.create(env, cfg, device="cuda", seed=43)
    gpu.load(path)
    fill_replay(env, gpu, obs)
    before = copy.deepcopy(gpu.critics.state_dict())
    gpu.update(4)
    result = gpu.update(4)
    assert np.isfinite(result["critic_loss"]) and np.isfinite(result["actor_loss"])
    assert any(not torch.equal(value, before[key]) for key, value in gpu.critics.state_dict().items())
    assert all(DIAGNOSTICS <= values.keys() for values in result["actor_diagnostics"].values())
