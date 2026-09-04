from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.crossgate_actor.actor import CrossGateActor as CrossGateActorNetwork
from algorithms.matd3.actor import Actor
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "configs" / "cooperative.yaml"


def _build() -> tuple[MultiMicrogridEnv, object]:
    base = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    base["_project_root"] = str(ROOT)
    spec = get_algorithm_spec("crossgate_actor")
    env = MultiMicrogridEnv(spec.build_environment_config(base), split="train")
    config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    return env, spec.create(env, config, seed=7)


def test_crossgate_plugin_uses_explicit_cross_microgrid_interface() -> None:
    env, algorithm = _build()
    assert env.action_dim == 2
    assert env.ess_sharing_control_mode == "explicit_offer"
    assert env.ess_sharing_observation_mode == "cross_microgrid"
    assert env.ess_offer_projection == "peer_demand_cap"
    assert env.ess_p2p_enabled
    assert set(algorithm.actors) == set(env.agent_ids)
    for mg in env.agent_ids:
        actor = algorithm.actors[mg]
        assert isinstance(actor, CrossGateActorNetwork)
        assert {name: len(indices) for name, indices in actor.feature_groups.items()} == {
            "local": 27, "market": 9, "peer": 12,
        }


def test_crossgate_starts_as_identity_and_retains_gate_gradients() -> None:
    env, algorithm = _build()
    mg = env.agent_ids[0]
    actor = algorithm.actors[mg]
    observation = torch.randn(8, env.observation_dims[mg])
    actions = actor(observation)
    diagnostics = actor.gate_diagnostics()
    assert actions.shape == (8, 2)
    assert torch.all(actions <= 1.0) and torch.all(actions >= -1.0)
    assert diagnostics["local_gate_mean"] == 0.5
    assert diagnostics["peer_gate_mean"] == 0.5
    actions.square().mean().backward()
    assert actor.local_gate.weight.grad is not None
    assert actor.peer_gate.weight.grad is not None
    assert torch.count_nonzero(actor.local_gate.weight.grad).item() > 0
    assert torch.count_nonzero(actor.peer_gate.weight.grad).item() > 0


def test_battery_head_is_local_and_detached_from_offer_branch() -> None:
    env, algorithm = _build()
    mg = env.agent_ids[0]
    actor = algorithm.actors[mg]
    first = torch.randn(1, env.observation_dims[mg])
    second = torch.randn(1, env.observation_dims[mg])
    second[:, actor._local_indices] = first[:, actor._local_indices]
    actions = actor(torch.cat([first, second], dim=0))
    assert torch.allclose(actions[0, 0], actions[1, 0], atol=1e-7)

    actor.zero_grad(set_to_none=True)
    actor(torch.randn(8, env.observation_dims[mg]))[:, 1].sum().backward()
    battery_modules = (
        actor.battery_local_encoder, actor.battery_backbone, actor.battery_head,
    )
    for module in battery_modules:
        for parameter in module.parameters():
            assert parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0


def test_crossgate_parameter_count_is_close_to_plain_actor() -> None:
    env, algorithm = _build()
    mg = env.agent_ids[0]
    plain = Actor(env.observation_dims[mg], env.action_dim, hidden_dims=(128, 128))
    plain_count = sum(parameter.numel() for parameter in plain.parameters())
    crossgate_count = algorithm.actor_parameter_counts()[mg]
    assert abs(crossgate_count - plain_count) / plain_count < 0.05


def test_peer_demand_projection_caps_and_masks_impossible_offers() -> None:
    env, _ = _build()
    cap = env._peer_demand_offer_cap(
        "mg1",
        load_demands={"mg1": 100.0, "mg2": 7.0, "mg3": 80.0},
        charge_demands={"mg1": 0.0, "mg2": 3.0, "mg3": 0.0},
        prices={"mg1": (0.5, 0.3), "mg2": (0.4, 0.2), "mg3": (0.2, 0.1)},
    )
    # MG2 is price eligible and contributes 10 kW. MG3 cannot buy from MG1
    # because its grid purchase price is below MG1's sell price.
    assert np.isclose(cap, 10.0)

    base = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    base["_project_root"] = str(ROOT)
    base["p2p"]["enabled"] = False
    spec = get_algorithm_spec("crossgate_actor")
    disabled = MultiMicrogridEnv(spec.build_environment_config(base), split="test")
    disabled.reset(day=str(disabled.data.valid_days[0].date()), seed=5, deterministic=True)
    done = False
    saw_uncapped_offer = False
    while not done:
        actions = {
            mg: np.asarray([1.0, 1.0], dtype=np.float32) for mg in disabled.agent_ids
        }
        _, _, done, info = disabled.step(actions)
        for mg in disabled.agent_ids:
            plan = info["plans"][mg]
            saw_uncapped_offer |= float(plan["ess_p2p_offer_uncapped_kw"]) > 1e-9
            assert plan["ess_p2p_offer_kw"] == 0.0
            assert plan["sharing_offer_ratio"] == 0.0
            assert plan["sharing_action_masked"]
    assert saw_uncapped_offer


def test_crossgate_training_and_checkpoint_round_trip(tmp_path: Path) -> None:
    env, algorithm = _build()
    observations, reset_info = env.reset(seed=7, deterministic=True)
    state = reset_info["global_state"]
    metrics = {}
    for _ in range(5):
        actions = algorithm.select_actions(observations, noise_std=0.05)
        next_observations, rewards, done, info = env.step(actions)
        algorithm.replay.add(
            observations, state, actions, rewards[env.agent_ids[0]],
            next_observations, info["global_state_after"], done,
        )
        metrics = algorithm.update(batch_size=2)
        observations, state = next_observations, info["global_state_after"]
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["crossgate_local_gate_mean"])
    assert np.isfinite(metrics["crossgate_peer_gate_mean"])
    assert metrics["actor_parameter_count"] > 0
    assert set(metrics["crossgate_gate_diagnostics"]) == set(env.agent_ids)

    checkpoint = tmp_path / "crossgate.pt"
    algorithm.save(checkpoint)
    _, restored = _build()
    restored.load(checkpoint)
    expected = algorithm.select_actions(observations, noise_std=0.0)
    actual = restored.select_actions(observations, noise_std=0.0)
    for mg in env.agent_ids:
        assert np.allclose(actual[mg], expected[mg])
