from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
DR_VARIANTS = ("pnn", "matd3_transformer", "pnn_transformer")


def build_variant(slug: str):
    spec = get_algorithm_spec(slug)
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(cfg))
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    settings.setdefault("logging", {})["diagnostic_interval"] = 2
    return spec, env, settings, spec.create(env, settings, seed=42)


@pytest.mark.parametrize("slug", DR_VARIANTS)
def test_variant_uses_dr_action_space_and_joint_projection(slug: str):
    _, env, _, model = build_variant(slug)
    assert env.action_design == "single_dispatch_dr"
    assert env.action_dim == 2
    assert model.action_dim == 2
    assert model.raw_action_dim == 2
    assert model.effective_action_dim == 3

    observations, info = env.reset(seed=42, deterministic=True)
    actions = model.select_actions(observations, noise_std=0.0)
    assert set(actions) == set(env.agent_ids)
    for mg in env.agent_ids:
        assert actions[mg].shape == (2,)
        assert np.isfinite(actions[mg]).all()
        assert (actions[mg] >= -1.0).all() and (actions[mg] <= 1.0).all()

    state = torch.as_tensor(info["global_state"], dtype=torch.float32).unsqueeze(0)
    raw = torch.as_tensor(
        np.stack([actions[mg] for mg in env.agent_ids]), dtype=torch.float32
    ).unsqueeze(0)
    effective, plans = model.project_actions(state, raw)
    assert effective.shape == (1, len(env.agent_ids), 3)
    assert len(plans) == len(env.agent_ids)
    assert torch.isfinite(effective).all()

    next_obs, rewards, terminal, step_info = env.step(actions)
    assert set(next_obs) == set(env.agent_ids)
    assert set(rewards) == set(env.agent_ids)
    assert step_info["action_design"] == "single_dispatch_dr"


@pytest.mark.parametrize("slug", DR_VARIANTS)
def test_variant_replay_critic_and_actor_update_use_projected_dr_actions(slug: str):
    _, env, _, model = build_variant(slug)
    observations, info = env.reset(seed=42, deterministic=True)
    actions = model.select_actions(observations, noise_std=0.0)
    next_obs, rewards, terminal, step_info = env.step(actions)
    model.replay.add(
        observations,
        info["global_state"],
        actions,
        rewards[env.agent_ids[0]],
        next_obs,
        step_info["global_state_after"],
        terminal,
    )

    first = model.update(1)
    second = model.update(1)
    assert first == {}
    assert np.isfinite(second["critic_loss"])
    assert np.isfinite(second["actor_loss"])


@pytest.mark.parametrize("slug", ("matd3_transformer", "pnn_transformer"))
def test_transformer_variants_receive_dr_state_and_hvac_forecast_gradients(slug: str):
    _, env, _, model = build_variant(slug)
    observations, _ = env.reset(seed=42, deterministic=True)
    dr_features = {"mg1": "silo_ratio", "mg2": "t_air", "mg3": "ev_soc"}

    for mg, feature in dr_features.items():
        schema = list(env.obs_builder.schema(mg))
        obs = torch.tensor(observations[mg], dtype=torch.float32, requires_grad=True)
        action = model.actors[mg](obs)
        action.sum().backward()
        grad = obs.grad[schema.index(feature)]
        assert torch.isfinite(grad)
        assert float(grad.abs()) > 0.0, f"{slug}/{mg} ignores {feature}"

    schema = list(env.obs_builder.schema("mg2"))
    assert "temperature_t+1" in schema
    obs = torch.tensor(observations["mg2"], dtype=torch.float32, requires_grad=True)
    model.actors["mg2"](obs).sum().backward()
    temp_grad = obs.grad[schema.index("temperature_t+1")]
    assert torch.isfinite(temp_grad)
    assert float(temp_grad.abs()) > 0.0, f"{slug}/mg2 ignores future HVAC temperature"


@pytest.mark.parametrize("slug", DR_VARIANTS)
def test_variant_checkpoint_marks_dr_action_semantics(slug: str, tmp_path: Path):
    _, env, settings, model = build_variant(slug)
    path = tmp_path / f"{slug}.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["algorithm_slug"] == slug
    assert payload["action_semantics"] == "single_dispatch_dr_v1"

    reloaded = get_algorithm_spec(slug).create(env, settings, seed=43)
    reloaded.load(path)
