from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import torch
import yaml

from algorithms.base import MultiAgentAlgorithm
from algorithms.pnn_transformer.actor import PnnTransformerDispatchActor
from algorithms.pnn_transformer.algorithm import PnnTransformer
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


def build_pnn_transformer():
    spec = get_algorithm_spec("pnn_transformer")
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(cfg))
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    return spec, env, settings, spec.create(env, settings, seed=42)


def test_pnn_transformer_registry_and_protocol():
    spec, env, _, model = build_pnn_transformer()
    assert spec.slug == "pnn_transformer"
    assert spec.display_name == "PNN-Transformer"
    assert type(model) is PnnTransformer
    assert model.action_dim == 2
    assert model.raw_action_dim == 2
    assert model.effective_action_dim == 3
    assert isinstance(model, MultiAgentAlgorithm)
    assert env.action_design == "single_dispatch_dr"


def test_actor_output_shape_and_finite():
    _, env, _, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        assert type(actor) is PnnTransformerDispatchActor
        single = torch.tensor(obs[mg], dtype=torch.float32)
        action = actor(single)
        assert action.shape == (2,)
        assert torch.isfinite(action).all()
        batch = actor(single.unsqueeze(0).repeat(8, 1))
        assert batch.shape == (8, 2)
        assert torch.isfinite(batch).all()
        assert (batch >= -1.0).all() and (batch <= 1.0).all()


def test_tri_stream_shapes():
    _, env, _, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        d_model = actor.embed_dim
        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(6, 1)
        components = actor.forward_components(batch_obs)
        assert components["tokens"].shape == (6, 10, d_model)
        assert components["attended"].shape == (6, 10, d_model)
        assert components["dispatch_rep"].shape == (6, d_model)
        assert components["product_signals"].shape == (6, 3 * d_model)
        assert components["local_rep"].shape == (6, d_model)
        assert components["fused"].shape == (6, 5 * d_model)
        assert components["action"].shape == (6, 2)


def test_pnn_product_semantics():
    _, env, _, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    mg = env.agent_ids[0]
    actor = model.actors[mg]
    first = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
    components1 = actor.forward_components(first)
    inverted = first.clone()
    inverted[0, actor._local_indices] *= -1.0
    components2 = actor.forward_components(inverted)
    diff = (components1["product_signals"] - components2["product_signals"]).abs().max()
    assert float(diff.detach()) > 1e-4


def test_transformer_depth_and_norm():
    _, env, _, model = build_pnn_transformer()
    for mg in env.agent_ids:
        actor = model.actors[mg]
        assert len(actor.transformer.layers) == 2
        assert isinstance(actor.transformer.norm, torch.nn.LayerNorm)
        assert actor.transformer.norm.normalized_shape == (actor.embed_dim,)


def test_gradient_flow_across_all_streams():
    _, env, _, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        action = actor(torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0))
        model.actor_optimizers[mg].zero_grad()
        (action ** 2).sum().backward()
        branches = {
            name: sum(
                parameter.grad.detach().norm(2).item()
                for parameter in params
                if parameter.grad is not None
            )
            for name, params in model._actor_branch_parameters[mg].items()
        }
        assert branches["transformer"] > 0.0
        assert branches["pnn_encoders"] > 0.0
        assert branches["local_tokenizer"] > 0.0
        assert branches["backbone"] > 0.0
        assert branches["head"] > 0.0


def test_save_and_load(tmp_path):
    spec, env, settings, model = build_pnn_transformer()
    path = tmp_path / "pnn_transformer_test.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["action_semantics"] == "single_dispatch_dr_v1"

    model2 = spec.create(env, settings, seed=42)
    model2.load(path)
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act1 = model.actors[mg](t_obs)
            act2 = model2.actors[mg](t_obs)
        assert torch.allclose(act1, act2, atol=1e-6)


def test_token_norm_balance():
    _, env, _, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        components = actor.forward_components(
            torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        )
        norms = components["tokens"][0].norm(dim=-1)
        for value in norms:
            assert 3.5 <= float(value.detach()) <= 7.0
        assert float((norms.max() / norms.min()).detach()) < 1.6


def test_dr_features_are_part_of_both_hybrid_streams():
    _, env, _, model = build_pnn_transformer()
    expected = {"mg1": "silo_ratio", "mg2": "t_air", "mg3": "ev_soc"}
    for mg, feature in expected.items():
        actor = model.actors[mg]
        local_names = [actor.schema[int(i)] for i in actor._local_indices]
        assert feature in local_names
        assert actor.local_encoder[0].in_features == actor.pnn_local_encoder[0].in_features
    mg2 = model.actors["mg2"]
    forecast_names = [mg2.schema[int(i)] for i in mg2._all_forecast_indices]
    assert "temperature_t+1" in forecast_names
    assert "forecast_valid_t+1" in forecast_names
    assert mg2.pnn_temporal_encoder[0].in_features == len(forecast_names)
