from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.base import MultiAgentAlgorithm
from algorithms.matd3_transformer.actor import TransformerDispatchActor
from algorithms.matd3_transformer.algorithm import Matd3Transformer
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


def build_matd3_transformer():
    spec = get_algorithm_spec("matd3_transformer")
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(cfg))
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    return spec, env, settings, spec.create(env, settings, seed=42)


def test_matd3_transformer_registry_and_protocol():
    spec, env, _, model = build_matd3_transformer()
    assert spec.slug == "matd3_transformer"
    assert spec.display_name == "MATD3-Transformer"
    assert type(model) is Matd3Transformer
    assert model.action_dim == 2
    assert model.raw_action_dim == 2
    assert model.effective_action_dim == 3
    assert isinstance(model, MultiAgentAlgorithm)
    assert env.action_design == "single_dispatch_dr"


def test_actor_output_shape_and_finite():
    _, env, _, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        assert type(actor) is TransformerDispatchActor
        single_obs = torch.tensor(obs[mg], dtype=torch.float32)
        single_action = actor(single_obs)
        assert single_action.shape == (2,)
        assert torch.isfinite(single_action).all()

        batch_obs = single_obs.unsqueeze(0).repeat(8, 1)
        batch_actions = actor(batch_obs)
        assert batch_actions.shape == (8, 2)
        assert torch.isfinite(batch_actions).all()
        assert (batch_actions >= -1.0).all() and (batch_actions <= 1.0).all()


def test_horizon_embedding_and_forecast_tokens():
    _, env, _, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        assert actor.horizon_embedding.weight.shape == (4, actor.embed_dim)
        assert not torch.equal(actor.horizon_embedding.weight[0], actor.horizon_embedding.weight[3])
        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(8, 1)
        components = actor.forward_components(batch_obs)
        assert components["forecast_tokens"].shape == (8, 4, actor.embed_dim)


def test_real_local_residual_and_interaction():
    _, env, _, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        d_model = actor.embed_dim
        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(4, 1)
        components = actor.forward_components(batch_obs)
        assert components["fused"].shape[-1] == 3 * d_model
        assert torch.allclose(components["fused"][:, d_model:2 * d_model], components["local_rep"])
        assert torch.allclose(components["fused"][:, 2 * d_model:], components["interact"])
        assert not torch.allclose(
            components["fused"][:, d_model:2 * d_model], components["attended"][:, 2, :]
        )


def test_no_attention_scale_dependency():
    _, env, _, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        assert not hasattr(actor, "attention_scale")
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        components = actor.forward_components(t_obs)
        assert torch.allclose(components["dispatch_rep"], components["attended"][:, 0, :])


def test_gradient_flow_all_branches():
    _, env, _, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        action = actor(t_obs)
        model.actor_optimizers[mg].zero_grad()
        (action ** 2).sum().backward()
        assert actor.horizon_embedding.weight.grad is not None
        assert torch.isfinite(actor.horizon_embedding.weight.grad).all()
        for name, module in actor.named_children():
            for parameter_name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None, f"Module {name}.{parameter_name} missing grad"
                    assert torch.isfinite(parameter.grad).all()


def test_transformer_diagnostics():
    _, env, _, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        actor(torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0))
        diag = actor.last_diagnostics
        assert diag["dispatch_norm"] > 0.0
        assert diag["local_norm"] > 0.0
        assert diag["norm_ratio"] > 0.0


def test_save_and_load(tmp_path):
    spec, env, settings, model = build_matd3_transformer()
    path = tmp_path / "model.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["action_semantics"] == "single_dispatch_dr_v1"

    model2 = spec.create(env, settings, seed=43)
    model2.load(path)
    for mg in env.agent_ids:
        for p1, p2 in zip(model.actors[mg].parameters(), model2.actors[mg].parameters()):
            assert torch.equal(p1, p2)


def test_disaggregated_peer_tokens_and_layers():
    _, env, _, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        d_model = actor.embed_dim
        assert len(actor.transformer.layers) == 2
        assert actor.peer_id_embedding.weight.shape == (len(env.agent_ids), d_model)
        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(8, 1)
        components = actor.forward_components(batch_obs)
        assert components["peer_tokens"].shape == (8, len(actor.peers), d_model)
        assert components["tokens"].shape == (8, 10, d_model)
        p0 = actor.peer_id_embedding.weight[actor._peer_agent_indices[0]]
        p1 = actor.peer_id_embedding.weight[actor._peer_agent_indices[1]]
        assert not torch.equal(p0, p1)


def test_token_norm_balance():
    _, env, _, model = build_matd3_transformer()
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


def test_dr_features_are_part_of_transformer_inputs():
    _, env, _, model = build_matd3_transformer()
    expected = {"mg1": "silo_ratio", "mg2": "t_air", "mg3": "ev_soc"}
    for mg, feature in expected.items():
        actor = model.actors[mg]
        local_names = [actor.schema[int(i)] for i in actor._local_indices]
        assert feature in local_names
    mg2 = model.actors["mg2"]
    forecast_names = [mg2.schema[int(i)] for i in mg2._all_forecast_indices]
    assert "temperature_t+1" in forecast_names
    assert "forecast_valid_t+1" in forecast_names
