from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import pytest
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
    spec, env, settings, model = build_pnn_transformer()
    assert spec.slug == "pnn_transformer"
    assert spec.display_name == "PNN-Transformer"
    assert type(model) is PnnTransformer
    assert model.action_dim == 1
    assert isinstance(model, MultiAgentAlgorithm)


def test_actor_output_shape():
    """Verify single-observation and batch forward passes produce exact expected shapes."""
    spec, env, settings, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        # Single sample input: [obs_dim] -> [action_dim]
        single_obs = torch.tensor(obs[mg], dtype=torch.float32)
        single_action = actor(single_obs)
        assert single_action.shape == (model.action_dim,)
        assert -1.0 <= float(single_action[0].detach()) <= 1.0

        # Batch input: [B, obs_dim] -> [B, action_dim]
        batch_obs = single_obs.unsqueeze(0).repeat(8, 1)
        batch_actions = actor(batch_obs)
        assert batch_actions.shape == (8, model.action_dim)
        assert (batch_actions >= -1.0).all() and (batch_actions <= 1.0).all()


def test_output_finite():
    """Verify Actor outputs are strictly finite without NaN / Inf."""
    spec, env, settings, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(16, 1)
        actions = actor(batch_obs)
        assert torch.isfinite(actions).all(), f"{mg}: actions contain non-finite values"


def test_tri_stream_shapes():
    """Verify internal shapes of Macro Transformer, Micro PNN Products, and Local Residual streams."""
    spec, env, settings, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        d_model = actor.embed_dim
        assert d_model == 32

        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(6, 1)
        components = actor.forward_components(batch_obs)

        # 1. Macro Transformer stream
        assert components["tokens"].shape == (6, 10, d_model)  # 1 dispatch + 1 hour + 1 local + 4 forecast + 2 peer + 1 market
        assert components["attended"].shape == (6, 10, d_model)
        assert components["dispatch_rep"].shape == (6, d_model)

        # 2. Micro PNN Product stream (temporal, spatial, market: 3 * d_model)
        assert components["product_signals"].shape == (6, 3 * d_model)

        # 3. Local Constraint Residual stream
        assert components["local_rep"].shape == (6, d_model)

        # 4. Tri-Stream Fusion: 32 + 96 + 32 = 160
        assert components["fused"].shape == (6, 5 * d_model)
        assert components["action"].shape == (6, 1)


def test_pnn_product_semantics():
    """Verify that PNN signed linear encoders preserve sign for bilinear interactions."""
    spec, env, settings, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    mg = env.agent_ids[0]
    actor = model.actors[mg]

    t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
    components1 = actor.forward_components(t_obs)

    # Invert local features sign (simulating switching from generation surplus to deficit)
    t_obs_inverted = t_obs.clone()
    t_obs_inverted[0, actor._local_indices] *= -1.0
    components2 = actor.forward_components(t_obs_inverted)

    # The product signals should differ significantly due to sign-sensitive Hadamard interactions
    diff = (components1["product_signals"] - components2["product_signals"]).abs().max()
    assert float(diff.detach()) > 1e-4, "PNN product signals did not respond to signed local state change"


def test_transformer_depth_and_norm():
    """Verify 2-layer Pre-LN Transformer structure with Final LayerNorm."""
    spec, env, settings, model = build_pnn_transformer()
    for mg in env.agent_ids:
        actor = model.actors[mg]
        assert len(actor.transformer.layers) == 2
        assert isinstance(actor.transformer.norm, torch.nn.LayerNorm)
        assert actor.transformer.norm.normalized_shape == (actor.embed_dim,)


def test_gradient_flow_across_all_streams():
    """Verify backpropagation gradients flow through Transformer, PNN products, local bypass, and backbone."""
    spec, env, settings, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        action = actor(t_obs)
        loss = (action ** 2).sum()

        model.actor_optimizers[mg].zero_grad()
        loss.backward()

        branches = {
            name: sum(p.grad.detach().norm(2).item() for p in params if p.grad is not None)
            for name, params in model._actor_branch_parameters[mg].items()
        }

        # Gradients must flow to all 3 streams
        assert branches["transformer"] > 0.0, f"{mg}: zero grad in transformer stream"
        assert branches["pnn_encoders"] > 0.0, f"{mg}: zero grad in PNN product encoders"
        assert branches["local_tokenizer"] > 0.0, f"{mg}: zero grad in local tokenizer / residual"
        assert branches["backbone"] > 0.0, f"{mg}: zero grad in dispatch backbone"
        assert branches["head"] > 0.0, f"{mg}: zero grad in dispatch head"


def test_save_and_load(tmp_path):
    """Verify checkpoint save and load integrity."""
    spec, env, settings, model = build_pnn_transformer()
    ckpt_path = tmp_path / "pnn_transformer_test.pt"
    model.save(ckpt_path)
    assert ckpt_path.exists()

    model2 = spec.create(env, settings, seed=42)
    model2.load(ckpt_path)

    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act1 = model.actors[mg](t_obs)
            act2 = model2.actors[mg](t_obs)
        assert torch.allclose(act1, act2, atol=1e-6)


def test_token_norm_balance():
    """Verify Token Norm Balancing: all 10 tokens entering self-attention have Euclidean norm ~ sqrt(D)."""
    spec, env, settings, model = build_pnn_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        components = actor.forward_components(t_obs)
        tokens = components["tokens"][0]  # [10, 32]
        token_norms = tokens.norm(dim=-1)

        # Each token norm must be in balanced range
        for i, norm_val in enumerate(token_norms):
            val = float(norm_val.detach())
            assert 3.5 <= val <= 7.0, f"Token {i} norm out of expected range: {val}"

        # Token norm ratio (max/min) must be strictly bounded below 1.6
        ratio = float((token_norms.max() / token_norms.min()).detach())
        assert ratio < 1.6, f"Token norm ratio too high: {ratio}"

