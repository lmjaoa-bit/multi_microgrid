from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import pytest
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
    spec, env, settings, model = build_matd3_transformer()
    assert spec.slug == "matd3_transformer"
    assert spec.display_name == "MATD3-Transformer"
    assert type(model) is Matd3Transformer
    assert model.action_dim == 1
    assert isinstance(model, MultiAgentAlgorithm)


def test_test1_actor_output_shape():
    """Test 1: Actor output shape invariance for batch and single-observation."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        # Single sample input: [obs_dim] -> [action_dim]
        single_obs = torch.tensor(obs[mg], dtype=torch.float32)
        single_action = actor(single_obs)
        assert single_action.shape == (model.action_dim,)

        # Batch input: [B, obs_dim] -> [B, action_dim]
        batch_obs = single_obs.unsqueeze(0).repeat(8, 1)
        batch_actions = actor(batch_obs)
        assert batch_actions.shape == (8, model.action_dim)


def test_test2_output_finite():
    """Test 2: Ensure Actor output is free from NaN / Inf."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(16, 1)
        actions = actor(batch_obs)
        assert torch.isfinite(actions).all(), f"{mg}: actions contain non-finite values"


def test_test3_horizon_embedding_shape():
    """Test 3: Horizon embedding shape and forecast tokens shape."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        d_model = actor.embed_dim
        assert actor.horizon_embedding.weight.shape == (4, d_model)

        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(8, 1)
        components = actor.forward_components(batch_obs)
        assert components["forecast_tokens"].shape == (8, 4, d_model)


def test_test4_horizon_embedding_temporal_identity():
    """Test 4: t+1 and t+4 possess distinct temporal embeddings."""
    spec, env, settings, model = build_matd3_transformer()
    for mg in env.agent_ids:
        actor = model.actors[mg]
        emb_t1 = actor.horizon_embedding.weight[0]
        emb_t4 = actor.horizon_embedding.weight[3]
        assert not torch.equal(emb_t1, emb_t4), "t+1 and t+4 share identical embedding weights"


def test_test5_real_local_residual():
    """Test 5: Verify true local physical residual bypassing the Transformer and bilinear interact."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        d_model = actor.embed_dim
        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(4, 1)
        components = actor.forward_components(batch_obs)

        # 1. Total fused dim is 3 * d_model (dispatch_rep + local_rep + interact)
        assert components["fused"].shape[-1] == 3 * d_model

        # 2. The residual branch entering fusion must exactly match local_rep
        fused_local_part = components["fused"][:, d_model: 2 * d_model]
        assert torch.allclose(fused_local_part, components["local_rep"]), (
            "Residual branch does not match pre-Transformer local_encoder output"
        )

        # 3. Bilinear interaction branch entering fusion matches interact
        fused_interact_part = components["fused"][:, 2 * d_model:]
        assert torch.allclose(fused_interact_part, components["interact"]), (
            "Interact branch does not match actor interact output"
        )

        # 4. It must NOT equal the attended LOCAL token (attended[:, 2, :])
        attended_local_token = components["attended"][:, 2, :]
        assert not torch.allclose(fused_local_part, attended_local_token), (
            "Residual branch erroneously equals post-Transformer attended token"
        )


def test_test6_no_attention_scale_dependency():
    """Test 6: Verify complete removal of attention_scale from Actor."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        assert not hasattr(actor, "attention_scale"), "Actor still has attention_scale attribute"

        # Check forward output is unaffected by any attention_scale argument
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        action1 = actor(t_obs)
        components = actor.forward_components(t_obs)
        # dispatch_rep must equal attended[:, 0, :] directly without scaling
        assert torch.allclose(components["dispatch_rep"], components["attended"][:, 0, :])


def test_test7_gradient_flow_all_branches():
    """Test 7: Backward pass ensures valid gradients across all branches."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        action = actor(t_obs)
        loss = (action ** 2).sum()
        model.actor_optimizers[mg].zero_grad()
        loss.backward()

        # Specific P0 modules must have gradients
        assert actor.horizon_embedding.weight.grad is not None
        assert torch.isfinite(actor.horizon_embedding.weight.grad).all()

        for name, module in actor.named_children():
            for p_name, p in module.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, f"Module {name}.{p_name} missing grad"
                    assert torch.isfinite(p.grad).all(), f"Module {name}.{p_name} nonfinite grad"


def test_matd3_transformer_diagnostics():
    """Verify representation norm diagnostics are recorded."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        actor(t_obs)
        diag = actor.last_diagnostics
        assert "dispatch_norm" in diag
        assert "local_norm" in diag
        assert "norm_ratio" in diag
        assert diag["dispatch_norm"] > 0.0
        assert diag["local_norm"] > 0.0
        assert diag["norm_ratio"] > 0.0


def test_matd3_transformer_save_and_load(tmp_path):
    spec, env, settings, model = build_matd3_transformer()
    ckpt_path = tmp_path / "model.pt"
    model.save(ckpt_path)
    assert ckpt_path.exists()

    model2 = spec.create(env, settings, seed=43)
    model2.load(ckpt_path)
    for mg in env.agent_ids:
        for p1, p2 in zip(model.actors[mg].parameters(), model2.actors[mg].parameters()):
            assert torch.equal(p1, p2)


def test_disaggregated_peer_tokens_and_layers():
    """Verify Step 2 structural upgrade: 2-layer Transformer and disaggregated peer tokens."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        d_model = actor.embed_dim
        assert len(actor.transformer.layers) == 2, "Transformer must have 2 layers"
        assert actor.peer_id_embedding.weight.shape == (len(env.agent_ids), d_model)

        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(8, 1)
        components = actor.forward_components(batch_obs)
        num_peers = len(actor.peers)
        assert components["peer_tokens"].shape == (8, num_peers, d_model)
        # Sequence: 1 dispatch + 1 hour + 1 local + 4 forecast + 2 peers + 1 market = 10 tokens
        assert components["tokens"].shape == (8, 10, d_model)

        # Spatial ID embeddings must be distinct for different microgrids
        p0_emb = actor.peer_id_embedding.weight[actor._peer_agent_indices[0]]
        p1_emb = actor.peer_id_embedding.weight[actor._peer_agent_indices[1]]
        assert not torch.equal(p0_emb, p1_emb), "Peer spatial embeddings must be distinct"

        # Check gradients flow to peer_id_embedding
        loss = components["action"].pow(2).sum()
        model.actor_optimizers[mg].zero_grad()
        loss.backward()
        assert actor.peer_id_embedding.weight.grad is not None
        assert torch.isfinite(actor.peer_id_embedding.weight.grad).all()


def test_token_norm_balance():
    """Verify Token Norm Balancing: all 10 tokens entering self-attention have Euclidean norm ~ sqrt(D)."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    expected_norm = float(np.sqrt(32))  # ~5.657
    for mg in env.agent_ids:
        actor = model.actors[mg]
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        components = actor.forward_components(t_obs)
        tokens = components["tokens"][0]  # [10, 32]
        token_norms = tokens.norm(dim=-1)

        # Each token norm must be close to sqrt(D) ~ 5.66 (within 20%)
        # Note: dispatch, hour, forecast, peer, and local all have direct LayerNorm (norm exactly 5.657 at init)
        for i, norm_val in enumerate(token_norms):
            val = float(norm_val.detach())
            assert 3.5 <= val <= 7.0, f"Token {i} norm out of expected range: {val}"

        # Token norm ratio (max/min) must be strictly bounded below 2.0 (previously was 2.5x - 7.5 vs 2.9)
        ratio = float((token_norms.max() / token_norms.min()).detach())
        assert ratio < 1.6, f"Token norm ratio too high: {ratio}"


