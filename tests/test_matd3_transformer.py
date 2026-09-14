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
        single_obs = torch.tensor(obs[mg], dtype=torch.float32)
        single_action = actor(single_obs)
        assert single_action.shape == (model.action_dim,)

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
    """Test 3: Eight-hour horizon embedding and forecast-token shape."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        d_model = actor.embed_dim
        assert actor.forecast_horizon == 8
        assert actor.horizon_embedding.weight.shape == (8, d_model)

        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(8, 1)
        components = actor.forward_components(batch_obs)
        assert components["forecast_tokens"].shape == (8, 8, d_model)


def test_invalid_forecast_tokens_do_not_affect_dispatch_representation():
    """Invalid forecast steps must be excluded from dispatch-token attention."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)

    for mg in env.agent_ids:
        actor = model.actors[mg]
        schema_map = {name: i for i, name in enumerate(actor.schema)}
        base = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        perturbed = base.clone()

        # Mark the last two forecast steps invalid, then change only their values.
        # A correct key-padding mask makes the dispatch representation invariant.
        masked_steps = range(actor.forecast_horizon - 1, actor.forecast_horizon + 1)
        for k in masked_steps:
            valid_name = f"forecast_valid_t+{k}"
            assert valid_name in schema_map
            base[:, schema_map[valid_name]] = 0.0
            perturbed[:, schema_map[valid_name]] = 0.0

            feature_names = [f"load_t+{k}", f"pv_t+{k}", f"wind_t+{k}", f"price_t+{k}"]
            feature_names.extend(f"{peer}_net_load_t+{k}" for peer in actor.peers)
            for feature_name in feature_names:
                perturbed[:, schema_map[feature_name]] += 100.0

        base_components = actor.forward_components(base)
        perturbed_components = actor.forward_components(perturbed)

        for components in (base_components, perturbed_components):
            padding_mask = components["attention_padding_mask"]
            assert padding_mask.shape == (1, actor.num_tokens)
            assert padding_mask[0, 3 + actor.forecast_horizon - 2]
            assert padding_mask[0, 3 + actor.forecast_horizon - 1]

        assert torch.allclose(
            base_components["dispatch_rep"],
            perturbed_components["dispatch_rep"],
            atol=1e-6,
            rtol=1e-6,
        ), f"{mg}: invalid forecast tokens leaked into dispatch attention"


def test_test4_horizon_embedding_temporal_identity():
    """Test 4: t+1 and t+8 possess distinct temporal embeddings."""
    spec, env, settings, model = build_matd3_transformer()
    for mg in env.agent_ids:
        actor = model.actors[mg]
        emb_t1 = actor.horizon_embedding.weight[0]
        emb_t8 = actor.horizon_embedding.weight[7]
        assert not torch.equal(emb_t1, emb_t8), "t+1 and t+8 share identical embedding weights"


def test_test5_real_local_residual():
    """Test 5: Verify true local physical residual bypassing the Transformer and bilinear interact."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        d_model = actor.embed_dim
        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(4, 1)
        components = actor.forward_components(batch_obs)

        assert components["fused"].shape[-1] == 3 * d_model

        fused_local_part = components["fused"][:, d_model: 2 * d_model]
        assert torch.allclose(fused_local_part, components["local_rep"]), (
            "Residual branch does not match pre-Transformer local_encoder output"
        )

        fused_interact_part = components["fused"][:, 2 * d_model:]
        assert torch.allclose(fused_interact_part, components["interact"]), (
            "Interact branch does not match actor interact output"
        )

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

        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        actor(t_obs)
        components = actor.forward_components(t_obs)
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

        assert actor.horizon_embedding.weight.grad is not None
        assert torch.isfinite(actor.horizon_embedding.weight.grad).all()

        for name, module in actor.named_children():
            for p_name, p in module.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, f"Module {name}.{p_name} missing grad"
                    assert torch.isfinite(p.grad).all(), f"Module {name}.{p_name} nonfinite grad"


def test_matd3_transformer_diagnostics():
    """Verify representation norm and forecast-validity diagnostics are recorded."""
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
        assert "forecast_valid_fraction" in diag
        assert diag["dispatch_norm"] > 0.0
        assert diag["local_norm"] > 0.0
        assert diag["norm_ratio"] > 0.0
        assert 0.0 <= diag["forecast_valid_fraction"] <= 1.0


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
    """Verify 2-layer Transformer, disaggregated peer tokens, and 8h sequence length."""
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
        # Sequence: 1 dispatch + 1 hour + 1 local + 8 forecast + 2 peers + 1 market = 14 tokens
        assert components["tokens"].shape == (8, 14, d_model)

        p0_emb = actor.peer_id_embedding.weight[actor._peer_agent_indices[0]]
        p1_emb = actor.peer_id_embedding.weight[actor._peer_agent_indices[1]]
        assert not torch.equal(p0_emb, p1_emb), "Peer spatial embeddings must be distinct"

        loss = components["action"].pow(2).sum()
        model.actor_optimizers[mg].zero_grad()
        loss.backward()
        assert actor.peer_id_embedding.weight.grad is not None
        assert torch.isfinite(actor.peer_id_embedding.weight.grad).all()


def test_token_norm_balance():
    """Verify Token Norm Balancing for all 14 valid tokens at episode start."""
    spec, env, settings, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    expected_norm = float(np.sqrt(32))  # ~5.657
    for mg in env.agent_ids:
        actor = model.actors[mg]
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        components = actor.forward_components(t_obs)
        tokens = components["tokens"][0]  # [14, 32] at t=0, all 8 forecast steps are valid
        token_norms = tokens.norm(dim=-1)

        for i, norm_val in enumerate(token_norms):
            val = float(norm_val.detach())
            assert 3.5 <= val <= 7.0, f"Token {i} norm out of expected range: {val}"

        ratio = float((token_norms.max() / token_norms.min()).detach())
        assert ratio < 1.6, f"Token norm ratio too high: {ratio}"
