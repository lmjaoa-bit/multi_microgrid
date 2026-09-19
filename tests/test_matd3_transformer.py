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
        assert actor.horizon_embedding.weight.shape == (actor.forecast_horizon, actor.embed_dim)
        assert not torch.equal(actor.horizon_embedding.weight[0], actor.horizon_embedding.weight[-1])
        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(8, 1)
        components = actor.forward_components(batch_obs)
        assert components["forecast_tokens"].shape == (8, actor.forecast_horizon, actor.embed_dim)


def test_shared_ess_dr_decoder_ablation():
    _, env, _, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        d_model = actor.embed_dim
        batch_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(4, 1)
        components = actor.forward_components(batch_obs)

        assert components["fused"].shape[-1] == 5 * d_model
        assert components["ess_action"].shape == (4, 1)
        assert components["dr_action"].shape == (4, 1)
        assert torch.allclose(
            components["action"],
            torch.cat([components["ess_action"], components["dr_action"]], dim=-1),
        )
        assert actor.dispatch_head[0].out_features == 2
        assert not hasattr(actor, "ess_head")
        assert not hasattr(actor, "dr_head")
        assert actor.native_dr_action_head
        assert not actor.native_dr_split_heads


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


def test_transformer_diagnostics_can_be_disabled_without_changing_action():
    _, env, _, model = build_matd3_transformer()
    observations, _ = env.reset(seed=42, deterministic=True)
    actor = model.actors[env.agent_ids[0]]
    observation = torch.tensor(
        observations[env.agent_ids[0]], dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        expected = actor(observation, collect_diagnostics=True)
        diagnostics = dict(actor.last_diagnostics)
        actual = actor(observation, collect_diagnostics=False)

    assert torch.equal(actual, expected)
    assert actor.last_diagnostics == diagnostics


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
        assert components["tokens"].shape == (8, actor.num_tokens, d_model)
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
        dr_names = [actor.schema[int(i)] for i in actor._dr_indices]
        assert feature in dr_names
    mg2 = model.actors["mg2"]
    forecast_names = [mg2.schema[int(i)] for i in mg2._all_forecast_indices]
    assert "temperature_t+1" in forecast_names
    assert "forecast_valid_t+1" in forecast_names


def test_spatiotemporal_microgrid_time_tokens():
    _, env, _, model = build_matd3_transformer()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg]
        batch = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0).repeat(3, 1)
        components = actor.forward_components(batch)
        peer_count = len(actor.peers)

        assert components["spatiotemporal_tokens"].shape == (
            3,
            actor.forecast_horizon,
            1 + peer_count,
            actor.embed_dim,
        )
        assert components["peer_future_tokens"].shape == (
            3,
            actor.forecast_horizon,
            peer_count,
            actor.embed_dim,
        )
        assert actor.num_future_tokens == actor.forecast_horizon * (1 + peer_count)
        assert actor.num_tokens == 6 + peer_count + actor.num_future_tokens

        # Tokens for different peers at the same horizon must have separate
        # spatial identities rather than being pre-mixed into one time token.
        first_horizon = components["peer_future_tokens"][:, 0]
        assert first_horizon.shape[1] == peer_count
        if peer_count > 1:
            assert not torch.equal(first_horizon[:, 0], first_horizon[:, 1])



def test_invalid_future_spatiotemporal_tokens_are_attention_masked():
    _, env, _, model = build_matd3_transformer()
    observations, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg].eval()
        horizon = actor.forecast_horizon
        schema = list(actor.schema)
        assert horizon >= 4
        sample = torch.tensor(observations[mg], dtype=torch.float32)

        # Simulate an episode boundary in a batch: the first sample has only
        # three valid future hours, while the second has two.
        batch = sample.unsqueeze(0).repeat(2, 1)
        for k in range(4, horizon + 1):
            batch[:, schema.index(f"forecast_valid_t+{k}")] = 0.0
        batch[1, schema.index("forecast_valid_t+3")] = 0.0

        components = actor.forward_components(batch, collect_diagnostics=False)
        mask = components["attention_padding_mask"]
        prefix = 5 + len(actor.peers)
        entities = 1 + len(actor.peers)
        assert mask.shape == (2, actor.num_tokens)
        assert mask.dtype == torch.bool
        assert not mask[:, :prefix].any()
        assert not mask[:, -1].any()  # market token
        assert not mask[0, prefix:prefix + 3 * entities].any()
        assert mask[0, prefix + 3 * entities:-1].all()
        assert not mask[1, prefix:prefix + 2 * entities].any()
        assert mask[1, prefix + 2 * entities:-1].all()

        # Invalid local AND peer forecasts may contain arbitrary data without
        # changing the ESS or DR action; their spatial/time embeddings must
        # also be excluded from attention.
        original = batch[0].clone()
        changed = original.clone()
        for k in range(4, horizon + 1):
            suffix = f"_t+{k}"
            for i, name in enumerate(schema):
                if name.endswith(suffix) and not name.startswith("forecast_valid_"):
                    changed[i] += 25.0 + k
        with torch.no_grad():
            expected = actor(original, collect_diagnostics=False)
            actual = actor(changed, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_invalid_future_inputs_have_zero_action_gradients():
    _, env, _, model = build_matd3_transformer()
    observations, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = model.actors[mg].eval()
        schema = list(actor.schema)
        obs = torch.tensor(observations[mg], dtype=torch.float32)
        invalid_step = actor.forecast_horizon
        obs[schema.index(f"forecast_valid_t+{invalid_step}")] = 0.0
        obs.requires_grad_(True)
        actor(obs, collect_diagnostics=False).sum().backward()
        assert obs.grad is not None
        for index in actor._all_forecast_indices.tolist():
            if schema[index].endswith(f"_t+{invalid_step}"):
                assert obs.grad[index].item() == 0.0
        for index in actor._all_peer_future_indices.tolist():
            if schema[index].endswith(f"_t+{invalid_step}"):
                assert obs.grad[index].item() == 0.0


def test_ess_dr_behavior_diagnostics_aggregate_hourly_and_terminal_penalty():
    from evaluation.evaluate_matd3 import build_dispatch_behavior_diagnostics

    def hourly_row(day: str, hour: int, ess: float, dr: float) -> dict:
        return {
            "date": day,
            "hour": hour,
            "mg1_raw_ess_action": ess,
            "mg1_raw_dr_action": dr,
            "mg1_battery_charge_kw": 2.0,
            "mg1_battery_discharge_kw": 3.0,
            "mg1_ess_to_local_load_kw": 1.0,
            "mg1_ess_p2p_offer_kw": 3.0,
            "mg1_ess_p2p_sell_kw": 2.0,
            "mg1_ess_p2p_unmatched_kw": 1.0,
            "mg1_flexible_load_kw": 4.0,
            "mg1_soc": 0.4,
            "mg1_ess_projected": 0,
            "mg1_dr_projected": 1,
        }

    hourly = {
        "day1": [hourly_row("day1", 0, 0.96, 0.5)],
        "day2": [hourly_row("day2", 0, 0.2, -0.96)],
    }
    daily = [
        {
            "soc_statistics_by_microgrid": {
                "mg1": {"terminal_soc": 0.4, "terminal_soc_target": 0.5}
            },
            "economic_breakdown_by_microgrid": {
                "mg1": {"terminal_soc_penalty": 7.0}
            },
        },
        {
            "soc_statistics_by_microgrid": {
                "mg1": {"terminal_soc": 0.3, "terminal_soc_target": 0.5}
            },
            "economic_breakdown_by_microgrid": {
                "mg1": {"terminal_soc_penalty": 9.0}
            },
        },
    ]
    diagnostics = build_dispatch_behavior_diagnostics(
        hourly, daily, ["mg1"], delta_t_hours=0.5
    )
    mg = diagnostics["by_microgrid"]["mg1"]
    assert diagnostics["days_count"] == 2
    assert diagnostics["steps_count"] == 2
    assert np.isclose(mg["ess_raw_action_saturation_rate"], 0.5)
    assert np.isclose(mg["dr_raw_action_saturation_rate"], 0.5)
    assert np.isclose(mg["ess_charge_energy_kwh_per_day"], 1.0)
    assert np.isclose(mg["ess_p2p_offer_energy_kwh_per_day"], 1.5)
    assert np.isclose(mg["ess_p2p_accepted_energy_kwh_per_day"], 1.0)
    assert np.isclose(mg["terminal_soc_mean"], 0.35)
    assert np.isclose(mg["terminal_soc_penalty_mean_per_day"], 8.0)
    assert len(mg["hourly_profile"]) == 1
    assert np.isclose(mg["hourly_profile"][0]["dr_projection_rate"], 1.0)

def test_peer_energy_summary_replaces_raw_peer_soc():
    _, env, _, model = build_matd3_transformer()
    assert env.ess_sharing_observation_mode == "peer_energy_summary"
    for mg in env.agent_ids:
        schema = env.obs_builder.schema(mg)
        actor = model.actors[mg]
        for peer in actor.peers:
            assert f"{peer}_soc" not in schema
            for suffix in (
                "energy_balance", "deficit", "surplus",
                "charge_flex", "discharge_flex", "previous_p2p_net",
            ):
                assert f"{peer}_{suffix}" in schema
            for k in range(1, actor.forecast_horizon + 1):
                assert f"{peer}_energy_balance_t+{k}" in schema
                assert f"{peer}_deficit_t+{k}" in schema
                assert f"{peer}_surplus_t+{k}" in schema
        assert actor.peer_representation == "energy_summary"
        assert actor.peer_feature_dim == 6
        assert actor.peer_encoder[0].in_features == 6
