from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.matd3_hierarchical_transformer.actor import (
    HierarchicalTransformerActor,
)
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


def _env_cfg():
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    cfg["reward"] = {
        "semantics_version": "generalized_system_cost_v2",
        "calibration_status": "diagnostic",
    }
    cfg["cost"].update(
        curtailment_price_per_kwh=cfg["cost"]["renewable_curtailment_cost_per_kwh"],
        carbon_price_per_kg=0.1,
        peak_price_per_kw=1.0,
        mill_unserved_price_per_t=10.0,
        comfort_price_per_degree_hour=2.0,
        ev_shortfall_price_per_kwh=cfg["microgrids"][2]["dr"]["departure_penalty_per_kwh"],
    )
    for mg in cfg["microgrids"]:
        mg["grid"]["export_limit_kw"] = 0.0
    cfg["microgrids"][1]["dr"]["control_semantics"] = "soft_comfort"
    cfg["microgrids"][2]["dr"]["control_semantics"] = "no_deadline_minimum"
    cfg["observation"]["noisy_forecast"] = False
    cfg["observation"]["evaluation_forecast_noise"] = False
    return cfg


def _actor(env, mg="mg1"):
    norms = {
        key: float(env.obs_builder.norm[mg][key])
        for key in ("load", "pv", "wind", "power")
    }
    return HierarchicalTransformerActor(
        env.obs_builder.schema(mg),
        env.agent_ids,
        mg,
        env.cfg_by_id[mg],
        power_normalizers=norms,
        forecast_horizon=8,
        include_peer_forecast=True,
        peer_forecast_features=("energy_balance",),
    ).eval()


def test_v2_forecast_contract_preserves_physical_feature_order():
    env = MultiMicrogridEnv(_env_cfg(), split="train")
    mg = "mg1"
    actor = HierarchicalTransformerActor(
        env.obs_builder.schema(mg),
        env.agent_ids,
        mg,
        env.cfg_by_id[mg],
        power_normalizers={"load": 100.0, "pv": 200.0, "wind": 300.0, "power": 100.0},
        forecast_horizon=8,
        include_peer_forecast=True,
        peer_forecast_features=("energy_balance",),
    ).eval()
    x = torch.zeros(1, len(actor.schema))
    x[:, actor.schema.index("hour_cos")] = 1.0
    for name, value in {
        "load_t+1": 0.6, "pv_t+1": 0.2, "wind_t+1": 0.1,
        "price_t+1": 0.7, "forecast_valid_t+1": 1.0,
        f"{actor.peers[0]}_energy_balance_t+1": 0.25,
        f"{actor.peers[1]}_energy_balance_t+1": -0.4,
    }.items():
        x[:, actor.schema.index(name)] = value
    assert actor.forecast_horizon == 8
    assert actor.forecast_feature_dim == 8
    assert actor.total_tokens == 14
    assert actor.peers == tuple(p for p in env.agent_ids if p != mg)
    features = actor._build_forecast_features(x)
    # (60 kW load - 40 kW PV - 30 kW wind) / 100 kW = -0.1.
    expected = torch.tensor([-0.1, 0.7, 0.0, 1.0, 0.2588190451, 0.9659258263, 0.25, -0.4])
    torch.testing.assert_close(features[0, 0], expected, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(features[:, 1:], torch.zeros(1, 7, 8), rtol=0.0, atol=0.0)


def test_invalid_forecasts_cannot_change_action():
    env = MultiMicrogridEnv(_env_cfg(), split="train")
    mg = "mg1"
    actor = _actor(env, mg)
    obs, _ = env.reset(seed=5, deterministic=True, forecast_noise=False)
    base = torch.as_tensor(obs[mg], dtype=torch.float32).unsqueeze(0).clone()
    peers = actor.peers

    for k in range(1, 9):
        base[:, actor.schema.index(f"forecast_valid_t+{k}")] = 0.0
        for name in (f"load_t+{k}", f"pv_t+{k}", f"wind_t+{k}", f"price_t+{k}"):
            base[:, actor.schema.index(name)] = 0.0
        for peer in peers:
            base[:, actor.schema.index(f"{peer}_energy_balance_t+{k}")] = 0.0
    with torch.no_grad():
        expected = actor(base)

    mutated = base.clone()
    for k in range(1, 9):
        mutated[:, actor.schema.index(f"load_t+{k}")] = float("nan")
        mutated[:, actor.schema.index(f"pv_t+{k}")] = 999.0
        for peer in peers:
            mutated[:, actor.schema.index(f"{peer}_energy_balance_t+{k}")] = float("nan")
    with torch.no_grad():
        actual = actor(mutated)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)


def test_valid_peer_changes_forecast_embedding():
    env = MultiMicrogridEnv(_env_cfg(), split="train")
    mg = "mg1"
    actor = _actor(env, mg)
    obs, _ = env.reset(seed=8, deterministic=True, forecast_noise=False)
    x = torch.as_tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
    x[:, actor.schema.index("forecast_valid_t+1")] = 1.0
    before = actor._build_forecast_features(x).clone()
    peer_name = f"{actor.peers[0]}_energy_balance_t+1"
    y = x.clone()
    y[:, actor.schema.index(peer_name)] += 0.5
    after = actor._build_forecast_features(y)
    assert not torch.equal(before[:, 0, :], after[:, 0, :])
    with torch.no_grad():
        emb_before = actor.forecast_encoder(before)
        emb_after = actor.forecast_encoder(after)
    assert not torch.equal(emb_before[:, 0, :], emb_after[:, 0, :])


def test_trade_encoder_consumes_peak_and_carbon():
    env = MultiMicrogridEnv(_env_cfg(), split="train")
    actor = _actor(env, "mg1")
    obs, _ = env.reset(seed=12, deterministic=True, forecast_noise=False)
    x = torch.as_tensor(obs["mg1"], dtype=torch.float32).unsqueeze(0)
    captured = []

    def hook(_module, args):
        captured.append(args[0].detach().clone())

    handle = actor.trade_encoder.register_forward_pre_hook(hook)
    try:
        actor._build_tokens(x)
        y = x.clone()
        y[:, actor.schema.index("team_load_peak_so_far")] += 0.25
        y[:, actor.schema.index("grid_carbon_factor")] += 0.25
        actor._build_tokens(y)
    finally:
        handle.remove()
    assert "team_load_peak_so_far" in [actor.schema[i] for i in actor.trade_indices]
    assert "grid_carbon_factor" in [actor.schema[i] for i in actor.trade_indices]
    assert len(captured) == 2
    assert not torch.equal(captured[0], captured[1])
