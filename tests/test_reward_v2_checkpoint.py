from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from algorithms.matd3_hierarchical_transformer.algorithm import (
    Matd3HierarchicalTransformer,
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


def _algorithm_config(slug: str):
    name = (
        "matd3_dispatch.yaml"
        if slug == "matd3_dispatch"
        else "matd3_hierarchical_transformer.yaml"
    )
    cfg = yaml.safe_load(
        (ROOT / "configs" / "algorithms" / name).read_text(encoding="utf-8")
    )
    cfg["algorithm"]["gamma"] = 1.0
    if slug == "matd3_hierarchical_transformer":
        cfg["algorithm"]["network"]["forecast_horizon"] = 8
        cfg["algorithm"]["network"]["include_peer_forecast"] = True
        cfg["algorithm"]["network"]["peer_forecast_features"] = ["energy_balance"]
    return cfg


def _build(slug: str, seed: int = 42):
    env = MultiMicrogridEnv(_env_cfg(), split="train")
    cfg = _algorithm_config(slug)
    cls = (
        Matd3Dispatch
        if slug == "matd3_dispatch"
        else Matd3HierarchicalTransformer
    )
    return env, cls.from_env(env, cfg, device="cpu", seed=seed)


def _snapshot(algorithm):
    result = {}
    for group in ("actors", "target_actors", "critics", "target_critics"):
        for name, tensor in getattr(algorithm, group).state_dict().items():
            result[f"{group}.{name}"] = tensor.detach().clone()
    return result


def _assert_same(before, after):
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key], after[key]), key


@pytest.mark.parametrize(
    "slug",
    ["matd3_dispatch", "matd3_hierarchical_transformer"],
)
def test_v2_checkpoint_roundtrip(slug, tmp_path):
    env, source = _build(slug, seed=42)
    obs, _ = env.reset(seed=101, deterministic=True, forecast_noise=False)
    expected = source.select_actions(obs, noise_std=0.0)
    path = tmp_path / f"{slug}.pt"
    source.save(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    contract = payload["reward_v2_contract"]
    assert contract["reward"]["semantics_version"] == "generalized_system_cost_v2"
    assert contract["gamma"] == pytest.approx(1.0)
    assert contract["action_semantics"] == "single_dispatch_dr_v4_instantaneous_physical"
    assert contract["signature"]
    if slug == "matd3_hierarchical_transformer":
        assert payload["actor_forecast_context_version"] == "peer_balance_h8_v2"

    _, restored = _build(slug, seed=999)
    restored.load(path)
    actual = restored.select_actions(obs, noise_std=0.0)
    for mg in env.agent_ids:
        np.testing.assert_allclose(expected[mg], actual[mg], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "slug,mutation",
    [
        ("matd3_dispatch", "reward"),
        ("matd3_dispatch", "schema"),
        ("matd3_dispatch", "normalizer"),
        ("matd3_dispatch", "gamma"),
        ("matd3_hierarchical_transformer", "peer_order"),
        ("matd3_hierarchical_transformer", "mask"),
        ("matd3_hierarchical_transformer", "forecast_version"),
    ],
)
def test_v2_contract_mutation_rejected_before_network_mutation(slug, mutation, tmp_path):
    env, source = _build(slug, seed=42)
    path = tmp_path / f"{slug}_{mutation}.pt"
    source.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    contract = payload["reward_v2_contract"]

    if mutation == "reward":
        contract["reward"]["carbon_price_per_kg"] += 1e-6
    elif mutation == "schema":
        mg = env.agent_ids[0]
        contract["observation_schemas"][mg][0] = "wrong_feature"
    elif mutation == "normalizer":
        mg = env.agent_ids[0]
        contract["normalizers"]["by_microgrid"][mg]["load"] += 1e-6
    elif mutation == "gamma":
        contract["gamma"] = 0.99
    elif mutation == "peer_order":
        mg = env.agent_ids[0]
        contract["peer_orders"][mg] = list(reversed(contract["peer_orders"][mg]))
    elif mutation == "mask":
        contract["mask_semantics"] = "wrong_mask"
    elif mutation == "forecast_version":
        payload["actor_forecast_context_version"] = "local_h8_v1"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    torch.save(payload, path)

    _, target = _build(slug, seed=777)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="contract|context"):
        target.load(path)
    _assert_same(before, _snapshot(target))


@pytest.mark.parametrize(
    "slug",
    ["matd3_dispatch", "matd3_hierarchical_transformer"],
)
def test_v2_rejects_missing_contract_before_mutation(slug, tmp_path):
    _, source = _build(slug, seed=42)
    path = tmp_path / f"{slug}_missing_contract.pt"
    source.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("reward_v2_contract", None)
    torch.save(payload, path)

    _, target = _build(slug, seed=9)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="contract"):
        target.load(path)
    _assert_same(before, _snapshot(target))


def test_v2_transformer_rejects_legacy_six_dim_checkpoint_before_mutation(tmp_path):
    legacy_cfg = yaml.safe_load(
        (ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8")
    )
    legacy_cfg["_project_root"] = str(ROOT)
    legacy_env = MultiMicrogridEnv(legacy_cfg, split="train")
    legacy_algo_cfg = yaml.safe_load(
        (ROOT / "configs/algorithms/matd3_hierarchical_transformer.yaml").read_text(
            encoding="utf-8"
        )
    )
    legacy_algo_cfg["algorithm"]["network"].pop("include_peer_forecast", None)
    legacy_algo_cfg["algorithm"]["network"].pop("peer_forecast_features", None)
    legacy = Matd3HierarchicalTransformer.from_env(
        legacy_env, legacy_algo_cfg, device="cpu", seed=42
    )
    path = tmp_path / "legacy_t.pt"
    legacy.save(path)

    _, target = _build("matd3_hierarchical_transformer", seed=99)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="contract"):
        target.load(path)
    _assert_same(before, _snapshot(target))
