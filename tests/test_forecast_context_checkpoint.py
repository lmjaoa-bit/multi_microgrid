from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.matd3_forecast_context.algorithm import Matd3ForecastContext
from algorithms.matd3_hierarchical_transformer.algorithm import (
    Matd3HierarchicalTransformer,
)
from envs.multi_microgrid_env import MultiMicrogridEnv


CONFIG_PATHS = {
    "A": Path("configs/algorithms/matd3_forecast_context.yaml"),
    "B": Path("configs/algorithms/matd3_forecast_context/b_peer_h8.yaml"),
    "C": Path("configs/algorithms/matd3_forecast_context/c_local_h24.yaml"),
    "D": Path("configs/algorithms/matd3_forecast_context/d_peer_h24.yaml"),
}


@pytest.fixture(scope="module")
def env() -> MultiMicrogridEnv:
    config = yaml.safe_load(
        Path("configs/cooperative.yaml").read_text(encoding="utf-8")
    )
    return MultiMicrogridEnv(copy.deepcopy(config), split="train")


def _config(group: str) -> dict:
    return yaml.safe_load(
        CONFIG_PATHS[group].read_text(encoding="utf-8")
    )


def _algorithm(
    env: MultiMicrogridEnv,
    group: str,
    seed: int = 42,
) -> Matd3ForecastContext:
    return Matd3ForecastContext.from_env(
        env,
        _config(group),
        device="cpu",
        seed=seed,
    )


def _snapshot(algorithm: Matd3ForecastContext) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for group_name in (
        "actors",
        "target_actors",
        "critics",
        "target_critics",
    ):
        group = getattr(algorithm, group_name)
        for name, value in group.state_dict().items():
            result[f"{group_name}.{name}"] = value.detach().clone()
    return result


def _assert_snapshot_equal(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> None:
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name


@pytest.mark.parametrize("group", ["A", "B", "C", "D"])
def test_checkpoint_roundtrip_records_full_contract(env, tmp_path, group):
    algorithm = _algorithm(env, group, seed=42)
    observations, _ = env.reset(seed=101)
    before = algorithm.select_actions(observations, noise_std=0.0)

    path = tmp_path / f"{group}.pt"
    algorithm.save(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    expected_horizon, expected_peer = Matd3ForecastContext.GROUP_MATRIX[group]
    assert checkpoint["algorithm_slug"] == "matd3_forecast_context"
    assert (
        checkpoint["forecast_context_version"]
        == "forecast_context_v1"
    )
    contract = checkpoint["forecast_context_contract"]
    assert contract["horizon"] == expected_horizon
    assert contract["include_peer_forecast"] is expected_peer
    assert contract["peer_forecast_features"] == ["energy_balance"]
    assert contract["agent_order"] == env.agent_ids
    assert contract["mask_semantics"] == "valid_lt_half_padding_v1"
    assert contract["local_balance_sign"] == "positive_deficit"
    assert contract["peer_balance_sign"] == "positive_surplus"
    assert checkpoint["forecast_context_signature"] == (
        Matd3ForecastContext._contract_signature(contract)
    )

    restored = _algorithm(env, group, seed=999)
    restored.load(path)
    after = restored.select_actions(observations, noise_std=0.0)
    for mg in env.agent_ids:
        np.testing.assert_allclose(
            before[mg],
            after[mg],
            rtol=1e-6,
            atol=1e-7,
        )


@pytest.mark.parametrize("target_group", ["B", "C", "D"])
def test_checkpoint_rejects_cross_group_load_before_network_mutation(
    env,
    tmp_path,
    target_group,
):
    source = _algorithm(env, "A")
    path = tmp_path / "source_a.pt"
    source.save(path)

    target = _algorithm(env, target_group, seed=777)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="contract|Forecast context"):
        target.load(path)
    _assert_snapshot_equal(before, _snapshot(target))


@pytest.mark.parametrize(
    "mutation",
    [
        "horizon",
        "peer_order",
        "normalizer",
        "architecture",
        "source_schema",
    ],
)
def test_checkpoint_contract_mutation_is_rejected_even_with_valid_signature(
    env,
    tmp_path,
    mutation,
):
    source = _algorithm(env, "A")
    path = tmp_path / f"mutated_{mutation}.pt"
    source.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    contract = payload["forecast_context_contract"]

    if mutation == "horizon":
        contract["horizon"] = 24
    elif mutation == "peer_order":
        mg = env.agent_ids[0]
        contract["per_agent"][mg]["peer_order"] = list(
            reversed(contract["per_agent"][mg]["peer_order"])
        )
    elif mutation == "normalizer":
        mg = env.agent_ids[0]
        contract["power_normalizers_by_agent"][mg]["load"] += 1e-5
    elif mutation == "architecture":
        contract["architecture"]["hidden_dim"] += 1
    elif mutation == "source_schema":
        mg = env.agent_ids[0]
        contract["per_agent"][mg]["local_source_schema_names"][0][0] = (
            "wrong_load_t+1"
        )
    else:  # pragma: no cover
        raise AssertionError(mutation)

    payload["forecast_context_signature"] = (
        Matd3ForecastContext._contract_signature(contract)
    )
    torch.save(payload, path)

    target = _algorithm(env, "A", seed=888)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="contract|normalizer|Forecast context"):
        target.load(path)
    _assert_snapshot_equal(before, _snapshot(target))


def test_checkpoint_rejects_tampered_signature(env, tmp_path):
    source = _algorithm(env, "A")
    path = tmp_path / "bad_signature.pt"
    source.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["forecast_context_signature"] = "0" * 64
    torch.save(payload, path)

    target = _algorithm(env, "A", seed=11)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="signature"):
        target.load(path)
    _assert_snapshot_equal(before, _snapshot(target))


def test_checkpoint_rejects_old_ordinary_t_checkpoint(env, tmp_path):
    ordinary_cfg = yaml.safe_load(
        Path(
            "configs/algorithms/matd3_hierarchical_transformer.yaml"
        ).read_text(encoding="utf-8")
    )
    old = Matd3HierarchicalTransformer.from_env(
        env,
        ordinary_cfg,
        device="cpu",
        seed=42,
    )
    path = tmp_path / "ordinary_t.pt"
    old.save(path)

    target = _algorithm(env, "A", seed=999)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="algorithm|forecast_context"):
        target.load(path)
    _assert_snapshot_equal(before, _snapshot(target))


def test_feature_metadata_validation_runs_before_load_state_dict(env, tmp_path):
    source = _algorithm(env, "A")
    path = tmp_path / "bad_feature_metadata.pt"
    source.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mg = env.agent_ids[0]
    payload["actor_feature_normalizers"][mg]["load"] += 1e-4

    # Make any accidental early load_state_dict visible.
    first_key = next(iter(payload["actors"]))
    payload["actors"][first_key] = payload["actors"][first_key] + 99.0
    torch.save(payload, path)

    target = _algorithm(env, "A", seed=314)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="normalizer"):
        target.load(path)
    _assert_snapshot_equal(before, _snapshot(target))
