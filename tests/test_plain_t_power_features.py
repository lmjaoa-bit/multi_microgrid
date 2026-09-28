from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.matd3_hierarchical_transformer.actor import HierarchicalTransformerActor
from algorithms.matd3_hierarchical_transformer.algorithm import Matd3HierarchicalTransformer
from envs.multi_microgrid_env import MultiMicrogridEnv


NORMALIZER_KEYS = ("load", "pv", "wind", "power")
FIXED_NORMALIZERS = {"load": 100.0, "pv": 500.0, "wind": 1000.0, "power": 1000.0}


@pytest.fixture(scope="module")
def configs() -> tuple[dict, dict]:
    env_cfg = yaml.safe_load(Path("configs/cooperative.yaml").read_text(encoding="utf-8"))
    algo_cfg = yaml.safe_load(
        Path("configs/algorithms/matd3_hierarchical_transformer.yaml").read_text(encoding="utf-8")
    )
    return env_cfg, algo_cfg


@pytest.fixture(scope="module")
def train_env(configs) -> MultiMicrogridEnv:
    env_cfg, _ = configs
    return MultiMicrogridEnv(copy.deepcopy(env_cfg), split="train")


def _build_actor(
    env: MultiMicrogridEnv,
    *,
    mg: str = "mg1",
    normalizers: dict[str, float] | None = None,
) -> HierarchicalTransformerActor:
    return HierarchicalTransformerActor(
        env.obs_builder.schema(mg),
        env.agent_ids,
        mg,
        env.cfg_by_id[mg],
        power_normalizers=FIXED_NORMALIZERS if normalizers is None else normalizers,
        forecast_horizon=8,
    )


def _empty_obs(actor: HierarchicalTransformerActor, batch_size: int = 1) -> torch.Tensor:
    return torch.zeros((batch_size, len(actor.schema)), dtype=torch.float32)


def _set_feature(obs: torch.Tensor, actor: HierarchicalTransformerActor, name: str, values) -> None:
    idx = actor.schema.index(name)
    obs[:, idx] = torch.as_tensor(values, dtype=obs.dtype).reshape(-1)


def _capture_pre_input(module: torch.nn.Module, store: dict[str, torch.Tensor], key: str):
    def hook(_module, args):
        store[key] = args[0].detach().clone()

    return module.register_forward_pre_hook(hook)


def _network_snapshot(algo: Matd3HierarchicalTransformer) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for group_name in ("actors", "target_actors", "critics", "target_critics"):
        group = getattr(algo, group_name)
        for name, value in group.state_dict().items():
            snapshot[f"{group_name}.{name}"] = value.detach().cpu().clone()
    return snapshot


def _assert_snapshot_equal(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> None:
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name


def test_current_supply_token_uses_physical_power_balance(train_env):
    actor = _build_actor(train_env)
    obs = _empty_obs(actor)
    _set_feature(obs, actor, "load", [0.5])
    _set_feature(obs, actor, "pv", [0.2])
    _set_feature(obs, actor, "wind", [0.0])

    captured: dict[str, torch.Tensor] = {}
    handle = _capture_pre_input(actor.supply_encoder, captured, "supply")
    try:
        actor._build_tokens(obs)
    finally:
        handle.remove()

    expected = torch.tensor([[0.5, 0.2, 0.0, 0.05, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(captured["supply"], expected, rtol=1e-6, atol=1e-7)


def test_forecast_token_uses_physical_net_load_for_batches_and_steps(train_env):
    actor = _build_actor(train_env)
    obs = _empty_obs(actor, batch_size=2)
    expected = torch.empty((2, actor.forecast_horizon), dtype=torch.float32)

    for batch in range(2):
        for k in range(1, actor.forecast_horizon + 1):
            load_kw = 120.0 + 10.0 * batch + 5.0 * (k - 1)
            pv_kw = 40.0 + 2.0 * (k - 1)
            wind_kw = 30.0 + 3.0 * batch
            obs[batch, actor.schema.index(f"load_t+{k}")] = load_kw / FIXED_NORMALIZERS["load"]
            obs[batch, actor.schema.index(f"pv_t+{k}")] = pv_kw / FIXED_NORMALIZERS["pv"]
            obs[batch, actor.schema.index(f"wind_t+{k}")] = wind_kw / FIXED_NORMALIZERS["wind"]
            obs[batch, actor.schema.index(f"price_t+{k}")] = 0.25 + 0.01 * k
            valid_name = f"forecast_valid_t+{k}"
            if valid_name in actor.schema:
                obs[batch, actor.schema.index(valid_name)] = 1.0
            expected[batch, k - 1] = (
                load_kw - pv_kw - wind_kw
            ) / FIXED_NORMALIZERS["power"]

    captured: dict[str, torch.Tensor] = {}
    handle = _capture_pre_input(actor.forecast_encoder, captured, "forecast")
    try:
        actor._build_tokens(obs)
    finally:
        handle.remove()

    torch.testing.assert_close(captured["forecast"][:, :, 0], expected, rtol=1e-6, atol=1e-7)
    assert captured["forecast"][0, 0, 0].item() == pytest.approx(0.05, abs=1e-7)


def test_physical_net_load_is_invariant_to_per_field_rescaling(train_env):
    norms_a = dict(FIXED_NORMALIZERS)
    norms_b = {"load": 200.0, "pv": 250.0, "wind": 2000.0, "power": 1000.0}
    actor_a = _build_actor(train_env, normalizers=norms_a)
    actor_b = _build_actor(train_env, normalizers=norms_b)

    load_kw = torch.tensor([[[50.0], [120.0]], [[80.0], [40.0]]])
    pv_kw = torch.tensor([[[100.0], [40.0]], [[0.0], [10.0]]])
    wind_kw = torch.tensor([[[0.0], [30.0]], [[20.0], [30.0]]])

    net_a = actor_a._physical_net_load(
        load_kw / norms_a["load"],
        pv_kw / norms_a["pv"],
        wind_kw / norms_a["wind"],
    )
    net_b = actor_b._physical_net_load(
        load_kw / norms_b["load"],
        pv_kw / norms_b["pv"],
        wind_kw / norms_b["wind"],
    )
    torch.testing.assert_close(net_a, net_b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("load_kw", "pv_kw", "wind_kw", "expected"),
    [
        (50.0, 0.0, 0.0, 0.05),
        (50.0, 0.0, 100.0, -0.05),
        (50.0, 100.0, 0.0, -0.05),
        (100.0, 100.0, 0.0, 0.0),
    ],
)
def test_physical_net_load_boundary_cases(train_env, load_kw, pv_kw, wind_kw, expected):
    actor = _build_actor(train_env)
    result = actor._physical_net_load(
        torch.tensor([[load_kw / FIXED_NORMALIZERS["load"]]], dtype=torch.float32),
        torch.tensor([[pv_kw / FIXED_NORMALIZERS["pv"]]], dtype=torch.float32),
        torch.tensor([[wind_kw / FIXED_NORMALIZERS["wind"]]], dtype=torch.float32),
    )
    assert result.item() == pytest.approx(expected, abs=1e-7)
    if expected == 0.0:
        assert torch.sign(-result).item() == 0.0


@pytest.mark.parametrize("missing_key", NORMALIZER_KEYS)
def test_actor_rejects_missing_power_normalizer(train_env, missing_key):
    bad = dict(FIXED_NORMALIZERS)
    bad.pop(missing_key)
    with pytest.raises(ValueError, match="normalizer|normalizers"):
        _build_actor(train_env, normalizers=bad)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("key", NORMALIZER_KEYS)
def test_actor_rejects_nonpositive_or_nonfinite_power_normalizer(train_env, key, bad_value):
    bad = dict(FIXED_NORMALIZERS)
    bad[key] = bad_value
    with pytest.raises(ValueError, match="normalizer|normalizers"):
        _build_actor(train_env, normalizers=bad)


def test_supply_and_forecast_keep_original_normalized_channels(train_env):
    actor = _build_actor(train_env)
    obs = _empty_obs(actor)
    current = {"load": 0.37, "pv": 0.61, "wind": 0.29}
    for name, value in current.items():
        _set_feature(obs, actor, name, [value])

    for k in range(1, actor.forecast_horizon + 1):
        obs[0, actor.schema.index(f"load_t+{k}")] = 0.11 * k
        obs[0, actor.schema.index(f"pv_t+{k}")] = 0.02 * k
        obs[0, actor.schema.index(f"wind_t+{k}")] = 0.03 * k
        valid_name = f"forecast_valid_t+{k}"
        if valid_name in actor.schema:
            obs[0, actor.schema.index(valid_name)] = 1.0

    captured: dict[str, torch.Tensor] = {}
    hs = _capture_pre_input(actor.supply_encoder, captured, "supply")
    hf = _capture_pre_input(actor.forecast_encoder, captured, "forecast")
    try:
        actor._build_tokens(obs)
    finally:
        hs.remove()
        hf.remove()

    torch.testing.assert_close(
        captured["supply"][0, :3],
        torch.tensor([current["load"], current["pv"], current["wind"]]),
    )
    assert captured["forecast"].shape[-1] == 6


def test_forecast_padding_mask_and_token_order_are_unchanged(train_env):
    actor = _build_actor(train_env)
    obs = _empty_obs(actor, batch_size=2)
    pattern = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    for k, valid in enumerate(pattern, start=1):
        name = f"forecast_valid_t+{k}"
        if name in actor.schema:
            obs[:, actor.schema.index(name)] = valid

    tokens, padding = actor._build_tokens(obs)
    assert tokens.shape == (2, 14, actor.embed_dim)
    assert padding.shape == (2, 14)
    assert actor.type_ids_tensor.tolist() == [0, 1, 2, 3, 4, 4] + [5] * 8
    expected_forecast_invalid = torch.tensor([v < 0.5 for v in pattern], dtype=torch.bool)
    assert torch.equal(padding[0, -8:].cpu(), expected_forecast_invalid)
    assert not padding[:, :6].any()


def test_power_normalizer_buffer_is_nonpersistent_and_moves_with_actor(train_env):
    actor = _build_actor(train_env)
    expected = torch.tensor([FIXED_NORMALIZERS[k] for k in NORMALIZER_KEYS], dtype=torch.float32)
    torch.testing.assert_close(actor.power_normalizer_tensor.cpu(), expected)
    assert "power_normalizer_tensor" not in actor.state_dict()

    if torch.cuda.is_available():
        cpu_actor = _build_actor(train_env).eval()
        gpu_actor = copy.deepcopy(cpu_actor).cuda().eval()
        assert gpu_actor.power_normalizer_tensor.device.type == "cuda"

        obs_cpu = _empty_obs(cpu_actor, batch_size=2)
        with torch.no_grad():
            cpu_out = cpu_actor(obs_cpu)
            gpu_out = gpu_actor(obs_cpu.cuda()).cpu()
        torch.testing.assert_close(cpu_out, gpu_out, rtol=2e-5, atol=2e-6)


def test_algorithm_uses_env_power_normalizers_and_train_statistics_for_all_splits(configs):
    env_cfg, algo_cfg = configs
    train_norms = None
    for split in ("train", "val", "test"):
        env = MultiMicrogridEnv(copy.deepcopy(env_cfg), split=split)
        norms = {
            mg: {key: float(env.obs_builder.norm[mg][key]) for key in NORMALIZER_KEYS}
            for mg in env.agent_ids
        }
        if train_norms is None:
            train_norms = norms
        else:
            assert norms == train_norms

        algo = Matd3HierarchicalTransformer.from_env(
            env, copy.deepcopy(algo_cfg), device="cpu", seed=42
        )
        for mg in env.agent_ids:
            expected = torch.tensor([norms[mg][key] for key in NORMALIZER_KEYS], dtype=torch.float32)
            torch.testing.assert_close(algo.actors[mg].power_normalizer_tensor.cpu(), expected)


def test_yaml_records_physical_power_feature_semantics(configs):
    _, algo_cfg = configs
    assert (
        algo_cfg["algorithm"]["network"]["feature_semantics"]
        == Matd3HierarchicalTransformer.ACTOR_FEATURE_SEMANTICS
        == "physical_power_balance_v1"
    )


def test_explicit_wrong_feature_semantics_is_rejected(configs):
    env_cfg, algo_cfg = configs
    env = MultiMicrogridEnv(copy.deepcopy(env_cfg), split="train")
    bad = copy.deepcopy(algo_cfg)
    bad.setdefault("algorithm", {}).setdefault("network", {})["feature_semantics"] = "legacy_mixed_scale"
    with pytest.raises(ValueError, match="feature semantics|feature_semantics"):
        Matd3HierarchicalTransformer.from_env(env, bad, device="cpu", seed=42)


def test_checkpoint_roundtrip_records_feature_metadata_and_preserves_actions(configs, tmp_path):
    env_cfg, algo_cfg = configs
    env = MultiMicrogridEnv(copy.deepcopy(env_cfg), split="train")
    algo = Matd3HierarchicalTransformer.from_env(
        env, copy.deepcopy(algo_cfg), device="cpu", seed=42
    )
    obs, _ = env.reset(seed=123)
    before = algo.select_actions(obs, noise_std=0.0)

    path = tmp_path / "plain_t.pt"
    algo.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["actor_feature_semantics"] == "physical_power_balance_v1"
    assert payload["actor_feature_normalizers"] == {
        mg: {key: float(env.obs_builder.norm[mg][key]) for key in NORMALIZER_KEYS}
        for mg in env.agent_ids
    }

    restored = Matd3HierarchicalTransformer.from_env(
        env, copy.deepcopy(algo_cfg), device="cpu", seed=999
    )
    restored.load(path)
    after = restored.select_actions(obs, noise_std=0.0)
    for mg in env.agent_ids:
        np.testing.assert_allclose(before[mg], after[mg], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_semantics",
        "wrong_semantics",
        "missing_normalizers",
        "missing_scale",
        "scale_mismatch",
        "agent_mismatch",
    ],
)
def test_checkpoint_rejects_incompatible_feature_metadata_before_loading_network_state(
    configs, tmp_path, mutation
):
    env_cfg, algo_cfg = configs
    env = MultiMicrogridEnv(copy.deepcopy(env_cfg), split="train")
    source = Matd3HierarchicalTransformer.from_env(
        env, copy.deepcopy(algo_cfg), device="cpu", seed=42
    )
    path = tmp_path / f"{mutation}.pt"
    source.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if mutation == "missing_semantics":
        payload.pop("actor_feature_semantics")
    elif mutation == "wrong_semantics":
        payload["actor_feature_semantics"] = "legacy_mixed_scale"
    elif mutation == "missing_normalizers":
        payload.pop("actor_feature_normalizers")
    elif mutation == "missing_scale":
        payload["actor_feature_normalizers"][env.agent_ids[0]].pop("load")
    elif mutation == "scale_mismatch":
        payload["actor_feature_normalizers"][env.agent_ids[0]]["load"] += 1e-6
    elif mutation == "agent_mismatch":
        payload["actor_feature_normalizers"] = {
            "mg1": payload["actor_feature_normalizers"]["mg1"],
            "mg2": payload["actor_feature_normalizers"]["mg2"],
            "mgX": payload["actor_feature_normalizers"]["mg3"],
        }
    else:  # pragma: no cover
        raise AssertionError(mutation)

    # Make a late validation bug observable: if load_state_dict runs first, parameters would change.
    first_key = next(iter(payload["actors"]))
    payload["actors"][first_key] = payload["actors"][first_key] + 7.0
    torch.save(payload, path)

    target = Matd3HierarchicalTransformer.from_env(
        env, copy.deepcopy(algo_cfg), device="cpu", seed=777
    )
    before = _network_snapshot(target)
    with pytest.raises(ValueError, match="feature|normalizer|agent"):
        target.load(path)
    after = _network_snapshot(target)
    _assert_snapshot_equal(before, after)
