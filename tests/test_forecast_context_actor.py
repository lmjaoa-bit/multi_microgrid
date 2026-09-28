from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from algorithms.matd3_forecast_context.actor import ForecastContextActor
from envs.multi_microgrid_env import MultiMicrogridEnv


@pytest.fixture(scope="module")
def env() -> MultiMicrogridEnv:
    config = yaml.safe_load(
        Path("configs/cooperative.yaml").read_text(encoding="utf-8")
    )
    return MultiMicrogridEnv(copy.deepcopy(config), split="train")


def _actor(
    env: MultiMicrogridEnv,
    *,
    mg: str = "mg1",
    horizon: int = 8,
    peer: bool = False,
    schema: list[str] | None = None,
) -> ForecastContextActor:
    return ForecastContextActor(
        env.obs_builder.schema(mg) if schema is None else schema,
        env.agent_ids,
        mg,
        env.cfg_by_id[mg],
        power_normalizers=env.obs_builder.norm[mg],
        forecast_horizon=horizon,
        include_peer_forecast=peer,
        peer_forecast_features=("energy_balance",),
    )


def _empty(actor: ForecastContextActor, batch: int = 1) -> torch.Tensor:
    return torch.zeros((batch, len(actor.schema)), dtype=torch.float32)


def _set(obs: torch.Tensor, actor: ForecastContextActor, name: str, value: float) -> None:
    obs[:, actor.schema.index(name)] = float(value)


@pytest.mark.parametrize(
    ("horizon", "peer", "expected_dim", "expected_tokens"),
    [
        (8, False, 6, 14),
        (8, True, 8, 14),
        (24, False, 6, 30),
        (24, True, 8, 30),
    ],
)
def test_matrix_shapes(env, horizon, peer, expected_dim, expected_tokens):
    actor = _actor(env, horizon=horizon, peer=peer)
    obs = _empty(actor, batch=2)
    for k in range(1, horizon + 1):
        _set(obs, actor, f"forecast_valid_t+{k}", 1.0)

    features = actor._build_forecast_features(obs)
    tokens, mask = actor._build_tokens(obs)

    assert features.shape == (2, horizon, expected_dim)
    assert tokens.shape == (2, expected_tokens, actor.embed_dim)
    assert mask.shape == (2, expected_tokens)
    assert not mask.any()
    assert actor.forecast_encoder[0].in_features == expected_dim


def test_forecast_feature_order_and_peer_sign_are_exact(env):
    actor = _actor(env, horizon=8, peer=True)
    obs = _empty(actor)
    # hour 0 => future hour 1 has sin/cos for +1 hour.
    _set(obs, actor, "hour_sin", 0.0)
    _set(obs, actor, "hour_cos", 1.0)

    norms = env.obs_builder.norm["mg1"]
    _set(obs, actor, "load_t+1", 120.0 / norms["load"])
    _set(obs, actor, "pv_t+1", 40.0 / norms["pv"])
    _set(obs, actor, "wind_t+1", 30.0 / norms["wind"])
    _set(obs, actor, "price_t+1", 0.37)
    _set(obs, actor, "forecast_valid_t+1", 1.0)

    peer0, peer1 = actor.peers
    _set(obs, actor, f"{peer0}_energy_balance_t+1", 0.25)
    _set(obs, actor, f"{peer1}_energy_balance_t+1", -0.40)

    features = actor._build_forecast_features(obs)
    expected_net = (120.0 - 40.0 - 30.0) / norms["power"]

    assert features[0, 0, 0].item() == pytest.approx(expected_net)
    assert features[0, 0, 1].item() == pytest.approx(0.37)
    assert features[0, 0, 2].item() == pytest.approx(0.0)
    assert features[0, 0, 3].item() == pytest.approx(1.0)
    assert features[0, 0, 4].item() == pytest.approx(
        torch.sin(torch.tensor(2.0 * torch.pi / 24.0)).item()
    )
    assert features[0, 0, 5].item() == pytest.approx(
        torch.cos(torch.tensor(2.0 * torch.pi / 24.0)).item()
    )
    # Peer energy balance is already normalized on each peer's power scale and
    # remains positive for surplus; do not rescale or flip it.
    assert features[0, 0, 6].item() == pytest.approx(0.25)
    assert features[0, 0, 7].item() == pytest.approx(-0.40)


def test_peer_future_has_differentiable_path_to_action(env):
    torch.manual_seed(123)
    actor = _actor(env, horizon=8, peer=True).eval()
    obs = _empty(actor)
    for k in range(1, 9):
        _set(obs, actor, f"forecast_valid_t+{k}", 1.0)
    peer0, peer1 = actor.peers
    _set(obs, actor, f"{peer0}_energy_balance_t+1", 0.2)
    _set(obs, actor, f"{peer1}_energy_balance_t+1", -0.1)
    obs.requires_grad_(True)

    loss = actor(obs).square().sum()
    loss.backward()

    indices = [
        actor.schema.index(f"{peer0}_energy_balance_t+1"),
        actor.schema.index(f"{peer1}_energy_balance_t+1"),
    ]
    gradient = obs.grad[0, indices].abs().sum().item()
    assert gradient > 0.0


def test_disabled_peer_future_is_not_read(env):
    torch.manual_seed(123)
    actor = _actor(env, horizon=8, peer=False).eval()
    obs_a = _empty(actor)
    for k in range(1, 9):
        _set(obs_a, actor, f"forecast_valid_t+{k}", 1.0)
    obs_b = obs_a.clone()
    for peer in actor.peers:
        for k in range(1, 9):
            obs_b[:, actor.schema.index(f"{peer}_energy_balance_t+{k}")] = (
                10.0 + k
            )

    with torch.no_grad():
        action_a = actor(obs_a)
        action_b = actor(obs_b)
    torch.testing.assert_close(action_a, action_b, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("peer", [False, True])
def test_invalid_forecast_token_content_cannot_change_action(env, peer):
    torch.manual_seed(321)
    actor = _actor(env, horizon=8, peer=peer).eval()
    obs_a = _empty(actor)
    obs_b = obs_a.clone()

    # Step 1 stays valid and identical. Every later step is padded but receives
    # different arbitrary finite values in obs_b.
    _set(obs_a, actor, "forecast_valid_t+1", 1.0)
    _set(obs_b, actor, "forecast_valid_t+1", 1.0)
    for k in range(2, 9):
        _set(obs_a, actor, f"forecast_valid_t+{k}", 0.0)
        _set(obs_b, actor, f"forecast_valid_t+{k}", 0.0)
        for name in (f"load_t+{k}", f"pv_t+{k}", f"wind_t+{k}", f"price_t+{k}"):
            obs_b[:, actor.schema.index(name)] = 17.0 + k
        if f"temperature_t+{k}" in actor.schema:
            obs_b[:, actor.schema.index(f"temperature_t+{k}")] = -9.0
        if peer:
            for peer_id in actor.peers:
                obs_b[
                    :,
                    actor.schema.index(f"{peer_id}_energy_balance_t+{k}"),
                ] = 23.0 - k

    with torch.no_grad():
        action_a = actor(obs_a)
        action_b = actor(obs_b)
    torch.testing.assert_close(action_a, action_b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("horizon", [8, 24])
@pytest.mark.parametrize("peer", [False, True])
def test_all_forecasts_masked_still_produces_finite_action(env, horizon, peer):
    actor = _actor(env, horizon=horizon, peer=peer).eval()
    obs = _empty(actor)
    # This represents the end-of-day case where every future token is invalid.
    for k in range(1, horizon + 1):
        _set(obs, actor, f"forecast_valid_t+{k}", 0.0)

    with torch.no_grad():
        output = actor(obs)
    assert torch.isfinite(output).all()


def test_peer_projection_adds_only_128_parameters_per_actor(env):
    torch.manual_seed(1)
    local = _actor(env, horizon=8, peer=False)
    torch.manual_seed(1)
    peer = _actor(env, horizon=8, peer=True)
    assert peer.parameter_count - local.parameter_count == 2 * 64


@pytest.mark.parametrize("peer", [False, True])
def test_horizon_extension_adds_no_trainable_parameters(env, peer):
    torch.manual_seed(1)
    h8 = _actor(env, horizon=8, peer=peer)
    torch.manual_seed(1)
    h24 = _actor(env, horizon=24, peer=peer)
    assert h8.parameter_count == h24.parameter_count


def test_hvac_requires_temperature_for_every_forecast_step(env):
    schema = env.obs_builder.schema("mg2")
    assert "temperature_t+1" in schema
    bad_schema = [name for name in schema if name != "temperature_t+1"]
    with pytest.raises(ValueError, match="temperature_t\\+1"):
        _actor(env, mg="mg2", schema=bad_schema)


def test_non_hvac_temperature_is_explicit_zero(env):
    actor = _actor(env, mg="mg1", horizon=8)
    assert "temperature_t+1" not in actor.schema
    obs = _empty(actor)
    for k in range(1, 9):
        _set(obs, actor, f"forecast_valid_t+{k}", 1.0)
    features = actor._build_forecast_features(obs)
    assert torch.equal(features[:, :, 2], torch.zeros_like(features[:, :, 2]))


def test_missing_validity_never_falls_back_to_dummy_field(env):
    schema = [
        name
        for name in env.obs_builder.schema("mg1")
        if name != "forecast_valid_t+1"
    ]
    with pytest.raises(ValueError, match="forecast_valid_t\\+1"):
        _actor(env, schema=schema)


@pytest.mark.parametrize("bad_horizon", [True, 0, -1, 7, 9, 23, 25, 8.0, "8"])
def test_invalid_horizon_rejected(env, bad_horizon):
    with pytest.raises(ValueError, match="forecast_horizon"):
        ForecastContextActor(
            env.obs_builder.schema("mg1"),
            env.agent_ids,
            "mg1",
            env.cfg_by_id["mg1"],
            power_normalizers=env.obs_builder.norm["mg1"],
            forecast_horizon=bad_horizon,
        )


@pytest.mark.parametrize("bad_peer", ["false", 0, 1, None])
def test_peer_switch_requires_bool(env, bad_peer):
    with pytest.raises(ValueError, match="include_peer_forecast"):
        ForecastContextActor(
            env.obs_builder.schema("mg1"),
            env.agent_ids,
            "mg1",
            env.cfg_by_id["mg1"],
            power_normalizers=env.obs_builder.norm["mg1"],
            forecast_horizon=8,
            include_peer_forecast=bad_peer,
        )
