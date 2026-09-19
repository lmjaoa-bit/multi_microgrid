"""Fast, data-independent checks for the isolated Transformer critic variant."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from algorithms.matd3_transformer_critic.critic import DeviceActionQ, TwinDeviceTransformerCritic
from algorithms.registry import get_algorithm_spec
from core.global_state import GlobalStateBuilder


ROOT = Path(__file__).resolve().parents[1]


class _Profiles:
    price_max = 1.5

    @staticmethod
    def normalizers(mg, dt):
        del mg, dt
        return {
            "load": 100.0, "pv": 600.0, "wind": 900.0, "power": 1000.0,
            "price": 1.5, "trade": 100.0,
        }


@pytest.fixture
def critic_inputs():
    config = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    schema = GlobalStateBuilder(config, _Profiles(), forecasts=None, delta_t=1.0).schema()
    normalizers = {mg["id"]: _Profiles.normalizers(mg["id"], 1.0) for mg in config["microgrids"]}
    state = torch.randn(2, len(schema), dtype=torch.float32)
    for k in range(1, 9):
        state[:, schema.index(f"forecast_valid_t+{k}")] = 1.0
    action = torch.randn(2, 9, dtype=torch.float32)
    return config, schema, normalizers, state, action


def test_plugin_and_dedicated_configuration():
    spec = get_algorithm_spec("matd3_transformer_critic")
    config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    assert spec.slug == config["algorithm"]["name"]
    assert config["algorithm"]["network"]["forecast_horizon"] == 8


def test_device_tokens_and_action_gradients(critic_inputs):
    cfg, schema, normalizers, state, action = critic_inputs
    critic = TwinDeviceTransformerCritic(
        schema, cfg["microgrids"], normalizers,
        embed_dim=16, heads=4, layers=1, horizon=8, hidden_dims=(32,),
    )
    action = action.requires_grad_()
    q1, q2 = critic(state, action)
    assert q1.shape == q2.shape == (2, 1)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
    assert critic.q1_network.state_encoder is not critic.q2_network.state_encoder
    q1.sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0
    assert critic.q1_network.cross_attention.in_proj_weight.grad is not None
    assert critic.q1_network.cross_attention.in_proj_weight.grad.abs().sum() > 0


def test_padding_does_not_leak_through_raw_state_skip(critic_inputs):
    cfg, schema, normalizers, state, action = critic_inputs
    network = DeviceActionQ(
        schema, cfg["microgrids"], normalizers,
        embed_dim=16, heads=4, layers=1, horizon=8, hidden_dims=(32,),
    ).eval()
    padded = state.clone()
    padded[:, schema.index("forecast_valid_t+8")] = 0.0
    changed = padded.clone()
    for mg in ("mg1", "mg2", "mg3"):
        changed[:, schema.index(f"{mg}_net_load_t+8")] += 200.0
        changed[:, schema.index(f"{mg}_price_t+8")] -= 100.0
        temp = f"{mg}_temperature_t+8"
        if temp in schema:
            changed[:, schema.index(temp)] += 100.0
    with torch.no_grad():
        q_original = network(padded, action)
        q_changed = network(changed, action)
    torch.testing.assert_close(q_original, q_changed, rtol=1e-5, atol=1e-5)


def test_forecast_and_joint_action_both_affect_q(critic_inputs):
    cfg, schema, normalizers, state, action = critic_inputs
    network = DeviceActionQ(
        schema, cfg["microgrids"], normalizers,
        embed_dim=16, heads=4, layers=1, horizon=8, hidden_dims=(32,),
    ).eval()
    forecast_changed = state.clone()
    forecast_changed[:, schema.index("mg2_net_load_t+3")] += 0.4
    action_changed = action.clone()
    action_changed[:, 3] += 0.4
    with torch.no_grad():
        reference = network(state, action)
        forecast = network(forecast_changed, action)
        acted = network(state, action_changed)
    assert not torch.allclose(reference, forecast, rtol=0.0, atol=1e-8)
    assert not torch.allclose(reference, acted, rtol=0.0, atol=1e-8)


def test_invalid_schema_and_horizon_rejected(critic_inputs):
    cfg, schema, normalizers, _, _ = critic_inputs
    with pytest.raises(ValueError, match="8h"):
        DeviceActionQ(schema, cfg["microgrids"], normalizers, horizon=4)
    broken = list(schema)
    broken.remove("mg1_soc")
    with pytest.raises(ValueError, match="mg1_soc"):
        DeviceActionQ(broken, cfg["microgrids"], normalizers)
