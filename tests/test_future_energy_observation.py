from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


def get_base_config():
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    return cfg


def test_4_hour_forecast_horizon():
    """测试1：4小时预测
    配置：future_horizon=4
    检查存在：energy_balance_t+1, t+2, t+3, t+4
    不存在：energy_balance_t+5
    """
    cfg = get_base_config()
    cfg["observation"]["future_horizon"] = 4
    cfg["observation"]["future_energy_features"] = {"enabled": True}
    env = MultiMicrogridEnv(cfg)
    obs, _ = env.reset(seed=42, deterministic=True)

    for mg in env.agent_ids:
        schema = env.obs_builder.schema(mg)
        for k in (1, 2, 3, 4):
            assert f"energy_balance_t+{k}" in schema
            assert f"renewable_power_t+{k}" in schema
            assert f"deficit_ratio_t+{k}" in schema
            assert f"surplus_ratio_t+{k}" in schema
        assert "energy_balance_t+5" not in schema
        assert "renewable_power_t+5" not in schema


def test_8_hour_forecast_horizon():
    """测试2：8小时预测
    配置：future_horizon=8
    检查存在：energy_balance_t+8
    """
    cfg = get_base_config()
    cfg["observation"]["future_horizon"] = 8
    cfg["observation"]["future_energy_features"] = {"enabled": True}
    env = MultiMicrogridEnv(cfg)
    obs, _ = env.reset(seed=42, deterministic=True)

    for mg in env.agent_ids:
        schema = env.obs_builder.schema(mg)
        assert "energy_balance_t+8" in schema
        assert "renewable_power_t+8" in schema
        assert "energy_balance_t+9" not in schema


def test_disabled_future_energy_features():
    """测试3：关闭future feature
    配置：future_energy_features.enabled=false
    检查不存在：
    renewable_power
    energy_balance
    energy_balance_t+k
    """
    cfg = get_base_config()
    cfg["observation"]["future_horizon"] = 4
    cfg["observation"]["future_energy_features"] = {"enabled": False}
    env = MultiMicrogridEnv(cfg)
    obs, _ = env.reset(seed=42, deterministic=True)

    for mg in env.agent_ids:
        schema = env.obs_builder.schema(mg)
        assert "renewable_power" not in schema
        assert "energy_balance" not in schema
        assert "deficit_ratio" not in schema
        assert "surplus_ratio" not in schema
        for k in range(1, 5):
            assert f"renewable_power_t+{k}" not in schema
            assert f"energy_balance_t+{k}" not in schema
            assert f"deficit_ratio_t+{k}" not in schema
            assert f"surplus_ratio_t+{k}" not in schema


def test_default_future_energy_features_enabled():
    """默认行为测试：未在配置中指定 future_energy_features 时，默认自动启用。"""
    cfg = get_base_config()
    cfg["observation"]["future_horizon"] = 4
    cfg["observation"].pop("future_energy_features", None)
    env = MultiMicrogridEnv(cfg)
    obs, _ = env.reset(seed=42, deterministic=True)

    for mg in env.agent_ids:
        schema = env.obs_builder.schema(mg)
        assert "renewable_power" in schema
        assert "energy_balance" in schema
        assert "energy_balance_t+1" in schema


@pytest.mark.parametrize("horizon", [4, 8, 12, 24])
def test_dynamic_horizons_transformer_compatibility(horizon: int):
    """测试多预测时域支持：4h, 8h, 12h, 24h
    无需修改 actor / transformer / observation 代码，仅修改 future_horizon。
    """
    spec = get_algorithm_spec("matd3_transformer")
    cfg = get_base_config()
    cfg["observation"]["future_horizon"] = horizon
    cfg["observation"]["future_energy_features"] = {"enabled": True}

    env = MultiMicrogridEnv(spec.build_environment_config(cfg))
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    model = spec.create(env, settings, seed=42)

    obs, _ = env.reset(seed=42, deterministic=True)
    actions = model.select_actions(obs, noise_std=0.0)

    for mg in env.agent_ids:
        assert mg in actions
        assert actions[mg].shape == (2,)
        actor = model.actors[mg]
        assert actor.forecast_horizon == horizon
        assert actor.horizon_embedding.weight.shape[0] == horizon

    next_obs, rewards, done, info = env.step(actions)
    for mg in env.agent_ids:
        assert np.isfinite(next_obs[mg]).all()
