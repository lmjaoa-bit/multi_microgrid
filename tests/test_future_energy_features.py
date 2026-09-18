from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from core.config_validation import validate_config
from core.observation import ObservationBuilder
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


def get_base_config():
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    return cfg


def test_config_validation_future_features():
    cfg = get_base_config()
    # Valid configurations
    cfg["observation"]["future_horizon"] = 8
    cfg["observation"]["future_energy_features"] = {"enabled": True}
    validate_config(cfg)

    # Invalid horizon
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["observation"]["future_horizon"] = 0
    with pytest.raises(ValueError, match="future_horizon"):
        validate_config(bad_cfg)

    bad_cfg["observation"]["future_horizon"] = "eight"
    with pytest.raises(ValueError, match="future_horizon"):
        validate_config(bad_cfg)

    # Invalid future_energy_features
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["observation"]["future_energy_features"] = "not_a_mapping"
    with pytest.raises(ValueError, match="future_energy_features must be a mapping"):
        validate_config(bad_cfg)

    bad_cfg["observation"]["future_energy_features"] = {"enabled": "yes"}
    with pytest.raises(ValueError, match="future_energy_features.enabled must be boolean"):
        validate_config(bad_cfg)


@pytest.mark.parametrize("horizon", [4, 8, 24])
def test_observation_builder_schema_and_values(horizon: int):
    cfg = get_base_config()
    cfg["observation"]["future_horizon"] = horizon
    cfg["observation"]["future_energy_features"] = {"enabled": True}
    env = MultiMicrogridEnv(cfg)
    obs, _ = env.reset(seed=42, deterministic=True)

    for mg in env.agent_ids:
        schema = env.obs_builder.schema(mg)
        obs_vec = obs[mg]
        assert len(schema) == len(obs_vec)

        # Current energy trend features
        assert "energy_balance" in schema
        assert "deficit_ratio" in schema
        assert "surplus_ratio" in schema

        # Future features for each k in 1..horizon
        for k in range(1, horizon + 1):
            assert f"renewable_power_t+{k}" in schema
            assert f"energy_balance_t+{k}" in schema
            assert f"deficit_ratio_t+{k}" in schema
            assert f"surplus_ratio_t+{k}" in schema

            # Check mathematical consistency of calculations
            f_res = obs_vec[schema.index(f"renewable_power_t+{k}")]
            f_bal = obs_vec[schema.index(f"energy_balance_t+{k}")]
            f_def = obs_vec[schema.index(f"deficit_ratio_t+{k}")]
            f_sur = obs_vec[schema.index(f"surplus_ratio_t+{k}")]
            f_pv = obs_vec[schema.index(f"pv_t+{k}")]
            f_wind = obs_vec[schema.index(f"wind_t+{k}")]
            f_load = obs_vec[schema.index(f"load_t+{k}")]

            assert np.isfinite(f_res) and np.isfinite(f_bal)
            assert np.isfinite(f_def) and np.isfinite(f_sur)
            assert f_def >= 0.0
            assert f_sur >= 0.0

            # When deficit is strictly positive, surplus must be 0, and vice versa
            if f_def > 1e-6:
                assert f_sur == pytest.approx(0.0, abs=1e-6)
            if f_sur > 1e-6:
                assert f_def == pytest.approx(0.0, abs=1e-6)

        # Check that k beyond horizon is not present
        assert f"renewable_power_t+{horizon + 1}" not in schema


def test_future_energy_features_disabled():
    cfg = get_base_config()
    cfg["observation"]["future_horizon"] = 4
    cfg["observation"]["future_energy_features"] = {"enabled": False}
    env = MultiMicrogridEnv(cfg)
    obs, _ = env.reset(seed=42, deterministic=True)

    for mg in env.agent_ids:
        schema = env.obs_builder.schema(mg)
        assert "energy_balance" not in schema
        assert "deficit_ratio" not in schema
        assert "surplus_ratio" not in schema
        for k in range(1, 5):
            assert f"renewable_power_t+{k}" not in schema
            assert f"energy_balance_t+{k}" not in schema


def test_transformer_actor_step_compatibility():
    spec = get_algorithm_spec("matd3_transformer")
    cfg = get_base_config()
    cfg["observation"]["future_horizon"] = 8
    cfg["observation"]["future_energy_features"] = {"enabled": True}
    env = MultiMicrogridEnv(spec.build_environment_config(cfg))
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    settings.get("algorithm", {}).setdefault("network", {})["forecast_horizon"] = 8
    model = spec.create(env, settings, seed=42)

    obs, _ = env.reset(seed=42, deterministic=True)
    actions = model.select_actions(obs, noise_std=0.0)

    for mg in env.agent_ids:
        assert mg in actions
        assert isinstance(actions[mg], np.ndarray)
        assert actions[mg].shape == (2,)
        assert np.all(actions[mg] >= -1.0) and np.all(actions[mg] <= 1.0)

    next_obs, rewards, done, info = env.step(actions)
    for mg in env.agent_ids:
        assert np.isfinite(next_obs[mg]).all()
