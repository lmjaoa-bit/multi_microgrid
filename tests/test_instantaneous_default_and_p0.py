from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from algorithms.matd3 import MATD3
from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from algorithms.matd3_transformer.algorithm import Matd3Transformer
from algorithms.pnn.algorithm import PNN
from algorithms.pnn_transformer.algorithm import PnnTransformer
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"


def _load() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_canonical_environment_is_the_only_instantaneous_projection_configuration() -> None:
    cfg = _load()
    assert "constraint_projection" not in cfg
    assert not (ROOT / "configs" / "cooperative_instantaneous_projection.yaml").exists()
    assert not (ROOT / "envs" / "instantaneous_projection_env.py").exists()


def test_default_environment_drops_cross_time_projection_but_keeps_instantaneous_safety() -> None:
    env = MultiMicrogridEnv(_load(), split="train")
    day = str(env.data.valid_days[0].date())
    env.reset(day=day, deterministic=True)

    battery = env.cores["mg1"].battery
    battery.soc = battery.soc_min
    assert env._battery_flex(23)["mg1"]["feasible_charge_min_ratio"] == pytest.approx(0.0)

    mill = env.dr_devices["mg1"]
    mill_plan = mill.plan(-1.0, 23)
    assert mill_plan["effective_duty"] == pytest.approx(0.0)
    assert mill.silo_min_t <= mill_plan["silo_after_t"] <= mill.silo_max_t
    assert abs(mill_plan["silo_after_t"] - mill.initial_silo_t) > mill.terminal_tolerance_t

    ev = env.dr_devices["mg3"]
    low, high = ev.feasible_bounds(6)
    assert low == pytest.approx(0.0)
    assert high > 0.0

    hvac = env.dr_devices["mg2"]

    def forbidden(*args, **kwargs):
        raise AssertionError("instantaneous HVAC feasibility must not solve terminal lookahead")

    hvac._solve_terminal_q = forbidden
    low, high = hvac.feasible_bounds(0)
    assert -hvac.cool_max_kw <= low <= high <= hvac.heat_max_kw


def test_default_environment_keeps_terminal_constraint_penalties_as_learning_signal() -> None:
    env = MultiMicrogridEnv(_load(), split="train")
    day = str(env.data.valid_days[0].date())
    env.reset(day=day, deterministic=True)
    done = False
    info = None
    rewards = None
    while not done:
        actions = {
            "mg1": np.asarray([0.0, -1.0], dtype=np.float32),
            "mg2": np.asarray([0.0, 0.0], dtype=np.float32),
            "mg3": np.asarray([0.0, -1.0], dtype=np.float32),
        }
        _, rewards, done, info = env.step(actions)

    assert info is not None and rewards is not None
    penalty = info["terminal_constraint_penalty"]
    assert penalty["mg3"]["ev_equivalent_kwh"] > 0.0
    assert penalty["mg3"]["total_penalty"] > 0.0
    assert info["costs"]["mg3"]["terminal_constraint_penalty"] == pytest.approx(
        penalty["mg3"]["total_penalty"]
    )
    assert rewards["mg1"] == pytest.approx(info["team_profit"] / env.accounting.reward_scale)


def test_p0_constraint_pressure_features_are_exposed_to_actor_and_centralized_critic() -> None:
    env = MultiMicrogridEnv(_load(), split="test")
    observations, info = env.reset(seed=42, deterministic=True)

    expected_local = {
        "mg1": {"terminal_soc_gap", "ess_time_to_terminal", "ess_required_avg_power", "ess_terminal_pressure",
                "silo_target_gap", "time_to_terminal", "mill_required_avg_duty", "mill_terminal_pressure"},
        "mg2": {"terminal_soc_gap", "ess_time_to_terminal", "ess_required_avg_power", "ess_terminal_pressure",
                "terminal_air_gap", "terminal_mass_gap", "hvac_time_to_terminal", "thermal_recovery_pressure"},
        "mg3": {"terminal_soc_gap", "ess_time_to_terminal", "ess_required_avg_power", "ess_terminal_pressure",
                "ev_remaining_energy", "ev_hours_to_departure", "deadline_pressure"},
    }
    for mg, names in expected_local.items():
        schema = set(env.obs_builder.schema(mg))
        assert names <= schema
        assert observations[mg].shape == (env.observation_dims[mg],)
        assert np.isfinite(observations[mg]).all()

    state_schema = set(env.global_state_builder.schema())
    for mg in env.agent_ids:
        assert f"{mg}_ess_time_to_terminal" in state_schema
        assert f"{mg}_ess_required_avg_power" in state_schema
        assert f"{mg}_ess_terminal_pressure" in state_schema
    assert {"mg1_mill_required_avg_duty", "mg1_mill_terminal_pressure"} <= state_schema
    assert {"mg2_terminal_air_gap", "mg2_terminal_mass_gap", "mg2_hvac_time_to_terminal",
            "mg2_thermal_recovery_pressure"} <= state_schema
    assert np.isfinite(info["global_state"]).all()


def test_all_trainable_algorithms_accept_optional_diagnostic_sampling_flag() -> None:
    for cls in (MATD3, Matd3Dispatch, PNN, Matd3Transformer, PnnTransformer):
        signature = inspect.signature(cls.update)
        assert "collect_diagnostics" in signature.parameters


def test_all_algorithm_configs_default_to_twenty_step_diagnostic_sampling() -> None:
    for name in ("matd3_dispatch", "pnn", "matd3_transformer", "pnn_transformer"):
        path = ROOT / "configs" / "algorithms" / f"{name}.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["logging"]["diagnostic_interval"] == 20
