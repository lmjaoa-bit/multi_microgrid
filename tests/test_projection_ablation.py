from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from components.commercial_hvac import CommercialHVACDR
from components.controllable_load import build_controllable_load
from components.ev_charging import EVChargingDR
from components.industrial_mill import IndustrialMillDR
from core.config_validation import validate_config
from envs.multi_microgrid_env import MultiMicrogridEnv
from experiments.profiles import experiment_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"


def _day_frame(*, tout: float = 18.0, ghi: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2018-01-02", periods=24, freq="h"),
        "mg2_temperature_c": np.full(24, tout),
        "mg2_ghi_w_per_m2": np.full(24, ghi),
    })


def _dr_configs() -> tuple[dict, dict, dict]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return tuple(deepcopy(mg["dr"]) for mg in config["microgrids"])


def test_projection_mode_validation_and_default_contract_are_explicit() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["constraint_projection"]["mode"] == "strong"
    validate_config(config)

    instantaneous = deepcopy(config)
    instantaneous["constraint_projection"]["mode"] = "instantaneous"
    validate_config(instantaneous)

    invalid = deepcopy(config)
    invalid["constraint_projection"]["mode"] = "oracle_magic"
    with pytest.raises(ValueError, match="constraint_projection.mode"):
        validate_config(invalid)

    spec = SimpleNamespace(slug="matd3_dispatch", environment_overrides={})
    contract = experiment_contract(spec, instantaneous, CONFIG)
    assert contract["constraint_projection_mode"] == "instantaneous"
    assert contract["physical_settings"]["constraint_projection"]["mode"] == "instantaneous"


def test_instantaneous_mill_projection_drops_terminal_reachability() -> None:
    mill_cfg, _, _ = _dr_configs()
    strong = build_controllable_load(
        mill_cfg, delta_t=1.0, episode_length=24, projection_mode="strong"
    )
    weak = build_controllable_load(
        mill_cfg, delta_t=1.0, episode_length=24, projection_mode="instantaneous"
    )
    assert isinstance(strong, IndustrialMillDR)
    assert isinstance(weak, IndustrialMillDR)
    frame = _day_frame()
    base = np.full(24, 48.6)
    strong.reset(frame, base)
    weak.reset(frame, base)

    strong_plan = strong.plan(-1.0, 23)
    weak_plan = weak.plan(-1.0, 23)
    assert abs(strong_plan["silo_after_t"] - strong.initial_silo_t) <= strong.terminal_tolerance_t + 1e-9
    assert weak_plan["effective_duty"] == pytest.approx(0.0)
    assert weak.silo_min_t <= weak_plan["silo_after_t"] <= weak.silo_max_t
    assert abs(weak_plan["silo_after_t"] - weak.initial_silo_t) > weak.terminal_tolerance_t


def test_instantaneous_ev_projection_does_not_force_departure_target() -> None:
    _, _, ev_cfg = _dr_configs()
    strong = build_controllable_load(
        ev_cfg, delta_t=1.0, episode_length=24, projection_mode="strong"
    )
    weak = build_controllable_load(
        ev_cfg, delta_t=1.0, episode_length=24, projection_mode="instantaneous"
    )
    assert isinstance(strong, EVChargingDR)
    assert isinstance(weak, EVChargingDR)
    frame = _day_frame()
    base = np.full(24, 57.52)
    strong.reset(frame, base)
    weak.reset(frame, base)

    strong_low, _ = strong.feasible_bounds(6)
    weak_low, weak_high = weak.feasible_bounds(6)
    assert strong_low > 0.0
    assert weak_low == pytest.approx(0.0)
    assert weak_high > 0.0
    weak.commit(weak.plan(-1.0, 6))
    assert weak.soc == pytest.approx(weak.initial_soc)


def test_instantaneous_hvac_projection_skips_terminal_lookahead_solver() -> None:
    _, hvac_cfg, _ = _dr_configs()
    hvac = build_controllable_load(
        hvac_cfg, delta_t=1.0, episode_length=24, projection_mode="instantaneous"
    )
    assert isinstance(hvac, CommercialHVACDR)
    hvac.reset(_day_frame(tout=18.0), np.full(24, 75.0))

    def forbidden(*args, **kwargs):
        raise AssertionError("instantaneous projection must not call the terminal lookahead solver")

    hvac._solve_terminal_q = forbidden
    low, high = hvac.feasible_bounds(0)
    assert -hvac.cool_max_kw <= low <= high <= hvac.heat_max_kw


def test_instantaneous_environment_penalizes_terminal_constraint_violations() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["constraint_projection"]["mode"] = "instantaneous"
    env = MultiMicrogridEnv(config, split="train")
    assert env.constraint_projection_mode == "instantaneous"
    observations, _ = env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    done = False
    info = None
    while not done:
        actions = {
            "mg1": np.asarray([0.0, -1.0], dtype=np.float32),
            "mg2": np.asarray([0.0, 0.0], dtype=np.float32),
            "mg3": np.asarray([0.0, -1.0], dtype=np.float32),
        }
        observations, rewards, done, info = env.step(actions)

    assert info is not None
    assert info["constraint_projection_mode"] == "instantaneous"
    penalty = info["terminal_constraint_penalty"]
    assert penalty["mg3"]["ev_equivalent_kwh"] > 0.0
    assert penalty["mg3"]["total_penalty"] > 0.0
    assert info["costs"]["mg3"]["terminal_soc_penalty"] == pytest.approx(
        penalty["mg3"]["total_penalty"]
    )
    assert rewards["mg1"] == pytest.approx(info["team_profit"] / env.accounting.reward_scale)
