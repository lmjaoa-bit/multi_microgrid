from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from core.config_validation import validate_config
from envs.instantaneous_projection_env import InstantaneousProjectionEnv
from envs.multi_microgrid_env import MultiMicrogridEnv
from experiments.profiles import experiment_contract


ROOT = Path(__file__).resolve().parents[1]
STRONG_CONFIG = ROOT / "configs" / "cooperative.yaml"
WEAK_CONFIG = ROOT / "configs" / "cooperative_instantaneous_projection.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_projection_configs_differ_only_by_projection_mode() -> None:
    strong = _load(STRONG_CONFIG)
    weak = _load(WEAK_CONFIG)
    assert strong["constraint_projection"]["mode"] == "strong"
    assert weak["constraint_projection"]["mode"] == "instantaneous"

    strong_cmp = deepcopy(strong)
    weak_cmp = deepcopy(weak)
    strong_cmp["constraint_projection"]["mode"] = "<mode>"
    weak_cmp["constraint_projection"]["mode"] = "<mode>"
    assert weak_cmp == strong_cmp

    validate_config(strong)
    validate_config(weak)
    invalid = deepcopy(strong)
    invalid["constraint_projection"]["mode"] = "oracle_magic"
    with pytest.raises(ValueError, match="constraint_projection.mode"):
        validate_config(invalid)


def test_contract_records_projection_mode() -> None:
    weak = _load(WEAK_CONFIG)
    spec = SimpleNamespace(slug="matd3_dispatch", environment_overrides={})
    contract = experiment_contract(spec, weak, WEAK_CONFIG)
    assert contract["constraint_projection_mode"] == "instantaneous"
    assert contract["physical_settings"]["constraint_projection"]["mode"] == "instantaneous"


def test_strong_and_instantaneous_ess_projection_do_not_contaminate_each_other() -> None:
    strong = MultiMicrogridEnv(_load(STRONG_CONFIG), split="train")
    weak = InstantaneousProjectionEnv(_load(WEAK_CONFIG), split="train")
    day = str(strong.data.valid_days[0].date())
    strong.reset(day=day, deterministic=True)
    weak.reset(day=day, deterministic=True)

    strong.cores["mg1"].battery.soc = strong.cores["mg1"].battery.soc_min
    weak.cores["mg1"].battery.soc = weak.cores["mg1"].battery.soc_min
    strong_min = strong._battery_flex(23)["mg1"]["feasible_charge_min_ratio"]
    weak_min = weak._battery_flex(23)["mg1"]["feasible_charge_min_ratio"]
    assert strong_min > 0.0
    assert weak_min == pytest.approx(0.0)


def test_instantaneous_dr_devices_drop_only_terminal_lookahead() -> None:
    strong = MultiMicrogridEnv(_load(STRONG_CONFIG), split="train")
    weak = InstantaneousProjectionEnv(_load(WEAK_CONFIG), split="train")
    day = str(strong.data.valid_days[0].date())
    strong.reset(day=day, deterministic=True)
    weak.reset(day=day, deterministic=True)

    strong_mill = strong.dr_devices["mg1"]
    weak_mill = weak.dr_devices["mg1"]
    strong_plan = strong_mill.plan(-1.0, 23)
    weak_plan = weak_mill.plan(-1.0, 23)
    assert abs(strong_plan["silo_after_t"] - strong_mill.initial_silo_t) <= strong_mill.terminal_tolerance_t + 1e-9
    assert weak_plan["effective_duty"] == pytest.approx(0.0)
    assert weak_mill.silo_min_t <= weak_plan["silo_after_t"] <= weak_mill.silo_max_t
    assert abs(weak_plan["silo_after_t"] - weak_mill.initial_silo_t) > weak_mill.terminal_tolerance_t

    strong_ev = strong.dr_devices["mg3"]
    weak_ev = weak.dr_devices["mg3"]
    strong_low, _ = strong_ev.feasible_bounds(6)
    weak_low, weak_high = weak_ev.feasible_bounds(6)
    assert strong_low > 0.0
    assert weak_low == pytest.approx(0.0)
    assert weak_high > 0.0

    weak_hvac = weak.dr_devices["mg2"]

    def forbidden(*args, **kwargs):
        raise AssertionError("instantaneous projection must not call terminal lookahead")

    weak_hvac._solve_terminal_q = forbidden
    low, high = weak_hvac.feasible_bounds(0)
    assert -weak_hvac.cool_max_kw <= low <= high <= weak_hvac.heat_max_kw


def test_instantaneous_environment_penalizes_terminal_constraint_violations() -> None:
    env = InstantaneousProjectionEnv(_load(WEAK_CONFIG), split="train")
    day = str(env.data.valid_days[0].date())
    observations, _ = env.reset(day=day, deterministic=True)
    done = False
    info = None
    rewards = None
    while not done:
        actions = {
            "mg1": np.asarray([0.0, -1.0], dtype=np.float32),
            "mg2": np.asarray([0.0, 0.0], dtype=np.float32),
            "mg3": np.asarray([0.0, -1.0], dtype=np.float32),
        }
        observations, rewards, done, info = env.step(actions)

    assert info is not None and rewards is not None
    assert info["constraint_projection_mode"] == "instantaneous"
    penalty = info["terminal_constraint_penalty"]
    assert penalty["mg3"]["ev_equivalent_kwh"] > 0.0
    assert penalty["mg3"]["total_penalty"] > 0.0
    assert info["costs"]["mg3"]["terminal_constraint_penalty"] == pytest.approx(
        penalty["mg3"]["total_penalty"]
    )
    assert rewards["mg1"] == pytest.approx(info["team_profit"] / env.accounting.reward_scale)
