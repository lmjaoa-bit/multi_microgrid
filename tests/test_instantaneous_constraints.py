from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"


@pytest.fixture()
def env() -> MultiMicrogridEnv:
    environment = MultiMicrogridEnv(CONFIG, split="train")
    day = str(environment.data.valid_days[0].date())
    environment.reset(day=day, deterministic=True)
    return environment


def test_ess_last_hour_bounds_only_protect_current_soc(env: MultiMicrogridEnv) -> None:
    battery = env.cores["mg1"].battery
    battery.soc = battery.soc_min

    bounds = env._battery_flex(23)["mg1"]

    assert bounds["feasible_charge_min_ratio"] == pytest.approx(0.0)
    assert bounds["feasible_discharge_max_ratio"] == pytest.approx(0.0)


def test_mill_last_hour_only_protects_next_silo_state(env: MultiMicrogridEnv) -> None:
    mill = env.dr_devices["mg1"]
    mill.silo_t = mill.initial_silo_t

    plan = mill.plan(-1.0, 23)

    assert plan["effective_duty"] == pytest.approx(0.0)
    assert mill.silo_min_t <= plan["silo_after_t"] <= mill.silo_max_t
    assert abs(plan["silo_after_t"] - mill.initial_silo_t) > mill.terminal_tolerance_t


def test_ev_minimum_charge_is_zero_before_departure(env: MultiMicrogridEnv) -> None:
    ev = env.dr_devices["mg3"]

    low, high = ev.feasible_bounds(6)

    assert low == pytest.approx(0.0)
    assert high > 0.0


def test_hvac_bounds_never_call_terminal_optimizer(
    env: MultiMicrogridEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hvac = env.dr_devices["mg2"]

    def forbidden(*_args, **_kwargs):
        pytest.fail("terminal optimization called")

    monkeypatch.setattr(hvac, "_solve_terminal_q", forbidden, raising=False)
    hvac._invalidate_bounds_cache()

    low, high = hvac.feasible_bounds(0)

    assert -hvac.cool_max_kw <= low <= high <= hvac.heat_max_kw


def test_canonical_config_has_no_projection_mode() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert "constraint_projection" not in config
    assert config["terminal_soc"]["mode"] == "penalty"


def test_terminal_penalty_reports_every_device(env: MultiMicrogridEnv) -> None:
    done = False
    while not done:
        actions = {
            mg: np.asarray([0.0, -1.0], dtype=np.float32)
            for mg in env.agent_ids
        }
        _, rewards, done, info = env.step(actions)

    breakdown = info["terminal_constraint_penalty"]
    assert set(breakdown) == set(env.agent_ids)
    expected_fields = {
        "ess_equivalent_kwh",
        "mill_equivalent_kwh",
        "hvac_equivalent_kwh",
        "ev_equivalent_kwh",
        "total_equivalent_kwh",
        "total_penalty",
    }
    assert all(expected_fields <= set(item) for item in breakdown.values())
    assert breakdown["mg1"]["mill_equivalent_kwh"] > 0.0
    assert breakdown["mg3"]["ev_equivalent_kwh"] > 0.0
    assert info["terminal_constraint_penalty_total"] == pytest.approx(
        sum(item["total_penalty"] for item in breakdown.values())
    )
    assert rewards["mg1"] == pytest.approx(
        info["team_profit"] / env.accounting.reward_scale
    )
