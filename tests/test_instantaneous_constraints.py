from __future__ import annotations

from pathlib import Path

import pytest

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
