from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.constraint_observations import ess_terminal_guidance
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"

ESS_FIELDS = {
    "terminal_soc_gap",
    "time_to_terminal",
    "required_avg_power_to_target",
    "terminal_pressure",
}
MILL_FIELDS = {
    "silo_target_gap",
    "time_remaining",
    "required_avg_production",
    "production_pressure",
}
HVAC_FIELDS = {
    "terminal_air_gap",
    "terminal_mass_gap",
    "time_remaining",
    "thermal_recovery_pressure",
}
EV_FIELDS = {"remaining_energy", "hours_to_departure", "deadline_pressure"}


@pytest.fixture()
def env() -> MultiMicrogridEnv:
    environment = MultiMicrogridEnv(CONFIG, split="train")
    day = str(environment.data.valid_days[0].date())
    environment.reset(day=day, deterministic=True)
    return environment


def test_local_and_global_schemas_expose_p0_fields(env: MultiMicrogridEnv) -> None:
    global_names = set(env.global_state_builder.schema())
    device_fields = {"mg1": MILL_FIELDS, "mg2": HVAC_FIELDS, "mg3": EV_FIELDS}

    for mg, fields in device_fields.items():
        assert ESS_FIELDS <= set(env.obs_builder.schema(mg))
        assert fields <= set(env.obs_builder.schema(mg))
        assert {f"{mg}_{name}" for name in ESS_FIELDS} <= global_names
        assert {f"{mg}_{name}" for name in fields} <= global_names


def test_ess_guidance_uses_dispatch_sign_and_efficiency(env: MultiMicrogridEnv) -> None:
    battery = env.cores["mg1"].battery
    battery.soc = battery.initial_soc - 0.10

    guidance = ess_terminal_guidance(battery, remaining_hours=4.0)

    expected_kw = -0.10 * battery.capacity_kwh / (battery.eta_charge * 4.0)
    expected_ratio = expected_kw / battery.max_charge_kw
    assert guidance["terminal_soc_gap"] == pytest.approx(-0.10)
    assert guidance["required_avg_power_kw"] == pytest.approx(expected_kw)
    assert guidance["required_avg_power_to_target"] == pytest.approx(expected_ratio)
    assert guidance["terminal_pressure"] == pytest.approx(abs(expected_ratio))


def test_ess_terminal_guidance_preserves_gap_without_dividing_by_zero(
    env: MultiMicrogridEnv,
) -> None:
    battery = env.cores["mg1"].battery
    battery.soc = battery.initial_soc + 0.10

    guidance = ess_terminal_guidance(battery, remaining_hours=0.0)

    assert guidance["terminal_soc_gap"] == pytest.approx(0.10)
    assert guidance["required_avg_power_kw"] == pytest.approx(0.0)
    assert guidance["required_avg_power_to_target"] == pytest.approx(0.0)
    assert guidance["terminal_pressure"] == pytest.approx(0.0)


def test_mill_guidance_reports_required_average_production(env: MultiMicrogridEnv) -> None:
    mill = env.dr_devices["mg1"]
    mill.silo_t = mill.initial_silo_t

    state = mill.state(20)

    expected_ratio = mill.downstream_tph / mill.production_tph
    assert state["time_remaining_h"] == pytest.approx(4.0)
    assert state["required_avg_production_tph"] == pytest.approx(mill.downstream_tph)
    assert state["production_pressure"] == pytest.approx(expected_ratio)


def test_hvac_guidance_uses_thermal_capacity_and_cop(env: MultiMicrogridEnv) -> None:
    hvac = env.dr_devices["mg2"]
    hvac.t_air = hvac.initial_t_air - 1.0
    hvac.t_mass = hvac.initial_t_mass - 0.5
    hvac._invalidate_bounds_cache()

    state = hvac.state(20)

    required_thermal_kw = (1.0 * hvac.c_air + 0.5 * hvac.c_mass) / 4.0
    expected_pressure = required_thermal_kw / hvac.cop_heat / hvac.heat_max_kw
    assert state["terminal_air_gap_c"] == pytest.approx(1.0)
    assert state["terminal_mass_gap_c"] == pytest.approx(0.5)
    assert state["time_remaining_h"] == pytest.approx(4.0)
    assert state["thermal_recovery_pressure"] == pytest.approx(expected_pressure)


def test_built_observations_and_state_match_schema_and_are_finite(
    env: MultiMicrogridEnv,
) -> None:
    observations = env._last_observations
    state = env.global_state()

    assert observations is not None
    for mg, observation in observations.items():
        assert observation.shape == (len(env.obs_builder.schema(mg)),)
        assert np.isfinite(observation).all()
    assert state.shape == (len(env.global_state_builder.schema()),)
    assert np.isfinite(state).all()
