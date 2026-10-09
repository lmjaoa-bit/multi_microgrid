from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from envs.multi_microgrid_env import MultiMicrogridEnv
from utils.metrics_collector import EpisodeMetricsCollector


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def env():
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    instance = MultiMicrogridEnv(cfg, split="test")
    instance.reset(day="2018-01-28", seed=42, deterministic=True, forecast_noise=True)
    return instance


def _targets(env):
    return {
        "mg1": {"battery_charge_kw": 0., "battery_discharge_kw": 0., "mill_duty": .4},
        "mg2": {"battery_charge_kw": 0., "battery_discharge_kw": 0.,
                "hvac_heating_kw": 0., "hvac_cooling_kw": 7.},
        "mg3": {"battery_charge_kw": 0., "battery_discharge_kw": 0., "flexible_load_kw": 0.},
    }


def test_inverse_mapping_uses_live_soc_and_ev_span(env):
    from diagnostics.oracle_action_replay import inverse_oracle_actions
    targets = _targets(env)
    battery = env.cores["mg1"].battery
    battery.soc = .11
    available = battery.available_discharge_kw()
    targets["mg1"]["battery_discharge_kw"] = available * .4
    low, high = env.dr_devices["mg3"].feasible_bounds(env.t)
    targets["mg3"]["flexible_load_kw"] = low + .3 * (high - low)
    actions, diagnostics = inverse_oracle_actions(env, targets)
    assert actions["mg1"] == pytest.approx([.4, -.2])
    assert actions["mg2"][1] == pytest.approx(-.2)
    assert actions["mg3"][1] == pytest.approx(-.4)
    assert diagnostics[0]["soc_before"] == pytest.approx(.11)
    assert diagnostics[0]["dispatch_request_error_kw"] == pytest.approx(0.)
    for mg in env.agent_ids:
        assert env.dr_devices[mg].plan(actions[mg][1], env.t)["flexible_load_kw"] == pytest.approx(
            targets[mg].get("flexible_load_kw", 28.5 * .4 if mg == "mg1" else 7.))


def test_unavailable_discharge_is_logged_without_state_injection(env):
    from diagnostics.oracle_action_replay import inverse_oracle_actions
    targets = _targets(env)
    targets["mg1"]["battery_discharge_kw"] = 20.
    original_soc = env.cores["mg1"].battery.soc
    actions, diagnostics = inverse_oracle_actions(env, targets)
    assert actions["mg1"][0] == 1.
    assert diagnostics[0]["dispatch_request_error_kw"] == pytest.approx(-20.)
    assert diagnostics[0]["dispatch_target_clipped"]
    assert env.cores["mg1"].battery.soc == original_soc


@pytest.mark.parametrize("invalid", ["simultaneous", "nonfinite", "negative"])
def test_inverse_mapping_rejects_invalid_device_targets(env, invalid):
    from diagnostics.oracle_action_replay import inverse_oracle_actions
    targets = _targets(env)
    if invalid == "simultaneous":
        targets["mg1"].update(battery_charge_kw=10., battery_discharge_kw=10.)
    elif invalid == "nonfinite":
        targets["mg2"]["hvac_cooling_kw"] = np.nan
    else:
        targets["mg1"]["battery_charge_kw"] = -1.
    with pytest.raises(ValueError):
        inverse_oracle_actions(env, targets)


def _feasible_schedule(env):
    collector = EpisodeMetricsCollector(env, capture_hourly=True)
    targets = []
    for t in range(24):
        actions = {mg: np.array([-.5 if t < 12 else .5, [.4, -.1, .2][i]])
                   for i, mg in enumerate(env.agent_ids)}
        _, rewards, _, info = env.step(actions)
        collector.record_step(actions, info, rewards)
        for mg in env.agent_ids:
            device = info["devices"][mg]
            dr = info["dr_plans"][mg] if "dr_plans" in info else info["plans"][mg]["dr_plan"]
            targets.append({
                "timestamp": str(info["timestamp"]), "microgrid": mg,
                "battery_charge_kw": device["charge_kw"],
                "battery_discharge_kw": device["discharge_kw"], "soc": device["soc"],
                "base_load_kw": info["plans"][mg]["base_load_kw"],
                "renewable_kw": info["plans"][mg]["renewable_kw"],
                "flexible_load_kw": dr["flexible_load_kw"],
                "mill_duty": dr.get("effective_duty", np.nan),
                "hvac_heating_kw": dr.get("heating_power_kw", np.nan),
                "hvac_cooling_kw": dr.get("cooling_power_kw", np.nan),
            })
    return pd.DataFrame(targets), collector.finalize_episode(), collector.hourly_rows


def test_full_day_feasible_schedule_replays_identical_physics_and_profit(env):
    from diagnostics.oracle_action_replay import replay_oracle_trajectory
    targets, original, hourly = _feasible_schedule(env)
    result = replay_oracle_trajectory(env, targets, day="2018-01-28", seed=42,
                                      forecast_noise=True)
    assert len(result["hourly_rows"]) == 24
    assert len(result["action_diagnostics"]) == 72
    assert result["metrics"]["team_profit"] == pytest.approx(original["team_profit"], abs=1e-6)
    for actual, expected in zip(result["hourly_rows"], hourly):
        for mg in env.agent_ids:
            for suffix in ("soc", "grid_import_kw", "p2p_sell_kw", "flexible_load_kw"):
                assert actual[f"{mg}_{suffix}"] == pytest.approx(expected[f"{mg}_{suffix}"], abs=1e-6)


def test_daily_diagnostic_reads_authoritative_service_schema(env):
    from diagnostics.oracle_action_replay import replay_oracle_trajectory
    from scripts.replay_oracle_actions import _day_metrics
    from optimization.common_constraints import DispatchResult
    from evaluation.service_metrics import summarize_day_services, SERVICE_METRICS_SCHEMA_VERSION
    targets, original, _ = _feasible_schedule(env)
    replay = replay_oracle_trajectory(env, targets, day="2018-01-28", seed=42, forecast_noise=True)
    oracle = DispatchResult("test", "kOptimal", -original["team_profit"],
                            original["team_profit"], {}, targets, {}, metadata={})
    actual = _day_metrics(replay, oracle, env.config)
    expected = summarize_day_services(replay["hourly_rows"], env.config, True, SERVICE_METRICS_SCHEMA_VERSION)
    assert actual["hvac_comfort_success_rate"] == pytest.approx(expected["hvac"]["all_boundary_comfort_rate"])
    assert actual["ev_departure_shortfall_kwh"] == pytest.approx(expected["ev"]["departure_shortfall_kwh"])


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "scenario"])
def test_replay_rejects_incomplete_duplicate_or_wrong_scenario(env, corruption):
    from diagnostics.oracle_action_replay import replay_oracle_trajectory
    targets, _, _ = _feasible_schedule(env)
    if corruption == "missing":
        targets = targets.iloc[:-1]
    elif corruption == "duplicate":
        targets = pd.concat([targets.iloc[:-1], targets.iloc[:1]])
    else:
        targets.loc[0, "base_load_kw"] += 1.
    with pytest.raises(ValueError):
        replay_oracle_trajectory(env, targets, day="2018-01-28", seed=42, forecast_noise=True)


def test_source_loader_rejects_duplicate_dates_and_incomplete_runs(tmp_path):
    from scripts.replay_oracle_actions import load_run_inputs
    (tmp_path / "config_snapshot.yaml").write_text(yaml.safe_dump({
        "environment": {"action_design": "single_dispatch_dr"}}), encoding="utf-8")
    (tmp_path / "test_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_status.json").write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    pd.DataFrame({"date": ["2018-01-28"] * 2, "test_seed": [42, 43],
                  "algorithm_team_profit": [-1., -2.]}).to_csv(tmp_path / "test_daily_metrics.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_run_inputs(tmp_path)
    pd.DataFrame({"date": ["2018-01-28"], "test_seed": [42],
                  "algorithm_team_profit": [-1.]}).to_csv(tmp_path / "test_daily_metrics.csv", index=False)
    (tmp_path / "run_status.json").write_text(json.dumps({"status": "TRAINED"}), encoding="utf-8")
    with pytest.raises(ValueError, match="COMPLETED"):
        load_run_inputs(tmp_path)


@pytest.mark.parametrize("column", ["generalized_reference_scenario_fingerprint",
                                    "generalized_reference_initial_state_fingerprint"])
@pytest.mark.parametrize("bad_value", ["different", None])
def test_pairing_rejects_different_or_missing_scenario_fingerprints(column, bad_value):
    from scripts.replay_oracle_actions import validate_paired_runs
    reference = pd.DataFrame({
        "date": ["2018-01-28"], "test_seed": [42],
        "generalized_reference_scenario_fingerprint": ["a" * 64],
        "generalized_reference_initial_state_fingerprint": ["b" * 64],
    })
    other = reference.copy()
    other.loc[0, column] = bad_value
    with pytest.raises(ValueError, match="fingerprint"):
        validate_paired_runs({}, reference, {}, other)


def test_pairing_aligns_dates_and_identical_fingerprints():
    from scripts.replay_oracle_actions import validate_paired_runs
    reference = pd.DataFrame({
        "date": ["2018-01-28", "2018-01-29"], "test_seed": [42, 43],
        "generalized_reference_scenario_fingerprint": ["a" * 64, "b" * 64],
        "generalized_reference_initial_state_fingerprint": ["c" * 64] * 2,
    })
    actual = validate_paired_runs({}, reference, {}, reference.iloc[::-1])
    pd.testing.assert_frame_equal(actual, reference)


@pytest.mark.parametrize("location", ["release", "source", "other_study"])
def test_output_validation_rejects_outside_selected_algorithm_studies(location):
    from scripts.replay_oracle_actions import validate_study_output
    run = ROOT / "outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/runs/source_run"
    candidates = {
        "release": ROOT / "outputs/releases/new_study",
        "source": run / "new_study",
        "other_study": ROOT / "outputs/working_runs/matd3_dispatch_q_guided/studies/new_study",
    }
    with pytest.raises(ValueError, match="studies"):
        validate_study_output(run, candidates[location])


def test_output_validation_accepts_new_selected_algorithm_study():
    from scripts.replay_oracle_actions import validate_study_output
    run = ROOT / "outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/runs/source_run"
    output = run.parents[1] / "studies/new_study"
    assert validate_study_output(run, output) == output.resolve()
