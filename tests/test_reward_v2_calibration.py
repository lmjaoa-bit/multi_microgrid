from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

import evaluation.reward_calibration as calibration
from experiments.training_pipeline import _validate_reward_run_mode
from evaluation.reward_calibration import (
    build_idle_policy,
    build_service_priority_policy,
    collect_calibration,
)


ROOT = Path(__file__).resolve().parents[1]


def _v2_config():
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    cfg["reward"] = {
        "semantics_version": "generalized_system_cost_v2",
        "calibration_status": "diagnostic",
    }
    cfg["cost"].update(
        curtailment_price_per_kwh=cfg["cost"]["renewable_curtailment_cost_per_kwh"],
        carbon_price_per_kg=0.1,
        peak_price_per_kw=0.5,
        mill_unserved_price_per_t=10.0,
        comfort_price_per_degree_hour=2.0,
        ev_shortfall_price_per_kwh=cfg["microgrids"][2]["dr"]["departure_penalty_per_kwh"],
    )
    for mg in cfg["microgrids"]:
        mg["grid"]["export_limit_kw"] = 0.0
    cfg["microgrids"][1]["dr"]["control_semantics"] = "soft_comfort"
    cfg["microgrids"][2]["dr"]["control_semantics"] = "no_deadline_minimum"
    return cfg


def test_calibration_rejects_test_split():
    with pytest.raises(ValueError, match="train or val"):
        collect_calibration(
            _v2_config(),
            split="test",
            days=["2018-01-01"],
            policy="idle",
            seed=42,
        )


def test_calibration_records_components_and_scenarios(monkeypatch):
    class DummyEnv:
        pass

    monkeypatch.setattr(calibration, "MultiMicrogridEnv", lambda *a, **k: DummyEnv())

    daily = []
    for factor in (1.0, 2.0):
        raw = {
            "grid_import_cost": 100.0 * factor,
            "battery_degradation_cost": 5.0 * factor,
            "curtailment_kwh": 10.0 * factor,
            "carbon_kg": 20.0 * factor,
            "delta_peak_kw": 30.0 * factor,
            "mill_unserved_t": 0.0,
            "comfort_degree_hours": 1.0 * factor,
            "ev_departure_shortfall_kwh": 2.0 * factor,
            "unmet_kwh": 0.0,
        }
        daily.append({
            "reward_v2_raw_components": raw,
            "generalized_system_cost": 200.0 * factor,
        })

    def fake_evaluate(env, days, policy_fn, **kwargs):
        assert days == ["2018-01-01", "2018-01-02"]
        assert callable(policy_fn)
        return (
            {
                "team_reward_mean": -3.0,
                "generalized_system_cost_mean": 300.0,
                "team_profit_mean": -120.0,
            },
            daily,
        )

    monkeypatch.setattr(calibration, "evaluate_policy_on_days", fake_evaluate)

    def policy(_obs):
        return {}

    policy.calibration_policy_name = "fixture_policy"
    report = collect_calibration(
        _v2_config(),
        split="val",
        days=["2018-01-01", "2018-01-02"],
        policy=policy,
        seed=123,
    )
    assert report["split"] == "val"
    assert report["days"] == ["2018-01-01", "2018-01-02"]
    assert report["seed"] == 123
    assert report["policy"] == "fixture_policy"
    assert report["config_hash"]
    expected_raw = {
        "grid_import_cost",
        "battery_degradation_cost",
        "curtailment_kwh",
        "carbon_kg",
        "delta_peak_kw",
        "mill_unserved_t",
        "comfort_degree_hours",
        "ev_departure_shortfall_kwh",
        "unmet_kwh",
    }
    assert set(report["raw_component_distributions"]) == expected_raw
    assert "configured_design_candidate" in report["weight_candidates"]
    assert "numerical_scale_reference_not_for_locking" in report["weight_candidates"]
    assert "reason" in report["weight_candidates"]["configured_design_candidate"]
    assert report["reward_scale"] > 0.0
    assert "unit_p95_reference_scale" in report["scale_diagnostic"]


def test_idle_actions_are_device_specific():
    class DummyEnv:
        agent_ids = ["mg1", "mg2", "mg3"]
        cfg_by_id = {
            "mg1": {"dr": {"type": "industrial_mill"}},
            "mg2": {"dr": {"type": "commercial_hvac"}},
            "mg3": {"dr": {"type": "ev_charging"}},
        }

    policy = build_idle_policy(DummyEnv())
    actions = policy({})
    np.testing.assert_array_equal(actions["mg1"], np.asarray([0.0, -1.0], np.float32))
    np.testing.assert_array_equal(actions["mg2"], np.asarray([0.0, 0.0], np.float32))
    np.testing.assert_array_equal(actions["mg3"], np.asarray([0.0, -1.0], np.float32))


def test_service_priority_rule_does_not_call_hvac_viability():
    class Mill:
        def feasible_bounds(self, t):
            return (0.2, 1.0)

    class HVAC:
        t_air = 20.0
        t_mass = 20.0
        heat_max_kw = 35.0
        cool_max_kw = 35.0
        def _state_comfort_bounds(self, t):
            return (21.0, 25.0)
        def _next_state(self, t_air, t_mass, q_hvac, t):
            return (t_air + 0.1 * q_hvac, t_mass)
        def _q_to_signed_electric(self, q_hvac):
            return q_hvac
        def prepare_viability(self, *args, **kwargs):
            raise AssertionError("future viability must not be called")

    class EV:
        cohort_mode = True
        eta = 0.93
        sessions = [{
            "group_id": "G1",
            "arrival_hour": 0,
            "departure_hour": 4,
            "max_charge_kw": 10.0,
        }]
        remaining = {"G1": 20.0}
        def _hour(self, t):
            return t
        def feasible_bounds(self, t):
            return (0.0, 10.0)

    class DummyEnv:
        agent_ids = ["mg1", "mg2", "mg3"]
        t = 1
        cfg_by_id = {
            "mg1": {"dr": {"type": "industrial_mill"}},
            "mg2": {"dr": {"type": "commercial_hvac"}},
            "mg3": {"dr": {"type": "ev_charging"}},
        }
        dr_devices = {"mg1": Mill(), "mg2": HVAC(), "mg3": EV()}

    policy = build_service_priority_policy(DummyEnv())
    actions = policy({})
    assert actions["mg2"][1] > 0.0
    assert -1.0 <= actions["mg3"][1] <= 1.0



def test_unlocked_reward_cannot_start_formal_run():
    cfg = _v2_config()
    with pytest.raises(ValueError, match="cannot start a formal run"):
        _validate_reward_run_mode(
            cfg, training_episodes=10, diagnostic_only=False
        )

    with pytest.raises(ValueError, match="at most 100"):
        _validate_reward_run_mode(
            cfg, training_episodes=101, diagnostic_only=True
        )

    diagnostic = _validate_reward_run_mode(
        cfg, training_episodes=100, diagnostic_only=True
    )
    assert diagnostic["run_purpose"] == "reward_calibration"
    assert diagnostic["formal_test_allowed"] is False

    locked = deepcopy(cfg)
    locked["reward"]["calibration_status"] = "locked"
    locked["reward"]["calibration_report_sha256"] = "0" * 64
    formal = _validate_reward_run_mode(
        locked, training_episodes=5000, diagnostic_only=False
    )
    assert formal["run_purpose"] == "formal_training"
    assert formal["formal_test_allowed"] is True
