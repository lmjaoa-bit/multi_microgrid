from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from core.config_validation import validate_config
from core.reward_v2 import (
    REWARD_V2_VERSION,
    calculate_reward_v2,
    resolve_carbon_factors,
    resolve_reward_v2_config,
)
from experiments.profiles import ROOT


def _legacy_config():
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg.pop("reward", None)
    for key in (
        "curtailment_price_per_kwh",
        "carbon_price_per_kg",
        "peak_price_per_kw",
        "mill_unserved_price_per_t",
        "comfort_price_per_degree_hour",
        "ev_shortfall_price_per_kwh",
    ):
        cfg.get("cost", {}).pop(key, None)
    return cfg


def _v2_config():
    cfg = deepcopy(_legacy_config())
    cfg["reward"] = {
        "semantics_version": REWARD_V2_VERSION,
        "calibration_status": "diagnostic",
    }
    cfg["cost"].update(
        curtailment_price_per_kwh=cfg["cost"]["renewable_curtailment_cost_per_kwh"],
        carbon_price_per_kg=0.1,
        peak_price_per_kw=1.0,
        mill_unserved_price_per_t=10.0,
        comfort_price_per_degree_hour=2.0,
        ev_shortfall_price_per_kwh=cfg["microgrids"][2]["dr"]["departure_penalty_per_kwh"],
    )
    for mg in cfg["microgrids"]:
        mg["grid"]["export_limit_kw"] = 0.0
    cfg["microgrids"][1]["dr"]["control_semantics"] = "soft_comfort"
    cfg["microgrids"][2]["dr"]["control_semantics"] = "no_deadline_minimum"
    return cfg


def test_legacy_reward_has_no_v2_contract():
    assert resolve_reward_v2_config(_legacy_config()) is None


def test_v2_alias_conflict_rejected():
    cfg = _v2_config()
    cfg["cost"]["curtailment_price_per_kwh"] = 9.0
    with pytest.raises(ValueError, match="Conflicting reward aliases"):
        resolve_reward_v2_config(cfg)

    cfg = _v2_config()
    cfg["cost"]["ev_shortfall_price_per_kwh"] = 9.0
    with pytest.raises(ValueError, match="Conflicting reward aliases"):
        resolve_reward_v2_config(cfg)


def test_v2_units_and_missing_values():
    cfg = _v2_config()
    resolved = resolve_reward_v2_config(cfg)
    assert resolved is not None
    assert resolved["reward_scale"] > 0
    validate_config(cfg)

    for key in (
        "carbon_price_per_kg",
        "peak_price_per_kw",
        "mill_unserved_price_per_t",
        "comfort_price_per_degree_hour",
    ):
        bad = _v2_config()
        del bad["cost"][key]
        with pytest.raises((TypeError, ValueError)):
            resolve_reward_v2_config(bad)

    bad = _v2_config()
    bad["cost"]["carbon_price_per_kg"] = np.nan
    with pytest.raises(ValueError):
        resolve_reward_v2_config(bad)

    bad = _v2_config()
    bad["time"]["episode_length"] = 23
    with pytest.raises(ValueError, match="24-hour"):
        resolve_reward_v2_config(bad)


def test_v2_requires_zero_export_and_soft_service_semantics():
    bad = _v2_config()
    bad["microgrids"][0]["grid"]["export_limit_kw"] = 1.0
    with pytest.raises(ValueError, match="export_limit_kw=0"):
        validate_config(bad)

    bad = _v2_config()
    bad["microgrids"][1]["dr"]["control_semantics"] = "legacy_comfort_projection"
    with pytest.raises(ValueError, match="soft_comfort"):
        validate_config(bad)

    bad = _v2_config()
    bad["microgrids"][2]["dr"]["control_semantics"] = "legacy_deadline_minimum"
    with pytest.raises(ValueError, match="no_deadline_minimum"):
        validate_config(bad)


def test_carbon_source_is_explicit():
    cfg = _v2_config()
    frame = pd.DataFrame({"grid_carbon_factor_kg_per_kwh": [0.4, 0.5]})
    assert resolve_carbon_factors(cfg, frame).tolist() == [0.4, 0.5]

    missing = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="grid_carbon_factor"):
        resolve_carbon_factors(cfg, missing)

    cfg["reward"]["fixed_carbon_factor_kg_per_kwh"] = 0.5
    assert resolve_carbon_factors(cfg, missing).tolist() == [0.5, 0.5]

    bad = pd.DataFrame({"grid_carbon_factor_kg_per_kwh": [0.5, np.nan]})
    with pytest.raises(ValueError):
        resolve_carbon_factors(cfg, bad)


def test_reward_v2_pure_numeric_fixture():
    raw = dict(zip(
        (
            "grid_import_cost",
            "battery_degradation_cost",
            "curtailment_kwh",
            "carbon_kg",
            "delta_peak_kw",
            "mill_unserved_t",
            "comfort_degree_hours",
            "ev_departure_shortfall_kwh",
            "unmet_kwh",
        ),
        [10, 2, 3, 4, 5, 6, 7, 8, 9],
    ))
    rates = {
        "curtailment_price_per_kwh": 1,
        "carbon_price_per_kg": 1,
        "peak_price_per_kw": 1,
        "mill_unserved_price_per_t": 1,
        "comfort_price_per_degree_hour": 1,
        "ev_shortfall_price_per_kwh": 1,
        "unmet_load_cost_per_kwh": 1,
        "reward_scale": 10,
    }
    result = calculate_reward_v2(raw=raw, rates=rates)
    assert result["generalized_system_cost"] == pytest.approx(54.0)
    assert result["team_reward"] == pytest.approx(-5.4)



def test_locked_v2_requires_calibration_report_hash():
    cfg = _v2_config()
    cfg["reward"]["calibration_status"] = "locked"
    with pytest.raises(ValueError, match="calibration_report_sha256"):
        resolve_reward_v2_config(cfg)

    cfg["reward"]["calibration_report_sha256"] = "a" * 64
    resolved = resolve_reward_v2_config(cfg)
    assert resolved["calibration_report_sha256"] == "a" * 64
