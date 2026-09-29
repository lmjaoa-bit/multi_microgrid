from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core.reward_v2 import calculate_reward_v2, resolve_carbon_factors
from optimization.common_constraints import build_dispatch_data, solve_dispatch
from optimization.cooperative_dispatch import solve_generalized_dispatch_reference


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


def _independent_cost_recompute(cfg, data, result):
    frame = result.trajectories
    carbon = resolve_carbon_factors(cfg, data.frame)
    by_id = {str(item["id"]): item for item in cfg["microgrids"]}
    raw = {
        "grid_import_cost": 0.0,
        "battery_degradation_cost": 0.0,
        "curtailment_kwh": 0.0,
        "carbon_kg": 0.0,
        "delta_peak_kw": 0.0,
        "mill_unserved_t": 0.0,
        "comfort_degree_hours": 0.0,
        "ev_departure_shortfall_kwh": 0.0,
        "unmet_kwh": 0.0,
    }
    peak = 0.0
    for t, timestamp in enumerate(data.timestamps):
        rows = frame.loc[frame["timestamp"] == timestamp]
        peak = max(peak, float(rows["load_kw"].sum()))
        for _, row in rows.iterrows():
            mg = str(row["microgrid"])
            grid = float(row["grid_import_kw"])
            buy = (
                float(data.buy_price[mg][t])
                if isinstance(data.buy_price, dict)
                else float(data.buy_price[t])
            )
            raw["grid_import_cost"] += grid * buy
            raw["battery_degradation_cost"] += (
                float(row["battery_charge_kw"]) + float(row["battery_discharge_kw"])
            ) * float(by_id[mg]["battery"]["degradation_cost_per_kwh"])
            raw["curtailment_kwh"] += float(row["curtailment_kw"])
            raw["carbon_kg"] += grid * float(carbon[t])
            comfort = row.get("comfort_degree_hours", 0.0)
            raw["comfort_degree_hours"] += (
                float(comfort) if pd.notna(comfort) else 0.0
            )
            raw["unmet_kwh"] += float(row["unmet_load_kw"])
            raw["ev_departure_shortfall_kwh"] += sum(
                float(value)
                for key, value in row.items()
                if str(key).startswith("ev_")
                and str(key).endswith("_departure_shortfall_kwh")
                and pd.notna(value)
            )
    raw["delta_peak_kw"] = peak
    rates = {
        "curtailment_price_per_kwh": cfg["cost"]["curtailment_price_per_kwh"],
        "carbon_price_per_kg": cfg["cost"]["carbon_price_per_kg"],
        "peak_price_per_kw": cfg["cost"]["peak_price_per_kw"],
        "mill_unserved_price_per_t": cfg["cost"]["mill_unserved_price_per_t"],
        "comfort_price_per_degree_hour": cfg["cost"]["comfort_price_per_degree_hour"],
        "ev_shortfall_price_per_kwh": cfg["cost"]["ev_shortfall_price_per_kwh"],
        "unmet_load_cost_per_kwh": cfg["cost"]["unmet_load_cost_per_kwh"],
        "reward_scale": cfg["cost"]["reward_scale"],
    }
    return calculate_reward_v2(raw=raw, rates=rates)


def test_generalized_solver_objective_matches_trajectory():
    cfg = _v2_config()
    _, data = build_dispatch_data(cfg, split="test", day="2018-01-28")
    result = solve_generalized_dispatch_reference(
        cfg, split="test", day="2018-01-28", solver_output=False
    )
    replayed = _independent_cost_recompute(cfg, data, result)
    tolerance = 1e-6 * max(1.0, abs(float(result.total_cost)))
    assert abs(
        replayed["generalized_system_cost"] - float(result.total_cost)
    ) <= tolerance
    assert result.metadata["objective_kind"] == "generalized_cost"
    assert result.metadata["hvac_comfort_semantics"] == "soft_penalty"
    assert result.metadata["replayed_trajectory_cost"] == pytest.approx(
        result.total_cost, rel=1e-6, abs=1e-6
    )


def test_oracle_soft_service_and_zero_export():
    cfg = _v2_config()
    result = solve_generalized_dispatch_reference(
        cfg, split="test", day="2018-01-28", solver_output=False
    )
    frame = result.trajectories
    assert float(frame["grid_export_kw"].abs().max()) <= 1e-8

    hvac = frame.loc[frame["microgrid"] == "mg2"]
    assert {
        "comfort_low_slack_c",
        "comfort_high_slack_c",
        "comfort_violation_c",
        "comfort_degree_hours",
    } <= set(hvac.columns)
    assert (hvac["comfort_low_slack_c"] >= -1e-8).all()
    assert (hvac["comfort_high_slack_c"] >= -1e-8).all()
    assert (
        hvac["hvac_heating_kw"] * hvac["hvac_cooling_kw"]
    ).abs().max() <= 1e-7

    ev = frame.loc[frame["microgrid"] == "mg3"]
    shortfall_columns = [
        name for name in ev.columns
        if name.startswith("ev_") and name.endswith("_departure_shortfall_kwh")
    ]
    assert shortfall_columns
    assert all((ev[name].fillna(0.0) >= -1e-8).all() for name in shortfall_columns)

    actual_peak = float(
        frame.groupby("timestamp", sort=False)["load_kw"].sum().max()
    )
    expected_peak_cost = actual_peak * float(cfg["cost"]["peak_price_per_kw"])
    assert result.cost_breakdown["cost_peak"] == pytest.approx(
        expected_peak_cost, rel=1e-6, abs=1e-6
    )


def test_accounting_coalition_oracle_keeps_v2_hvac_comfort_soft():
    cfg = _v2_config()
    cfg["microgrids"][1]["dr"]["occupied_comfort_c"] = [60.0, 61.0]
    cfg["microgrids"][1]["dr"]["unoccupied_comfort_c"] = [60.0, 61.0]
    _, data = build_dispatch_data(cfg, split="test", day="2018-01-28")

    result = solve_dispatch(
        cfg,
        data,
        data.agent_ids,
        cooperative=True,
        include_dr=True,
        objective_kind="accounting_profit",
    )

    assert result.status == "kOptimal"
    assert result.metadata["hvac_comfort_semantics"] == (
        "soft_service_unpriced_in_accounting_objective"
    )
    hvac = result.trajectories.loc[result.trajectories["microgrid"] == "mg2"]
    assert "comfort_violation_c" in hvac
    assert float(hvac["comfort_violation_c"].max()) > 1.0


def test_solver_incumbent_metadata_is_not_presented_as_bound():
    cfg = _v2_config()
    result = solve_generalized_dispatch_reference(
        cfg, split="test", day="2018-01-28", solver_output=False
    )
    meta = result.metadata
    assert meta["feasible_incumbent"] is True
    assert np.isfinite(float(meta["solver_objective_value"]))
    assert meta["solver_objective_value"] == pytest.approx(
        result.total_cost, rel=1e-6, abs=1e-6
    )
    # The dual bound is separate evidence. A feasible incumbent must never be
    # copied into the bound field merely to make a "lower bound" available.
    if meta["solver_objective_bound"] is not None:
        assert np.isfinite(float(meta["solver_objective_bound"]))
        assert float(meta["solver_objective_bound"]) <= (
            float(meta["solver_objective_value"]) + 1e-6
        )
    if meta["proven_optimal"]:
        assert meta["solver_mip_gap"] is None or float(meta["solver_mip_gap"]) <= 1e-7
