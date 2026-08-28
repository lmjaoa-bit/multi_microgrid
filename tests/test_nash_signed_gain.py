from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.evaluate_nash import BestResponseResult, _summary_row


def test_signed_and_profitable_gain_distinction():
    selected = BestResponseResult(
        focal_agent="mg1",
        restart=0,
        seed=42,
        initialization="equilibrium",
        best_validation_cost=100.0,
        best_episode=0,
        policy_state={},
        training_rows=[],
    )

    baseline_rows = [
        {
            "mg1_cost": 100.0,
            "mg1_reward": -10.0,
            "system_cost": 300.0,
            "p2p_kwh": 50.0,
            "grid_import_kwh": 20.0,
            "grid_export_kwh": 10.0,
            "carbon_kg": 5.0,
        }
    ]

    # Scenario A: BR retraining is worse than baseline (e.g. cost 105.0)
    br_worse_rows = [
        {
            "mg1_cost": 105.0,
            "mg1_reward": -10.5,
            "system_cost": 310.0,
            "p2p_kwh": 50.0,
            "grid_import_kwh": 20.0,
            "grid_export_kwh": 10.0,
            "carbon_kg": 5.0,
        }
    ]

    summary_worse = _summary_row(
        focal_agent="mg1",
        baseline_rows=baseline_rows,
        br_rows=br_worse_rows,
        selected=selected,
        restart_count=1,
        epsilon_threshold_pct=1.0,
    )

    assert pytest.approx(summary_worse["unilateral_gain_signed_abs"]) == -5.0
    assert summary_worse["unilateral_gain_signed_pct"] < 0.0
    assert pytest.approx(summary_worse["unilateral_gain_abs"]) == 0.0
    assert pytest.approx(summary_worse["unilateral_gain_pct"]) == 0.0
    assert summary_worse["br_beats_or_matches_baseline"] == 0
    assert summary_worse["best_br_is_baseline_policy"] == 1

    # Scenario B: BR retraining improves over baseline (e.g. cost 90.0)
    selected_better = BestResponseResult(
        focal_agent="mg1",
        restart=1,
        seed=42,
        initialization="equilibrium",
        best_validation_cost=90.0,
        best_episode=15,
        policy_state={},
        training_rows=[],
    )
    br_better_rows = [
        {
            "mg1_cost": 90.0,
            "mg1_reward": -9.0,
            "system_cost": 280.0,
            "p2p_kwh": 50.0,
            "grid_import_kwh": 20.0,
            "grid_export_kwh": 10.0,
            "carbon_kg": 5.0,
        }
    ]

    summary_better = _summary_row(
        focal_agent="mg1",
        baseline_rows=baseline_rows,
        br_rows=br_better_rows,
        selected=selected_better,
        restart_count=1,
        epsilon_threshold_pct=1.0,
    )

    assert pytest.approx(summary_better["unilateral_gain_signed_abs"]) == 10.0
    assert summary_better["unilateral_gain_signed_pct"] > 0.0
    assert pytest.approx(summary_better["unilateral_gain_abs"]) == 10.0
    assert pytest.approx(summary_better["unilateral_gain_pct"]) == 10.0
    assert summary_better["br_beats_or_matches_baseline"] == 1
    assert summary_better["best_br_is_baseline_policy"] == 0
