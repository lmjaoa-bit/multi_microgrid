from __future__ import annotations

import math
from pathlib import Path
import pytest
import yaml

from experiments.checkpoint_selector import CheckpointCandidate, CheckpointSelector


def test_profit_only_mode_legacy_ranking():
    selector = CheckpointSelector({"mode": "profit_only", "objective": "team_profit_mean"})
    assert selector.mode == "profit_only"

    c1 = selector.evaluate({"team_profit_mean": 100.0}, episode=100)
    assert c1.eligible is True
    assert c1.total_violation == 0.0
    assert c1.objective_value == 100.0

    c2 = selector.evaluate({"team_profit_mean": 105.0}, episode=200)
    assert selector.is_better(c2, c1) is True
    assert selector.is_better(c1, c2) is False

    # Tie on profit: later episode wins
    c3 = selector.evaluate({"team_profit_mean": 105.0}, episode=300)
    assert selector.is_better(c3, c2) is True
    assert selector.is_better(c2, c3) is False


def test_constrained_profit_eligible_outranks_ineligible():
    cfg = {
        "mode": "constrained_profit",
        "objective": "team_profit_mean",
        "constraints": {
            "strict_feasible_day_rate": {"min": 1.0},
            "unmet_energy_mean_kwh": {"max": 0.0, "atol": 1e-9},
            "mill_downstream_service_completion_rate": {"min": 1.0},
        },
    }
    selector = CheckpointSelector(cfg)

    # Ineligible candidate with very high profit (violation on unmet energy)
    ineligible_high_profit = selector.evaluate(
        {
            "team_profit_mean": 200.0,
            "strict_feasible_day_rate": 1.0,
            "unmet_energy_mean_kwh": 5.0,  # violates max 0.0
            "mill_downstream_service_completion_rate": 1.0,
        },
        episode=100,
    )
    assert ineligible_high_profit.eligible is False
    assert ineligible_high_profit.total_violation > 0.0

    # Eligible candidate with lower profit
    eligible_lower_profit = selector.evaluate(
        {
            "team_profit_mean": 150.0,
            "strict_feasible_day_rate": 1.0,
            "unmet_energy_mean_kwh": 0.0,
            "mill_downstream_service_completion_rate": 1.0,
        },
        episode=200,
    )
    assert eligible_lower_profit.eligible is True
    assert eligible_lower_profit.total_violation <= 1e-9

    # Eligible candidate MUST outrank ineligible candidate despite lower profit
    assert selector.is_better(eligible_lower_profit, ineligible_high_profit) is True
    assert selector.is_better(ineligible_high_profit, eligible_lower_profit) is False


def test_constrained_profit_among_eligible_higher_objective_wins():
    cfg = {
        "mode": "constrained_profit",
        "objective": "team_profit_mean",
        "constraints": {
            "strict_feasible_day_rate": {"min": 1.0},
        },
    }
    selector = CheckpointSelector(cfg)

    c1 = selector.evaluate({"team_profit_mean": 100.0, "strict_feasible_day_rate": 1.0}, episode=100)
    c2 = selector.evaluate({"team_profit_mean": 120.0, "strict_feasible_day_rate": 1.0}, episode=200)

    assert selector.is_better(c2, c1) is True
    assert selector.is_better(c1, c2) is False


def test_constrained_profit_among_ineligible_lower_violation_wins():
    cfg = {
        "mode": "constrained_profit",
        "objective": "team_profit_mean",
        "constraints": {
            "strict_feasible_day_rate": {"min": 1.0},
            "unmet_energy_mean_kwh": {"max": 0.0, "atol": 1e-9},
        },
    }
    selector = CheckpointSelector(cfg)

    # Candidate 1: strict_feasible = 0.5 -> viol = (1.0 - 0.5) / 1.0 = 0.5
    c1 = selector.evaluate(
        {"team_profit_mean": 100.0, "strict_feasible_day_rate": 0.5, "unmet_energy_mean_kwh": 0.0},
        episode=100,
    )
    # Candidate 2: strict_feasible = 0.8 -> viol = (1.0 - 0.8) / 1.0 = 0.2
    c2 = selector.evaluate(
        {"team_profit_mean": 80.0, "strict_feasible_day_rate": 0.8, "unmet_energy_mean_kwh": 0.0},
        episode=200,
    )

    # c2 has lower violation (0.2 < 0.5), so c2 outranks c1 even though c1 has higher profit
    assert selector.is_better(c2, c1) is True
    assert selector.is_better(c1, c2) is False


def test_constrained_profit_missing_and_nonfinite_metrics():
    cfg = {
        "mode": "constrained_profit",
        "objective": "team_profit_mean",
        "constraints": {
            "strict_feasible_day_rate": {"min": 1.0},
            "unmet_energy_mean_kwh": {"max": 0.0},
        },
    }
    selector = CheckpointSelector(cfg)

    # Missing metric
    c_missing = selector.evaluate({"team_profit_mean": 100.0, "strict_feasible_day_rate": 1.0}, episode=100)
    assert c_missing.eligible is False
    assert math.isinf(c_missing.total_violation)

    # NaN metric
    c_nan = selector.evaluate(
        {"team_profit_mean": 100.0, "strict_feasible_day_rate": float("nan"), "unmet_energy_mean_kwh": 0.0},
        episode=200,
    )
    assert c_nan.eligible is False
    assert math.isinf(c_nan.total_violation)


def test_constrained_profit_tie_breaking():
    cfg = {
        "mode": "constrained_profit",
        "objective": "team_profit_mean",
        "constraints": {
            "strict_feasible_day_rate": {"min": 1.0},
        },
    }
    selector = CheckpointSelector(cfg)

    # Both ineligible with exact same violation
    c1 = selector.evaluate(
        {"team_profit_mean": 100.0, "strict_feasible_day_rate": 0.5}, episode=100
    )
    # Same violation, higher objective
    c2 = selector.evaluate(
        {"team_profit_mean": 110.0, "strict_feasible_day_rate": 0.5}, episode=200
    )
    assert selector.is_better(c2, c1) is True

    # Same violation, same objective, later episode
    c3 = selector.evaluate(
        {"team_profit_mean": 110.0, "strict_feasible_day_rate": 0.5}, episode=300
    )
    assert selector.is_better(c3, c2) is True


def test_current_service_selection_ignores_disabled_terminal_and_production_coverage():
    selector = CheckpointSelector({
        "mode": "constrained_profit",
        "objective": "team_profit_mean",
        "constraints": {
            "strict_feasible_day_rate": {"min": 1.0},
            "unmet_energy_mean_kwh": {"max": 0.0, "atol": 1e-7},
            "mill_downstream_service_completion_rate": {"min": 1.0, "atol": 1e-7},
            "ev_departure_success_rate": {"min": 1.0, "atol": 1e-7},
        },
    })
    candidate = selector.evaluate({
        "team_profit_mean": 10.0,
        "strict_feasible_day_rate": 1.0,
        "unmet_energy_mean_kwh": 0.0,
        "mill_downstream_service_completion_rate": 1.0,
        "mill_production_coverage_ratio": 0.7056,
        "mill_terminal_success_rate": None,
        "hvac_terminal_joint_success_rate": None,
        "ev_departure_success_rate": 1.0,
    }, episode=100)
    assert candidate.eligible is True
    assert candidate.constraint_failures == []


def test_current_service_selection_rejects_missing_required_service_metric():
    selector = CheckpointSelector({
        "mode": "constrained_profit",
        "objective": "team_profit_mean",
        "constraints": {
            "strict_feasible_day_rate": {"min": 1.0},
            "mill_downstream_service_completion_rate": {"min": 1.0},
        },
    })
    candidate = selector.evaluate({
        "team_profit_mean": 100.0,
        "strict_feasible_day_rate": 1.0,
    }, episode=100)
    assert candidate.eligible is False
    assert math.isinf(candidate.total_violation)


def test_current_algorithms_share_identical_service_checkpoint_selection():
    root = Path(__file__).resolve().parents[1]
    selections = []
    for name in (
        "matd3_dispatch.yaml",
        "matd3_hierarchical_transformer.yaml",
    ):
        cfg = yaml.safe_load(
            (root / "configs" / "algorithms" / name).read_text(encoding="utf-8")
        )
        selections.append(cfg["validation"]["selection"])
    assert selections[0] == selections[1]
    selection = selections[0]
    assert selection["mode"] == "constrained_profit"
    assert set(selection["constraints"]) == {
        "strict_feasible_day_rate",
        "unmet_energy_mean_kwh",
        "hvac_comfort_rate",
        "mill_downstream_service_completion_rate",
        "ev_departure_success_rate",
    }


def test_constrained_profit_rejects_first_ineligible_as_best():
    selector = CheckpointSelector({
        "mode": "constrained_profit",
        "objective": "team_profit_mean",
        "constraints": {"strict_feasible_day_rate": {"min": 1.0}},
    })
    candidate = selector.evaluate(
        {"team_profit_mean": 999.0, "strict_feasible_day_rate": 0.0},
        episode=100,
    )
    assert candidate.eligible is False
    assert selector.is_better(candidate, None) is False


def test_constrained_profit_accepts_first_eligible_as_best():
    selector = CheckpointSelector({
        "mode": "constrained_profit",
        "objective": "team_profit_mean",
        "constraints": {"strict_feasible_day_rate": {"min": 1.0}},
    })
    candidate = selector.evaluate(
        {"team_profit_mean": 10.0, "strict_feasible_day_rate": 1.0},
        episode=100,
    )
    assert candidate.eligible is True
    assert selector.is_better(candidate, None) is True
