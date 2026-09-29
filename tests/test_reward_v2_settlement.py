from __future__ import annotations

import pandas as pd
import pytest

import evaluation.coalition_evaluator as coalition_module
from evaluation.coalition_evaluator import evaluate_day_references
from evaluation.settlement import settle_test_day
from optimization.common_constraints import DispatchData, DispatchResult


def _dummy_data():
    ids = ["mg1", "mg2", "mg3"]
    return DispatchData(
        timestamps=["2026-01-01T00:00:00"],
        agent_ids=ids,
        load_kw={mg: [0.0] for mg in ids},
        renewable_kw={mg: [0.0] for mg in ids},
        buy_price={mg: [1.0] for mg in ids},
        sell_price={mg: [0.0] for mg in ids},
        frame=pd.DataFrame({"timestamp": ["2026-01-01T00:00:00"]}),
    )


def _dummy_result(members, objective_kind):
    members = list(members)
    accounting_profit = float(10 * len(members))
    return DispatchResult(
        mode="dummy",
        status="kOptimal",
        total_cost=123.0 if objective_kind == "generalized_cost" else -accounting_profit,
        total_profit=accounting_profit,
        profit_by_microgrid={mg: accounting_profit / len(members) for mg in members},
        trajectories=pd.DataFrame(),
        pair_flows_kwh={},
        cost_breakdown={},
        metadata={"objective_kind": objective_kind},
    )


def test_seven_accounting_calls_plus_one_generalized_call(monkeypatch):
    calls = []

    monkeypatch.setattr(
        coalition_module,
        "build_dispatch_data",
        lambda *args, **kwargs: (
            {"microgrids": [{"id": mg, "dr": {"type": "dummy"}} for mg in ("mg1", "mg2", "mg3")]},
            _dummy_data(),
        ),
    )

    def fake_solve(_cfg, _data, members, cooperative, **kwargs):
        objective_kind = kwargs.get("objective_kind", "accounting_profit")
        calls.append((tuple(members), objective_kind))
        return _dummy_result(members, objective_kind)

    monkeypatch.setattr(coalition_module, "solve_dispatch", fake_solve)
    references = evaluate_day_references(
        {}, split="test", day="2026-01-01", include_generalized=True
    )

    accounting = [item for item in calls if item[1] == "accounting_profit"]
    generalized = [item for item in calls if item[1] == "generalized_cost"]
    assert len(accounting) == 7
    assert len(generalized) == 1
    assert generalized[0][0] == ("mg1", "mg2", "mg3")
    assert references.accounting.valid is True
    assert references.accounting.values["v_grand_coalition"] == pytest.approx(30.0)
    assert references.generalized is not None
    assert references.generalized.total_cost == pytest.approx(123.0)
    # The generalized cost must not overwrite the accounting grand-coalition value.
    assert references.accounting.values["v_grand_coalition"] != references.generalized.total_cost


def test_accounting_transfers_conserve_actual_profit():
    values = {
        "v_mg1": 10.0,
        "v_mg2": 20.0,
        "v_mg3": 30.0,
        "v_mg1_mg2": 35.0,
        "v_mg1_mg3": 50.0,
        "v_mg2_mg3": 60.0,
        "v_grand_coalition": 75.0,
    }
    physical = {"mg1": 12.0, "mg2": 23.0, "mg3": 40.0}
    result = settle_test_day(
        values,
        matd3_joint_profit=75.0,
        physical_profit_by_microgrid=physical,
    )
    assert result["equal_budget_balance_error"] == pytest.approx(0.0, abs=1e-10)
    assert result["equal_team_profit_invariance_error"] == pytest.approx(0.0, abs=1e-10)
    if result["shapley_allocation_valid"]:
        assert result["shapley_budget_balance_error"] == pytest.approx(0.0, abs=1e-10)
        assert result["shapley_team_profit_invariance_error"] == pytest.approx(0.0, abs=1e-10)


def test_missing_coalition_and_negative_surplus_are_visible():
    missing = {
        "v_mg1": 10.0,
        "v_mg2": 20.0,
        "v_mg3": None,
        "v_mg1_mg2": 35.0,
        "v_mg1_mg3": 45.0,
        "v_mg2_mg3": 55.0,
        "v_grand_coalition": 70.0,
    }
    unavailable = settle_test_day(
        missing,
        matd3_joint_profit=60.0,
        physical_profit_by_microgrid={"mg1": 15.0, "mg2": 20.0, "mg3": 25.0},
    )
    assert unavailable["v_mg3"] is None
    assert unavailable["independent_total_profit"] is None
    assert unavailable["shapley_allocation_valid"] is False

    values = {
        "v_mg1": 20.0,
        "v_mg2": 20.0,
        "v_mg3": 20.0,
        "v_mg1_mg2": 40.0,
        "v_mg1_mg3": 40.0,
        "v_mg2_mg3": 40.0,
        "v_grand_coalition": 55.0,
    }
    negative = settle_test_day(
        values,
        matd3_joint_profit=50.0,
        physical_profit_by_microgrid={"mg1": 15.0, "mg2": 15.0, "mg3": 20.0},
    )
    assert negative["matd3_cooperation_surplus"] == pytest.approx(-10.0)
    assert negative["negative_cooperation_surplus"] is True
    assert negative["matd3_cooperation_acceptable"] is False
    assert negative["equal_surplus_mg1"] == pytest.approx(-10.0 / 3.0)
