from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from allocation.nash_bargaining import solve_nash_bargaining
from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.coalition_evaluator import evaluate_day_coalitions
from evaluation.compare_independent_cooperative import compare_dispatches
from evaluation.settlement import settle_test_day
from evaluation.shapley_allocator import calculate_shapley
from optimization.cooperative_dispatch import solve_cooperative_dispatch
from optimization.common_constraints import DispatchData, solve_dispatch
from optimization.independent_dispatch import solve_independent_dispatch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"


def test_independent_baseline_does_not_mutate_cooperative_environment() -> None:
    env = MultiMicrogridEnv(CONFIG)
    env.reset(seed=5, deterministic=True)
    state, soc, time = env.global_state(), {mg: env.cores[mg].battery.soc for mg in env.agent_ids}, env.t
    result = solve_independent_dispatch(CONFIG, split="test")
    assert np.array_equal(env.global_state(), state)
    assert {mg: env.cores[mg].battery.soc for mg in env.agent_ids} == soc
    assert env.t == time
    assert result.pair_flows_kwh == {}


def test_cooperative_oracle_power_balance() -> None:
    result = solve_cooperative_dispatch(CONFIG, split="test")
    frame = result.trajectories
    residual = (frame["renewable_kw"] + frame["battery_discharge_kw"] + frame["grid_import_kw"]
                + frame["p2p_import_kw"] + frame["unmet_load_kw"] - frame["load_kw"]
                - frame["battery_charge_kw"] - frame["grid_export_kw"]
                - frame["p2p_export_kw"] - frame["curtailment_kw"])
    assert residual.abs().max() < 1e-6
    assert not ((frame["grid_import_kw"] > 1e-8) & (frame["p2p_export_kw"] > 1e-8)).any()
    assert max(result.pair_flows_kwh.values(), default=0.0) <= 24 * 60.0 + 1e-6


def test_renewable_only_dispatch_rejects_battery_sourced_p2p_exports() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ids = ["mg1", "mg2"]
    data = DispatchData(
        timestamps=["test-hour"],
        agent_ids=ids,
        load_kw={"mg1": np.asarray([0.0]), "mg2": np.asarray([20.0])},
        renewable_kw={"mg1": np.asarray([0.0]), "mg2": np.asarray([0.0])},
        buy_price=np.asarray([1.0]),
        sell_price=np.asarray([0.5]),
    )
    initial_soc = {"mg1": 0.9, "mg2": 0.1}
    unrestricted = solve_dispatch(
        config,
        data,
        ids,
        cooperative=True,
        initial_soc_by_microgrid=initial_soc,
        apply_terminal_condition=False,
    )
    renewable_only = solve_dispatch(
        config,
        data,
        ids,
        cooperative=True,
        initial_soc_by_microgrid=initial_soc,
        apply_terminal_condition=False,
        p2p_export_source="renewable_only",
    )

    unrestricted_seller = unrestricted.trajectories.query("microgrid == 'mg1'").iloc[0]
    renewable_only_seller = renewable_only.trajectories.query("microgrid == 'mg1'").iloc[0]
    assert unrestricted_seller["p2p_export_kw"] > 0.0
    assert np.isclose(renewable_only_seller["p2p_export_kw"], 0.0)
    assert np.isclose(renewable_only_seller["battery_discharge_kw"], 0.0)


def test_comparison_exports_coordination_redispatch() -> None:
    summary, frame = compare_dispatches(CONFIG, split="test")
    assert "cooperation_gain" in summary
    assert "coordination_battery_action_delta_kw" in frame
    assert {"local_soc", "cooperative_soc", "p2p_import_kw"} <= set(frame)


def test_nash_bargaining_budget_balance_and_individual_rationality() -> None:
    result = solve_nash_bargaining({"mg1": 10.0, "mg2": 20.0, "mg3": 30.0}, 90.0,
                                   {"mg1": 5.0, "mg2": 25.0, "mg3": 60.0})
    assert result.individual_rational
    assert result.budget_balanced
    assert all(abs(value - 10.0) < 1e-8 for value in result.individual_cooperation_gain.values())
    assert abs(sum(result.allocated_cooperative_profit.values()) - 90.0) < 1e-8


def test_three_player_shapley_efficiency_and_actual_surplus_settlement() -> None:
    values = {
        "v_mg1": 10.0, "v_mg2": 20.0, "v_mg3": 30.0,
        "v_mg1_mg2": 40.0, "v_mg1_mg3": 50.0, "v_mg2_mg3": 60.0,
        "v_grand_coalition": 90.0,
    }
    shapley = calculate_shapley(values)
    assert np.allclose(
        [shapley["shapley_value_mg1"], shapley["shapley_value_mg2"], shapley["shapley_value_mg3"]],
        [20.0, 30.0, 40.0],
    )
    assert abs(float(shapley["shapley_efficiency_error"])) < 1e-10

    settlement = settle_test_day(
        values,
        matd3_joint_profit=75.0,
        physical_profit_by_microgrid={"mg1": 10.0, "mg2": 20.0, "mg3": 45.0},
    )
    assert settlement["shapley_allocation_valid"]
    assert np.allclose(
        [settlement[f"shapley_allocated_surplus_mg{index}"] for index in (1, 2, 3)],
        [5.0, 5.0, 5.0],
    )
    assert abs(settlement["shapley_budget_balance_error"]) < 1e-10
    assert abs(settlement["equal_budget_balance_error"]) < 1e-10
    assert abs(settlement["shapley_team_profit_invariance_error"]) < 1e-10
    assert settlement["shapley_all_ir_satisfied"]


def test_nonpositive_milp_cooperation_gain_marks_shapley_invalid() -> None:
    values = {
        "v_mg1": 10.0, "v_mg2": 20.0, "v_mg3": 30.0,
        "v_mg1_mg2": 30.0, "v_mg1_mg3": 40.0, "v_mg2_mg3": 50.0,
        "v_grand_coalition": 55.0,
    }
    settlement = settle_test_day(
        values,
        matd3_joint_profit=70.0,
        physical_profit_by_microgrid={"mg1": 15.0, "mg2": 20.0, "mg3": 35.0},
    )
    assert not settlement["shapley_allocation_valid"]
    assert settlement["shapley_settled_profit_mg1"] is None
    assert settlement["shapley_all_ir_satisfied"] is None
    assert settlement["equal_all_ir_satisfied"]


def test_coalition_evaluator_solves_all_seven_values_on_same_day() -> None:
    day = "2018-01-28"
    evaluation = evaluate_day_coalitions(CONFIG, split="test", day=day)
    independent = solve_independent_dispatch(CONFIG, split="test", day=day)
    assert evaluation.valid
    assert evaluation.failure_count == 0
    assert len(evaluation.values) == 7
    assert abs(float(evaluation.independent_total_profit) - independent.total_profit) < 1e-7
