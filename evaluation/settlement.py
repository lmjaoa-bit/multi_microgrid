from __future__ import annotations

from typing import Any, Mapping

from evaluation.shapley_allocator import allocate_actual_surplus
from utils.metrics_collector import safe_ratio


def settle_test_day(
    coalition_values: Mapping[str, float | None],
    *,
    matd3_joint_profit: float,
    physical_profit_by_microgrid: Mapping[str, float],
    tolerance: float = 1e-8,
    ir_epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Calculate Equal and Shapley settlements without changing physical dispatch."""
    output: dict[str, Any] = dict(coalition_values)
    singles = [coalition_values.get(f"v_mg{index}") for index in (1, 2, 3)]
    if any(value is None for value in singles):
        output.update(_unavailable_settlement(matd3_joint_profit, physical_profit_by_microgrid))
        return output

    disagreement = {f"mg{index}": float(coalition_values[f"v_mg{index}"]) for index in (1, 2, 3)}
    independent_total = float(sum(disagreement.values()))
    actual_surplus = float(matd3_joint_profit) - independent_total
    output.update({
        "independent_total_profit": independent_total,
        "independent_milp_profit": independent_total,
        "matd3_joint_profit": float(matd3_joint_profit),
        "matd3_cooperation_surplus": actual_surplus,
        "matd3_cooperation_gain_pct": (
            safe_ratio(actual_surplus, abs(independent_total)) * 100.0
            if abs(independent_total) > tolerance else None
        ),
        "negative_cooperation_surplus": actual_surplus < -tolerance,
        "positive_cooperation_surplus": actual_surplus > tolerance,
        "matd3_cooperation_acceptable": actual_surplus >= -tolerance,
    })
    for index in (1, 2, 3):
        output[f"physical_profit_mg{index}"] = float(physical_profit_by_microgrid[f"mg{index}"])

    equal_share = actual_surplus / 3.0
    equal_transfers: list[float] = []
    equal_settled: list[float] = []
    equal_ir: list[bool] = []
    for index in (1, 2, 3):
        mg = f"mg{index}"
        settled = disagreement[mg] + equal_share
        transfer = settled - float(physical_profit_by_microgrid[mg])
        ir_gain = settled - disagreement[mg]
        satisfied = ir_gain >= -ir_epsilon
        output[f"equal_surplus_{mg}"] = equal_share
        output[f"equal_allocated_surplus_{mg}"] = equal_share
        output[f"equal_settled_profit_{mg}"] = settled
        output[f"equal_transfer_{mg}"] = transfer
        output[f"equal_ir_gain_{mg}"] = ir_gain
        output[f"equal_ir_satisfied_{mg}"] = satisfied
        equal_transfers.append(transfer)
        equal_settled.append(settled)
        equal_ir.append(satisfied)
    output["equal_all_ir_satisfied"] = all(equal_ir)
    output["equal_budget_balance_error"] = sum(equal_transfers)
    output["equal_budget_balance_satisfied"] = abs(sum(equal_transfers)) <= tolerance
    output["equal_team_profit_invariance_error"] = sum(equal_settled) - float(matd3_joint_profit)

    shapley = allocate_actual_surplus(
        coalition_values,
        actual_surplus,
        tolerance=tolerance,
    )
    output.update(shapley)
    if not bool(shapley["shapley_allocation_valid"]):
        output.update(_invalid_shapley_settlement())
        return output

    transfers: list[float] = []
    settled_values: list[float] = []
    ir_values: list[bool] = []
    final_errors: list[float] = []
    for index in (1, 2, 3):
        mg = f"mg{index}"
        allocated = float(shapley[f"shapley_allocated_surplus_{mg}"])
        settled = disagreement[mg] + allocated
        physical = float(physical_profit_by_microgrid[mg])
        transfer = settled - physical
        ir_gain = settled - disagreement[mg]
        satisfied = ir_gain >= -ir_epsilon
        output[f"shapley_settled_profit_{mg}"] = settled
        output[f"shapley_transfer_payment_{mg}"] = transfer
        output[f"shapley_ir_gain_{mg}"] = ir_gain
        output[f"shapley_ir_satisfied_{mg}"] = satisfied
        output[f"shapley_final_settlement_error_{mg}"] = physical + transfer - settled
        transfers.append(transfer)
        settled_values.append(settled)
        ir_values.append(satisfied)
        final_errors.append(physical + transfer - settled)
    budget_error = float(sum(transfers))
    output["shapley_all_ir_satisfied"] = all(ir_values)
    output["shapley_budget_balance_error"] = budget_error
    output["shapley_budget_balance_satisfied"] = abs(budget_error) <= tolerance
    output["shapley_team_profit_invariance_error"] = sum(settled_values) - float(matd3_joint_profit)
    output["shapley_final_settlement_max_error"] = max(abs(value) for value in final_errors)
    return output


def _invalid_shapley_settlement() -> dict[str, Any]:
    result: dict[str, Any] = {
        "shapley_all_ir_satisfied": None,
        "shapley_budget_balance_error": None,
        "shapley_budget_balance_satisfied": None,
        "shapley_team_profit_invariance_error": None,
        "shapley_final_settlement_max_error": None,
    }
    for index in (1, 2, 3):
        mg = f"mg{index}"
        for prefix in (
            "shapley_settled_profit_",
            "shapley_transfer_payment_",
            "shapley_ir_gain_",
            "shapley_ir_satisfied_",
            "shapley_final_settlement_error_",
        ):
            result[f"{prefix}{mg}"] = None
    return result


def _unavailable_settlement(
    joint_profit: float,
    physical: Mapping[str, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "independent_total_profit": None,
        "independent_milp_profit": None,
        "matd3_joint_profit": float(joint_profit),
        "matd3_cooperation_surplus": None,
        "matd3_cooperation_gain_pct": None,
        "negative_cooperation_surplus": None,
        "positive_cooperation_surplus": None,
        "matd3_cooperation_acceptable": False,
        "equal_all_ir_satisfied": False,
        "equal_budget_balance_error": None,
        "equal_budget_balance_satisfied": False,
        "equal_team_profit_invariance_error": None,
        "shapley_allocation_valid": False,
    }
    for index in (1, 2, 3):
        mg = f"mg{index}"
        result[f"physical_profit_{mg}"] = float(physical[mg])
        for prefix in (
            "equal_surplus_", "equal_allocated_surplus_", "equal_settled_profit_",
            "equal_transfer_", "equal_ir_gain_", "equal_ir_satisfied_",
        ):
            result[f"{prefix}{mg}"] = None
    result.update(_invalid_shapley_settlement())
    return result
