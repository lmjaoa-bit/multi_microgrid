from __future__ import annotations

import warnings
from typing import Any, Mapping


REQUIRED_VALUES = (
    "v_mg1",
    "v_mg2",
    "v_mg3",
    "v_mg1_mg2",
    "v_mg1_mg3",
    "v_mg2_mg3",
    "v_grand_coalition",
)


def calculate_shapley(values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Return the exact three-player Shapley values and consistency errors."""
    if any(values.get(key) is None for key in REQUIRED_VALUES):
        return {
            "shapley_value_mg1": None,
            "shapley_value_mg2": None,
            "shapley_value_mg3": None,
            "shapley_efficiency_error": None,
        }
    v1, v2, v3 = (float(values[f"v_mg{index}"]) for index in (1, 2, 3))
    v12 = float(values["v_mg1_mg2"])
    v13 = float(values["v_mg1_mg3"])
    v23 = float(values["v_mg2_mg3"])
    v123 = float(values["v_grand_coalition"])
    phi1 = v1 / 3.0 + (v12 - v2) / 6.0 + (v13 - v3) / 6.0 + (v123 - v23) / 3.0
    phi2 = v2 / 3.0 + (v12 - v1) / 6.0 + (v23 - v3) / 6.0 + (v123 - v13) / 3.0
    phi3 = v3 / 3.0 + (v13 - v1) / 6.0 + (v23 - v2) / 6.0 + (v123 - v12) / 3.0
    return {
        "shapley_value_mg1": phi1,
        "shapley_value_mg2": phi2,
        "shapley_value_mg3": phi3,
        "shapley_efficiency_error": phi1 + phi2 + phi3 - v123,
    }


def allocate_actual_surplus(
    values: Mapping[str, float | None],
    matd3_cooperation_surplus: float,
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Use MILP marginal-contribution weights to allocate only MATD3's surplus."""
    shapley = calculate_shapley(values)
    output: dict[str, Any] = dict(shapley)
    if any(shapley[f"shapley_value_mg{index}"] is None for index in (1, 2, 3)):
        output.update(_invalid_allocation_fields())
        return output

    gains = {
        f"mg{index}": float(shapley[f"shapley_value_mg{index}"]) - float(values[f"v_mg{index}"])
        for index in (1, 2, 3)
    }
    milp_surplus = float(values["v_grand_coalition"]) - sum(
        float(values[f"v_mg{index}"]) for index in (1, 2, 3)
    )
    gain_total = float(sum(gains.values()))
    output["milp_cooperation_surplus"] = milp_surplus
    output["shapley_gain_balance_error"] = gain_total - milp_surplus
    for index in (1, 2, 3):
        output[f"shapley_cooperation_gain_mg{index}"] = gains[f"mg{index}"]

    if gain_total <= tolerance:
        warnings.warn(
            "MILP Shapley cooperation gain is non-positive; Shapley surplus allocation is invalid",
            RuntimeWarning,
            stacklevel=2,
        )
        output.update(_invalid_allocation_fields(keep_gain_fields=True))
        output["milp_cooperation_surplus"] = milp_surplus
        output["shapley_gain_balance_error"] = gain_total - milp_surplus
        for index in (1, 2, 3):
            output[f"shapley_cooperation_gain_mg{index}"] = gains[f"mg{index}"]
        return output

    weights = {mg: gain / gain_total for mg, gain in gains.items()}
    allocated = {mg: weight * float(matd3_cooperation_surplus) for mg, weight in weights.items()}
    output["shapley_allocation_valid"] = True
    for index in (1, 2, 3):
        mg = f"mg{index}"
        output[f"shapley_weight_{mg}"] = weights[mg]
        output[f"shapley_allocated_surplus_{mg}"] = allocated[mg]
    output["shapley_weight_sum_error"] = sum(weights.values()) - 1.0
    output["shapley_surplus_allocation_error"] = (
        sum(allocated.values()) - float(matd3_cooperation_surplus)
    )
    return output


def _invalid_allocation_fields(*, keep_gain_fields: bool = False) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "shapley_allocation_valid": False,
        "shapley_weight_sum_error": None,
        "shapley_surplus_allocation_error": None,
    }
    if not keep_gain_fields:
        fields.update({"milp_cooperation_surplus": None, "shapley_gain_balance_error": None})
    for index in (1, 2, 3):
        mg = f"mg{index}"
        if not keep_gain_fields:
            fields[f"shapley_cooperation_gain_{mg}"] = None
        fields[f"shapley_weight_{mg}"] = None
        fields[f"shapley_allocated_surplus_{mg}"] = None
    return fields
