import pytest

from algorithms.matd3_shapley_q.coalition import (
    all_coalitions,
    exact_shapley_values,
    normalize_characteristic_values,
)


AGENTS = ("mg1", "mg2", "mg3")


def positive_cooperation_values():
    return {
        frozenset(): 80.0,
        frozenset({"mg1"}): 84.0,
        frozenset({"mg2"}): 83.0,
        frozenset({"mg3"}): 82.0,
        frozenset({"mg1", "mg2"}): 91.0,
        frozenset({"mg1", "mg3"}): 89.0,
        frozenset({"mg2", "mg3"}): 88.0,
        frozenset(AGENTS): 100.0,
    }


def test_characteristic_normalization_uses_nonzero_independent_value():
    physical = positive_cooperation_values()
    normalized = normalize_characteristic_values(physical, AGENTS)
    assert physical[frozenset()] != 0.0
    assert normalized[frozenset()] == pytest.approx(0.0)
    assert normalized[frozenset(AGENTS)] == pytest.approx(20.0)


def test_exact_shapley_efficiency_targets_cooperation_surplus():
    values = positive_cooperation_values()
    phi = exact_shapley_values(values, AGENTS)
    assert sum(phi.values()) == pytest.approx(
        values[frozenset(AGENTS)] - values[frozenset()]
    )


def test_exact_shapley_is_zero_when_cooperation_changes_no_value():
    values = {coalition: 80.0 for coalition in all_coalitions(AGENTS)}
    phi = exact_shapley_values(values, AGENTS)
    assert phi == pytest.approx({mg: 0.0 for mg in AGENTS})


def test_exact_shapley_sum_is_positive_for_positive_cooperation():
    values = positive_cooperation_values()
    phi = exact_shapley_values(values, AGENTS)
    assert sum(phi.values()) > 0.0
