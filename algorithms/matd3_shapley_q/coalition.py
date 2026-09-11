"""Coalition-value normalization and exact Shapley allocation utilities."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

Coalition = frozenset[str]


def all_coalitions(agents: Sequence[str]) -> list[Coalition]:
    """Return every coalition, including the empty and grand coalitions."""
    ids = tuple(agents)
    if len(set(ids)) != len(ids):
        raise ValueError("agents must contain unique identifiers")
    return [
        frozenset(coalition)
        for size in range(len(ids) + 1)
        for coalition in itertools.combinations(ids, size)
    ]


def normalize_characteristic_values(
    physical_values: Mapping[Coalition, float],
    agents: Sequence[str],
) -> dict[Coalition, float]:
    """Express each physical characteristic value relative to ``J(empty)``."""
    expected = set(all_coalitions(agents))
    if set(physical_values) != expected:
        raise ValueError("physical_values must contain every coalition exactly once")
    baseline = float(physical_values[frozenset()])
    return {
        coalition: float(value) - baseline
        for coalition, value in physical_values.items()
    }


def enumerate_predecessor_coalitions(
    active: str, agents: Sequence[str]
) -> list[Coalition]:
    """Enumerate all predecessor coalitions for one active agent."""
    ids = tuple(agents)
    if active not in ids:
        raise ValueError(f"active agent {active!r} is not in agents")
    predecessors = tuple(agent for agent in ids if agent != active)
    return [
        frozenset(coalition)
        for size in range(len(predecessors) + 1)
        for coalition in itertools.combinations(predecessors, size)
    ]


def shapley_weight(predecessor_size: int, agent_count: int) -> float:
    """Return the standard predecessor marginal-contribution weight."""
    if agent_count < 1 or not 0 <= predecessor_size < agent_count:
        raise ValueError("predecessor_size must be in [0, agent_count)")
    return math.factorial(predecessor_size) * math.factorial(
        agent_count - predecessor_size - 1
    ) / math.factorial(agent_count)


def exact_shapley_values(
    physical_values: Mapping[Coalition, float],
    agents: Sequence[str],
) -> dict[str, float]:
    """Calculate exact Shapley values from a complete physical-value mapping."""
    normalized = normalize_characteristic_values(physical_values, agents)
    result: dict[str, float] = {}
    for active in agents:
        total = 0.0
        for predecessors in enumerate_predecessor_coalitions(active, agents):
            joined = predecessors | {active}
            total += shapley_weight(len(predecessors), len(agents)) * (
                normalized[joined] - normalized[predecessors]
            )
        result[active] = total
    return result


__all__ = [
    "Coalition",
    "all_coalitions",
    "enumerate_predecessor_coalitions",
    "exact_shapley_values",
    "normalize_characteristic_values",
    "shapley_weight",
]
