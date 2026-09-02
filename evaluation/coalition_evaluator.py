from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from optimization.common_constraints import (
    DispatchResult,
    build_dispatch_data,
    solve_dispatch,
)


COALITION_KEYS = (
    "v_mg1",
    "v_mg2",
    "v_mg3",
    "v_mg1_mg2",
    "v_mg1_mg3",
    "v_mg2_mg3",
    "v_grand_coalition",
)


@dataclass(frozen=True)
class CoalitionEvaluation:
    values: dict[str, float | None]
    statuses: dict[str, str]
    results: dict[str, DispatchResult]
    valid: bool
    failure_count: int

    @property
    def independent_total_profit(self) -> float | None:
        singletons = [self.values.get(f"v_mg{index}") for index in (1, 2, 3)]
        if any(value is None for value in singletons):
            return None
        return float(sum(float(value) for value in singletons if value is not None))

    @property
    def grand_result(self) -> DispatchResult | None:
        return self.results.get("v_grand_coalition")

    def flat_status_fields(self) -> dict[str, str]:
        return {f"{key}_solve_status": self.statuses.get(key, "not_solved") for key in COALITION_KEYS}


def _coalition_specs(agent_ids: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
    ids = list(agent_ids)
    if ids != ["mg1", "mg2", "mg3"]:
        raise ValueError(f"Shapley coalition evaluation requires agent_ids ['mg1', 'mg2', 'mg3']; got {ids}")
    return [
        ("v_mg1", ("mg1",)),
        ("v_mg2", ("mg2",)),
        ("v_mg3", ("mg3",)),
        ("v_mg1_mg2", ("mg1", "mg2")),
        ("v_mg1_mg3", ("mg1", "mg3")),
        ("v_mg2_mg3", ("mg2", "mg3")),
        ("v_grand_coalition", ("mg1", "mg2", "mg3")),
    ]


def evaluate_day_coalitions(
    config: str | Path | Mapping[str, Any],
    *,
    split: str = "test",
    day: str,
    solver_output: bool = False,
) -> CoalitionEvaluation:
    """Solve all seven non-empty coalitions on one shared 24-hour scenario.

    The dispatch data is built once and reused by every solve, which guarantees
    identical load, renewable, price, horizon, and configured initial SOC data.
    """
    cfg, data = build_dispatch_data(config, split=split, day=day)
    values: dict[str, float | None] = {}
    statuses: dict[str, str] = {}
    results: dict[str, DispatchResult] = {}
    for key, members in _coalition_specs(data.agent_ids):
        try:
            result = solve_dispatch(
                cfg,
                data,
                members,
                cooperative=len(members) > 1,
                solver_output=solver_output,
            )
        except Exception as exc:  # one failed coalition invalidates only this day's allocation
            values[key] = None
            statuses[key] = f"failed:{type(exc).__name__}:{exc}"
            continue
        values[key] = float(result.total_profit)
        statuses[key] = str(result.status)
        results[key] = result
    failure_count = sum(value is None for value in values.values())
    return CoalitionEvaluation(
        values=values,
        statuses=statuses,
        results=results,
        valid=failure_count == 0,
        failure_count=failure_count,
    )


def dispatch_energy_metrics(result: DispatchResult | None) -> dict[str, float | None]:
    if result is None or result.trajectories.empty:
        return {
            "renewable_available_kwh": None,
            "renewable_used_kwh": None,
            "renewable_curtailment_kwh": None,
            "renewable_utilization_pct": None,
        }
    frame: pd.DataFrame = result.trajectories
    available = float(frame["renewable_kw"].sum())
    curtailed = float(frame["curtailment_kw"].sum())
    used = available - curtailed
    utilization = used / available * 100.0 if available > 1e-12 else None
    return {
        "renewable_available_kwh": available,
        "renewable_used_kwh": used,
        "renewable_curtailment_kwh": curtailed,
        "renewable_utilization_pct": utilization,
    }
