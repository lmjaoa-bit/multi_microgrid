from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from allocation.nash_bargaining import solve_nash_bargaining
from optimization.cooperative_dispatch import solve_cooperative_dispatch
from optimization.independent_dispatch import solve_independent_dispatch


def evaluate_nash_bargaining(config: str | Path | Mapping[str, Any], split: str = "test",
                             day: str | None = None) -> dict[str, object]:
    independent = solve_independent_dispatch(config, split, day)
    cooperative = solve_cooperative_dispatch(config, split, day)
    return solve_nash_bargaining(independent.profit_by_microgrid, cooperative.total_profit,
                                 cooperative.profit_by_microgrid).as_dict()
