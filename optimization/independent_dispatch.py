from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from optimization.common_constraints import DispatchResult, build_dispatch_data, solve_dispatch


def solve_independent_dispatch(config: str | Path | Mapping[str, Any], split: str = "test",
                               day: str | None = None, solver_output: bool = False) -> DispatchResult:
    """Solve three local counterfactuals without changing a cooperative environment."""
    cfg, data = build_dispatch_data(config, split, day)
    results = [
        solve_dispatch(
            cfg,
            data,
            [mg],
            cooperative=False,
            solver_output=solver_output,
            include_dr=bool(next(
                (item.get("dr") for item in cfg["microgrids"] if item["id"] == mg),
                None,
            )),
        )
        for mg in data.agent_ids
    ]
    profits = {mg: result.profit_by_microgrid[mg]
               for result in results for mg in result.profit_by_microgrid}
    trajectories = pd.concat([result.trajectories for result in results], ignore_index=True)
    total = float(sum(profits.values()))
    return DispatchResult("independent", "+".join(result.status for result in results), -total,
                          total, profits, trajectories, {})
