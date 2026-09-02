from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from optimization.common_constraints import DispatchResult, build_dispatch_data, solve_dispatch


def solve_cooperative_dispatch(config: str | Path | Mapping[str, Any], split: str = "test",
                               day: str | None = None, solver_output: bool = False) -> DispatchResult:
    """Solve the centralized cooperative social-planner oracle."""
    cfg, data = build_dispatch_data(config, split, day)
    return solve_dispatch(cfg, data, data.agent_ids, cooperative=True, solver_output=solver_output)
