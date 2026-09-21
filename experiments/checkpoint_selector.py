from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class ConstraintSpec:
    name: str
    min_val: float | None = None
    max_val: float | None = None
    atol: float = 0.0


@dataclass
class CheckpointCandidate:
    episode: int
    eligible: bool
    total_violation: float
    objective_value: float
    constraint_values: dict[str, float]
    constraint_failures: list[str]
    selection_mode: str
    is_best: bool = False


class CheckpointSelector:
    """Selects validation checkpoints using either profit_only or constrained_profit."""

    def __init__(self, selection_config: Mapping[str, Any] | None = None) -> None:
        cfg = dict(selection_config or {})
        self.mode = str(cfg.get("mode", "profit_only")).lower()
        self.objective_key = str(cfg.get("objective", "team_profit_mean"))
        raw_constraints = cfg.get("constraints", {})
        self.constraints: list[ConstraintSpec] = []
        if isinstance(raw_constraints, Mapping):
            for name, spec in raw_constraints.items():
                if isinstance(spec, Mapping):
                    min_val = float(spec["min"]) if "min" in spec and spec["min"] is not None else None
                    max_val = float(spec["max"]) if "max" in spec and spec["max"] is not None else None
                    atol = float(spec.get("atol", 0.0))
                    self.constraints.append(
                        ConstraintSpec(name=str(name), min_val=min_val, max_val=max_val, atol=atol)
                    )
        self.epsilon = 1e-9

    def evaluate(self, summary: Mapping[str, Any], episode: int) -> CheckpointCandidate:
        """Evaluate a validation summary against configured constraints."""
        objective_raw = summary.get(self.objective_key, 0.0)
        try:
            objective_value = float(objective_raw)
            if not math.isfinite(objective_value):
                objective_value = -float("inf")
        except (TypeError, ValueError):
            objective_value = -float("inf")

        if self.mode == "profit_only":
            return CheckpointCandidate(
                episode=int(episode),
                eligible=True,
                total_violation=0.0,
                objective_value=objective_value,
                constraint_values={},
                constraint_failures=[],
                selection_mode="profit_only",
            )

        # constrained_profit mode
        constraint_values: dict[str, float] = {}
        constraint_failures: list[str] = []
        total_violation = 0.0
        has_nonfinite_required = False

        for c in self.constraints:
            val_raw = summary.get(c.name)
            if val_raw is None:
                has_nonfinite_required = True
                constraint_failures.append(f"{c.name}: missing")
                continue
            try:
                val = float(val_raw)
                if not math.isfinite(val):
                    has_nonfinite_required = True
                    constraint_failures.append(f"{c.name}: nonfinite ({val_raw})")
                    continue
            except (TypeError, ValueError):
                has_nonfinite_required = True
                constraint_failures.append(f"{c.name}: invalid ({val_raw})")
                continue

            constraint_values[c.name] = val

            # Check min constraint: max((min - value - atol) / max(abs(min), eps), 0)
            if c.min_val is not None:
                effective_gap = c.min_val - val - c.atol
                if effective_gap > 0.0:
                    denom = max(abs(c.min_val), self.epsilon)
                    viol = max(0.0, effective_gap / denom)
                    total_violation += viol
                    constraint_failures.append(
                        f"{c.name}: {val:.6g} < min {c.min_val:.6g} (viol: {viol:.4g})"
                    )

            # Check max constraint: max((value - max - atol) / max(abs(max), 1, eps), 0)
            if c.max_val is not None:
                effective_gap = val - c.max_val - c.atol
                if effective_gap > 0.0:
                    denom = max(abs(c.max_val), 1.0, self.epsilon)
                    viol = max(0.0, effective_gap / denom)
                    total_violation += viol
                    constraint_failures.append(
                        f"{c.name}: {val:.6g} > max {c.max_val:.6g} (viol: {viol:.4g})"
                    )

        if has_nonfinite_required:
            total_violation = float("inf")
            eligible = False
        else:
            eligible = len(constraint_failures) == 0 and total_violation <= self.epsilon

        return CheckpointCandidate(
            episode=int(episode),
            eligible=eligible,
            total_violation=total_violation,
            objective_value=objective_value,
            constraint_values=constraint_values,
            constraint_failures=constraint_failures,
            selection_mode=self.mode,
        )

    def is_better(
        self, candidate: CheckpointCandidate, current_best: CheckpointCandidate | None
    ) -> bool:
        """Determines if candidate outranks current_best according to lexicographic rules."""
        if current_best is None:
            return True

        if self.mode == "profit_only":
            if candidate.objective_value > current_best.objective_value:
                return True
            elif candidate.objective_value < current_best.objective_value:
                return False
            return candidate.episode >= current_best.episode

        # constrained_profit:
        # 1. eligible candidates outrank ineligible candidates
        if candidate.eligible and not current_best.eligible:
            return True
        if not candidate.eligible and current_best.eligible:
            return False

        # 2. among eligible candidates, higher objective wins
        if candidate.eligible and current_best.eligible:
            if candidate.objective_value > current_best.objective_value:
                return True
            elif candidate.objective_value < current_best.objective_value:
                return False
            return candidate.episode >= current_best.episode

        # 3. among ineligible candidates, lower normalized violation wins
        if candidate.total_violation < current_best.total_violation:
            return True
        elif candidate.total_violation > current_best.total_violation:
            return False

        # 4. exact ties use higher objective, then later episode number
        if candidate.objective_value > current_best.objective_value:
            return True
        elif candidate.objective_value < current_best.objective_value:
            return False
        return candidate.episode >= current_best.episode
