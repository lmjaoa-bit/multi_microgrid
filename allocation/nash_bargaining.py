from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class NashBargainingResult:
    baseline_profit: dict[str, float]
    physical_cooperative_profit: dict[str, float]
    allocated_cooperative_profit: dict[str, float]
    transfer_payment: dict[str, float]
    individual_cooperation_gain: dict[str, float]
    cooperative_gain: float

    @property
    def individual_rational(self) -> bool:
        return all(self.allocated_cooperative_profit[mg] + 1e-9 >= self.baseline_profit[mg]
                   for mg in self.baseline_profit)

    @property
    def budget_balanced(self) -> bool:
        return abs(sum(self.transfer_payment.values())) <= 1e-8

    def as_dict(self) -> dict[str, object]:
        return {"baseline_profit": self.baseline_profit,
                "physical_cooperative_profit": self.physical_cooperative_profit,
                "allocated_cooperative_profit": self.allocated_cooperative_profit,
                "transfer_payment": self.transfer_payment,
                "individual_cooperation_gain": self.individual_cooperation_gain,
                "cooperative_gain": self.cooperative_gain,
                "individual_rational": self.individual_rational,
                "budget_balanced": self.budget_balanced}


def solve_nash_bargaining(baseline_profit: Mapping[str, float], cooperative_total_profit: float,
                          physical_cooperative_profit: Mapping[str, float] | None = None) -> NashBargainingResult:
    """Closed-form symmetric Nash bargaining allocation.

    Dispatch is already fixed.  This function only allocates the incremental
    alliance gain and calculates budget-balanced ex-post transfers.
    """
    ids = sorted(str(mg) for mg in baseline_profit)
    if not ids:
        raise ValueError("At least one participant is required")
    base = {mg: float(baseline_profit[mg]) for mg in ids}
    gain = float(cooperative_total_profit) - sum(base.values())
    if gain < -1e-8:
        raise ValueError("Nash bargaining is infeasible because cooperative gain is negative")
    gain = max(0.0, gain)
    equal_gain = gain / len(ids)
    allocated = {mg: base[mg] + equal_gain for mg in ids}
    allocated[ids[-1]] += float(cooperative_total_profit) - sum(allocated.values())
    if physical_cooperative_profit is None:
        physical = dict(allocated)
    else:
        physical = {mg: float(physical_cooperative_profit[mg]) for mg in ids}
        if not np.isclose(sum(physical.values()), cooperative_total_profit, atol=1e-7):
            raise ValueError("Physical cooperative profits must sum to cooperative_total_profit")
    transfers = {mg: allocated[mg] - physical[mg] for mg in ids}
    transfers[ids[-1]] -= sum(transfers.values())
    result = NashBargainingResult(base, physical, allocated, transfers,
                                  {mg: allocated[mg] - base[mg] for mg in ids}, gain)
    if not result.individual_rational or not result.budget_balanced:
        raise RuntimeError("Nash bargaining invariants failed")
    return result
