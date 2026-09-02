from __future__ import annotations


def profit_optimality_gap(matd3_profit: float, milp_profit: float) -> float:
    """Non-negative relative profit gap; zero means the oracle was matched."""
    return max(0.0, (float(milp_profit) - float(matd3_profit)) / max(abs(float(milp_profit)), 1e-9))
