from __future__ import annotations

from typing import Any

from algorithms.matd3_hierarchical_transformer_predictive_dr.training import update_predictive_dr


def update_p2p_value(algo: Any, batch_size: int) -> dict[str, Any]:
    return update_predictive_dr(algo, batch_size)
