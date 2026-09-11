from __future__ import annotations

from algorithms.matd3_shapley_q.algorithm import Matd3ShapleyQ
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_shapley_q",
    display_name="Matd3ShapleyQ",
    algorithm_class=Matd3ShapleyQ,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_shapley_q.yaml",
    description="Experimental variant derived from MATD3.",
)
