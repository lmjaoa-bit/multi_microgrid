from __future__ import annotations

from algorithms.matd3_hierarchical_transformer_feasible_dr_v2.algorithm import (
    Matd3HierarchicalTransformerFeasibleDRV2,
)
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_hierarchical_transformer_feasible_dr_v2",
    display_name="MATD3 Hierarchical Transformer Feasible DR V2",
    algorithm_class=Matd3HierarchicalTransformerFeasibleDRV2,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_hierarchical_transformer_feasible_dr_v2.yaml",
    description="MATD3 Hierarchical Transformer with feasible DR and adaptive reference v2.",
)
