from __future__ import annotations

from algorithms.matd3_hierarchical_transformer_feasible_dr.algorithm import (
    Matd3HierarchicalTransformerFeasibleDR,
)
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_hierarchical_transformer_feasible_dr",
    display_name="MATD3 Hierarchical Transformer Feasible DR",
    algorithm_class=Matd3HierarchicalTransformerFeasibleDR,
    default_config_path=(
        PROJECT_ROOT
        / "configs"
        / "algorithms"
        / "matd3_hierarchical_transformer_feasible_dr.yaml"
    ),
    description="Hierarchical Transformer with feasible HVAC and deadline-safe EV DR actions.",
)
