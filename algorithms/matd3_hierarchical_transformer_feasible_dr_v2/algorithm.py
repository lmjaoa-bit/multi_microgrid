from __future__ import annotations

from algorithms.matd3_hierarchical_transformer_feasible_dr.algorithm import (
    Matd3HierarchicalTransformerFeasibleDR,
)


class Matd3HierarchicalTransformerFeasibleDRV2(Matd3HierarchicalTransformerFeasibleDR):
    """V2 adaptive-reference feasible DR Transformer."""

    action_semantics = "single_dispatch_dr_adaptive_reference_v2"
