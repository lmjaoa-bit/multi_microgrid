from __future__ import annotations

from algorithms.matd3_hierarchical_transformer.algorithm import Matd3HierarchicalTransformer


class Matd3HierarchicalTransformerFeasibleDR(Matd3HierarchicalTransformer):
    """Hierarchical Transformer with feasible MG2/MG3 DR actions."""

    action_semantics = "single_dispatch_dr_feasible_v1"
