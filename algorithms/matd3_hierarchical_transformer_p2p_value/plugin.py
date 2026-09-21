from __future__ import annotations

from algorithms.matd3_hierarchical_transformer_p2p_value.algorithm import (
    Matd3HierarchicalTransformerP2PValue,
)
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT

ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_hierarchical_transformer_p2p_value",
    display_name="MATD3 Hierarchical Transformer P2P Value",
    algorithm_class=Matd3HierarchicalTransformerP2PValue,
    default_config_path=(
        PROJECT_ROOT
        / "configs"
        / "algorithms"
        / "matd3_hierarchical_transformer_p2p_value.yaml"
    ),
    description=(
        "Predictive feasible-DR hierarchical Transformer with an "
        "action-conditioned P2P fill-value auxiliary objective."
    ),
)
