from __future__ import annotations

from algorithms.matd3_hierarchical_transformer_predictive_dr.algorithm import (
    Matd3HierarchicalTransformerPredictiveDr,
)
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_hierarchical_transformer_predictive_dr",
    display_name="MATD3 Hierarchical Transformer Predictive DR",
    algorithm_class=Matd3HierarchicalTransformerPredictiveDr,
    default_config_path=(
        PROJECT_ROOT
        / "configs"
        / "algorithms"
        / "matd3_hierarchical_transformer_predictive_dr_hvac.yaml"
    ),
    description=(
        "Hierarchical categorical-token spatio-temporal Transformer with finite-horizon "
        "predictive 2R2C HVAC reference mapping."
    ),
)
