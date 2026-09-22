from algorithms.matd3_hierarchical_transformer_peak_risk.algorithm import Matd3HierarchicalTransformerPeakRisk
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT

ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_hierarchical_transformer_peak_risk",
    display_name="MATD3 Hierarchical Transformer Peak Risk",
    algorithm_class=Matd3HierarchicalTransformerPeakRisk,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_hierarchical_transformer_peak_risk.yaml",
    description="Predictive DR Transformer with an actor peak-risk token and a shared twin maximum-load critic.",
)
