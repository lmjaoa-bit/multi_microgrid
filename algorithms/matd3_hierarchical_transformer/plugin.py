from __future__ import annotations

from algorithms.diagnostic_sampling import install_diagnostic_sampling
from algorithms.matd3_hierarchical_transformer.algorithm import Matd3HierarchicalTransformer
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT

install_diagnostic_sampling(
    Matd3HierarchicalTransformer, variant="matd3_hierarchical_transformer"
)

ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_hierarchical_transformer",
    display_name="普通 T",
    algorithm_class=Matd3HierarchicalTransformer,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_hierarchical_transformer.yaml",
    description="MATD3 with 14-token hierarchical spatio-temporal transformer actors, physical bilateral gates, and decoupled dual-action heads.",
    environment_overrides={"ess_sharing": {"observation_mode": "peer_energy_summary"}},
)
