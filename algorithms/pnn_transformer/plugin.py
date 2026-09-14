from __future__ import annotations

from algorithms.pnn_transformer.algorithm import PnnTransformer
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT
from algorithms.runtime_helpers import install_single_dispatch_runtime_optimizations


install_single_dispatch_runtime_optimizations(
    PnnTransformer, variant="pnn_transformer"
)

ALGORITHM_SPEC = AlgorithmSpec(
    slug="pnn_transformer",
    display_name="PNN-Transformer",
    algorithm_class=PnnTransformer,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "pnn_transformer.yaml",
    description="Organic hybrid of Product-based Neural Network (PNN) and Spatio-Temporal Transformer with local battery residual bypass.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {
            "control_mode": "single_dispatch",
            "observation_mode": "cross_microgrid",
            "offer_projection": "peer_demand_cap",
        },
    },
)
