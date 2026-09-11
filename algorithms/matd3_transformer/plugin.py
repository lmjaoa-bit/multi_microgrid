from __future__ import annotations

from algorithms.matd3_transformer.algorithm import Matd3Transformer
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_transformer",
    display_name="MATD3-Transformer",
    algorithm_class=Matd3Transformer,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_transformer.yaml",
    description="Spatio-Temporal Transformer dispatch actors with 24h diurnal periodic embedding and structured forecast attention.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {
            "control_mode": "single_dispatch",
            "observation_mode": "cross_microgrid",
            "offer_projection": "peer_demand_cap",
        },
    },
)
