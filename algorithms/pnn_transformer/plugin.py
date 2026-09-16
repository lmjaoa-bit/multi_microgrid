from __future__ import annotations

from algorithms.dr_environment_adapter import install_dr_environment_adapter
from algorithms.pnn_transformer.algorithm import PnnTransformer
from algorithms.pnn_transformer.runtime import install_historical_actor_runtime_optimizations
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT
from algorithms.runtime_helpers import install_single_dispatch_runtime_optimizations


install_historical_actor_runtime_optimizations()
install_dr_environment_adapter(
    PnnTransformer, slug="pnn_transformer", actor_kind="pnn_transformer"
)
install_single_dispatch_runtime_optimizations(
    PnnTransformer, variant="pnn_transformer"
)

ALGORITHM_SPEC = AlgorithmSpec(
    slug="pnn_transformer",
    display_name="PNN-Transformer",
    algorithm_class=PnnTransformer,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "pnn_transformer.yaml",
    description="PNN-Transformer hybrid for joint ESS dispatch and heterogeneous controllable-load scheduling.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "action_design": "single_dispatch_dr",
        "ess_sharing": {
            "control_mode": "single_dispatch",
            "observation_mode": "cross_microgrid",
            "offer_projection": "peer_demand_cap",
        },
    },
)
