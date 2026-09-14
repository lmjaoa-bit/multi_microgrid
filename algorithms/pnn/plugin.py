from algorithms.pnn.algorithm import PNN
from algorithms.pnn.runtime import optimized_pnn_update
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT
from algorithms.runtime_helpers import install_single_dispatch_runtime_optimizations


install_single_dispatch_runtime_optimizations(PNN, variant="pnn")
PNN.update = optimized_pnn_update

ALGORITHM_SPEC = AlgorithmSpec(
    slug="pnn",
    display_name="PNN",
    algorithm_class=PNN,
    default_config_path=PROJECT_ROOT / "configs/algorithms/pnn.yaml",
    description="Single signed dispatch with pure PNN feature product interactions (no gating).",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {
            "control_mode": "single_dispatch",
            "observation_mode": "cross_microgrid",
            "offer_projection": "peer_demand_cap",
        },
    },
)
