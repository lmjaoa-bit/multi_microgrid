from algorithms.crossgate.algorithm import Crossgate
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT

ALGORITHM_SPEC = AlgorithmSpec(
    slug="crossgate",
    display_name="CrossGate",
    algorithm_class=Crossgate,
    default_config_path=PROJECT_ROOT / "configs/algorithms/crossgate.yaml",
    description="single signed dispatch; local-first mapping; CrossGate features.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {"control_mode": "single_dispatch",
                        "observation_mode": "cross_microgrid",
                        "offer_projection": "peer_demand_cap"},
    },
)
