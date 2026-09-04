from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT

ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_dispatch",
    display_name="MATD3",
    algorithm_class=Matd3Dispatch,
    default_config_path=PROJECT_ROOT / "configs/algorithms/matd3_dispatch.yaml",
    description="MATD3 baseline: full observation to canonical flat MLP Actor; no grouped encoders or gates.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {"control_mode": "single_dispatch",
                        "observation_mode": "cross_microgrid",
                        "offer_projection": "peer_demand_cap"},
    },
)
