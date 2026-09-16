from algorithms.dr_environment_adapter import install_dr_environment_adapter
from algorithms.pnn.algorithm import PNN
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


install_dr_environment_adapter(PNN, slug="pnn", actor_kind="pnn")

ALGORITHM_SPEC = AlgorithmSpec(
    slug="pnn",
    display_name="PNN",
    algorithm_class=PNN,
    default_config_path=PROJECT_ROOT / "configs/algorithms/pnn.yaml",
    description="PNN actor for joint ESS dispatch and heterogeneous controllable-load scheduling.",
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
