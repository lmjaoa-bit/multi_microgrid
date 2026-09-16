from __future__ import annotations

from algorithms.dr_environment_adapter import install_dr_environment_adapter
from algorithms.matd3_transformer.algorithm import Matd3Transformer
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


install_dr_environment_adapter(
    Matd3Transformer, slug="matd3_transformer", actor_kind="transformer"
)

ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_transformer",
    display_name="MATD3-Transformer",
    algorithm_class=Matd3Transformer,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_transformer.yaml",
    description="Spatio-Temporal Transformer actors for joint ESS dispatch and heterogeneous controllable-load scheduling.",
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
