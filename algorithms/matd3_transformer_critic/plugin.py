"""Lazy plugin: the shared pipeline owns training, evaluation and all outputs."""
from algorithms.matd3_transformer_critic.algorithm import Matd3TransformerCritic
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_transformer_critic",
    display_name="MATD3-Transformer-Critic",
    algorithm_class=Matd3TransformerCritic,
    default_config_path=PROJECT_ROOT / "configs/algorithms/matd3_transformer_critic.yaml",
    description=(
        "Flat MATD3 actors with per-agent device-aware, action-conditioned "
        "spatiotemporal Transformer twin critics and projected ESS/DR actions."
    ),
    environment_overrides={
        "env": {"mode": "cooperative"},
        "action_design": "single_dispatch_dr",
        "ess_sharing": {
            "control_mode": "single_dispatch",
            "observation_mode": "peer_energy_summary",
            "offer_projection": "peer_demand_cap",
        },
    },
)
