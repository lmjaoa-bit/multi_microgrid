from __future__ import annotations

from algorithms.matd3_dispatch_q_guided.algorithm import Matd3DispatchQGuided
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_dispatch_q_guided",
    display_name="MATD3 Q-guided dispatch",
    algorithm_class=Matd3DispatchQGuided,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_dispatch_q_guided.yaml",
    description="Dispatch exploration with twin-critic-guided recovery of ineffective battery intents.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {
            "control_mode": "single_dispatch",
            "observation_mode": "peer_energy_summary",
            "offer_projection": "peer_demand_cap",
        },
    },
)
