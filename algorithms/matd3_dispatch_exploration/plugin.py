from __future__ import annotations

from algorithms.matd3_dispatch_exploration.algorithm import Matd3DispatchExploration
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_dispatch_exploration",
    display_name="MATD3 dispatch exploration",
    algorithm_class=Matd3DispatchExploration,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_dispatch_exploration.yaml",
    description="MATD3 dispatch with delayed learning and occasional random dispatch exploration.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {
            "control_mode": "single_dispatch",
            "observation_mode": "peer_energy_summary",
            "offer_projection": "peer_demand_cap",
        },
    },
)
