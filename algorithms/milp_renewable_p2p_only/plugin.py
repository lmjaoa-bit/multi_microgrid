from __future__ import annotations

from algorithms.milp_renewable_p2p_only.algorithm import MilpRenewableP2pOnly
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="milp_renewable_p2p_only",
    display_name="MILP Renewable P2P Only MPC",
    algorithm_class=MilpRenewableP2pOnly,
    default_config_path=(
        PROJECT_ROOT / "configs" / "algorithms" / "milp_renewable_p2p_only.yaml"
    ),
    description=(
        "Centralized receding-horizon MILP with the same observations and battery control "
        "as ESS Sharing MPC, while physically restricting P2P exports to renewable surplus."
    ),
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {
            "control_mode": "explicit_offer",
            "observation_mode": "cross_microgrid",
            "ess_p2p_enabled": False,
        },
    },
)
