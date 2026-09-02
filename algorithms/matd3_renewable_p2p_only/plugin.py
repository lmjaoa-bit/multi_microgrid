from __future__ import annotations

from algorithms.matd3_renewable_p2p_only.algorithm import Matd3RenewableP2pOnly
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_renewable_p2p_only",
    display_name="MATD3 Renewable P2P Only",
    algorithm_class=Matd3RenewableP2pOnly,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_renewable_p2p_only.yaml",
    description=(
        "Sharing-matched two-dimensional MATD3 ablation that retains renewable P2P "
        "but physically disables ESS exports to peers."
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
