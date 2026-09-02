from __future__ import annotations

from algorithms.milp_ess_sharing_mpc.algorithm import MilpEssSharingMpc
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="milp_ess_sharing_mpc",
    display_name="MILP ESS Sharing MPC",
    algorithm_class=MilpEssSharingMpc,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "milp_ess_sharing_mpc.yaml",
    description=(
        "Centralized receding-horizon MILP using the same current state, four-hour "
        "forecast window, and two-dimensional ESS-sharing control as MATD3 ESS Sharing."
    ),
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {
            "control_mode": "explicit_offer",
            "observation_mode": "cross_microgrid",
            "ess_p2p_enabled": True,
        },
    },
)
