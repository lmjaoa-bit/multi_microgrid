from __future__ import annotations

from algorithms.milp_mpc.algorithm import MilpMpc
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="milp_mpc",
    display_name="MilpMpc",
    algorithm_class=MilpMpc,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "milp_mpc.yaml",
    description="Experimental variant derived from MATD3.",
)
