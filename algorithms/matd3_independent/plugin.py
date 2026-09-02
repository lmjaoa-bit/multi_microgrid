from __future__ import annotations

from algorithms.matd3_independent.algorithm import Matd3Independent
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_independent",
    display_name="MATD3 Independent",
    algorithm_class=Matd3Independent,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_independent.yaml",
    description="One-dimensional MATD3 with P2P disabled; every microgrid operates independently.",
    environment_overrides={
        "env": {"mode": "independent"},
        "ess_sharing": {"control_mode": "residual_auto", "observation_mode": "local"},
    },
)
