from __future__ import annotations

from algorithms.matd3_ess_sharing.algorithm import Matd3EssSharing
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_ess_sharing",
    display_name="MATD3 ESS Spatiotemporal Sharing",
    algorithm_class=Matd3EssSharing,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_ess_sharing.yaml",
    description="Two-dimensional ESS offer control with cross-microgrid forecasts and clearing feedback.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {"control_mode": "explicit_offer", "observation_mode": "cross_microgrid"},
    },
)
