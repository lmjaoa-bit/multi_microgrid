from __future__ import annotations

from algorithms.matd3_ess_local_obs.algorithm import Matd3EssLocalObs
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_ess_local_obs",
    display_name="MATD3 ESS Local Observation",
    algorithm_class=Matd3EssLocalObs,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_ess_local_obs.yaml",
    description="Two-dimensional ESS offer control using local observations and market feedback only.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {"control_mode": "explicit_offer", "observation_mode": "local"},
    },
)
