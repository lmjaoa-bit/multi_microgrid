from __future__ import annotations

from algorithms.matd3_dr_counterfactual.algorithm import Matd3DrCounterfactual
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_dr_counterfactual",
    display_name="MATD3-Q DR Counterfactual",
    algorithm_class=Matd3DrCounterfactual,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_dr_counterfactual.yaml",
    description="MATD3-Q with training-only real-return DR value contrasts and actor guidance.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {"control_mode": "single_dispatch",
                        "observation_mode": "peer_energy_summary",
                        "offer_projection": "peer_demand_cap"},
    },
)
