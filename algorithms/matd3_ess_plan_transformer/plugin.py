from __future__ import annotations

from algorithms.matd3_ess_plan_transformer.algorithm import Matd3EssPlanTransformer
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_ess_plan_transformer",
    display_name="储能规划 T",
    algorithm_class=Matd3EssPlanTransformer,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_ess_plan_transformer.yaml",
    description="H24 ESS plan actor with forecast-only storage supervision and unchanged dueling critic.",
    environment_overrides={"ess_sharing": {"observation_mode": "peer_energy_summary"}},
)
