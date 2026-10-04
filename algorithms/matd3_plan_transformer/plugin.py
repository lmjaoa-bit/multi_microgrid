from __future__ import annotations

from algorithms.matd3_plan_transformer.algorithm import Matd3PlanTransformer
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_plan_transformer",
    display_name="Plan-Transformer",
    algorithm_class=Matd3PlanTransformer,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_plan_transformer.yaml",
    description="Two-layer NOW cross-attention with a causally supervised auxiliary H24 SOC/ESS plan.",
)
