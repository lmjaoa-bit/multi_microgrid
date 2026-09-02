from __future__ import annotations

from algorithms.matd3.matd3 import MATD3
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3",
    display_name="MATD3",
    algorithm_class=MATD3,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3.yaml",
    description="Canonical cooperative multi-agent TD3 baseline; keep unchanged as the reference algorithm.",
)
