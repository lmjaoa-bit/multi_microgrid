from __future__ import annotations

from algorithms.matd3_ess_full_offer.algorithm import Matd3EssFullOffer
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_ess_full_offer",
    display_name="MATD3 ESS Full Offer",
    algorithm_class=Matd3EssFullOffer,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_ess_full_offer.yaml",
    description="One-dimensional MATD3 that offers all flexible ESS discharge to P2P clearing.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {"control_mode": "fixed_full_offer", "observation_mode": "local"},
    },
)
