from __future__ import annotations

from algorithms.crossgate_actor.algorithm import CrossGateActor
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="crossgate_actor",
    display_name="CROSSGATE Actor",
    algorithm_class=CrossGateActor,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "crossgate_actor.yaml",
    description="Local battery control with a CrossGate-conditioned, demand-capped ESS offer head.",
    environment_overrides={
        "env": {"mode": "cooperative"},
        "ess_sharing": {
            "control_mode": "explicit_offer",
            "observation_mode": "cross_microgrid",
            "offer_projection": "peer_demand_cap",
        },
    },
)
