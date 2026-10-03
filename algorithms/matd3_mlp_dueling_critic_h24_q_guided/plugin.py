from __future__ import annotations

from algorithms.matd3_mlp_dueling_critic_h24_q_guided.algorithm import Matd3MlpDuelingCriticH24QGuided
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT

ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_mlp_dueling_critic_h24_q_guided",
    display_name="Dueling Critic T-Q H24",
    algorithm_class=Matd3MlpDuelingCriticH24QGuided,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_mlp_dueling_critic_h24_q_guided.yaml",
    description="H24 semantic MLP actors with dueling critics: Transformer V(s) plus action-direct MLP A(s,a), zero-action centred; ESS Q guidance retained.",
    environment_overrides={"ess_sharing": {"observation_mode": "peer_energy_summary"}},
)
