from __future__ import annotations

from algorithms.matd3_mlp_transformer_critic_h24_q_guided.algorithm import Matd3MlpTransformerCriticH24QGuided
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_mlp_transformer_critic_h24_q_guided",
    display_name="Critic T-Q H24",
    algorithm_class=Matd3MlpTransformerCriticH24QGuided,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_mlp_transformer_critic_h24_q_guided.yaml",
    description="H24 semantic MLP actors, independent H24 Transformer twin critics, and conservative ESS Q guidance.",
    environment_overrides={"ess_sharing": {"observation_mode": "peer_energy_summary"}},
)
