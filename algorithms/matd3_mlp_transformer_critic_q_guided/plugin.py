from __future__ import annotations

from algorithms.matd3_mlp_transformer_critic_q_guided.algorithm import Matd3MlpTransformerCriticQGuided
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_mlp_transformer_critic_q_guided",
    display_name="Critic T-Q",
    algorithm_class=Matd3MlpTransformerCriticQGuided,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_mlp_transformer_critic_q_guided.yaml",
    description="H8 semantic MLP actors, independent H24 Transformer twin critics, and conservative ESS Q guidance.",
    environment_overrides={"ess_sharing": {"observation_mode": "peer_energy_summary"}},
)
