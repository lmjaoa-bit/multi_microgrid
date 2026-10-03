from __future__ import annotations

# Importing the ordinary-T plugin installs its shared sampled update, including
# the generic actor_auxiliary_loss hook used by this subclass.
import algorithms.matd3_hierarchical_transformer.plugin as _ordinary_t_plugin  # noqa: F401
from algorithms.matd3_hierarchical_transformer_q_guided.algorithm import (
    Matd3HierarchicalTransformerQGuided,
)
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_hierarchical_transformer_q_guided",
    display_name="普通 T（Q 引导实验）",
    algorithm_class=Matd3HierarchicalTransformerQGuided,
    default_config_path=(
        PROJECT_ROOT
        / "configs"
        / "algorithms"
        / "matd3_hierarchical_transformer_q_guided.yaml"
    ),
    description=(
        "Ordinary T with conservative twin-critic-guided ESS logit recovery "
        "for dispatch intents blocked by battery limits."
    ),
    environment_overrides={
        "ess_sharing": {"observation_mode": "peer_energy_summary"}
    },
)
