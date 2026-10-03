from __future__ import annotations

import algorithms.matd3_hierarchical_transformer_q_guided.plugin as _q_guided_plugin  # noqa: F401
from algorithms.matd3_hierarchical_transformer_q_readout.algorithm import Matd3HierarchicalTransformerQReadout
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_hierarchical_transformer_q_readout",
    display_name="普通 T（Q 引导与设备预测读取）",
    algorithm_class=Matd3HierarchicalTransformerQReadout,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_hierarchical_transformer_q_readout.yaml",
    description="T-Q retaining H8 base-load forecasts with separate ESS/DR attention readouts.",
    environment_overrides={"ess_sharing": {"observation_mode": "peer_energy_summary"}},
)
