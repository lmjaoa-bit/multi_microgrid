from __future__ import annotations

from algorithms.diagnostic_sampling import install_diagnostic_sampling
from algorithms.matd3_forecast_context.algorithm import Matd3ForecastContext
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT


install_diagnostic_sampling(
    Matd3ForecastContext,
    variant="matd3_forecast_context",
)

ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_forecast_context",
    display_name="普通 T（预测上下文实验）",
    algorithm_class=Matd3ForecastContext,
    default_config_path=(
        PROJECT_ROOT / "configs" / "algorithms" / "matd3_forecast_context.yaml"
    ),
    description=(
        "Ordinary-T MATD3 forecast-context ablation with configurable local horizon "
        "and peer future energy-balance inputs."
    ),
    environment_overrides={
        "ess_sharing": {"observation_mode": "peer_energy_summary"}
    },
)
