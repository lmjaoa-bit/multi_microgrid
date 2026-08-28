from __future__ import annotations

from pathlib import Path

from utils.config import compose_experiment_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_CONFIG = ROOT / "configs" / "multi_microgrid.yaml"
DEFAULT_ALGORITHM_CONFIG = (
    ROOT / "configs" / "algorithms" / "conditional_matd3.yaml"
)
ALGORITHM_NAME = "conditional_matd3"


def load_runtime_config(
    env_config: str | Path = DEFAULT_ENV_CONFIG,
    algorithm_config: str | Path = DEFAULT_ALGORITHM_CONFIG,
) -> dict:
    """Return the complete runtime config for this algorithm only."""
    runtime, _ = compose_experiment_config(env_config, algorithm_config)
    return runtime
