from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from envs.multi_microgrid_env import MultiMicrogridEnv


def projection_mode(config: str | Path | Mapping[str, Any]) -> str:
    if isinstance(config, (str, Path)):
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    else:
        cfg = config
    return str(cfg.get("constraint_projection", {}).get("mode", "strong"))


def build_environment(
    config: str | Path | dict[str, Any],
    *,
    split: str = "train",
) -> MultiMicrogridEnv:
    """Build the baseline or projection-ablation environment explicitly."""
    mode = projection_mode(config)
    if mode == "strong":
        return MultiMicrogridEnv(config, split=split)
    if mode == "instantaneous":
        from envs.instantaneous_projection_env import InstantaneousProjectionEnv
        return InstantaneousProjectionEnv(config, split=split)
    raise ValueError(f"Unsupported constraint_projection.mode {mode!r}")
