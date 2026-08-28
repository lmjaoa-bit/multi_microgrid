from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _project_root_from_config(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if parent.name == "configs":
            return parent.parent
    return path.parent


def _read_yaml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {resolved}")
    return resolved, loaded


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an environment or legacy combined configuration."""
    resolved, config = _read_yaml(path)
    config["_project_root"] = str(_project_root_from_config(resolved))
    return config


def load_algorithm_config(path: str | Path) -> dict[str, Any]:
    """Load one algorithm's independent hyperparameter/training configuration."""
    resolved, config = _read_yaml(path)
    algorithm = config.get("algorithm")
    if not isinstance(algorithm, Mapping) or not algorithm.get("name"):
        raise ValueError(
            f"Algorithm configuration must define algorithm.name: {resolved}"
        )
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError(f"training must be a mapping: {resolved}")
    config["_config_path"] = str(resolved)
    return config


def compose_experiment_config(
    env_config_path: str | Path,
    algorithm_config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compose shared environment settings with one self-contained algorithm config.

    The returned runtime config is a deep copy. Algorithm-specific code receives
    only its own ``algorithm`` section; it never needs parameters embedded in a
    predecessor algorithm's environment file.
    """
    environment = load_config(env_config_path)
    algorithm_document = load_algorithm_config(algorithm_config_path)
    runtime = copy.deepcopy(environment)
    runtime["algorithm"] = copy.deepcopy(algorithm_document["algorithm"])
    return runtime, algorithm_document


def runtime_config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a portable effective config with absolute external-data paths."""
    snapshot = copy.deepcopy(dict(config))
    project_root = Path(str(snapshot.pop("_project_root", Path.cwd())))
    data = snapshot.get("data", {})
    for key in ("train_path", "val_path", "test_path"):
        if key not in data:
            continue
        path = Path(str(data[key]))
        data[key] = str(path if path.is_absolute() else (project_root / path).resolve())
    return snapshot
