from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import re
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Mapping, Type

import algorithms
from algorithms.base import MultiAgentAlgorithm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class AlgorithmSpec:
    slug: str
    display_name: str
    algorithm_class: Type[Any]
    default_config_path: Path
    description: str = ""
    environment_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _SLUG_PATTERN.fullmatch(self.slug):
            raise ValueError(f"Invalid algorithm slug: {self.slug!r}")
        object.__setattr__(self, "default_config_path", Path(self.default_config_path).resolve())
        object.__setattr__(self, "environment_overrides", deepcopy(dict(self.environment_overrides)))

    def build_environment_config(self, base_config: Mapping[str, Any]) -> dict[str, Any]:
        """Return a run-local environment config with this method's declared controls applied."""
        result = deepcopy(dict(base_config))

        def merge(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
            for key, value in updates.items():
                if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
                    nested = dict(target[key])
                    merge(nested, value)
                    target[key] = nested
                else:
                    target[key] = deepcopy(value)

        merge(result, self.environment_overrides)
        return result

    def create(
        self,
        env: Any,
        config: Mapping[str, Any],
        *,
        device: str = "cpu",
        seed: int | None = None,
    ) -> MultiAgentAlgorithm:
        factory = getattr(self.algorithm_class, "from_env", None)
        if not callable(factory):
            raise TypeError(f"{self.display_name} must implement classmethod from_env")
        algorithm = factory(env, config, device=device, seed=seed)
        validate_algorithm_instance(algorithm, self.slug)
        return algorithm


def validate_algorithm_instance(algorithm: Any, slug: str = "algorithm") -> None:
    missing = [name for name in ("select_actions", "update", "save", "load")
               if not callable(getattr(algorithm, name, None))]
    if not hasattr(algorithm, "replay"):
        missing.append("replay")
    elif not callable(getattr(algorithm.replay, "add", None)):
        missing.append("replay.add")
    if missing:
        raise TypeError(f"Algorithm {slug!r} does not satisfy the shared pipeline contract: {missing}")


def get_algorithm_spec(slug: str) -> AlgorithmSpec:
    """Load only the requested plugin so broken experiments cannot affect other algorithms."""
    normalized = str(slug).strip().lower().replace("-", "_")
    if not _SLUG_PATTERN.fullmatch(normalized):
        raise KeyError(f"Invalid algorithm name: {slug!r}")
    module_name = f"algorithms.{normalized}.plugin"
    if importlib.util.find_spec(module_name) is None:
        available = ", ".join(available_algorithm_names()) or "none"
        raise KeyError(f"Unknown algorithm {normalized!r}. Available: {available}")
    module = importlib.import_module(module_name)
    spec = getattr(module, "ALGORITHM_SPEC", None)
    if not isinstance(spec, AlgorithmSpec):
        raise TypeError(f"{module_name} must expose ALGORITHM_SPEC: AlgorithmSpec")
    if spec.slug != normalized:
        raise ValueError(f"Plugin folder {normalized!r} declares mismatched slug {spec.slug!r}")
    if not spec.default_config_path.exists():
        raise FileNotFoundError(f"Default config not found for {spec.slug}: {spec.default_config_path}")
    return spec


def available_algorithm_names() -> list[str]:
    names: list[str] = []
    for item in pkgutil.iter_modules(algorithms.__path__):
        if not item.ispkg or item.name.startswith("_"):
            continue
        if importlib.util.find_spec(f"algorithms.{item.name}.plugin") is not None:
            names.append(item.name)
    return sorted(names)
