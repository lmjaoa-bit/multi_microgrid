from __future__ import annotations

import ast
from pathlib import Path

from algorithms.conditional_matd3 import (
    DEFAULT_ALGORITHM_CONFIG,
    DEFAULT_ENV_CONFIG,
    load_runtime_config,
)
from utils.config import load_algorithm_config, load_config


ROOT = Path(__file__).resolve().parents[1]


def _algorithm_packages() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "algorithms").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )


def test_each_algorithm_has_own_config_and_training_entrypoint():
    packages = _algorithm_packages()
    assert packages
    for package in packages:
        name = package.name
        config_path = ROOT / "configs" / "algorithms" / f"{name}.yaml"
        train_path = ROOT / "scripts" / f"train_{name}.py"
        assert config_path.is_file(), f"Missing independent config for {name}"
        assert train_path.is_file(), f"Missing independent training script for {name}"
        config = load_algorithm_config(config_path)
        assert config["algorithm"]["name"] == name
        assert "training" in config


def test_algorithm_packages_do_not_import_other_algorithm_packages():
    package_names = {path.name for path in _algorithm_packages()}
    for package in _algorithm_packages():
        for source_path in package.glob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            foreign = {
                imported.split(".")[1]
                for imported in imports
                if imported.startswith("algorithms.")
                and len(imported.split(".")) > 1
                and imported.split(".")[1] in package_names
                and imported.split(".")[1] != package.name
            }
            assert not foreign, (
                f"{source_path} imports predecessor/foreign algorithms: "
                f"{sorted(foreign)}"
            )


def test_environment_and_algorithm_configs_are_separate_and_composable():
    environment = load_config(DEFAULT_ENV_CONFIG)
    algorithm = load_algorithm_config(DEFAULT_ALGORITHM_CONFIG)
    runtime = load_runtime_config()
    assert "algorithm" not in environment
    assert "episodes" in algorithm["training"]
    assert int(algorithm["training"]["episodes"]) > 0
    assert algorithm["algorithm"]["exploration_noise"]
    assert runtime["algorithm"] == algorithm["algorithm"]
    assert runtime["microgrids"] == environment["microgrids"]
