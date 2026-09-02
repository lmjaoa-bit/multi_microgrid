from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.registry import get_algorithm_spec


SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _class_name(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_"))


def create_variant(
    slug: str,
    *,
    base: str = "matd3",
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path]:
    normalized = slug.strip().lower().replace("-", "_")
    if not SLUG_PATTERN.fullmatch(normalized) or normalized.startswith("_"):
        raise ValueError("Algorithm name must match [a-z][a-z0-9_]*")
    base_spec = get_algorithm_spec(base)
    package_dir = project_root / "algorithms" / normalized
    config_path = project_root / "configs" / "algorithms" / f"{normalized}.yaml"
    if package_dir.exists() or config_path.exists():
        raise FileExistsError(f"Algorithm already exists: {normalized}")

    class_name = _class_name(normalized)
    package_dir.mkdir(parents=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        f"from algorithms.{normalized}.algorithm import {class_name}\n\n"
        f"__all__ = [\"{class_name}\"]\n",
        encoding="utf-8",
    )
    (package_dir / "algorithm.py").write_text(
        "from __future__ import annotations\n\n"
        "from algorithms.matd3 import MATD3\n\n\n"
        f"class {class_name}(MATD3):\n"
        f"    \"\"\"Experimental {class_name} variant. Keep algorithms/matd3 unchanged.\"\"\"\n\n"
        "    # Override only the behavior under experiment. If network definitions change,\n"
        "    # place the new actor/critic modules in this package and import them here.\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package_dir / "plugin.py").write_text(
        "from __future__ import annotations\n\n"
        f"from algorithms.{normalized}.algorithm import {class_name}\n"
        "from algorithms.registry import AlgorithmSpec, PROJECT_ROOT\n\n\n"
        "ALGORITHM_SPEC = AlgorithmSpec(\n"
        f"    slug=\"{normalized}\",\n"
        f"    display_name=\"{class_name}\",\n"
        f"    algorithm_class={class_name},\n"
        f"    default_config_path=PROJECT_ROOT / \"configs\" / \"algorithms\" / \"{normalized}.yaml\",\n"
        f"    description=\"Experimental variant derived from {base_spec.display_name}.\",\n"
        ")\n",
        encoding="utf-8",
    )
    config_text = base_spec.default_config_path.read_text(encoding="utf-8")
    config_text = re.sub(
        r"(?m)^(\s*name:\s*).+$",
        rf"\g<1>{normalized}",
        config_text,
        count=1,
    )
    config_path.write_text(config_text, encoding="utf-8")
    return package_dir, config_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an isolated MATD3-derived algorithm package")
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", default="matd3")
    args = parser.parse_args(argv)
    package, config = create_variant(args.name, base=args.base)
    print(f"Created algorithm package: {package}")
    print(f"Created independent config: {config}")
    print(f"Run: python scripts/run_experiment.py --algorithm {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
