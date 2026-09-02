from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics_collector import write_json


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _unique_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination
    suffix = 1
    while True:
        candidate = destination.with_name(f"{destination.stem}_{suffix}{destination.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _replace_path_prefix(value: Any, old_prefix: str, new_prefix: str) -> tuple[Any, bool]:
    changed = False
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            converted, item_changed = _replace_path_prefix(item, old_prefix, new_prefix)
            result[key] = converted
            changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        result_list = []
        for item in value:
            converted, item_changed = _replace_path_prefix(item, old_prefix, new_prefix)
            result_list.append(converted)
            changed = changed or item_changed
        return result_list, changed
    if isinstance(value, str):
        candidates = (
            (old_prefix, new_prefix),
            (old_prefix.replace("\\", "/"), new_prefix.replace("\\", "/")),
        )
        converted = value
        for old, new in candidates:
            if old in converted:
                converted = converted.replace(old, new)
        return converted, converted != value
    return value, False


def repair_run_json_paths(run_dir: Path, algorithm_root: Path) -> list[str]:
    old_prefix = str((algorithm_root / run_dir.name).resolve())
    new_prefix = str(run_dir.resolve())
    repaired: list[str] = []
    for path in run_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        converted, changed = _replace_path_prefix(payload, old_prefix, new_prefix)
        if changed:
            write_json(path, converted)
            repaired.append(str(path))
    return repaired


def organize_algorithm_results(algorithm_root: Path, *, apply: bool = False) -> dict[str, Any]:
    root = _inside(algorithm_root, algorithm_root)
    runs_root = _inside(root / "runs", root)
    legacy_root = _inside(root / "legacy", root)
    moves: list[tuple[Path, Path]] = []
    if not root.exists():
        return {"algorithm_root": str(root), "moves": [], "applied": apply, "exists": False}

    for item in root.iterdir():
        if item.name in {"runs", "legacy", "latest_run.json"}:
            continue
        if item.is_dir() and (item / "run_metadata.json").exists():
            destination = _unique_destination(runs_root / item.name)
        else:
            destination = _unique_destination(legacy_root / item.name)
        moves.append((_inside(item, root), _inside(destination, root)))

    repaired_json_files: list[str] = []
    if apply:
        runs_root.mkdir(parents=True, exist_ok=True)
        legacy_root.mkdir(parents=True, exist_ok=True)
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))

        run_dirs = [path for path in runs_root.iterdir()
                    if path.is_dir() and (path / "run_metadata.json").exists()]
        for run_dir in run_dirs:
            repaired_json_files.extend(repair_run_json_paths(run_dir, root))
        if run_dirs:
            latest = max(run_dirs, key=lambda path: path.stat().st_mtime)
            metadata = json.loads((latest / "run_metadata.json").read_text(encoding="utf-8"))
            write_json(root / "latest_run.json", {
                "algorithm_slug": metadata.get("algorithm_slug", root.name),
                "algorithm_display_name": metadata.get("algorithm_display_name", metadata.get("algorithm")),
                "run_id": latest.name,
                "run_directory": str(latest.resolve()),
            })
    return {
        "algorithm_root": str(root),
        "moves": [{"from": str(source), "to": str(destination)} for source, destination in moves],
        "applied": apply,
        "repaired_json_files": repaired_json_files,
        "exists": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Organize legacy result files without deleting them")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--algorithm")
    parser.add_argument("--apply", action="store_true", help="Apply the displayed moves")
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    roots = [output_root / args.algorithm] if args.algorithm else [
        path for path in output_root.iterdir() if path.is_dir()
    ]
    reports = [organize_algorithm_results(root, apply=args.apply) for root in roots]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
