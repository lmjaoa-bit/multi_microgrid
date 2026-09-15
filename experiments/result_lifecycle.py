from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class PromotionError(RuntimeError):
    """Raised when a run cannot be promoted safely."""


_REQUIRED_ARTIFACTS = (
    "config_snapshot.yaml",
    "run_metadata.json",
    "training_summary.json",
    "summary.json",
    "test_daily_metrics.csv",
    "checkpoints/best_model.pt",
)

REGISTRY_FIELDS = (
    "release_name",
    "run_id",
    "algorithm_slug",
    "seed",
    "train_episodes",
    "test_episodes",
    "git_branch",
    "git_commit",
    "git_dirty",
    "config_hash",
    "best_validation_episode",
    "best_validation_score",
    "promoted_at",
    "release_run_directory",
)

_RELEASE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _run_git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def capture_git_identity(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    try:
        branch_raw = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
        commit = _run_git(root, "rev-parse", "HEAD") or None
        status = _run_git(root, "status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.SubprocessError):
        return {"git_branch": None, "git_commit": None, "git_dirty": None}
    return {
        "git_branch": None if branch_raw == "HEAD" else (branch_raw or None),
        "git_commit": commit,
        "git_dirty": bool(status),
    }


def update_run_metadata_git_identity(run_dir: str | Path, identity: Mapping[str, Any]) -> None:
    path = Path(run_dir) / "run_metadata.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"Cannot update Git identity in {path}: {exc}") from exc
    for key in ("git_branch", "git_commit", "git_dirty"):
        payload[key] = identity.get(key)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_release_name(name: str) -> str:
    value = str(name).strip()
    if value in {"", ".", ".."} or not _RELEASE_NAME_RE.fullmatch(value):
        raise PromotionError(f"Unsafe release name: {name!r}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PromotionError(f"Cannot read required file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PromotionError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromotionError(f"Expected JSON object in {path}")
    return payload


def _inspect_run(run_dir: Path, *, require_directory_name: bool) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise PromotionError(f"Run directory does not exist: {run_dir}")
    for relative in _REQUIRED_ARTIFACTS:
        path = run_dir / relative
        if not path.exists():
            raise PromotionError(f"Missing required artifact: {relative}")

    metadata = _read_json(run_dir / "run_metadata.json")
    training_summary = _read_json(run_dir / "training_summary.json")
    summary = _read_json(run_dir / "summary.json")

    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise PromotionError("run_metadata.json is missing run_id")
    if require_directory_name and run_dir.name != run_id:
        raise PromotionError(f"run_id {run_id!r} does not match directory name {run_dir.name!r}")
    if summary.get("run_id") != run_id:
        raise PromotionError("summary.json run_id does not match run_metadata.json run_id")

    algorithm_slug = metadata.get("algorithm_slug")
    if not algorithm_slug or summary.get("algorithm_slug") != algorithm_slug:
        raise PromotionError("algorithm_slug is missing or inconsistent across metadata and summary")

    git_commit = metadata.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise PromotionError("run metadata does not contain a Git commit")

    checkpoint_selection = training_summary.get("checkpoint_selection") or summary.get("checkpoint_selection") or {}
    if not isinstance(checkpoint_selection, dict):
        raise PromotionError("checkpoint_selection must be a JSON object")
    if checkpoint_selection.get("selection_metric") == "validation_team_profit_mean":
        if not (run_dir / "validation_history.json").exists():
            raise PromotionError("Missing required artifact: validation_history.json")

    if not ((run_dir / "test_summary.json").exists() or (run_dir / "test_evaluation.json").exists()):
        raise PromotionError("Missing formal test artifact: test_summary.json or test_evaluation.json")
    try:
        test_days_count = int(summary.get("test_days_count", 0))
    except (TypeError, ValueError) as exc:
        raise PromotionError("summary.json contains invalid test_days_count") from exc
    if test_days_count <= 0:
        raise PromotionError("Formal testing is incomplete: test_days_count must be positive")

    return {
        "run_id": run_id,
        "algorithm_slug": str(algorithm_slug),
        "algorithm_display_name": metadata.get("algorithm_display_name") or summary.get("algorithm_display_name") or str(algorithm_slug),
        "seed": metadata.get("seed"),
        "train_episodes": metadata.get("train_episodes", training_summary.get("episodes")),
        "test_episodes": metadata.get("test_episodes", test_days_count),
        "experiment_profile": (metadata.get("experiment_contract") or {}).get("experiment_profile"),
        "config_hash": metadata.get("config_hash"),
        "git_branch": metadata.get("git_branch"),
        "git_commit": git_commit,
        "git_dirty": metadata.get("git_dirty"),
        "best_checkpoint": checkpoint_selection.get("best_checkpoint", "checkpoints/best_model.pt"),
        "best_validation_episode": checkpoint_selection.get("best_episode"),
        "best_validation_score": checkpoint_selection.get("best_validation_score"),
    }


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    return _inspect_run(Path(run_dir), require_directory_name=True)


def _replace_path_prefix(value: Any, old_prefix: str, new_prefix: str) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        result = {}
        for key, item in value.items():
            converted, item_changed = _replace_path_prefix(item, old_prefix, new_prefix)
            result[key] = converted
            changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        changed = False
        result = []
        for item in value:
            converted, item_changed = _replace_path_prefix(item, old_prefix, new_prefix)
            result.append(converted)
            changed = changed or item_changed
        return result, changed
    if isinstance(value, str):
        converted = value
        candidates = (
            (old_prefix, new_prefix),
            (old_prefix.replace("\\", "/"), new_prefix.replace("\\", "/")),
        )
        for old, new in candidates:
            if old and old in converted:
                converted = converted.replace(old, new)
        return converted, converted != value
    return value, False


def _repair_copied_json_paths(temp_dir: Path, source: Path, final_dir: Path) -> list[str]:
    repaired: list[str] = []
    for name in (
        "run_metadata.json",
        "training_summary.json",
        "summary.json",
        "test_evaluation.json",
        "test_summary.json",
    ):
        path = temp_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PromotionError(f"Cannot repair copied metadata path in {path}: {exc}") from exc
        converted, changed = _replace_path_prefix(payload, str(source), str(final_dir.resolve()))
        if changed:
            path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
            repaired.append(name)
    return repaired


def _registry_rows(registry_path: Path) -> list[dict[str, str]]:
    if not registry_path.exists():
        return []
    with registry_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_registry_row(registry_path: Path, manifest: Mapping[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _registry_rows(registry_path)
    key = (str(manifest["release_name"]), str(manifest["run_id"]))
    if any((row.get("release_name"), row.get("run_id")) == key for row in rows):
        raise PromotionError(f"Registry already contains release/run pair {key[0]}/{key[1]}")
    exists = registry_path.exists()
    with registry_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: manifest.get(field) for field in REGISTRY_FIELDS})


def _render_readme(manifest: Mapping[str, Any]) -> str:
    warning = ""
    if manifest.get("allow_dirty_override"):
        warning = "\n> **WARNING:** This release was promoted with dirty or unknown Git state via `--allow-dirty`.\n"
    return f"""# {manifest['release_name']} — {manifest['run_id']}

{warning}
- Algorithm: `{manifest['algorithm_display_name']}` (`{manifest['algorithm_slug']}`)
- Seed: `{manifest['seed']}`
- Train episodes: `{manifest['train_episodes']}`
- Test episodes: `{manifest['test_episodes']}`
- Git branch: `{manifest['git_branch']}`
- Git commit: `{manifest['git_commit']}`
- Git dirty: `{manifest['git_dirty']}`
- Config hash: `{manifest['config_hash']}`
- Best validation episode: `{manifest['best_validation_episode']}`
- Best validation score: `{manifest['best_validation_score']}`
- Promoted at: `{manifest['promoted_at']}`
- Source working run: `{manifest['source_run_directory']}`

This directory is a retained research result. Treat its source artifacts as immutable evidence.
"""


def promote_run(
    run_dir: str | Path,
    releases_root: str | Path,
    release_name: str,
    *,
    allow_dirty: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    source = Path(run_dir).resolve()
    releases_root = Path(releases_root).resolve()
    release_name = validate_release_name(release_name)
    info = inspect_run(source)

    dirty = info.get("git_dirty")
    if dirty is not False and not allow_dirty:
        state = "dirty" if dirty is True else "unknown"
        raise PromotionError(f"Run Git state is {state}; commit/clean the code or use --allow-dirty")

    release_group = releases_root / release_name
    final_dir = release_group / info["run_id"]
    temp_dir = release_group / f".{info['run_id']}.promoting"
    if final_dir.exists():
        raise PromotionError(f"Release destination already exists: {final_dir}")
    if temp_dir.exists():
        raise PromotionError(f"Temporary promotion destination already exists: {temp_dir}")

    registry_path = releases_root / "registry.csv"
    key = (release_name, info["run_id"])
    if any((row.get("release_name"), row.get("run_id")) == key for row in _registry_rows(registry_path)):
        raise PromotionError(f"Registry already contains release/run pair {release_name}/{info['run_id']}")

    release_group.mkdir(parents=True, exist_ok=True)
    promoted_at = (now or datetime.now().astimezone()).isoformat()
    manifest = {
        **info,
        "release_name": release_name,
        "promoted_at": promoted_at,
        "allow_dirty_override": bool(allow_dirty and dirty is not False),
        "source_run_directory": str(source),
        "release_run_directory": str(final_dir.resolve()),
        "repaired_json_paths": [],
    }

    try:
        shutil.copytree(source, temp_dir)
        repaired_json_paths = _repair_copied_json_paths(temp_dir, source, final_dir)
        copied_info = _inspect_run(temp_dir, require_directory_name=False)
        manifest["repaired_json_paths"] = repaired_json_paths
        if copied_info["run_id"] != info["run_id"]:
            raise PromotionError("Copied release failed run_id verification")
        (temp_dir / "release_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (temp_dir / "README.md").write_text(_render_readme(manifest), encoding="utf-8")
        temp_dir.rename(final_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    try:
        append_registry_row(registry_path, manifest)
    except Exception as exc:
        raise PromotionError(
            f"Release finalized at {final_dir}, but registry update failed: {exc}"
        ) from exc
    return manifest
