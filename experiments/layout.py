from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil


DEFAULT_WORKING_OUTPUT_ROOT = Path("outputs") / "working_runs"


def run_root(path: str | Path) -> Path:
    """Accept either the public run directory or its internal artifacts directory."""
    root = Path(path).resolve()
    if root.name == "artifacts" and (root.parent / "figures").is_dir():
        return root.parent
    return root


def artifact_directory(path: str | Path) -> Path:
    """Read new two-folder runs and legacy flat runs through the same interface."""
    root = run_root(path)
    candidate = root / "artifacts"
    return candidate if candidate.is_dir() else root


def figure_directory(path: str | Path) -> Path:
    return run_root(path) / "figures"


def organize_run_artifacts(path: str | Path) -> Path:
    """Migrate a completed working run in place, verifying every moved file.

    The caller backs up the run first. Releases and active training runs are
    rejected; destinations are preflighted and never overwritten.
    """
    root = run_root(path)
    if (root / "release_manifest.json").exists() or "releases" in root.parts:
        raise ValueError("Retained releases are immutable")
    if not root.is_dir():
        raise FileNotFoundError(root)
    status_path = artifact_directory(root) / "run_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") in {"RUNNING", "EVALUATING"}:
            raise ValueError("Cannot reorganize an active training run")
    entries = [entry for entry in root.iterdir() if entry.name not in {"figures", "artifacts"}]
    target = root / "artifacts"
    if not entries:
        target.mkdir(exist_ok=True)
        (root / "figures").mkdir(exist_ok=True)
        return target
    before = {}
    for entry in entries:
        destination = target / entry.name
        if entry.is_symlink() or not entry.resolve().is_relative_to(root):
            raise ValueError(f"Source escapes the run directory: {entry}")
        if destination.exists() or not destination.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe or occupied migration destination: {destination}")
        files = [entry] if entry.is_file() else list(entry.rglob("*"))
        for file in files:
            if file.is_symlink() or not file.resolve().is_relative_to(root):
                raise ValueError(f"Linked artifact cannot be moved: {file}")
            if file.is_file():
                before[file.relative_to(root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    target.mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    for entry in entries:
        shutil.move(str(entry), str(target / entry.name))
    for relative, digest in before.items():
        if hashlib.sha256((target / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Moved artifact verification failed: {relative}")

    moved_names = {entry.name for entry in entries}

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            prefix = root.as_posix() + "/"
            relative = normalized[len(prefix):] if normalized.startswith(prefix) else normalized
            if relative.split("/")[0] in moved_names:
                if normalized.startswith(prefix):
                    new = root / "artifacts" / Path(relative)
                    return str(new) if "\\" in value else new.as_posix()
                return "artifacts/" + relative
        return value

    repaired = []
    for file in target.rglob("*.json"):
        payload = json.loads(file.read_text(encoding="utf-8"))
        converted = relocate(payload)
        if converted != payload:
            file.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
            repaired.append(file.relative_to(target).as_posix())
    audit = {"layout": "figures + artifacts", "moved_files_sha256_before": before,
             "moved_files_verified_before_path_repair": True, "path_repaired_json": repaired}
    (target / "layout_migration.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return target


@dataclass(frozen=True)
class ExperimentLayout:
    """Stable per-algorithm output layout shared by every experiment."""

    algorithm_root: Path
    runs_root: Path
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    figures_dir: Path
    hourly_dir: Path

    @property
    def artifacts_dir(self) -> Path:
        return artifact_directory(self.run_dir)

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        algorithm_slug: str,
        display_name: str,
        seed: int,
        *,
        algorithm_root_override: str | Path | None = None,
    ) -> "ExperimentLayout":
        algorithm_root = (
            Path(algorithm_root_override).resolve()
            if algorithm_root_override is not None
            else Path(output_root).resolve() / algorithm_slug
        )
        runs_root = algorithm_root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        normalized = "".join(
            character for character in display_name.upper()
            if character.isalnum() or character in {"-", "_"}
        ) or algorithm_slug.upper()
        base = f"{normalized}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_seed{int(seed)}"
        run_dir = runs_root / base
        suffix = 1
        while run_dir.exists():
            run_dir = runs_root / f"{base}_{suffix}"
            suffix += 1
        run_dir.mkdir()
        artifacts = run_dir / "artifacts"
        artifacts.mkdir()
        checkpoints = artifacts / "checkpoints"
        figures = run_dir / "figures"
        hourly = artifacts / "test_hourly_trajectories"
        checkpoints.mkdir()
        figures.mkdir()
        return cls(
            algorithm_root=algorithm_root,
            runs_root=runs_root,
            run_id=run_dir.name,
            run_dir=run_dir.resolve(),
            checkpoints_dir=checkpoints.resolve(),
            figures_dir=figures.resolve(),
            hourly_dir=hourly.resolve(),
        )
