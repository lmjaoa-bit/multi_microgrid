from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, MutableMapping


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def collect_git_provenance(project_root: Path | str) -> dict[str, Any]:
    """Capture the Git state that identifies the code used for a run."""
    root = Path(project_root).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "branch", "--show-current")
    status = _git_output(root, "status", "--porcelain")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": None if commit is None else bool(status),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _same_file_content(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    chunk_size = 1024 * 1024
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(chunk_size)
            right_chunk = right_handle.read(chunk_size)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _deduplicate_evaluation(run_dir: Path) -> None:
    evaluation_path = run_dir / "test_evaluation.json"
    legacy_summary_path = run_dir / "test_summary.json"

    if evaluation_path.exists() and legacy_summary_path.exists():
        try:
            if _read_json(evaluation_path) == _read_json(legacy_summary_path):
                legacy_summary_path.unlink()
        except (OSError, json.JSONDecodeError):
            pass

    if not evaluation_path.exists():
        return
    try:
        evaluation = _read_json(evaluation_path)
    except (OSError, json.JSONDecodeError):
        return
    if (
        isinstance(evaluation, dict)
        and "algorithm" in evaluation
        and evaluation.get("matd3") == evaluation.get("algorithm")
    ):
        evaluation.pop("matd3", None)
        _write_json(evaluation_path, evaluation)


def _deduplicate_checkpoint(
    run_dir: Path,
    summary: MutableMapping[str, Any],
) -> None:
    best_checkpoint = run_dir / "checkpoints" / "best_model.pt"
    final_checkpoint = run_dir / "checkpoints" / "final_model.pt"
    if not (best_checkpoint.exists() and final_checkpoint.exists()):
        return
    if not _same_file_content(best_checkpoint, final_checkpoint):
        return

    final_checkpoint.unlink()
    selection = summary.setdefault("checkpoint_selection", {})
    if isinstance(selection, dict):
        best_reference = selection.get("best_checkpoint", "checkpoints/best_model.pt")
        selection["final_checkpoint"] = best_reference
        selection["final_checkpoint_deduplicated"] = True
    summary["checkpoint"] = summary.get("best_checkpoint", str(best_checkpoint))

    training_summary_path = run_dir / "training_summary.json"
    if training_summary_path.exists():
        try:
            training_summary = _read_json(training_summary_path)
        except (OSError, json.JSONDecodeError):
            training_summary = None
        if isinstance(training_summary, dict):
            training_selection = training_summary.get("checkpoint_selection")
            if isinstance(training_selection, dict):
                best_reference = training_selection.get(
                    "best_checkpoint", "checkpoints/best_model.pt"
                )
                training_selection["final_checkpoint"] = best_reference
                training_selection["final_checkpoint_deduplicated"] = True
                _write_json(training_summary_path, training_summary)


def finalize_completed_run(
    summary: MutableMapping[str, Any],
    *,
    project_root: Path | str,
    provenance: Mapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Remove redundant completed-run artifacts and persist Git provenance."""
    root = Path(project_root).resolve()
    run_directory = summary.get("run_directory")
    if not run_directory:
        raise ValueError("summary must contain run_directory")
    run_dir = Path(str(run_directory))
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    metadata_path = run_dir / "run_metadata.json"
    try:
        metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update(dict(provenance) if provenance is not None else collect_git_provenance(root))
    _write_json(metadata_path, metadata)

    history_json = run_dir / "training_history.json"
    history_journal = run_dir / "training_history.jsonl"
    if history_json.exists() and history_journal.exists():
        history_journal.unlink()

    _deduplicate_evaluation(run_dir)

    if (
        "algorithm_ess_terminal_soc" in summary
        and summary.get("matd3_ess_terminal_soc") == summary.get("algorithm_ess_terminal_soc")
    ):
        summary.pop("matd3_ess_terminal_soc", None)

    _deduplicate_checkpoint(run_dir, summary)
    _write_json(run_dir / "summary.json", summary)
    return summary
