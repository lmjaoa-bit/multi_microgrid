from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime
from pathlib import Path
from experiments.layout import artifact_directory, run_root
from typing import Any, Mapping, cast

from algorithms.base import RecoverableAlgorithmProtocol
from utils.metrics_collector import json_ready


RUN_STATUSES = {"RUNNING", "TRAINED", "EVALUATING", "COMPLETED", "FAILED"}


def config_text_hash(text: str) -> str:
    """Hash configuration content independently of platform newline translation."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean, invalid = json_ready(payload)
    if invalid:
        raise ValueError(f"Non-finite lifecycle fields: {invalid}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_recoverable(algorithm: Any) -> RecoverableAlgorithmProtocol:
    if not isinstance(algorithm, RecoverableAlgorithmProtocol):
        raise ValueError(
            f"Algorithm {type(algorithm).__name__} does not support resumable training"
        )
    return cast(RecoverableAlgorithmProtocol, algorithm)


def write_run_status(run_dir: str | Path, status: str, **details: Any) -> None:
    if status not in RUN_STATUSES:
        raise ValueError(f"Invalid run status: {status}")
    path = artifact_directory(run_dir) / "run_status.json"
    history: list[dict[str, Any]] = []
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        history = list(previous.get("history", []))
    event = {
        "status": status,
        "timestamp": datetime.now().astimezone().isoformat(),
        **details,
    }
    history.append(event)
    _atomic_json(path, {**event, "history": history})


def write_training_error(
    run_dir: str | Path,
    error: BaseException,
    *,
    completed_episode: int,
    update_step: int,
    durable_recovery_point: str | None,
) -> None:
    _atomic_json(
        artifact_directory(run_dir) / "training_error.json",
        {
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "completed_episode": int(completed_episode),
            "update_step": int(update_step),
            "durable_recovery_point": durable_recovery_point,
            "timestamp": datetime.now().astimezone().isoformat(),
        },
    )


def validate_resume_directory(run_dir: str | Path, runs_root: str | Path) -> Path:
    resolved = run_root(run_dir)
    expected_root = Path(runs_root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Resume run directory not found: {resolved}")
    if not resolved.is_relative_to(expected_root):
        raise ValueError(
            f"Resume run must be inside the selected algorithm runs directory: {expected_root}"
        )
    return resolved
