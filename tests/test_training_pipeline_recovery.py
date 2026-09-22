from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.training_pipeline import build_parser
from experiments.training_recovery import (
    config_text_hash,
    ensure_recoverable,
    validate_resume_directory,
    write_run_status,
    write_training_error,
)
from utils.training_history import TrainingHistoryJournal


def test_parser_accepts_resume_run_as_path() -> None:
    args = build_parser().parse_args(["--resume-run", "saved/run"])
    assert args.resume_run == Path("saved/run")


def test_config_hash_ignores_windows_newline_translation() -> None:
    assert config_text_hash("a: 1\nb: 2\n") == config_text_hash("a: 1\r\nb: 2\r\n")


def test_nonrecoverable_algorithm_is_rejected_clearly() -> None:
    with pytest.raises(ValueError, match="does not support resumable training"):
        ensure_recoverable(object())


def test_run_status_records_complete_transition_history(tmp_path: Path) -> None:
    for status in ("RUNNING", "TRAINED", "EVALUATING", "COMPLETED"):
        write_run_status(tmp_path, status, completed_episode=3)
    payload = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETED"
    assert [row["status"] for row in payload["history"]] == [
        "RUNNING", "TRAINED", "EVALUATING", "COMPLETED"
    ]
    assert all("timestamp" in row for row in payload["history"])


def test_training_error_and_failed_status_are_durable(tmp_path: Path) -> None:
    write_run_status(tmp_path, "RUNNING", completed_episode=4)
    try:
        raise RuntimeError("deliberate failure")
    except RuntimeError as exc:
        write_training_error(
            tmp_path,
            exc,
            completed_episode=4,
            update_step=88,
            durable_recovery_point="checkpoints/training_state/manifest.json",
        )
        write_run_status(tmp_path, "FAILED", completed_episode=4)
    error = json.loads((tmp_path / "training_error.json").read_text(encoding="utf-8"))
    assert error["exception_type"] == "RuntimeError"
    assert error["message"] == "deliberate failure"
    assert "RuntimeError: deliberate failure" in error["traceback"]
    assert error["completed_episode"] == 4
    assert error["update_step"] == 88
    assert error["durable_recovery_point"].endswith("manifest.json")


def test_run_status_atomic_write_supports_long_windows_run_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / ("algorithm_" + "a" * 40) / "runs" / ("run_" + "b" * 61)
    run_dir.mkdir(parents=True)
    write_run_status(run_dir, "RUNNING", completed_episode=0)
    assert json.loads((run_dir / "run_status.json").read_text())["status"] == "RUNNING"


def test_resume_directory_must_be_inside_selected_algorithm_runs(tmp_path: Path) -> None:
    runs = tmp_path / "outputs" / "working_runs" / "peak" / "runs"
    valid = runs / "run-1"
    valid.mkdir(parents=True)
    assert validate_resume_directory(valid, runs) == valid.resolve()
    outside = tmp_path / "other" / "run-2"
    outside.mkdir(parents=True)
    with pytest.raises(ValueError, match="selected algorithm"):
        validate_resume_directory(outside, runs)


def test_history_journal_resume_appends_existing_run(tmp_path: Path) -> None:
    path = tmp_path / "training_history.jsonl"
    first = TrainingHistoryJournal(path)
    first.append({"episode": 1})
    resumed = TrainingHistoryJournal(path, resume=True)
    resumed.append({"episode": 2})
    assert [json.loads(line)["episode"] for line in path.read_text().splitlines()] == [1, 2]


def test_history_journal_resume_discards_records_after_durable_cursor(tmp_path: Path) -> None:
    path = tmp_path / "training_history.jsonl"
    journal = TrainingHistoryJournal(path)
    for episode in (1, 2, 3):
        journal.append({"episode": episode})
    resumed = TrainingHistoryJournal(path, resume=True, completed_records=1)
    resumed.append({"episode": 2})
    assert [json.loads(line)["episode"] for line in path.read_text().splitlines()] == [1, 2]
