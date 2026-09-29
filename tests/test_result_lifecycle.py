from __future__ import annotations

import csv
import json
import subprocess
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.layout import DEFAULT_WORKING_OUTPUT_ROOT
from experiments.result_lifecycle import (
    PromotionError,
    _run_git,
    capture_git_identity,
    inspect_run,
    promote_run,
)
from scripts.run_experiment import build_managed_parser


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_complete_run(root: Path, *, run_id: str = "PNN-TRANSFORMER_test_seed42", dirty=False) -> Path:
    run_dir = root / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best_model.pt").write_bytes(b"checkpoint")
    (run_dir / "config_snapshot.yaml").write_text("algorithm: test\n", encoding="utf-8")
    (run_dir / "test_daily_metrics.csv").write_text("day,profit\n1,1.0\n", encoding="utf-8")
    (run_dir / "test_summary.json").write_text("{}", encoding="utf-8")
    _write_json(run_dir / "validation_history.json", [{"episode": 100, "team_profit_mean": 1.0}])
    _write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "algorithm_slug": "pnn_transformer",
            "algorithm_display_name": "PNN-Transformer",
            "seed": 42,
            "train_episodes": 5000,
            "test_episodes": 55,
            "config_hash": "config-hash",
            "git_branch": "codex/controllable-devices",
            "git_commit": "abc123",
            "git_dirty": dirty,
            "experiment_contract": {"experiment_profile": None},
        },
    )
    checkpoint_selection = {
        "selection_metric": "validation_team_profit_mean",
        "best_episode": 300,
        "best_validation_score": -809.2,
        "best_checkpoint": "checkpoints/best_model.pt",
        "final_checkpoint": "checkpoints/final_model.pt",
    }
    _write_json(
        run_dir / "training_summary.json",
        {"episodes": 5000, "checkpoint_selection": checkpoint_selection},
    )
    _write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "algorithm_slug": "pnn_transformer",
            "algorithm_display_name": "PNN-Transformer",
            "checkpoint_selection": checkpoint_selection,
            "test_days_count": 55,
        },
    )
    return run_dir


def test_managed_parser_defaults_to_working_runs():
    assert DEFAULT_WORKING_OUTPUT_ROOT == Path("outputs") / "working_runs"
    args = build_managed_parser().parse_args([])
    assert args.output_root == DEFAULT_WORKING_OUTPUT_ROOT


def test_managed_parser_preserves_explicit_output_root(tmp_path: Path):
    args = build_managed_parser().parse_args(["--output-root", str(tmp_path)])
    assert args.output_root == tmp_path


def test_git_output_is_decoded_as_utf8_on_windows(monkeypatch, tmp_path: Path):
    calls = {}

    def fake_run(command, **kwargs):
        calls.update(kwargs)
        return SimpleNamespace(stdout="分支/工作区\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _run_git(tmp_path, "status") == "分支/工作区"
    assert calls["encoding"] == "utf-8"
    assert calls["errors"] == "replace"


def test_capture_git_identity_reports_branch_commit_and_dirty(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("clean", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "test-branch"], cwd=tmp_path, check=True, capture_output=True)

    clean = capture_git_identity(tmp_path)
    assert clean["git_branch"] == "test-branch"
    assert clean["git_commit"]
    assert clean["git_dirty"] is False

    (tmp_path / "untracked.txt").write_text("dirty", encoding="utf-8")
    dirty = capture_git_identity(tmp_path)
    assert dirty["git_dirty"] is True


def test_inspect_run_rejects_inconsistent_run_id(tmp_path: Path):
    run_dir = _make_complete_run(tmp_path)
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    payload["run_id"] = "wrong-id"
    _write_json(run_dir / "summary.json", payload)
    with pytest.raises(PromotionError, match="run_id"):
        inspect_run(run_dir)


def test_promote_clean_run_copies_and_registers_without_deleting_source(tmp_path: Path):
    source = _make_complete_run(tmp_path / "working_runs" / "pnn_transformer" / "runs")
    releases = tmp_path / "releases"
    now = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    manifest = promote_run(source, releases, "baseline_v1", now=now)

    final_dir = releases / "baseline_v1" / source.name
    assert source.exists()
    assert final_dir.exists()
    assert (final_dir / "release_manifest.json").exists()
    assert (final_dir / "README.md").exists()
    assert manifest["release_name"] == "baseline_v1"
    assert manifest["allow_dirty_override"] is False

    with (releases / "registry.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == source.name
    assert rows[0]["release_name"] == "baseline_v1"


def test_promote_rejects_dirty_run_unless_explicitly_allowed(tmp_path: Path):
    source = _make_complete_run(tmp_path / "working_runs" / "pnn_transformer" / "runs", dirty=True)
    releases = tmp_path / "releases"
    with pytest.raises(PromotionError, match="dirty"):
        promote_run(source, releases, "dirty_v1")
    assert not (releases / "dirty_v1" / source.name).exists()
    assert source.exists()

    manifest = promote_run(source, releases, "dirty_v1", allow_dirty=True)
    assert manifest["git_dirty"] is True
    assert manifest["allow_dirty_override"] is True
    readme = (releases / "dirty_v1" / source.name / "README.md").read_text(encoding="utf-8")
    assert "WARNING" in readme


def test_promote_rejects_existing_destination_instead_of_overwriting(tmp_path: Path):
    source = _make_complete_run(tmp_path / "working_runs" / "pnn_transformer" / "runs")
    releases = tmp_path / "releases"
    promote_run(source, releases, "baseline_v1")
    sentinel = releases / "baseline_v1" / source.name / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(PromotionError, match="already exists"):
        promote_run(source, releases, "baseline_v1")
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_promote_rejects_missing_required_artifact_and_preserves_source(tmp_path: Path):
    source = _make_complete_run(tmp_path / "working_runs" / "pnn_transformer" / "runs")
    (source / "checkpoints" / "best_model.pt").unlink()
    with pytest.raises(PromotionError, match="best_model.pt"):
        promote_run(source, tmp_path / "releases", "broken_v1")
    assert source.exists()


def test_release_name_rejects_path_like_values():
    from experiments.result_lifecycle import validate_release_name
    for value in ("", ".", "..", "../bad", "a/b", r"a\\b"):
        with pytest.raises(PromotionError):
            validate_release_name(value)


def test_promote_unknown_git_state_requires_explicit_override(tmp_path: Path):
    source = _make_complete_run(tmp_path / "working_runs" / "pnn_transformer" / "runs", dirty=None)
    releases = tmp_path / "releases"
    with pytest.raises(PromotionError, match="unknown"):
        promote_run(source, releases, "unknown_v1")
    manifest = promote_run(source, releases, "unknown_v1", allow_dirty=True)
    assert manifest["allow_dirty_override"] is True


def test_same_release_can_hold_multiple_run_ids_and_registry_rows(tmp_path: Path):
    root = tmp_path / "working_runs" / "pnn_transformer" / "runs"
    first = _make_complete_run(root, run_id="PNN-TRANSFORMER_seed42")
    second = _make_complete_run(root, run_id="PNN-TRANSFORMER_seed99")
    second_meta = json.loads((second / "run_metadata.json").read_text(encoding="utf-8"))
    second_meta["seed"] = 99
    _write_json(second / "run_metadata.json", second_meta)
    releases = tmp_path / "releases"
    promote_run(first, releases, "flexible_load_v1")
    promote_run(second, releases, "flexible_load_v1")
    with (releases / "registry.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["run_id"] for row in rows} == {first.name, second.name}


def test_update_run_metadata_git_identity_preserves_existing_fields(tmp_path: Path):
    from experiments.result_lifecycle import update_run_metadata_git_identity
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "run_metadata.json", {"run_id": "run", "seed": 42, "git_commit": "old"})
    update_run_metadata_git_identity(
        run_dir,
        {"git_branch": "feature", "git_commit": "new", "git_dirty": False},
    )
    payload = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "run"
    assert payload["seed"] == 42
    assert payload["git_branch"] == "feature"
    assert payload["git_commit"] == "new"
    assert payload["git_dirty"] is False


def test_promote_repairs_copied_json_paths_to_release_directory(tmp_path: Path):
    source = _make_complete_run(tmp_path / "working_runs" / "pnn_transformer" / "runs")
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_directory"] = str(source)
    summary["best_checkpoint"] = str(source / "checkpoints" / "best_model.pt")
    summary["nested"] = {"test_evaluation": str(source / "test_evaluation.json")}
    _write_json(summary_path, summary)

    releases = tmp_path / "releases"
    promote_run(source, releases, "portable_v1")
    final_dir = releases / "portable_v1" / source.name
    copied = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))

    assert copied["run_directory"] == str(final_dir.resolve())
    assert copied["best_checkpoint"] == str((final_dir / "checkpoints" / "best_model.pt").resolve())
    assert copied["nested"]["test_evaluation"] == str((final_dir / "test_evaluation.json").resolve())
    assert str(source) not in json.dumps(copied)
