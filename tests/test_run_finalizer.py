from __future__ import annotations

import json
import subprocess
from pathlib import Path

from utils.run_finalizer import collect_git_provenance, finalize_completed_run


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_collect_git_provenance_reports_branch_commit_and_dirty(tmp_path: Path) -> None:
    subprocess.run(
        ["git", "init", "-b", "feature/test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("v1", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    clean = collect_git_provenance(tmp_path)
    assert clean["git_branch"] == "feature/test"
    assert len(clean["git_commit"]) == 40
    assert clean["git_dirty"] is False

    (tmp_path / "tracked.txt").write_text("v2", encoding="utf-8")
    dirty = collect_git_provenance(tmp_path)
    assert dirty["git_dirty"] is True


def test_finalize_completed_run_removes_redundant_artifacts_and_updates_metadata(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "outputs" / "matd3_shapley_q" / "runs" / "run-1"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)

    _write_json(run_dir / "run_metadata.json", {"git_commit": "old"})
    _write_json(run_dir / "training_history.json", [{"episode": 1}])
    (run_dir / "training_history.jsonl").write_text('{"episode":1}\n', encoding="utf-8")
    evaluation = {
        "algorithm": {"score": 1},
        "matd3": {"score": 1},
        "comparison": {"x": 1},
    }
    _write_json(run_dir / "test_evaluation.json", evaluation)
    _write_json(run_dir / "test_summary.json", evaluation)
    _write_json(
        run_dir / "training_summary.json",
        {
            "checkpoint_selection": {
                "best_checkpoint": "checkpoints/best_model.pt",
                "final_checkpoint": "checkpoints/final_model.pt",
            }
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "run_directory": str(run_dir),
            "checkpoint": str(checkpoints / "final_model.pt"),
            "best_checkpoint": str(checkpoints / "best_model.pt"),
            "checkpoint_selection": {
                "best_checkpoint": "checkpoints/best_model.pt",
                "final_checkpoint": "checkpoints/final_model.pt",
            },
            "algorithm_ess_terminal_soc": {"mg1": 0.1},
            "matd3_ess_terminal_soc": {"mg1": 0.1},
        },
    )
    payload = b"identical checkpoint bytes"
    (checkpoints / "best_model.pt").write_bytes(payload)
    (checkpoints / "final_model.pt").write_bytes(payload)

    returned_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    finalize_completed_run(
        returned_summary,
        project_root=tmp_path,
        provenance={
            "git_commit": "abc123",
            "git_branch": "feature/test",
            "git_dirty": False,
        },
    )

    assert not (run_dir / "test_summary.json").exists()
    assert not (run_dir / "training_history.jsonl").exists()
    assert not (checkpoints / "final_model.pt").exists()

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["git_commit"] == "abc123"
    assert metadata["git_branch"] == "feature/test"
    assert metadata["git_dirty"] is False

    test_eval = json.loads((run_dir / "test_evaluation.json").read_text(encoding="utf-8"))
    assert "matd3" not in test_eval
    assert test_eval["algorithm"] == {"score": 1}

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint"] == str(checkpoints / "best_model.pt")
    assert summary["checkpoint_selection"]["final_checkpoint"] == "checkpoints/best_model.pt"
    assert summary["checkpoint_selection"]["final_checkpoint_deduplicated"] is True
    assert "matd3_ess_terminal_soc" not in summary


def test_finalize_keeps_distinct_final_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "matd3" / "runs" / "run-2"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    _write_json(run_dir / "run_metadata.json", {})
    _write_json(run_dir / "training_history.json", [])
    _write_json(
        run_dir / "test_evaluation.json",
        {"algorithm": {"score": 1}, "matd3": {"score": 2}},
    )
    _write_json(
        run_dir / "summary.json",
        {
            "run_directory": str(run_dir),
            "checkpoint": str(checkpoints / "final_model.pt"),
            "best_checkpoint": str(checkpoints / "best_model.pt"),
            "checkpoint_selection": {
                "best_checkpoint": "checkpoints/best_model.pt",
                "final_checkpoint": "checkpoints/final_model.pt",
            },
        },
    )
    (checkpoints / "best_model.pt").write_bytes(b"best")
    (checkpoints / "final_model.pt").write_bytes(b"final")

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    finalize_completed_run(summary, project_root=tmp_path, provenance={})

    assert (checkpoints / "final_model.pt").exists()
    test_eval = json.loads((run_dir / "test_evaluation.json").read_text(encoding="utf-8"))
    assert test_eval["matd3"] == {"score": 2}


def test_run_experiment_finalizes_using_startup_git_provenance(monkeypatch) -> None:
    from scripts import run_experiment

    parsed_args = object()
    summary = {"run_directory": "outputs/example/runs/run-1"}
    provenance = {
        "git_commit": "abc123",
        "git_branch": "feature/test",
        "git_dirty": False,
    }
    calls = {}

    class Parser:
        def parse_args(self, argv):
            calls["argv"] = argv
            return parsed_args

    monkeypatch.setattr(run_experiment, "build_parser", lambda: Parser())
    monkeypatch.setattr(run_experiment, "train", lambda args: summary)
    monkeypatch.setattr(run_experiment, "collect_git_provenance", lambda root: provenance)

    def fake_finalize(returned_summary, *, project_root, provenance):
        calls["summary"] = returned_summary
        calls["project_root"] = project_root
        calls["provenance"] = provenance

    monkeypatch.setattr(run_experiment, "finalize_completed_run", fake_finalize)

    assert run_experiment.main(["--algorithm", "matd3_shapley_q"]) == 0
    assert calls["argv"] == ["--algorithm", "matd3_shapley_q"]
    assert calls["summary"] is summary
    assert calls["project_root"] == run_experiment.PROJECT_ROOT
    assert calls["provenance"] is provenance
