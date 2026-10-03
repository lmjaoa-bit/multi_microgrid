from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest


def test_formal_queue_uses_nine_fresh_runs_and_locked_configurations():
    from scripts.run_dueling_h24_study import build_jobs
    jobs = build_jobs()
    assert len(jobs) == 9
    assert len({(job["algorithm"], job["seed"]) for job in jobs}) == 9
    assert {job["seed"] for job in jobs} == {42, 43, 44}
    assert all(job["episodes"] == 5000 and job["test_episodes"] == 55 for job in jobs)
    assert len({job["environment_hash"] for job in jobs}) == 1
    assert jobs[0]["environment_hash"] == "745ca262da0cebcdf5fc4f1f1b3bf3b7f60d5136161a4b2f858e813b6c988bf2"


def daily(shift=0.):
    return {seed: [dict(date="2018-01-01", algorithm_team_profit=10. + shift),
                   dict(date="2018-01-02", algorithm_team_profit=20. + shift)]
            for seed in (42, 43, 44)}


def test_pairing_aligns_dates_and_constant_bootstrap_is_exact():
    from diagnostics.dueling_study_report import paired_deltas, paired_bootstrap
    left, right = daily(2.), daily()
    for rows in right.values():
        rows.reverse()
    values = paired_deltas(left, right)
    np.testing.assert_array_equal(values, np.full((3, 2), 2.))
    result = paired_bootstrap(values, repetitions=100)
    assert result["mean"] == result["ci_low"] == result["ci_high"] == 2.
    assert result["win_rate"] == 1. and result["pairs"] == 6


@pytest.mark.parametrize("problem", ["missing_date", "duplicate_date", "missing_seed", "scenario"])
def test_pairing_rejects_incompatible_evidence(problem):
    from diagnostics.dueling_study_report import paired_deltas
    left, right = daily(), daily()
    if problem == "missing_date":
        right[43].pop()
    elif problem == "duplicate_date":
        right[43][1]["date"] = right[43][0]["date"]
    elif problem == "missing_seed":
        del right[43]
    else:
        left[43][0]["generalized_reference_scenario_fingerprint"] = "left"
        right[43][0]["generalized_reference_scenario_fingerprint"] = "right"
    with pytest.raises(ValueError):
        paired_deltas(left, right)


@pytest.mark.parametrize("values", [np.empty((3, 0)), np.array([[float("nan")]]), np.ones(3)])
def test_bootstrap_rejects_invalid_data(values):
    from diagnostics.dueling_study_report import paired_bootstrap
    with pytest.raises(ValueError):
        paired_bootstrap(values)


def test_mechanism_uses_current_matd3q_and_performance_requires_positive_ci():
    from diagnostics.dueling_study_report import judge_study
    dueling = dict(dispatch_action_grad_rms=.12, dispatch_saturation_rate=.3, critic_adv_share=.1)
    result = judge_study(dueling, dict(dispatch_action_grad_rms=.11), 2., -.1)
    assert result["mechanism"] and not result["performance"] and result["conclusion"] == "部分成立"
    result = judge_study(dueling, dict(dispatch_action_grad_rms=.13), 2., .1)
    assert not result["mechanism"] and result["conclusion"] == "不成立"
    assert judge_study(dueling, dict(dispatch_action_grad_rms=.11), 2., .1)["conclusion"] == "继续投入"
    assert judge_study({}, {}, 2., .1)["conclusion"] == "待完成"


def test_partial_report_leaves_research_verdict_pending(tmp_path):
    from diagnostics.dueling_study_report import write_report
    path = tmp_path / "report.md"
    write_report(dict(git_commit="abc", jobs=[]), path)
    assert "研究判定：待完成" in path.read_text(encoding="utf-8")


@pytest.fixture
def git_repo(tmp_path):
    def command(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()
    command("init", "--quiet")
    command("config", "user.name", "Study test")
    command("config", "user.email", "study-test@example.invalid")
    (tmp_path / "source.txt").write_text("source")
    command("add", "source.txt")
    command("commit", "--quiet", "-m", "fixture")
    return tmp_path, command("rev-parse", "HEAD")


def test_dirty_or_changed_git_state_is_rejected(git_repo):
    from scripts.run_dueling_h24_study import assert_clean_identity
    root, commit = git_repo
    assert assert_clean_identity(root, commit) == commit
    with pytest.raises(RuntimeError, match="commit"):
        assert_clean_identity(root, "wrong")
    (root / "untracked.txt").write_text("new work")
    with pytest.raises(RuntimeError, match="dirty"):
        assert_clean_identity(root, commit)


def test_run_verification_rejects_dirty_metadata_before_accepting_results(tmp_path):
    from scripts.run_dueling_h24_study import validate_run
    (tmp_path / "run_metadata.json").write_text(json.dumps(dict(git_dirty=True, git_commit="abc")))
    with pytest.raises(ValueError, match="dirty"):
        validate_run(tmp_path, dict(algorithm="x", seed=42), "abc")


def test_current_process_liveness_probe_is_read_only():
    import os
    from scripts.run_dueling_h24_study import _process_alive
    assert _process_alive(os.getpid())
    assert not _process_alive(-1)


def test_stale_lock_recovery_rejects_live_controller_or_child(tmp_path, monkeypatch):
    import scripts.run_dueling_h24_study as queue
    lock, manifest = tmp_path / "study.lock", tmp_path / "study_manifest.json"
    manifest.write_text(json.dumps(dict(jobs=[dict(status="RUNNING", pid=22)])))
    lock.write_text("11")
    monkeypatch.setattr(queue, "_process_alive", lambda pid: pid == 11)
    with pytest.raises(RuntimeError, match="running"):
        queue._acquire_lock(lock, manifest, resume=True)
    monkeypatch.setattr(queue, "_process_alive", lambda pid: pid == 22)
    with pytest.raises(RuntimeError, match="running"):
        queue._acquire_lock(lock, manifest, resume=True)
    monkeypatch.setattr(queue, "_process_alive", lambda pid: False)
    queue._acquire_lock(lock, manifest, resume=True)
    assert lock.exists()


@pytest.mark.parametrize("failure", ["validation", "interrupt", "early_exit"])
def test_queue_failure_stops_owned_child_and_records_failed_attempt(tmp_path, monkeypatch, failure):
    import torch
    import scripts.run_dueling_h24_study as queue
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(queue, "assert_clean_identity", lambda **kwargs: "abc")
    config = tmp_path / "formal.yaml"
    config.write_text("formal")
    job = dict(algorithm=queue.DUELING, seed=42, status="PENDING", command=["unused"],
               algorithm_config=str(config), algorithm_config_sha256=queue._sha(config))
    jobs = [dict(job, seed=42 + index) for index in range(9)]
    monkeypatch.setattr(queue, "build_jobs", lambda: jobs)
    monkeypatch.setattr(queue, "write_report", lambda *args: None)
    monkeypatch.setattr(queue, "_identify_run", lambda *args: None if failure == "early_exit" else tmp_path / "run")
    monkeypatch.setattr(queue, "validate_run", lambda *args: (_ for _ in ()).throw(ValueError("validation failed")))

    class FakeProcess:
        pid = 123
        terminated = False
        exited = False

        def wait(self, **kwargs):
            if failure == "interrupt" and not self.terminated:
                raise KeyboardInterrupt("interrupted")
            self.exited = True
            return 1 if failure == "early_exit" else 0

        def poll(self):
            return 0 if self.exited else None

        def terminate(self):
            self.terminated = True

    process = FakeProcess()
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *args, **kwargs: process)
    study = tmp_path / "outputs/working_runs" / queue.DUELING / "studies/test"
    with pytest.raises((ValueError, KeyboardInterrupt, RuntimeError)):
        queue.execute_study(study)
    saved = json.loads((study / "study_manifest.json").read_text(encoding="utf-8"))
    assert saved["status"] == saved["jobs"][0]["status"] == "FAILED"
    assert len(saved["jobs"][0]["attempts"]) == 1
    assert saved["jobs"][0]["attempts"][0]["status"] == "FAILED"
    assert not (study / "study.lock").exists()
    if failure == "interrupt":
        assert process.terminated and process.exited
    if failure == "early_exit":
        original_log = study / saved["jobs"][0]["log"]
        original_log.write_text("first failure evidence", encoding="utf-8")
        with pytest.raises(RuntimeError):
            queue.execute_study(study, resume=True)
        retried = json.loads((study / "study_manifest.json").read_text(encoding="utf-8"))["jobs"][0]
        assert len(retried["attempts"]) == 2
        assert retried["log"] != str(original_log)
        assert original_log.read_text(encoding="utf-8") == "first failure evidence"
