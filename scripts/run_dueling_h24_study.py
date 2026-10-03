"""Sequential formal experiments using the existing shared pipeline only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.dueling_study_report import DUELING, H24, MATD3Q, write_report

ENVIRONMENT_HASH = "745ca262da0cebcdf5fc4f1f1b3bf3b7f60d5136161a4b2f858e813b6c988bf2"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def assert_clean_identity(root=ROOT, expected_commit=None):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError("Study commit changed; refusing to mix source revisions")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Working tree is dirty; formal experiments require clean source")
    return commit


def _process_alive(pid):
    """Read process status; never use os.kill(pid, 0) on Windows."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            if error in (87, 1168):
                return False
            if error == 5:  # Access denied means liveness cannot safely be ruled out.
                return True
            raise ctypes.WinError(error)
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _acquire_lock(lock_path, manifest_path, *, resume):
    if lock_path.exists():
        if not resume:
            raise RuntimeError("Study lock already exists; use --resume after the previous process exits")
        owner = int(lock_path.read_text(encoding="utf-8"))
        previous = _read(manifest_path)
        children = [job.get("pid") for job in previous["jobs"] if job.get("status") in {"RUNNING", "FAILED"}]
        if any(_process_alive(pid) for pid in (owner, *children)):
            raise RuntimeError("Previous controller or training process is still running; refusing duplicate training")
        lock_path.unlink()  # Only this explicit, verified stale lock is removed.
    lock = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(lock, str(os.getpid()).encode())
    finally:
        os.close(lock)


def build_jobs():
    import yaml
    from algorithms.registry import get_algorithm_spec
    from experiments.profiles import experiment_contract
    base_path = ROOT / "configs/cooperative.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    jobs = []
    for seed in (42, 43, 44):
        for slug in (DUELING, H24, MATD3Q):
            spec = get_algorithm_spec(slug)
            cfg = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
            contract = experiment_contract(spec, spec.build_environment_config(base), base_path)
            if cfg["algorithm"]["name"] != slug or cfg["training"] != dict(episodes=5000, test_episodes=55):
                raise ValueError(f"Formal config changed: {slug}")
            if contract["resolved_environment_hash"] != ENVIRONMENT_HASH:
                raise ValueError(f"Environment hash changed: {slug}")
            command = [sys.executable, "-u", str(ROOT / "scripts/run_experiment.py"), "--algorithm", slug,
                       "--config", str(base_path), "--algorithm-config", str(spec.default_config_path),
                       "--seed", str(seed), "--device", "cuda", "--no-dashboard"]
            jobs.append(dict(algorithm=slug, seed=seed, episodes=5000, test_episodes=55,
                             environment_hash=ENVIRONMENT_HASH, algorithm_config=str(spec.default_config_path),
                             algorithm_config_sha256=_sha(spec.default_config_path),
                             environment_config_sha256=_sha(base_path), command=command, status="PENDING"))
    return jobs


def validate_run(path, job, commit):
    root = Path(path).resolve()
    meta = _read(root / "run_metadata.json")
    if meta.get("git_dirty") is not False:
        raise ValueError("Run metadata records dirty or unknown Git state")
    if meta.get("git_commit") != commit:
        raise ValueError("Run commit does not match the study")
    if meta.get("algorithm_slug") != job["algorithm"] or meta.get("seed") != job["seed"]:
        raise ValueError("Run plugin or seed mismatch")
    if meta.get("train_episodes") != job["episodes"] or meta.get("test_episodes") != job["test_episodes"]:
        raise ValueError("Run training/test scale mismatch")
    if meta.get("device") != "cuda" or meta.get("algorithm_config_path") != job["algorithm_config"]:
        raise ValueError("Run did not use its formal CUDA configuration")
    summary, test = _read(root / "summary.json"), _read(root / "test_summary.json")
    if not summary.get("formal_test_completed") or not summary.get("plotting_succeeded"):
        raise ValueError("Formal test or plotting incomplete")
    if test["experiment_contract"]["resolved_environment_hash"] != job["environment_hash"]:
        raise ValueError("Run environment mismatch")
    for key in ("algorithm_config_sha256", "environment_config_sha256"):
        if test[key] != job[key]:
            raise ValueError(f"Run {key} mismatch")
    best = root / "checkpoints/best_model.pt"
    if Path(test["checkpoint_used"]).resolve() != best or test["checkpoint_sha256"] != _sha(best):
        raise ValueError("Formal test did not use the verified best checkpoint")
    if not test["deterministic_policy"] or test["test_days_count"] != 55 or not test["coalition_milp_enabled"]:
        raise ValueError("Deterministic 55-day coalition evaluation incomplete")
    if test["milp_failure_count"] or test["shapley_valid_day_count"] + test["shapley_invalid_day_count"] != 55:
        raise ValueError("MILP/settlement incomplete")
    for key in ("equal_settled_profit_mean_by_microgrid", "shapley_settled_profit_mean_by_microgrid"):
        if set(test[key]) != {"mg1", "mg2", "mg3"}:
            raise ValueError("Missing settlement agents")
    figures = [{p.stem for p in (root / "figures").glob(f"*.{extension}") if not p.stem.startswith("01b_")}
               for extension in ("png", "pdf")]
    if len(figures[0]) != 15 or figures[0] != figures[1]:
        raise ValueError("Canonical 15 figure pairs incomplete")
    with (root / "test_daily_metrics.csv").open(encoding="utf-8", newline="") as handle:
        daily = list(csv.DictReader(handle))
    if len(daily) != 55 or len({row["date"] for row in daily}) != 55:
        raise ValueError("Missing or duplicated formal test dates")
    if not all(row.get("generalized_reference_scenario_fingerprint") and
               row.get("generalized_reference_initial_state_fingerprint") for row in daily):
        raise ValueError("Missing paired scenario fingerprints")
    for row in daily:
        for key in ("algorithm_team_profit", "gain_vs_idle_abs", "algorithm_generalized_system_cost", "renewable_curtailment_kwh"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite daily metric: {key}")
    if len(_read(root / "training_history.json")) != 5000:
        raise ValueError("Incomplete training history")
    if job["algorithm"] == DUELING:
        with (root / "actor_diagnostics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for key in ("critic_v_mean", "critic_adv_std", "critic_adv_abs_mean", "critic_adv_share",
                    "critic_dqda_dispatch_rms", "critic_dqda_dr_rms", "critic_v_adv_corr"):
            if not rows or not all(math.isfinite(float(row[key + "_mean"])) and
                                   float(row[key + "_nonfinite_count"]) == 0 for row in rows):
                raise ValueError(f"Missing/nonfinite dueling diagnostic: {key}")
    return root


def _identify_run(job, before):
    parent = ROOT / "outputs/working_runs" / job["algorithm"] / "runs"
    candidates = []
    for path in parent.glob("*"):
        if str(path) in before or not (path / "run_metadata.json").exists():
            continue
        meta = _read(path / "run_metadata.json")
        if meta.get("algorithm_slug") == job["algorithm"] and meta.get("seed") == job["seed"]:
            candidates.append(path)
    if len(candidates) > 1:
        raise RuntimeError("Ambiguous new run directories")
    return candidates[0] if candidates else None


def _last_episode(path):
    history = Path(path) / "training_history.jsonl"
    if not history.exists():
        return 0
    with history.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - 65536))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            return int(json.loads(line)["episode"])
        except (json.JSONDecodeError, KeyError):
            continue
    return 0


def execute_study(study_dir, *, resume=False):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this formal study")
    directory = Path(study_dir).resolve()
    allowed = ROOT / "outputs/working_runs" / DUELING / "studies"
    if not directory.is_relative_to(allowed.resolve()) or directory == allowed.resolve():
        raise ValueError("Study evidence must be in this algorithm's working studies directory")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path, lock_path = directory / "study_manifest.json", directory / "study.lock"
    if manifest_path.exists() != resume:
        raise ValueError("Use --resume for an existing study; a new study cannot overwrite evidence")
    commit = assert_clean_identity()
    if resume:
        manifest = _read(manifest_path)
        assert_clean_identity(expected_commit=manifest["git_commit"])
        expected = {(j["algorithm"], j["seed"], j["algorithm_config_sha256"]) for j in build_jobs()}
        actual = {(j["algorithm"], j["seed"], j["algorithm_config_sha256"]) for j in manifest["jobs"]}
        if actual != expected or len(manifest["jobs"]) != 9:
            raise ValueError("Study plan changed")
    else:
        manifest = dict(git_commit=commit, created_at=_now(), pid=os.getpid(), environment_hash=ENVIRONMENT_HASH,
                        projection_gradient_semantics="zero_residual_local_discharge_v1",
                        historical_seed42_reused=False, status="PREPARED", jobs=build_jobs())
    _acquire_lock(lock_path, manifest_path, resume=resume)
    process, active_job, attempt = None, None, None
    try:
        _write(manifest_path, manifest)
        write_report(manifest, directory / "report.md")
        for number, job in enumerate(manifest["jobs"], 1):
            assert_clean_identity(expected_commit=manifest["git_commit"])
            if job["status"] == "COMPLETED":
                validate_run(job["run_directory"], job, commit)
                continue
            if _sha(job["algorithm_config"]) != job["algorithm_config_sha256"]:
                raise ValueError("Formal YAML changed")
            parent = ROOT / "outputs/working_runs" / job["algorithm"] / "runs"
            before = {str(p) for p in parent.glob("*")}
            attempts = job.setdefault("attempts", [])
            job.pop("run_directory", None)
            job.pop("pid", None)
            log_path = directory / f"{number:02d}_{job['algorithm']}_seed{job['seed']}_attempt{len(attempts) + 1}.log"
            job.update(status="RUNNING", started_at=_now(), log=str(log_path), completed_episode=0)
            attempt = dict(number=len(attempts) + 1, status="RUNNING", log=str(log_path), started_at=job["started_at"])
            attempts.append(attempt)
            active_job = job
            manifest.update(status="RUNNING", active_job=number, pid=os.getpid())
            _write(manifest_path, manifest)
            environment = os.environ.copy()
            environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MKL_THREADING_LAYER="SEQUENTIAL", OMP_NUM_THREADS="1")
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen(job["command"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                job["pid"] = process.pid
                _write(manifest_path, manifest)
                while True:
                    try:
                        code = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        code = None
                    run = Path(job["run_directory"]) if job.get("run_directory") else _identify_run(job, before)
                    if run:
                        job.update(run_directory=str(run), completed_episode=_last_episode(run))
                        attempt["run_directory"] = str(run)
                    manifest["updated_at"] = _now()
                    _write(manifest_path, manifest)
                    if code is not None:
                        break
            if code != 0 or not job.get("run_directory"):
                job.update(status="FAILED", exit_code=code)
                raise RuntimeError(f"Formal run failed: {job['algorithm']} seed {job['seed']}; see {log_path}")
            validate_run(job["run_directory"], job, commit)
            job.update(status="COMPLETED", completed_at=_now(), completed_episode=5000)
            attempt.update(status="COMPLETED", completed_at=job["completed_at"])
            active_job, attempt = None, None
            _write(manifest_path, manifest)
            write_report(manifest, directory / "report.md")
        manifest.update(status="COMPLETED", completed_at=_now(), active_job=None)
        _write(manifest_path, manifest)
        write_report(manifest, directory / "report.md")
    except BaseException as exc:
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=30)
            except Exception as cleanup_error:
                manifest["child_cleanup_error"] = str(cleanup_error)
        if active_job is not None:
            active_job.update(status="FAILED", error=str(exc), completed_at=_now())
        if attempt is not None:
            attempt.update(status="FAILED", error=str(exc), completed_at=_now())
        manifest.update(status="FAILED", error=str(exc), updated_at=_now())
        _write(manifest_path, manifest)
        raise
    finally:
        if process is None or process.poll() is not None:
            lock_path.unlink()  # Keep the lock if child exit could not be confirmed.
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Regenerate the report from saved study evidence only")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_jobs(), ensure_ascii=False, indent=2))
    elif args.study_dir is None:
        parser.error("--study-dir is required when executing")
    elif args.report_only:
        write_report(_read(args.study_dir / "study_manifest.json"), args.study_dir / "report.md")
    else:
        execute_study(args.study_dir, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
