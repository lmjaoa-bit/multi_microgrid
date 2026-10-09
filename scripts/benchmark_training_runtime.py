from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from algorithms.base import TrainingContextConsumerProtocol
from envs.multi_microgrid_env import MultiMicrogridEnv
from utils.runtime_benchmark import (
    apply_runtime_recommendation,
    cpu_thread_candidates,
    estimate_training_seconds,
    select_best_result,
)
from utils.torch_runtime import configure_torch_runtime


RESULT_PREFIX = "RUNTIME_BENCHMARK_JSON="


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark steady-state DRL training runtime on CPU and CUDA without changing algorithm math."
    )
    parser.add_argument(
        "--algorithm",
        default="matd3_hierarchical_transformer",
        help="Any registered algorithm slug",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "cooperative.yaml"
    )
    parser.add_argument("--algorithm-config", type=Path)
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Timed steady-state episodes per candidate after replay/model warmup",
    )
    parser.add_argument("--warmup-updates", type=int, default=8)
    parser.add_argument("--sample-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip the optional informational CUDA torch.compile candidate",
    )

    parser.add_argument("--compare-parent", help="Registered corrected baseline algorithm")
    parser.add_argument("--compare-candidate", help="Registered peak-risk candidate")
    parser.add_argument("--compare-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compare-updates", type=int, default=1000)
    parser.add_argument("--compare-repetitions", type=int, default=3)
    parser.add_argument("--compare-episodes", type=int, default=100)
    parser.add_argument(
        "--compare-active-peak", action="store_true",
        help="Measure the mid-ramp peak Actor branch (synthetic counter advancement, no extra training)",
    )
    parser.add_argument("--compare-active-step", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--compare-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    parser.add_argument("--cpu-threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--interop-threads", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--compile-actors", action="store_true", help=argparse.SUPPRESS)
    return parser


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _load_configs(args: argparse.Namespace):
    spec = get_algorithm_spec(args.algorithm)
    config_path = Path(args.config).resolve()
    algorithm_path = Path(args.algorithm_config or spec.default_config_path).resolve()
    base_env_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_env_config["_project_root"] = str(config_path.parent.parent)
    env_config = spec.build_environment_config(base_env_config)
    algorithm_cfg = yaml.safe_load(algorithm_path.read_text(encoding="utf-8"))
    algorithm_cfg.setdefault("runtime", {})["cpu_threads"] = int(args.cpu_threads)
    algorithm_cfg["runtime"]["interop_threads"] = int(args.interop_threads)
    return spec, env_config, algorithm_cfg, algorithm_path


def _maybe_compile_actors(algorithm) -> None:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build")
    for mg in algorithm.agent_ids:
        algorithm.actors[mg] = torch.compile(
            algorithm.actors[mg], mode="reduce-overhead", fullgraph=False
        )
        algorithm.target_actors[mg] = torch.compile(
            algorithm.target_actors[mg], mode="reduce-overhead", fullgraph=False
        )


def _random_actions(env, rng: np.random.Generator):
    return {
        mg: rng.uniform(-1.0, 1.0, env.action_dim).astype(np.float32)
        for mg in env.agent_ids
    }


def _capture_context(algorithm, env, observations, state, episode, total_steps, day):
    if isinstance(algorithm, TrainingContextConsumerProtocol):
        algorithm.capture_training_context(env, observations, state,
            {"episode_index": episode, "total_steps": total_steps, "day": str(day)})


def _prefill_replay(env, algorithm, batch_size: int, seed: int, *, learning_starts: int = 0) -> None:
    rng = np.random.default_rng(seed + 900_000)
    threshold = max(batch_size, learning_starts)
    episode = 0
    while len(algorithm.replay) < threshold:
        observations, info = env.reset(seed=seed + 800_000 + episode)
        state = info["global_state"]
        done = False
        while not done and len(algorithm.replay) < threshold:
            _capture_context(algorithm, env, observations, state, episode, len(algorithm.replay), info["day"])
            actions = _random_actions(env, rng)
            next_observations, rewards, done, step_info = env.step(actions)
            algorithm.replay.add(
                observations,
                state,
                actions,
                float(rewards[env.agent_ids[0]]),
                next_observations,
                step_info["global_state_after"],
                done,
            )
            observations = next_observations
            state = step_info["global_state_after"]
        episode += 1


def _worker(args: argparse.Namespace) -> dict[str, object]:
    if args.cpu_threads is None or args.device is None:
        raise ValueError("worker requires --device and --cpu-threads")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    spec, env_config, algorithm_cfg, algorithm_path = _load_configs(args)
    runtime = configure_torch_runtime(algorithm_cfg, args.device)
    env = MultiMicrogridEnv(env_config, split="train")
    algorithm = spec.create(env, algorithm_cfg, device=args.device, seed=args.seed)
    settings = algorithm_cfg["algorithm"]
    batch_size = int(settings.get("batch_size", 128))
    exploration = float(settings.get("exploration_noise", 0.15))

    compile_started = time.perf_counter()
    if args.compile_actors:
        _maybe_compile_actors(algorithm)
    compile_setup_seconds = time.perf_counter() - compile_started

    learning_starts = int(settings.get("learning_starts", 0))
    _prefill_replay(env, algorithm, batch_size, args.seed, learning_starts=learning_starts)
    for _ in range(max(0, int(args.warmup_updates))):
        algorithm.update(batch_size)
    _sync(args.device)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rollout_seconds = 0.0
    environment_seconds = 0.0
    replay_add_seconds = 0.0
    update_seconds = 0.0
    update_count = 0
    step_count = 0
    context_seconds = 0.0
    prefill_steps = len(algorithm.replay)
    updates_before = getattr(algorithm, "update_steps", None)

    total_started = time.perf_counter()
    for episode in range(int(args.episodes)):
        observations, reset_info = env.reset(seed=args.seed + 1_000_000 + episode)
        state = reset_info["global_state"]
        done = False
        while not done:
            started = time.perf_counter()
            _capture_context(algorithm, env, observations, state, episode, prefill_steps + step_count, reset_info["day"])
            context_seconds += time.perf_counter() - started
            _sync(args.device)
            started = time.perf_counter()
            actions = algorithm.select_actions(observations, exploration)
            _sync(args.device)
            rollout_seconds += time.perf_counter() - started

            started = time.perf_counter()
            next_observations, rewards, done, info = env.step(actions)
            environment_seconds += time.perf_counter() - started

            started = time.perf_counter()
            algorithm.replay.add(
                observations,
                state,
                actions,
                float(rewards[env.agent_ids[0]]),
                next_observations,
                info["global_state_after"],
                done,
            )
            replay_add_seconds += time.perf_counter() - started

            _sync(args.device)
            started = time.perf_counter()
            algorithm.update(batch_size)
            _sync(args.device)
            update_seconds += time.perf_counter() - started
            update_count += 1
            step_count += 1
            observations = next_observations
            state = info["global_state_after"]

    _sync(args.device)
    total_seconds = time.perf_counter() - total_started
    actual_updates = (algorithm.update_steps - updates_before) if updates_before is not None else None
    if actual_updates is not None and actual_updates != update_count:
        raise RuntimeError(f"Benchmark measured {actual_updates} actual updates for {update_count} attempts")

    _sync(args.device)
    sample_started = time.perf_counter()
    for _ in range(max(1, int(args.sample_iterations))):
        algorithm.replay.sample(batch_size, algorithm.agent_ids, algorithm.device)
    _sync(args.device)
    replay_sample_seconds = time.perf_counter() - sample_started

    seconds_per_episode = total_seconds / max(1, int(args.episodes))
    result: dict[str, object] = {
        "status": "ok",
        "algorithm": args.algorithm,
        "device": args.device,
        "cpu_threads": runtime["cpu_threads"],
        "interop_threads": runtime["interop_threads"],
        "compile_actors": bool(args.compile_actors),
        "compile_setup_seconds": compile_setup_seconds,
        "episodes": int(args.episodes),
        "steps": step_count,
        "updates": update_count,
        "successful_updates": actual_updates,
        "replay_prefill_steps": prefill_steps,
        "learning_starts": learning_starts,
        "context_protocol_enabled": isinstance(algorithm, TrainingContextConsumerProtocol),
        "total_seconds": total_seconds,
        "seconds_per_episode": seconds_per_episode,
        "rollout_ms_per_step": 1000.0 * rollout_seconds / max(1, step_count),
        "context_ms_per_step": 1000.0 * context_seconds / max(1, step_count),
        "environment_ms_per_step": 1000.0 * environment_seconds / max(1, step_count),
        "replay_add_ms_per_step": 1000.0 * replay_add_seconds / max(1, step_count),
        "update_ms_per_step": 1000.0 * update_seconds / max(1, update_count),
        "replay_sample_ms": 1000.0 * replay_sample_seconds / max(1, int(args.sample_iterations)),
        "updates_per_second": update_count / max(update_seconds, 1e-12),
        "estimated_500_seconds": estimate_training_seconds(seconds_per_episode, 500),
        "estimated_5000_seconds": estimate_training_seconds(seconds_per_episode, 5000),
        "algorithm_config": str(algorithm_path),
    }
    if args.device == "cuda":
        result["cuda_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
        result["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    return result


def _candidate_label(
    device: str, cpu_threads: int, interop_threads: int, compile_actors: bool
) -> str:
    suffix = "-compile" if compile_actors else ""
    return f"{device}-cpu{cpu_threads}-interop{interop_threads}{suffix}"


def _run_candidate(
    args: argparse.Namespace,
    *,
    device: str,
    cpu_threads: int,
    interop_threads: int,
    compile_actors: bool = False,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--algorithm", args.algorithm,
        "--config", str(Path(args.config).resolve()),
        "--episodes", str(args.episodes),
        "--warmup-updates", str(args.warmup_updates),
        "--sample-iterations", str(args.sample_iterations),
        "--seed", str(args.seed),
        "--device", device,
        "--cpu-threads", str(cpu_threads),
        "--interop-threads", str(interop_threads),
    ]
    if args.algorithm_config:
        command.extend(["--algorithm-config", str(Path(args.algorithm_config).resolve())])
    if compile_actors:
        command.append("--compile-actors")

    label = _candidate_label(device, cpu_threads, interop_threads, compile_actors)
    started = time.perf_counter()
    process = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - started
    if process.returncode != 0:
        return {
            "label": label,
            "status": "error",
            "device": device,
            "cpu_threads": cpu_threads,
            "interop_threads": interop_threads,
            "compile_actors": compile_actors,
            "seconds_per_episode": None,
            "wall_seconds": wall_seconds,
            "error": process.stderr.strip()[-4000:] or process.stdout.strip()[-4000:],
        }

    payload = None
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX):])
            break
    if payload is None:
        return {
            "label": label,
            "status": "error",
            "device": device,
            "cpu_threads": cpu_threads,
            "interop_threads": interop_threads,
            "compile_actors": compile_actors,
            "seconds_per_episode": None,
            "wall_seconds": wall_seconds,
            "error": "worker returned no benchmark JSON",
        }
    payload["label"] = label
    payload["wall_seconds"] = wall_seconds
    return payload


def _print_row(row: dict[str, object]) -> None:
    if row.get("status") != "ok":
        print(f"{row['label']:<32} ERROR")
        return
    print(
        f"{row['label']:<32} {float(row['seconds_per_episode']):>8.3f} s/ep | "
        f"update {float(row['update_ms_per_step']):>7.2f} ms | "
        f"rollout {float(row['rollout_ms_per_step']):>6.2f} ms | "
        f"replay {float(row['replay_sample_ms']):>6.2f} ms"
    )


def _driver(args: argparse.Namespace) -> int:
    logical_cpus = os.cpu_count() or 1
    threads = cpu_thread_candidates(logical_cpus)
    cuda_available = torch.cuda.is_available()
    results: list[dict[str, object]] = []

    print(
        f"Runtime benchmark: {args.algorithm} | logical CPUs={logical_cpus} | "
        f"CUDA={cuda_available}"
    )
    print("Stage 1: CPU-thread sweep (interop=1) on every available device")
    for device in (["cpu", "cuda"] if cuda_available else ["cpu"]):
        for cpu_threads in threads:
            row = _run_candidate(
                args, device=device, cpu_threads=cpu_threads, interop_threads=1
            )
            results.append(row)
            _print_row(row)

    best_stage1 = select_best_result(results)
    if best_stage1 is None:
        raise RuntimeError("No successful runtime benchmark candidates")

    print("Stage 2: inter-op sweep on the fastest eager device/thread pair")
    for interop_threads in (2, 4):
        if interop_threads > logical_cpus:
            continue
        row = _run_candidate(
            args,
            device=str(best_stage1["device"]),
            cpu_threads=int(best_stage1["cpu_threads"]),
            interop_threads=interop_threads,
        )
        results.append(row)
        _print_row(row)

    if cuda_available and not args.no_compile:
        successful_cuda = [
            row
            for row in results
            if row.get("status") == "ok"
            and row.get("device") == "cuda"
            and not row.get("compile_actors")
        ]
        best_cuda = select_best_result(successful_cuda)
        if best_cuda is not None:
            print("Stage 3: informational CUDA torch.compile Actor candidate")
            row = _run_candidate(
                args,
                device="cuda",
                cpu_threads=int(best_cuda["cpu_threads"]),
                interop_threads=int(best_cuda["interop_threads"]),
                compile_actors=True,
            )
            results.append(row)
            _print_row(row)

    best = select_best_result(results)
    assert best is not None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or (
        PROJECT_ROOT
        / "outputs"
        / "runtime_benchmarks"
        / f"{args.algorithm}_{timestamp}.json"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    spec = get_algorithm_spec(args.algorithm)
    source_algorithm_path = Path(args.algorithm_config or spec.default_config_path).resolve()
    source_algorithm_cfg = yaml.safe_load(source_algorithm_path.read_text(encoding="utf-8"))
    recommended_cfg = apply_runtime_recommendation(source_algorithm_cfg, best)
    recommended_config_path = output.with_name(output.stem + "_recommended.yaml")
    recommended_config_path.write_text(
        yaml.safe_dump(recommended_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    report = {
        "algorithm": args.algorithm,
        "logical_cpus": logical_cpus,
        "cuda_available": cuda_available,
        "benchmark_episodes_per_candidate": args.episodes,
        "results": results,
        "best": best,
        "recommended_device": best["device"],
        "recommended_runtime": {
            "cpu_threads": best["cpu_threads"],
            "interop_threads": best["interop_threads"],
        },
        "recommended_algorithm_config": str(recommended_config_path),
        "recommended_compile_actors": False,
        "estimated_500_seconds": best["estimated_500_seconds"],
        "estimated_5000_seconds": best["estimated_5000_seconds"],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    training_command = (
        f'python scripts/run_experiment.py --algorithm {args.algorithm} '
        f'--device {best["device"]} --algorithm-config "{recommended_config_path}"'
    )

    print("\nBEST PRODUCTION-EAGER RESULT")
    _print_row(dict(best))
    print(f"Recommended device: {best['device']}")
    print(
        "Recommended runtime: "
        f"cpu_threads={best['cpu_threads']}, interop_threads={best['interop_threads']}"
    )
    print(f"Estimated 500 episodes: {float(best['estimated_500_seconds']) / 60.0:.2f} min")
    print(f"Estimated 5000 episodes: {float(best['estimated_5000_seconds']) / 3600.0:.2f} h")
    print(f"JSON report: {output}")
    print(f"Recommended config: {recommended_config_path}")
    print("Recommended training command:")
    print(training_command)
    return 0



def _comparison_worker(args: argparse.Namespace) -> dict[str, object]:
    """Run a comparable fixed-replay microbenchmark in its own process.

    The 100-episode segment measures actual policy/environment/update time with
    the normal diagnostic schedule and recovery semantics retained.
    """
    if args.device is None or args.cpu_threads is None:
        raise ValueError("comparison worker requires --device and --cpu-threads")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    spec, env_cfg, algorithm_cfg, cfg_path = _load_configs(args)
    configure_torch_runtime(algorithm_cfg, args.device)
    env = MultiMicrogridEnv(env_cfg, split="train")
    algorithm = spec.create(env, algorithm_cfg, device=args.device, seed=args.seed)
    settings = algorithm_cfg["algorithm"]
    batch_size = int(settings.get("batch_size", 128))
    _prefill_replay(env, algorithm, batch_size, args.seed,
                    learning_starts=int(settings.get("learning_starts", 0)))
    for _ in range(max(8, int(args.warmup_updates))):
        algorithm.update(batch_size)
    if args.compare_active_step:
        # Both workers enter the same update phase. The cost Critic is only
        # warm-started: this measures active-branch compute, not convergence.
        if args.compare_active_step <= algorithm.update_steps:
            raise ValueError("active-peak comparison step must exceed actual warmup steps")
        algorithm.update_steps = args.compare_active_step
        if getattr(algorithm, "peak_critic_enabled", False):
            peak_cfg = algorithm.peak_risk_config
            algorithm.peak_critic_updates = max(
                algorithm.peak_critic_updates, int(peak_cfg["critic_warmup_updates"]))
    _sync(args.device)

    projection_counts = {"due": 0, "not_due": 0}
    replay_counts = {"due": 0, "not_due": 0}
    phase = ["not_due"]
    original_projection = algorithm.project_actions
    original_sample = algorithm.replay.sample

    def projected(*values, **kwargs):
        projection_counts[phase[0]] += 1
        return original_projection(*values, **kwargs)

    def sampled(*values, **kwargs):
        replay_counts[phase[0]] += 1
        return original_sample(*values, **kwargs)

    algorithm.project_actions = projected
    algorithm.replay.sample = sampled
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elapsed_by_phase = {"due": [], "not_due": []}
    for _ in range(args.compare_updates):
        due = (algorithm.update_steps + 1) % algorithm.policy_delay == 0
        phase[0] = "due" if due else "not_due"
        _sync(args.device)
        started = time.perf_counter()
        algorithm.update(batch_size)
        _sync(args.device)
        elapsed_by_phase[phase[0]].append((time.perf_counter() - started) * 1000.0)
    algorithm.project_actions = original_projection
    algorithm.replay.sample = original_sample

    # Identical seed and episode count, including environment, replay and
    # algorithm update time; no formal-test day is used here.
    training_start = time.perf_counter()
    for ep in range(args.compare_episodes):
        observations, info = env.reset(seed=args.seed + 900_000 + ep)
        state = info["global_state"]
        done = False
        while not done:
            _capture_context(algorithm, env, observations, state, ep, len(algorithm.replay), info["day"])
            actions = algorithm.select_actions(
                observations, noise_std=float(settings.get("exploration_noise", 0.15)))
            next_obs, rewards, done, step_info = env.step(actions)
            algorithm.replay.add(observations, state, actions,
                float(rewards[env.agent_ids[0]]), next_obs,
                step_info["global_state_after"], done)
            algorithm.update(batch_size)
            observations, state = next_obs, step_info["global_state_after"]
    _sync(args.device)
    training_seconds = time.perf_counter() - training_start

    # Inference only; the shared formal evaluator does not execute the cost critic.
    obs, _ = env.reset(seed=args.seed + 650_000)
    _sync(args.device)
    eval_started = time.perf_counter()
    for _ in range(args.compare_updates):
        algorithm.select_actions(obs, noise_std=0.0)
    _sync(args.device)
    eval_ms = (time.perf_counter() - eval_started) * 1000 / args.compare_updates

    snapshot_seconds = None
    if args.compare_episodes and hasattr(algorithm, "save_training_state"):
        with tempfile.TemporaryDirectory(prefix="peak_risk_speed_") as directory:
            _sync(args.device)
            started = time.perf_counter()
            algorithm.save_training_state(
                Path(directory), {"completed_episode": args.compare_episodes,
                                  "update_steps": algorithm.update_steps})
            _sync(args.device)
            snapshot_seconds = time.perf_counter() - started
    peak_memory_mb = (torch.cuda.max_memory_allocated() / 1024**2
                      if args.device == "cuda" else None)
    parent_reward_critic = next(iter(algorithm.critics.values()))
    critic_size = sum(p.numel() for p in parent_reward_critic.parameters())
    peak = getattr(algorithm, "peak_critic", None)
    peak_size = sum(p.numel() for p in peak.parameters()) if peak is not None else 0
    return {
        "status": "ok", "algorithm": args.algorithm, "device": args.device,
        "algorithm_config": str(cfg_path), "seed": args.seed,
        "synthetic_active_peak_step": args.compare_active_step or None,
        "update_calls": args.compare_updates, "end_to_end_episodes": args.compare_episodes,
        "due_update_ms": float(np.median(elapsed_by_phase["due"])),
        "not_due_update_ms": float(np.median(elapsed_by_phase["not_due"])),
        "due_update_mean_ms": float(np.mean(elapsed_by_phase["due"])),
        "not_due_update_mean_ms": float(np.mean(elapsed_by_phase["not_due"])),
        # An exact observed mean, weighted by actual update-phase call counts.
        "weighted_update_mean_ms": float(np.mean(
            elapsed_by_phase["due"] + elapsed_by_phase["not_due"])),
        "median_update_ms": float(np.median(
            elapsed_by_phase["due"] + elapsed_by_phase["not_due"])),
        "projector_calls": projection_counts, "replay_samples": replay_counts,
        "actor_due_calls": len(elapsed_by_phase["due"]),
        "actor_not_due_calls": len(elapsed_by_phase["not_due"]),
        "training_seconds": training_seconds, "deterministic_eval_ms": eval_ms,
        "snapshot_seconds": snapshot_seconds, "gpu_peak_memory_mb": peak_memory_mb,
        "peak_critic_parameter_fraction": peak_size / max(critic_size, 1),
    }


def _comparison_subprocess(args: argparse.Namespace, algorithm: str) -> dict[str, object]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--compare-worker",
        "--algorithm", algorithm, "--config", str(Path(args.config).resolve()),
        "--compare-updates", str(args.compare_updates),
        "--compare-episodes", str(args.compare_episodes),
        "--warmup-updates", str(args.warmup_updates),
        "--compare-active-step", str(args.compare_active_step),
        "--seed", str(args.seed), "--device", args.compare_device,
        "--cpu-threads", str(args.cpu_threads or 4), "--interop-threads",
        str(args.interop_threads),
    ]
    process = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(
            f"Benchmark worker {algorithm} failed: {process.stderr[-6000:] or process.stdout[-6000:]}"
        )
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise RuntimeError(f"Benchmark worker {algorithm} returned no report")


def _comparison_driver(args: argparse.Namespace) -> int:
    if not args.compare_parent or not args.compare_candidate:
        raise ValueError("Both --compare-parent and --compare-candidate are required")
    if args.algorithm_config:
        raise ValueError("Comparison uses each registered algorithm's own configuration")
    if args.compare_updates < 1000 or args.compare_repetitions < 3 or args.compare_episodes < 100:
        raise ValueError("Production comparison requires >=1000 updates, 3 repetitions and 100 episodes")
    parent_spec = get_algorithm_spec(args.compare_parent)
    candidate_spec = get_algorithm_spec(args.compare_candidate)
    parent_cfg = yaml.safe_load(parent_spec.default_config_path.read_text(encoding="utf-8"))
    candidate_cfg = yaml.safe_load(candidate_spec.default_config_path.read_text(encoding="utf-8"))
    for section in ("training", "validation", "logging"):
        if parent_cfg.get(section) != candidate_cfg.get(section):
            raise ValueError(f"{section} schedule differs between parent and candidate")
    for field in ("batch_size", "network", "policy_delay", "warmup_steps",
                  "replay_capacity", "target_policy_noise", "target_noise_clip"):
        if parent_cfg["algorithm"].get(field) != candidate_cfg["algorithm"].get(field):
            raise ValueError(f"algorithm.{field} differs between parent and candidate")
    if args.compare_active_peak:
        peak_cfg = candidate_cfg["algorithm"].get("peak_risk", {})
        if not peak_cfg.get("enabled") or not peak_cfg.get("critic_enabled"):
            raise ValueError("--compare-active-peak requires an enabled peak-risk candidate")
        args.compare_active_step = int(peak_cfg["actor_start_update"]) + max(
            1, int(peak_cfg["actor_weight_ramp_updates"]) // 2)
        # Same actor-update parity for both algorithms.
        args.compare_active_step += (-args.compare_active_step) % int(
            candidate_cfg["algorithm"]["policy_delay"])
        if args.compare_active_step <= max(8, int(args.warmup_updates)):
            raise ValueError("active-peak step must exceed the warmup update count")

    parent_rows = []
    candidate_rows = []
    for _ in range(args.compare_repetitions):
        parent_rows.append(_comparison_subprocess(args, args.compare_parent))
        candidate_rows.append(_comparison_subprocess(args, args.compare_candidate))

    def median(rows, field):
        return float(np.median([float(row[field]) for row in rows]))

    def overhead(field):
        p = median(parent_rows, field)
        c = median(candidate_rows, field)
        if p <= 0:
            raise ValueError(f"invalid parent benchmark {field}={p}")
        return 100.0 * (c / p - 1.0)

    def paired_overheads(field):
        ratios = []
        for index, (p_row, c_row) in enumerate(zip(parent_rows, candidate_rows)):
            p, c = float(p_row[field]), float(c_row[field])
            if not (np.isfinite(p) and np.isfinite(c)) or p <= 0:
                raise ValueError(f"invalid paired benchmark {field} at repetition {index}")
            ratios.append(100.0 * (c / p - 1.0))
        return ratios

    # Do not take a pooled median of alternating policy-due/non-due steps.
    # Per-worker means reflect all 1000 actual calls; median of paired percent
    # changes is less sensitive to between-run GPU load than ratio of medians.
    weighted_overheads = paired_overheads("weighted_update_mean_ms")
    paired_training_overheads = paired_overheads("training_seconds")
    paired_eval_overheads = paired_overheads("deterministic_eval_ms")

    def count_per_call(rows, key, phase):
        return float(np.mean([
            row[key][phase] / max(int(row["actor_due_calls"] if phase == "due"
                                              else row["actor_not_due_calls"]), 1)
            for row in rows
        ]))

    projector_parity = all(
        count_per_call(parent_rows, "projector_calls", phase)
        == count_per_call(candidate_rows, "projector_calls", phase)
        for phase in ("due", "not_due")
    )
    replay_parity = all(
        count_per_call(parent_rows, "replay_samples", phase)
        == count_per_call(candidate_rows, "replay_samples", phase) == 1.0
        for phase in ("due", "not_due")
    )
    recovery_ratio = float(np.median([
        100.0 * float(row["snapshot_seconds"]) / max(float(row["training_seconds"]), 1e-12)
        for row in candidate_rows
    ]))
    memory_overhead = (overhead("gpu_peak_memory_mb")
                       if args.compare_device == "cuda" else None)
    report = {
        "device": args.compare_device, "parent": args.compare_parent,
        "candidate": args.compare_candidate,
        "update_calls_per_repetition": args.compare_updates,
        "end_to_end_episodes": args.compare_episodes,
        "repetitions": args.compare_repetitions,
        "synthetic_active_peak_step": args.compare_active_step or None,
        "active_peak_measurement_note": (
            "Only counters advanced to trigger the mid-ramp Actor penalty; the peak Critic "
            "was not trained for those steps. Throughput check only, not a learning benchmark."
            if args.compare_active_step else None),
        "comparison_method": "median_of_paired_percent_overheads",
        "median_update_overhead_pct": overhead("median_update_ms"),
        "median_update_overhead_note": "Legacy pooled median: informational only; NOT a pass/fail criterion.",
        "weighted_update_overhead_pct": float(np.median(weighted_overheads)),
        "paired_weighted_update_overheads_pct": weighted_overheads,
        "actor_due_update_overhead_pct": float(np.median(paired_overheads("due_update_mean_ms"))),
        "actor_not_due_update_overhead_pct": float(np.median(paired_overheads("not_due_update_mean_ms"))),
        "end_to_end_training_overhead_pct": float(np.median(paired_training_overheads)),
        "paired_end_to_end_training_overheads_pct": paired_training_overheads,
        "recovery_snapshot_overhead_pct": recovery_ratio,
        "deterministic_evaluation_overhead_pct": float(np.median(paired_eval_overheads)),
        "paired_deterministic_evaluation_overheads_pct": paired_eval_overheads,
        "peak_critic_parameter_fraction": median(candidate_rows, "peak_critic_parameter_fraction"),
        "projector_call_parity": projector_parity,
        "replay_sample_parity": replay_parity,
        "gpu_peak_memory_overhead_pct": memory_overhead,
        "update_overhead_target_pass": float(np.median(weighted_overheads)) <= 15.0,
        "update_overhead_hard_pass": float(np.median(weighted_overheads)) <= 20.0,
        "parent_runs": parent_rows, "candidate_runs": candidate_rows,
    }
    output = args.output or (
        PROJECT_ROOT / "outputs" / "runtime_benchmarks" /
        f"peak_risk_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved comparative benchmark: {output}")
    return 0 if report["update_overhead_hard_pass"] and projector_parity and replay_parity else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1 or args.warmup_updates < 0 or args.sample_iterations < 1:
        raise ValueError(
            "episodes/sample-iterations must be positive and warmup-updates non-negative"
        )
    if args.compare_worker:
        try:
            result = _comparison_worker(args)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0
    if args.compare_parent or args.compare_candidate:
        return _comparison_driver(args)
    if args.worker:
        try:
            result = _worker(args)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0
    return _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
