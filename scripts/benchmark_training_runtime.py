from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from utils.runtime_benchmark import (
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
    parser.add_argument("--algorithm", default="matd3_transformer",
                        choices=("matd3_transformer", "pnn", "pnn_transformer"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "cooperative.yaml")
    parser.add_argument("--algorithm-config", type=Path)
    parser.add_argument("--episodes", type=int, default=5,
                        help="Timed steady-state episodes per candidate after replay/model warmup")
    parser.add_argument("--warmup-updates", type=int, default=8)
    parser.add_argument("--sample-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip the optional CUDA torch.compile candidate")

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


def _prefill_replay(env, algorithm, batch_size: int, seed: int) -> None:
    rng = np.random.default_rng(seed + 900_000)
    episode = 0
    while len(algorithm.replay) < batch_size:
        observations, info = env.reset(seed=seed + 800_000 + episode)
        state = info["global_state"]
        done = False
        while not done and len(algorithm.replay) < batch_size:
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

    _prefill_replay(env, algorithm, batch_size, args.seed)
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

    total_started = time.perf_counter()
    for episode in range(int(args.episodes)):
        observations, reset_info = env.reset(seed=args.seed + 1_000_000 + episode)
        state = reset_info["global_state"]
        done = False
        while not done:
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
        "total_seconds": total_seconds,
        "seconds_per_episode": seconds_per_episode,
        "rollout_ms_per_step": 1000.0 * rollout_seconds / max(1, step_count),
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


def _candidate_label(device: str, cpu_threads: int, interop_threads: int, compile_actors: bool) -> str:
    suffix = "-compile" if compile_actors else ""
    return f"{device}-cpu{cpu_threads}-interop{interop_threads}{suffix}"


def _run_candidate(args: argparse.Namespace, *, device: str, cpu_threads: int,
                   interop_threads: int, compile_actors: bool = False) -> dict[str, object]:
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

    print(f"Runtime benchmark: {args.algorithm} | logical CPUs={logical_cpus} | CUDA={cuda_available}")
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
            row for row in results
            if row.get("status") == "ok" and row.get("device") == "cuda"
            and not row.get("compile_actors")
        ]
        best_cuda = select_best_result(successful_cuda)
        if best_cuda is not None:
            print("Stage 3: optional CUDA torch.compile Actor candidate")
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
        PROJECT_ROOT / "outputs" / "runtime_benchmarks" /
        f"{args.algorithm}_{timestamp}.json"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

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
        "recommended_compile_actors": bool(best.get("compile_actors", False)),
        "estimated_500_seconds": best["estimated_500_seconds"],
        "estimated_5000_seconds": best["estimated_5000_seconds"],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nBEST")
    _print_row(dict(best))
    print(f"Recommended device: {best['device']}")
    print(
        "Recommended runtime: "
        f"cpu_threads={best['cpu_threads']}, interop_threads={best['interop_threads']}"
    )
    print(f"Actor compile: {bool(best.get('compile_actors', False))}")
    print(f"Estimated 500 episodes: {float(best['estimated_500_seconds']) / 60.0:.2f} min")
    print(f"Estimated 5000 episodes: {float(best['estimated_5000_seconds']) / 3600.0:.2f} h")
    print(f"JSON report: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1 or args.warmup_updates < 0 or args.sample_iterations < 1:
        raise ValueError("episodes/sample-iterations must be positive and warmup-updates non-negative")
    if args.worker:
        try:
            result = _worker(args)
        except Exception as exc:  # worker must return a machine-readable failure to the driver
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0
    return _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
