from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from algorithms.registry import get_algorithm_spec
from algorithms.base import TrainingContextConsumerProtocol, AlgorithmDiagnosticsProtocol
from envs.multi_microgrid_env import MultiMicrogridEnv
from core.reward_v2 import resolve_reward_v2_config
from evaluation.evaluate_matd3 import evaluate, evaluate_policy_on_days
from evaluation.plotting import generate_figures
from experiments.checkpoint_selector import CheckpointCandidate, CheckpointSelector
from experiments.layout import artifact_directory, ExperimentLayout
from experiments.profiles import resolve_training_args, experiment_contract
from experiments.training_recovery import (
    config_text_hash,
    ensure_recoverable,
    validate_resume_directory,
    write_run_status,
    write_training_error,
)
from utils.metrics_collector import EpisodeMetricsCollector, write_json
from utils.training_history import TrainingHistoryJournal
from utils.torch_runtime import configure_torch_runtime
try:
    from utils.training_dashboard import AlgorithmTrainingDashboard
except ModuleNotFoundError as exc:
    if exc.name != "rich":
        raise

    class AlgorithmTrainingDashboard:  # type: ignore[no-redef]
        """Text-only fallback when the optional Rich dashboard is unavailable."""

        enabled = False

        def __init__(self, **_: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def update(self, _: Mapping[str, Any]) -> None:
            pass

        def set_error(self, reason: str) -> None:
            print(f"ERROR: {reason}")

        def notify_checkpoint(self, episode: int, profit: float, path: str = "") -> None:
            print(f"Validation checkpoint saved at episode {episode}: {profit:.4f} ({path})")

        def render_final_summary(self, **_: Any) -> None:
            pass


def _capture_training_context(algorithm, env, observations, state, metadata):
    if isinstance(algorithm, TrainingContextConsumerProtocol):
        algorithm.capture_training_context(env, observations, state, metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the shared train-test-settlement-plot pipeline")
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Registered algorithm plugin slug (for example, matd3 or matd3_hierarchical_transformer)",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--experiment", help="Named bundle from configs/experiment_profiles.yaml")
    parser.add_argument("--policy", choices=("mlp", "transformer"), help="Network under the selected/default action design")
    parser.add_argument("--action-design", help="Select an action design, paired with --policy or the default policy")
    parser.add_argument("--algorithm-config", type=Path,
                        help="Override the algorithm plugin's default YAML config")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"),
                        help="All algorithms are isolated below OUTPUT_ROOT/<algorithm>/runs/.")
    parser.add_argument("--output", type=Path,
                        help="Deprecated compatibility override for one algorithm's output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-episodes", type=int)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--validation-days", type=int)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--no-milp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "Reward V2 calibration/diagnostic run: max 100 complete-day "
            "episodes and no formal test/settlement/plot stage"
        ),
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Resume a supported interrupted run in its existing directory",
    )
    return parser


def _validate_reward_run_mode(
    env_config: Mapping[str, Any],
    *,
    training_episodes: int,
    diagnostic_only: bool,
) -> dict[str, Any]:
    """Fail closed when Reward V2 coefficients have not been locked."""
    reward_v2 = resolve_reward_v2_config(env_config)
    if reward_v2 is None:
        if diagnostic_only:
            raise ValueError("--diagnostic-only requires Reward V2")
        return {
            "reward_v2": None,
            "run_purpose": "formal_training",
            "formal_test_allowed": True,
        }

    status = str(reward_v2["calibration_status"])
    if status == "diagnostic" and not diagnostic_only:
        raise ValueError(
            "Reward V2 calibration_status=diagnostic cannot start a formal run; "
            "use --diagnostic-only for calibration work"
        )
    if diagnostic_only and int(training_episodes) > 100:
        raise ValueError(
            "Reward V2 diagnostic runs are limited to at most 100 complete-day episodes"
        )
    return {
        "reward_v2": reward_v2,
        "run_purpose": (
            "reward_calibration" if diagnostic_only else "formal_training"
        ),
        "formal_test_allowed": not diagnostic_only,
    }


def _validate_checkpoint_selection_contract(
    *, validation_enabled: bool, formal_test_allowed: bool
) -> None:
    """Formal test runs must select their checkpoint from validation results."""
    if formal_test_allowed and not validation_enabled:
        raise ValueError(
            "Formal training requires validation to select best_model.pt; "
            "--no-validation is only allowed for diagnostic-only runs"
        )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _mean(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _moving(history: Sequence[Mapping[str, Any]], field: str, window: int) -> float | None:
    return _mean([item.get(field) for item in history[-window:]])


def _diagnostic_means(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_names = (
        "critic_loss", "critic1_loss", "critic2_loss", "actor_loss",
        "q1_mean", "q2_mean", "q1_q2_gap", "target_q_mean",
        "td_error_mean", "td_error_std", "actor_grad_norm", "critic_grad_norm",
        "actor_parameter_count",
        "crossgate_local_gate_mean", "crossgate_local_gate_std",
        "crossgate_local_gate_low_rate", "crossgate_local_gate_high_rate",
        "crossgate_peer_gate_mean", "crossgate_peer_gate_std",
        "crossgate_peer_gate_low_rate", "crossgate_peer_gate_high_rate",
    )
    result = {name: _mean([item.get(name) for item in items]) for name in scalar_names}
    for source, target in (("critic_losses", "critic_losses"), ("actor_losses", "actor_losses")):
        ids = sorted({mg for item in items for mg in item.get(source, {})})
        result[target] = {
            mg: _mean([item.get(source, {}).get(mg) for item in items]) for mg in ids
        }
    q_items = [item["q_stats"] for item in items if "q_stats" in item]
    result["q_statistics"] = q_items[-1] if q_items else {}
    gate_items = [item["crossgate_gate_diagnostics"] for item in items
                  if item.get("crossgate_gate_diagnostics")]
    gate_agents = sorted({mg for item in gate_items for mg in item})
    result["crossgate_gate_diagnostics"] = {
        mg: {
            field: _mean([
                item.get(mg, {}).get(field) for item in gate_items
            ])
            for field in (
                "local_gate_mean", "local_gate_std",
                "local_gate_low_rate", "local_gate_high_rate",
                "peer_gate_mean", "peer_gate_std",
                "peer_gate_low_rate", "peer_gate_high_rate",
            )
        }
        for mg in gate_agents
    }
    count_items = [item["actor_parameter_counts"] for item in items
                   if item.get("actor_parameter_counts")]
    count_agents = sorted({mg for item in count_items for mg in item})
    result["actor_parameter_counts"] = {
        mg: int(count_items[-1][mg]) for mg in count_agents
    } if count_items else {}
    # Preserve algorithm-supplied nested diagnostics without looking inside its
    # networks. Missing delayed Actor updates must not become zero gradients.
    def nested_mean(values):
        values = [value for value in values if value is not None]
        if not values:
            return None
        if isinstance(values[0], Mapping):
            keys = sorted({key for value in values for key in value})
            return {key: nested_mean([value.get(key) for value in values]) for key in keys}
        return _mean(values)

    for name in ("actor_grad_norms", "critic_grad_norms", "actor_branch_grad_norms",
                 "actor_layer_grad_norms", "actor_diagnostics"):
        result[name] = nested_mean([item.get(name) for item in items]) or {}
    # New algorithms may report nested metrics without exposing network internals.
    # Missing delayed updates remain None rather than an invented zero.
    for name in ("peak", "runtime"):
        if any(name in item for item in items):
            result[name] = nested_mean([item.get(name) for item in items])
    details = [item["actor_diagnostics"] for item in items if item.get("actor_diagnostics")]
    result["actor_diagnostic_statistics"] = {}
    for mg in sorted({mg for item in details for mg in item}):
        statistics = {}
        for field in sorted({field for item in details for field in item.get(mg, {})}):
            values = [item[mg][field] for item in details if field in item.get(mg, {})]
            finite = np.asarray([float(value) for value in values if np.isfinite(float(value))])
            statistics[field] = {
                "count": len(values), "nonfinite_count": len(values) - len(finite),
                "mean": float(finite.mean()) if len(finite) else None,
                "p95": float(np.quantile(finite, .95)) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None,
            }
        result["actor_diagnostic_statistics"][mg] = statistics
    return result


def _append_actor_diagnostics(path: Path, episode: int, record: Mapping[str, Any]) -> None:
    """Human-readable per-agent CSV, flushed once per episode for live inspection."""
    rows = []
    for mg, fields in record.get("actor_diagnostic_statistics", {}).items():
        row = {"episode": episode, "microgrid": mg}
        for field, stats in fields.items():
            row.update({f"{field}_{stat}": value for stat, value in stats.items()})
        rows.append(row)
    if rows:
        exists = path.exists()
        if exists:
            with path.open("r", encoding="utf-8") as handle:
                fieldnames = next(csv.reader(handle), [])
        else:
            fieldnames = list(dict.fromkeys(k for row in rows for k in row))

        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)


def _validation_record(
    summary: Mapping[str, Any],
    episode: int,
    candidate: CheckpointCandidate | None = None,
    is_best: bool = False,
) -> dict[str, Any]:
    stats = summary["team_profit_statistics"]
    rec: dict[str, Any] = {
        "episode": int(episode),
        "team_profit_mean": stats["mean"],
        "team_profit_std": stats["std"],
        "team_profit_median": stats["median"],
        "profit_by_microgrid_mean": summary["profit_by_microgrid_mean"],
        "renewable_utilization_rate": summary["renewable_utilization_rate"],
        "p2p_total_energy_kwh": summary["p2p_total_energy_kwh"],
        "feasible_rate": summary["feasible_rate"],
        "is_best": bool(is_best),
    }
    if "team_reward_mean" in summary:
        rec["team_reward_mean"] = summary["team_reward_mean"]
    if "generalized_system_cost_mean" in summary:
        rec["generalized_system_cost_mean"] = summary["generalized_system_cost_mean"]
    if candidate is not None and candidate.selection_mode != "profit_only":
        rec.update({
            "selection_mode": candidate.selection_mode,
            "constraint_eligible": candidate.eligible,
            "constraint_violation_total": candidate.total_violation,
            "constraint_values": candidate.constraint_values,
            "constraint_failures": candidate.constraint_failures,
        })
    return rec


def train(args: argparse.Namespace) -> dict[str, object]:
    args = resolve_training_args(args)
    spec = get_algorithm_spec(args.algorithm)
    config_path = Path(args.config).resolve()
    algorithm_path = Path(args.algorithm_config or spec.default_config_path).resolve()
    base_env_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_env_config["_project_root"] = str(config_path.parent.parent)
    env_config = spec.build_environment_config(base_env_config)
    algorithm_cfg = yaml.safe_load(algorithm_path.read_text(encoding="utf-8"))
    declared_name = str(algorithm_cfg.get("algorithm", {}).get("name", spec.slug)).lower()
    if declared_name != spec.slug:
        raise ValueError(
            f"Algorithm config declares {declared_name!r}, but selected plugin is {spec.slug!r}"
        )
    contract = experiment_contract(spec, env_config, config_path,
                                   selected_experiment=args.selected_experiment)
    print(f"Experiment: {contract['experiment_profile'] or 'explicit algorithm'} | "
          f"Policy: {contract['policy']} | Action: {contract['action_design']['id']} "
          f"({contract['action_design']['control_mode']}) | "
          f"Environment: {contract['environment_profile']} [{contract['physical_config_hash'][:12]}]")
    training_cfg = algorithm_cfg.get("training", {})
    validation_cfg = algorithm_cfg.get("validation", {})
    logging_cfg = algorithm_cfg.get("logging", {})
    runtime_settings = configure_torch_runtime(algorithm_cfg, args.device)
    incremental_history = bool(logging_cfg.get("incremental_history", False))
    snapshot_interval = logging_cfg.get("history_snapshot_interval", 500)
    if incremental_history and (isinstance(snapshot_interval, bool)
                                or not isinstance(snapshot_interval, int)
                                or snapshot_interval < 1):
        raise ValueError("logging.history_snapshot_interval must be a positive integer")

    training_episodes = int(args.episodes if args.episodes is not None
                            else training_cfg.get("episodes", 5000))
    test_episodes = int(args.test_episodes if args.test_episodes is not None
                        else training_cfg.get("test_episodes", 55))
    validation_interval = int(
        args.validation_interval if args.validation_interval is not None
        else validation_cfg.get("interval", 100)
    )
    validation_enabled = bool(validation_cfg.get("enabled", True)) and not args.no_validation
    validation_days_limit = (
        int(args.validation_days) if args.validation_days is not None
        else validation_cfg.get("max_days")
    )
    if training_episodes <= 0 or test_episodes <= 0:
        raise ValueError("Training and test episode counts must be positive")
    reward_run_mode = _validate_reward_run_mode(
        env_config,
        training_episodes=training_episodes,
        diagnostic_only=bool(getattr(args, "diagnostic_only", False)),
    )
    _validate_checkpoint_selection_contract(
        validation_enabled=validation_enabled,
        formal_test_allowed=bool(reward_run_mode["formal_test_allowed"]),
    )
    if validation_enabled and validation_interval <= 0:
        raise ValueError("validation.interval must be positive")
    output_override = getattr(args, "output", None)
    resume_requested = getattr(args, "resume_run", None) is not None
    # Check support before creating directories or changing saved run evidence.
    env = MultiMicrogridEnv(env_config, split="train")
    algorithm = spec.create(env, algorithm_cfg, device=args.device, seed=args.seed)
    recoverable = None
    if hasattr(algorithm, "save_training_state") and hasattr(algorithm, "load_training_state"):
        recoverable = ensure_recoverable(algorithm)
    if resume_requested and recoverable is None:
        raise ValueError(f"Algorithm {spec.slug} does not support resumable training")
    if output_override is not None:
        expected_algorithm_root = Path(output_override).resolve()
    else:
        expected_algorithm_root = Path(getattr(args, "output_root", "outputs")).resolve() / spec.slug
    if resume_requested:
        resume_dir = validate_resume_directory(
            Path(args.resume_run), expected_algorithm_root / "runs"
        )
        resume_artifacts = artifact_directory(resume_dir)
        checkpoints = resume_artifacts / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        figures = resume_dir / "figures"
        figures.mkdir(exist_ok=True)
        layout = ExperimentLayout(
            algorithm_root=expected_algorithm_root,
            runs_root=expected_algorithm_root / "runs",
            run_id=resume_dir.name,
            run_dir=resume_dir,
            checkpoints_dir=checkpoints.resolve(),
            figures_dir=figures.resolve(),
            hourly_dir=(resume_artifacts / "test_hourly_trajectories").resolve(),
        )
    else:
        layout = ExperimentLayout.create(
            getattr(args, "output_root", "outputs"),
            spec.slug,
            spec.display_name,
            args.seed,
            algorithm_root_override=output_override,
        )
    algorithm_root = layout.algorithm_root
    run_id, run_dir = layout.run_id, layout.artifacts_dir
    checkpoints_dir = layout.checkpoints_dir
    best_checkpoint = checkpoints_dir / "best_model.pt"
    final_checkpoint = checkpoints_dir / "final_model.pt"

    snapshot = {"environment": env_config, "algorithm": algorithm_cfg, "experiment_contract": contract}
    snapshot_text = yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False)
    config_hash = config_text_hash(snapshot_text)
    start_time = datetime.now().astimezone()
    new_metadata: dict[str, Any] = {
        "run_id": run_id,
        "experiment_contract": contract,
        "algorithm": spec.display_name,
        "algorithm_slug": spec.slug,
        "algorithm_display_name": spec.display_name,
        "seed": int(args.seed),
        "start_time": start_time.isoformat(), "end_time": None,
        "config_path": str(config_path), "algorithm_config_path": str(algorithm_path),
        "environment_overrides": dict(spec.environment_overrides),
        "ess_sharing_control_mode": env_config.get("ess_sharing", {}).get("control_mode"),
        "ess_sharing_observation_mode": env_config.get("ess_sharing", {}).get("observation_mode"),
        "ess_p2p_enabled": env_config.get("ess_sharing", {}).get("ess_p2p_enabled", True),
        "ess_offer_projection": env_config.get("ess_sharing", {}).get("offer_projection", "none"),
        "mpc_forecast_horizon": algorithm_cfg.get("mpc", {}).get("forecast_horizon"),
        "forecast_horizon_hours": env_config.get("observation", {}).get("future_horizon"),
        "evaluation_forecast_noise": env_config.get("observation", {}).get(
            "evaluation_forecast_noise", False,
        ),
        "config_hash": config_hash, "device": str(args.device),
        "python_version": platform.python_version(), "train_episodes": training_episodes,
        "test_episodes": test_episodes, "git_commit": _git_commit(),
        "runtime": runtime_settings,
        "run_purpose": reward_run_mode["run_purpose"],
        "formal_test_completed": False,
        "reward_semantics_version": (
            reward_run_mode["reward_v2"]["semantics_version"]
            if reward_run_mode["reward_v2"] is not None else "legacy"
        ),
        "reward_calibration_status": (
            reward_run_mode["reward_v2"]["calibration_status"]
            if reward_run_mode["reward_v2"] is not None else None
        ),
    }
    if resume_requested:
        snapshot_path = run_dir / "config_snapshot.yaml"
        metadata_path = run_dir / "run_metadata.json"
        if not snapshot_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Resume run is missing config_snapshot.yaml or run_metadata.json")
        saved_hash = config_text_hash(snapshot_path.read_text(encoding="utf-8"))
        if saved_hash != config_hash:
            raise ValueError(
                f"Resume config hash mismatch: saved {saved_hash}, requested {config_hash}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("algorithm_slug") != spec.slug:
            raise ValueError("Resume algorithm_slug does not match selected algorithm")
        if int(metadata.get("seed", -1)) != int(args.seed):
            raise ValueError("Resume seed does not match the saved run")
        if int(metadata.get("train_episodes", -1)) != training_episodes:
            raise ValueError("Resume episode count does not match the saved run")
        metadata["resume_count"] = int(metadata.get("resume_count", 0)) + 1
        metadata["end_time"] = None
    else:
        (run_dir / "config_snapshot.yaml").write_text(snapshot_text, encoding="utf-8")
        metadata = new_metadata
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(algorithm_root / "latest_run.json", {
        "algorithm_slug": spec.slug,
        "algorithm_display_name": spec.display_name,
        "run_id": run_id,
        "run_directory": str(layout.run_dir),
    })
    write_run_status(run_dir, "RUNNING", completed_episode=0, resumed=resume_requested)

    if isinstance(algorithm, AlgorithmDiagnosticsProtocol):
        algorithm.configure_diagnostics(run_dir / 'auxiliary_plan',
            {'algorithm_slug': spec.slug, 'training_seed': args.seed})
    settings = algorithm_cfg["algorithm"]
    warmup = int(settings.get("warmup_steps", 1000))
    batch_size = int(settings.get("batch_size", 128))
    exploration = float(settings.get("exploration_noise", 0.15))
    replay_cap = int(settings.get("replay_capacity", 200_000))
    rng = np.random.default_rng(args.seed)
    ids = env.agent_ids
    recovery_cfg = dict(settings.get("training_recovery", {}))
    recovery_enabled = recoverable is not None and bool(recovery_cfg.get("enabled", True))
    recovery_interval = int(
        recovery_cfg.get("snapshot_interval_episodes", validation_interval)
    )
    if recovery_enabled and recovery_interval <= 0:
        raise ValueError("training_recovery.snapshot_interval_episodes must be positive")
    recovery_dir = checkpoints_dir / "training_state"
    durable_recovery_point: str | None = None

    validation_env = MultiMicrogridEnv(env_config, split="val") if validation_enabled else None
    validation_days = (
        [str(day.date()) for day in validation_env.data.valid_days] if validation_env else []
    )
    if validation_days_limit is not None:
        validation_days = validation_days[:max(1, int(validation_days_limit))]

    progress: dict[str, Any] = {}
    if resume_requested:
        progress = recoverable.load_training_state(recovery_dir)  # type: ignore[union-attr]
        durable_recovery_point = str(recovery_dir / "manifest.json")

    start_episode = int(progress.get("completed_episode", 0))
    history_path = run_dir / "training_history.json"
    history: list[dict[str, Any]] = (
        list(json.loads(history_path.read_text(encoding="utf-8")))
        if resume_requested and history_path.is_file()
        else []
    )
    if len(history) < start_episode:
        raise ValueError(
            f"Training history has {len(history)} rows, shorter than recovery cursor {start_episode}"
        )
    history = history[:start_episode]
    if resume_requested:
        write_json(history_path, history)
    journal_path = run_dir / "training_history.jsonl"
    history_journal = None
    if incremental_history:
        history_journal = TrainingHistoryJournal(
            journal_path,
            resume=resume_requested,
            completed_records=start_episode if resume_requested else None,
        )
    validation_path = run_dir / "validation_history.json"
    validation_history: list[dict[str, Any]] = (
        list(json.loads(validation_path.read_text(encoding="utf-8")))
        if resume_requested and validation_path.is_file()
        else []
    )
    validation_cursor = int(progress.get("validation_history_length", len(validation_history)))
    if len(validation_history) < validation_cursor:
        raise ValueError("Validation history is shorter than its recovery cursor")
    validation_history = validation_history[:validation_cursor]
    if resume_requested:
        write_json(validation_path, validation_history)

    total_steps = int(progress.get("total_steps", 0))
    if "pipeline_rng_state" in progress:
        rng.bit_generator.state = progress["pipeline_rng_state"]
    selector = CheckpointSelector(validation_cfg.get("selection"))
    best_candidate_raw = progress.get("best_candidate")
    best_candidate = (
        CheckpointCandidate(**best_candidate_raw) if best_candidate_raw is not None else None
    )
    best_validation_score_raw = progress.get("best_validation_score")
    best_validation_score = (
        float(best_validation_score_raw)
        if best_validation_score_raw is not None
        else -float("inf")
    )
    best_episode_raw = progress.get("best_episode")
    best_episode = int(best_episode_raw) if best_episode_raw is not None else None
    prior_training_seconds = float(progress.get("training_seconds", 0.0))
    train_started = time.perf_counter()

    dashboard = AlgorithmTrainingDashboard(
        total_episodes=training_episodes, warmup_steps=warmup,
        replay_capacity=replay_cap,
        rolling_windows=logging_cfg.get("rolling_windows", (50, 100, 500)),
        refresh_hz=float(logging_cfg.get("dashboard_refresh_hz", 4.0)),
        enabled=(not args.no_dashboard) and bool(logging_cfg.get("dashboard", True)),
        agent_ids=ids,
        initial_soc={mg: env.cores[mg].battery.initial_soc for mg in ids},
        soc_min={mg: env.cores[mg].battery.soc_min for mg in ids},
        soc_tolerance=float(env.config["terminal_soc"]["tolerance"]),
        algorithm_name=spec.display_name,
    )
    dashboard.start()
    try:
        for episode_index in range(start_episode, training_episodes):
            episode_started = time.perf_counter()
            observations, reset_info = env.reset(seed=args.seed + episode_index)
            state = reset_info["global_state"]
            collector = EpisodeMetricsCollector(env, capture_hourly=False)
            diagnostics: list[dict[str, Any]] = []
            done = False
            while not done:
                _capture_training_context(algorithm, env, observations, state,
                    {'episode_index': episode_index, 'total_steps': total_steps,
                     'day': str(reset_info['day'])})
                if total_steps < warmup:
                    actions = {
                        mg: rng.uniform(-1.0, 1.0, env.action_dim).astype(np.float32) for mg in ids
                    }
                else:
                    actions = algorithm.select_actions(observations, exploration)
                for mg in ids:
                    if not np.isfinite(actions[mg]).all():
                        dashboard.set_error(f"NaN / Inf action generated for {mg}")
                        raise RuntimeError(f"NaN / Inf action generated for {mg}")

                next_observations, rewards, done, info = env.step(actions)
                team_reward = float(rewards[ids[0]])
                if not math.isfinite(team_reward):
                    dashboard.set_error("NaN / Inf team reward")
                    raise RuntimeError("NaN / Inf team reward")
                algorithm.replay.add(
                    observations, state, actions, team_reward,
                    next_observations, info["global_state_after"], done,
                )
                update_metrics = algorithm.update(batch_size)
                if update_metrics:
                    diagnostics.append(update_metrics)
                collector.record_step(actions, info, rewards)
                observations, state = next_observations, info["global_state_after"]
                total_steps += 1

            ep_record = collector.finalize_episode()
            ep_record.update({
                "episode": episode_index + 1, "total_steps": total_steps, "steps": total_steps,
                "replay_size": len(algorithm.replay),
                "exploration_noise": exploration if total_steps >= warmup else 1.0,
                "episode_seconds": time.perf_counter() - episode_started,
            })
            ep_record.update(_diagnostic_means(diagnostics))
            _append_actor_diagnostics(run_dir / "actor_diagnostics.csv", episode_index + 1, ep_record)
            economics = ep_record["economic_breakdown_by_microgrid"]
            if ep_record.get("reward_v2_cost_breakdown") is not None:
                ep_record["reward_decomposition"] = dict(
                    ep_record["reward_v2_cost_breakdown"]
                )
            else:
                ep_record["reward_decomposition"] = {
                    "grid_import_cost": sum(x["grid_import_cost"] for x in economics.values()),
                    "grid_export_revenue": sum(x["grid_export_revenue"] for x in economics.values()),
                    "degradation_cost": sum(x["battery_degradation_cost"] for x in economics.values()),
                    "unmet_load_cost": sum(x["unmet_load_cost"] for x in economics.values()),
                    "curtailment_cost": sum(x["renewable_curtailment_cost"] for x in economics.values()),
                    "terminal_soc_penalty": sum(x["terminal_soc_penalty"] for x in economics.values()),
                }
            history.append(ep_record)
            rolling = {
                "team_profit_ma_50": _moving(history, "team_profit", 50),
                "team_profit_ma_100": _moving(history, "team_profit", 100),
                "team_profit_ma_500": _moving(history, "team_profit", 500),
                "team_reward_ma_100": _moving(history, "team_reward", 100),
                "actor_loss_ma_100": _moving(history, "actor_loss", 100),
                "critic_loss_ma_100": _moving(history, "critic_loss", 100),
                "p2p_energy_ma_100": _moving(history, "p2p_total_energy_kwh", 100),
                "renewable_utilization_ma_100": _moving(history, "renewable_utilization_rate", 100),
            }
            ep_record.update(rolling)

            should_validate = validation_enabled and (
                (episode_index + 1) % validation_interval == 0
                or episode_index + 1 == training_episodes
            )
            if should_validate and validation_env is not None:
                validation_noise = (
                    0.0
                    if bool(validation_cfg.get("deterministic_policy", True))
                    else exploration
                )
                policy = lambda obs: algorithm.select_actions(
                    obs, noise_std=validation_noise
                )
                val_summary, _ = evaluate_policy_on_days(
                    validation_env, validation_days, policy, seed=args.seed + 100_000,
                    deterministic_environment=bool(validation_cfg.get("fixed_scenarios", True)),
                    forecast_noise=bool(
                        env_config.get("observation", {}).get(
                            "evaluation_forecast_noise", False,
                        )
                    ),
                )
                candidate = selector.evaluate(val_summary, episode=episode_index + 1)
                is_best = selector.is_better(candidate, best_candidate)
                candidate.is_best = is_best
                if is_best:
                    best_candidate = candidate
                    best_validation_score = candidate.objective_value
                    best_episode = episode_index + 1
                    algorithm.save(best_checkpoint)
                    dashboard.notify_checkpoint(best_episode, candidate.objective_value, str(best_checkpoint))
                validation_history.append(_validation_record(
                    val_summary, episode_index + 1, candidate=candidate, is_best=is_best,
                ))
                write_json(run_dir / "validation_history.json", validation_history)
                ep_record["best_validation_objective"] = best_validation_score
                if selector.objective_key == "team_profit_mean":
                    ep_record["best_validation_profit"] = best_validation_score

            if history_journal is not None:
                history_journal.append(ep_record)
                save_history = (episode_index + 1) % snapshot_interval == 0
            else:
                save_history = should_validate or (episode_index + 1) % 10 == 0
            if save_history or episode_index + 1 == training_episodes:
                write_json(run_dir / "training_history.json", history)
            recovery_due = recovery_enabled and (
                (episode_index + 1) % recovery_interval == 0
                or episode_index + 1 == training_episodes
            )
            if recovery_due and recoverable is not None:
                # History and validation artifacts are committed before the
                # recovery cursor that refers to them.
                write_json(run_dir / "training_history.json", history)
                write_json(run_dir / "validation_history.json", validation_history)
                recovery_progress = {
                    "completed_episode": episode_index + 1,
                    "total_steps": total_steps,
                    "validation_history_length": len(validation_history),
                    "best_candidate": asdict(best_candidate) if best_candidate is not None else None,
                    "best_validation_score": (
                        best_validation_score if math.isfinite(best_validation_score) else None
                    ),
                    "best_episode": best_episode,
                    "pipeline_rng_state": rng.bit_generator.state,
                    "training_seconds": prior_training_seconds + time.perf_counter() - train_started,
                }
                recoverable.save_training_state(recovery_dir, recovery_progress)
                durable_recovery_point = str(recovery_dir / "manifest.json")
                write_run_status(
                    run_dir,
                    "RUNNING",
                    completed_episode=episode_index + 1,
                    total_steps=total_steps,
                    durable_recovery_point=durable_recovery_point,
                    resumed=resume_requested,
                )
            dashboard.update(ep_record)
            if not dashboard.enabled and ((episode_index + 1) % 25 == 0 or episode_index == 0
                                          or should_validate):
                print(
                    f"Episode {episode_index + 1}/{training_episodes} | "
                    f"Profit {ep_record['team_profit']:.2f} | "
                    f"MA100 {ep_record['team_profit_ma_100']:.2f} | "
                    f"Best validation {best_validation_score:.2f}"
                )
    except BaseException as exc:
        if history:
            write_json(run_dir / "training_history.json", history)
        write_training_error(
            run_dir,
            exc,
            completed_episode=len(history),
            update_step=int(getattr(algorithm, "update_steps", 0)),
            durable_recovery_point=durable_recovery_point,
        )
        write_run_status(
            run_dir,
            "FAILED",
            completed_episode=len(history),
            total_steps=total_steps,
            durable_recovery_point=durable_recovery_point,
        )
        raise
    finally:
        dashboard.stop()

    write_run_status(
        run_dir,
        "TRAINED",
        completed_episode=training_episodes,
        total_steps=total_steps,
        durable_recovery_point=durable_recovery_point,
    )

    if isinstance(algorithm, AlgorithmDiagnosticsProtocol):
        algorithm.set_diagnostic_phase(None, {'training_complete': True})
    algorithm.save(final_checkpoint)
    if validation_enabled and selector.requires_eligible_checkpoint:
        if best_candidate is None or not best_candidate.eligible:
            error = RuntimeError(
                "No physically eligible validation checkpoint was found; "
                "refusing to use any existing/final checkpoint as best_model.pt."
            )
            write_training_error(
                run_dir,
                error,
                completed_episode=training_episodes,
                update_step=int(getattr(algorithm, "update_steps", 0)),
                durable_recovery_point=durable_recovery_point,
            )
            write_run_status(
                run_dir,
                "FAILED",
                completed_episode=training_episodes,
                total_steps=total_steps,
                durable_recovery_point=durable_recovery_point,
            )
            raise error
        if not best_checkpoint.exists():
            error = FileNotFoundError(
                "Eligible validation metadata exists but checkpoints/best_model.pt is missing"
            )
            write_training_error(
                run_dir,
                error,
                completed_episode=training_episodes,
                update_step=int(getattr(algorithm, "update_steps", 0)),
                durable_recovery_point=durable_recovery_point,
            )
            write_run_status(
                run_dir,
                "FAILED",
                completed_episode=training_episodes,
                total_steps=total_steps,
                durable_recovery_point=durable_recovery_point,
            )
            raise error
    elif not best_checkpoint.exists():
        algorithm.save(best_checkpoint)
        best_episode = training_episodes
        fallback_key = (
            "team_reward" if selector.objective_key == "team_reward_mean"
            else "team_profit"
        )
        best_validation_score = float(history[-1][fallback_key])
    training_seconds = prior_training_seconds + time.perf_counter() - train_started
    checkpoint_selection = {
        "selection_mode": selector.mode,
        "selection_metric": (
            "validation_team_profit_mean"
            if selector.mode == "profit_only" and validation_enabled
            else (selector.objective_key if validation_enabled else "final_model_fallback_validation_disabled")
        ),
        "best_episode": best_episode,
        "best_validation_score": best_validation_score,
        "best_checkpoint": "checkpoints/best_model.pt",
        "final_checkpoint": "checkpoints/final_model.pt",
    }
    if selector.mode != "profit_only":
        checkpoint_selection.update({
            "constraint_eligible": best_candidate.eligible if best_candidate is not None else False,
            "constraint_violation_total": best_candidate.total_violation if best_candidate is not None else float("inf"),
            "constraint_failures": best_candidate.constraint_failures if best_candidate is not None else ["no_validation_run"],
        })
    profits = np.asarray([item["team_profit"] for item in history], dtype=np.float64)
    final_100 = profits[-100:]
    final_500 = profits[-500:]
    training_summary = {
        "experiment_contract": contract,
        "episodes": training_episodes, "total_steps": total_steps, "steps": total_steps,
        "best_episode": best_episode,
        "final_100_profit_mean": float(np.mean(final_100)),
        "final_100_profit_std": float(np.std(final_100)),
        "final_500_profit_mean": float(np.mean(final_500)),
        "final_500_profit_std": float(np.std(final_500)),
        "best_validation_objective": best_validation_score,
        "best_validation_profit_mean": (
            best_validation_score
            if selector.objective_key == "team_profit_mean" else None
        ),
        "checkpoint_selection": checkpoint_selection,
        "training_seconds": training_seconds,
        "seconds_per_episode_mean": training_seconds / training_episodes,
    }
    write_json(run_dir / "training_summary.json", training_summary)
    metadata["end_time"] = datetime.now().astimezone().isoformat()
    write_json(run_dir / "run_metadata.json", metadata)

    if not reward_run_mode["formal_test_allowed"]:
        diagnostics_dir = run_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            **training_summary,
            "algorithm_slug": spec.slug,
            "algorithm_display_name": spec.display_name,
            "run_id": run_id,
            "run_directory": str(layout.run_dir),
            "run_purpose": "reward_calibration",
            "formal_test_completed": False,
            "checkpoint": str(final_checkpoint),
            "best_checkpoint": str(best_checkpoint),
            "diagnostics_directory": str(diagnostics_dir),
        }
        metadata["run_purpose"] = "reward_calibration"
        metadata["formal_test_completed"] = False
        write_json(run_dir / "run_metadata.json", metadata)
        write_json(run_dir / "summary.json", summary)
        write_json(algorithm_root / "latest_run.json", {
            "algorithm_slug": spec.slug,
            "algorithm_display_name": spec.display_name,
            "run_id": run_id,
            "run_directory": str(layout.run_dir),
        })
        write_run_status(
            run_dir,
            "COMPLETED",
            completed_episode=training_episodes,
            total_steps=total_steps,
            durable_recovery_point=durable_recovery_point,
            run_purpose="reward_calibration",
            formal_test_completed=False,
        )
        return summary

    write_run_status(
        run_dir,
        "EVALUATING",
        completed_episode=training_episodes,
        total_steps=total_steps,
        durable_recovery_point=durable_recovery_point,
    )
    try:
        test_result = evaluate(
            config_path, algorithm_path, best_checkpoint, split="test",
            run_milp=not args.no_milp, output_dir=run_dir,
            checkpoint_metadata=checkpoint_selection, test_seed=args.seed,
            max_days=test_episodes, device=args.device, algorithm_name=spec.slug,
            experiment_identity=contract,
        )
    except BaseException as exc:
        write_training_error(
            run_dir,
            exc,
            completed_episode=training_episodes,
            update_step=int(getattr(algorithm, "update_steps", 0)),
            durable_recovery_point=durable_recovery_point,
        )
        write_run_status(
            run_dir,
            "FAILED",
            completed_episode=training_episodes,
            total_steps=total_steps,
            durable_recovery_point=durable_recovery_point,
        )
        raise
    metadata["formal_test_completed"] = True
    write_json(run_dir / "run_metadata.json", metadata)
    summary = {
        **training_summary,
        "algorithm_slug": spec.slug,
        "algorithm_display_name": spec.display_name,
        "run_id": run_id, "run_directory": str(layout.run_dir),
        "best_team_profit": float(np.max(profits)),
        "last_team_profit": float(profits[-1]),
        "checkpoint": str(final_checkpoint), "best_checkpoint": str(best_checkpoint),
        "test_days_count": test_result["test_days_count"],
        "comparison": test_result["comparison"],
        "algorithm_ess_terminal_soc": test_result["algorithm"]["ess"]["terminal_soc_mean"],
        "matd3_ess_terminal_soc": test_result["matd3"]["ess"]["terminal_soc_mean"],
        "test_evaluation": str(run_dir / "test_evaluation.json"),
        "run_purpose": "formal_training",
        "formal_test_completed": True,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(algorithm_root / "latest_run.json", {
        "algorithm_slug": spec.slug,
        "algorithm_display_name": spec.display_name,
        "run_id": run_id,
        "run_directory": str(layout.run_dir),
    })

    try:
        figure_paths = generate_figures(run_dir)
        summary["plotting_succeeded"] = True
        summary["figures_generated"] = [str(path.relative_to(layout.run_dir)) for path in figure_paths]
        summary["plotting_error"] = None
    except Exception as exc:  # test artifacts and checkpoints remain authoritative and untouched
        summary["plotting_succeeded"] = False
        summary["figures_generated"] = []
        summary["plotting_error"] = f"{type(exc).__name__}: {exc}"
        print(f"TRAINING/TEST SUCCEEDED, PLOTTING FAILED: {type(exc).__name__}: {exc}")
    write_json(run_dir / "summary.json", summary)
    write_run_status(
        run_dir,
        "COMPLETED",
        completed_episode=training_episodes,
        total_steps=total_steps,
        durable_recovery_point=durable_recovery_point,
    )

    dashboard.render_final_summary(
        checkpoint_path=str(best_checkpoint),
        metrics_path=str(run_dir / "training_history.json"),
        evaluation_path=str(run_dir / "test_summary.json"),
    )
    idle_stats = test_result["idle"]["team_profit_statistics"]
    algorithm_stats = test_result["team_profit_statistics"]
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run ID: {run_id}\nSeed: {args.seed}\nEpisodes: {training_episodes}\nTotal Steps: {total_steps}")
    print(
        f"Best validation episode: {best_episode}\n"
        f"Best validation objective ({selector.objective_key}): "
        f"{best_validation_score:.4f}"
    )
    print(f"Final-100 profit: {training_summary['final_100_profit_mean']:.4f}")
    print(f"Final-500 profit: {training_summary['final_500_profit_mean']:.4f}")
    print(f"Best checkpoint: {best_checkpoint}\nFinal checkpoint: {final_checkpoint}")
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Checkpoint used: {test_result['checkpoint_used']}\nTest days: {test_result['test_days_count']}")
    print(f"{spec.display_name} mean ± std: {algorithm_stats['mean']:.4f} ± {algorithm_stats['std']:.4f}")
    print(f"Idle mean ± std: {idle_stats['mean']:.4f} ± {idle_stats['std']:.4f}")
    print(f"Win rate vs Idle: {test_result['comparisons']['vs_idle']['win_rate']:.2%}")
    print(f"Results: {layout.run_dir}")
    print("=" * 60)
    return summary


def main(argv: list[str] | None = None) -> int:
    train(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
