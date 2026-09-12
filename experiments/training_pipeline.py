from __future__ import annotations

import argparse
import csv
import hashlib
import math
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.evaluate_matd3 import evaluate, evaluate_policy_on_days
from evaluation.plotting import generate_figures
from experiments.layout import ExperimentLayout
from experiments.profiles import resolve_training_args, experiment_contract
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the shared train-test-settlement-plot pipeline")
    parser.add_argument("--algorithm", default=None,
                        help="Registered algorithm plugin name, matd3 (ordinary MLP) or crossgate")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--experiment", help="Named bundle from configs/experiment_profiles.yaml")
    parser.add_argument("--policy", choices=("mlp", "crossgate"), help="Network under the selected/default action design")
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
    parser.add_argument("--warmup-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--no-milp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-dashboard", action="store_true")
    return parser


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


def _algorithm_replay_extras(
    algorithm: Any, step_idx: int, team_reward: float
) -> dict[str, float]:
    """Return optional algorithm-owned replay fields without inspecting internals."""
    getter = getattr(algorithm, "get_independent_step_data", None)
    if getter is None:
        return {}
    baseline = getter(step_idx)
    independent_reward = float(baseline["reward"])
    return {
        "independent_reward": independent_reward,
        "cooperation_surplus_reward": float(team_reward) - independent_reward,
        "independent_return_to_go": float(baseline["return_to_go"]),
        "next_independent_return_to_go": float(baseline["next_return_to_go"]),
    }


def _algorithm_episode_metrics(
    algorithm: Any, episode_record: Mapping[str, Any]
) -> dict[str, float]:
    """Collect optional episode metrics through an algorithm-level public hook."""
    finalize = getattr(algorithm, "episode_metrics", None)
    if finalize is None:
        return {}
    metrics = {name: float(value) for name, value in finalize().items()}
    grand_return = float(episode_record["team_reward"])
    grand_profit = float(episode_record["team_profit"])
    independent_return = metrics["independent_baseline_return"]
    independent_profit = metrics["independent_baseline_profit"]
    return {
        "grand_team_return": grand_return,
        "grand_team_profit": grand_profit,
        **metrics,
        "cooperation_surplus_return": grand_return - independent_return,
        "cooperation_surplus_profit": grand_profit - independent_profit,
    }


def _runtime_hyperparameters(
    args: argparse.Namespace, settings: Mapping[str, Any]
) -> tuple[int, int]:
    """Resolve run-local smoke overrides without mutating the YAML mapping."""
    warmup_override = getattr(args, "warmup_steps", None)
    batch_override = getattr(args, "batch_size", None)
    warmup = int(warmup_override if warmup_override is not None
                 else settings.get("warmup_steps", 1000))
    batch_size = int(batch_override if batch_override is not None
                     else settings.get("batch_size", 128))
    if warmup < 0:
        raise ValueError("warmup_steps must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return warmup, batch_size


def _training_diagnostic_summary(
    history: Sequence[Mapping[str, Any]], window: int = 100
) -> dict[str, float | None]:
    def final_mean(field: str) -> float | None:
        return _mean([record.get(field) for record in history[-window:]])

    return {
        "final_100_grand_return_mean": final_mean("grand_team_return"),
        "final_100_independent_return_mean": final_mean("independent_baseline_return"),
        "final_100_cooperation_surplus_return_mean": final_mean("cooperation_surplus_return"),
        "final_100_shapley_q_sum_mean": final_mean("shapley_q_sum"),
        "final_100_efficiency_residual_mean": final_mean("efficiency_residual"),
        "final_100_grand_decomposition_residual_mean": final_mean(
            "grand_decomposition_residual"
        ),
    }


def _diagnostic_means(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_names = (
        "critic_loss", "critic1_loss", "critic2_loss", "actor_loss",
        "q1_mean", "q2_mean", "q1_q2_gap", "target_q_mean",
        "td_error_mean", "td_error_std", "actor_grad_norm", "critic_grad_norm",
        "actor_parameter_count",
        "shapley_q_mg1", "shapley_q_mg2", "shapley_q_mg3", "shapley_q_sum",
        "cooperation_surplus_q_target", "efficiency_residual",
        "grand_decomposition_residual",
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
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)


def _validation_record(summary: Mapping[str, Any], episode: int, is_best: bool) -> dict[str, Any]:
    stats = summary["team_profit_statistics"]
    return {
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
    if validation_enabled and validation_interval <= 0:
        raise ValueError("validation.interval must be positive")

    output_override = getattr(args, "output", None)
    layout = ExperimentLayout.create(
        getattr(args, "output_root", "outputs"),
        spec.slug,
        spec.display_name,
        args.seed,
        algorithm_root_override=output_override,
    )
    algorithm_root = layout.algorithm_root
    run_id, run_dir = layout.run_id, layout.run_dir
    checkpoints_dir = layout.checkpoints_dir
    best_checkpoint = checkpoints_dir / "best_model.pt"
    final_checkpoint = checkpoints_dir / "final_model.pt"

    snapshot = {"environment": env_config, "algorithm": algorithm_cfg, "experiment_contract": contract}
    snapshot_text = yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False)
    (run_dir / "config_snapshot.yaml").write_text(snapshot_text, encoding="utf-8")
    config_hash = hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()
    start_time = datetime.now().astimezone()
    metadata: dict[str, Any] = {
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
    }
    write_json(run_dir / "run_metadata.json", metadata)

    env = MultiMicrogridEnv(env_config, split="train")
    algorithm = spec.create(env, algorithm_cfg, device=args.device, seed=args.seed)
    settings = algorithm_cfg["algorithm"]
    warmup, batch_size = _runtime_hyperparameters(args, settings)
    exploration = float(settings.get("exploration_noise", 0.15))
    replay_cap = int(settings.get("replay_capacity", 200_000))
    rng = np.random.default_rng(args.seed)
    ids = env.agent_ids

    validation_env = MultiMicrogridEnv(env_config, split="val") if validation_enabled else None
    validation_days = (
        [str(day.date()) for day in validation_env.data.valid_days] if validation_env else []
    )
    if validation_days_limit is not None:
        validation_days = validation_days[:max(1, int(validation_days_limit))]

    history: list[dict[str, Any]] = []
    history_journal = (TrainingHistoryJournal(run_dir / "training_history.jsonl")
                       if incremental_history else None)
    validation_history: list[dict[str, Any]] = []
    total_steps = 0
    best_validation_score = -float("inf")
    best_episode: int | None = None
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
        for episode_index in range(training_episodes):
            episode_started = time.perf_counter()
            episode_seed = args.seed + episode_index
            observations, reset_info = env.reset(seed=episode_seed)
            episode_start = getattr(algorithm, "on_episode_start", None)
            if episode_start is not None:
                episode_start(env, seed=episode_seed)
            state = reset_info["global_state"]
            collector = EpisodeMetricsCollector(env, capture_hourly=False)
            diagnostics: list[dict[str, Any]] = []
            done = False
            episode_step = 0
            while not done:
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
                replay_extras = _algorithm_replay_extras(
                    algorithm, episode_step, team_reward
                )
                algorithm.replay.add(
                    observations, state, actions, team_reward,
                    next_observations, info["global_state_after"], done,
                    **replay_extras,
                )
                update_metrics = algorithm.update(batch_size)
                if update_metrics:
                    diagnostics.append(update_metrics)
                collector.record_step(actions, info, rewards)
                observations, state = next_observations, info["global_state_after"]
                total_steps += 1
                episode_step += 1

            ep_record = collector.finalize_episode()
            ep_record.update(_algorithm_episode_metrics(algorithm, ep_record))
            ep_record.update({
                "episode": episode_index + 1, "total_steps": total_steps, "steps": total_steps,
                "replay_size": len(algorithm.replay),
                "exploration_noise": exploration if total_steps >= warmup else 1.0,
                "episode_seconds": time.perf_counter() - episode_started,
            })
            ep_record.update(_diagnostic_means(diagnostics))
            _append_actor_diagnostics(run_dir / "actor_diagnostics.csv", episode_index + 1, ep_record)
            economics = ep_record["economic_breakdown_by_microgrid"]
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
                policy = lambda obs: algorithm.select_actions(obs, noise_std=0.0)
                val_summary, _ = evaluate_policy_on_days(
                    validation_env, validation_days, policy, seed=args.seed + 100_000,
                    deterministic_environment=bool(validation_cfg.get("fixed_scenarios", True)),
                    forecast_noise=bool(
                        env_config.get("observation", {}).get(
                            "evaluation_forecast_noise", False,
                        )
                    ),
                )
                score = float(val_summary["team_profit_mean"])
                is_best = score > best_validation_score
                if is_best:
                    best_validation_score = score
                    best_episode = episode_index + 1
                    algorithm.save(best_checkpoint)
                    dashboard.notify_checkpoint(best_episode, score, str(best_checkpoint))
                validation_history.append(_validation_record(
                    val_summary, episode_index + 1, is_best,
                ))
                write_json(run_dir / "validation_history.json", validation_history)
                ep_record["best_validation_profit"] = best_validation_score

            if history_journal is not None:
                history_journal.append(ep_record)
                save_history = (episode_index + 1) % snapshot_interval == 0
            else:
                save_history = should_validate or (episode_index + 1) % 10 == 0
            if save_history or episode_index + 1 == training_episodes:
                write_json(run_dir / "training_history.json", history)
            dashboard.update(ep_record)
            if not dashboard.enabled and ((episode_index + 1) % 25 == 0 or episode_index == 0
                                          or should_validate):
                print(
                    f"Episode {episode_index + 1}/{training_episodes} | "
                    f"Profit {ep_record['team_profit']:.2f} | "
                    f"MA100 {ep_record['team_profit_ma_100']:.2f} | "
                    f"Best validation {best_validation_score:.2f}"
                )
    finally:
        dashboard.stop()
        # A catchable interruption still leaves a compatible snapshot. A hard
        # process kill can be recovered from complete lines in the journal.
        if history_journal is not None and history and sys.exc_info()[0] is not None:
            write_json(run_dir / "training_history.json", history)

    algorithm.save(final_checkpoint)
    if not best_checkpoint.exists():
        algorithm.save(best_checkpoint)
        best_episode = training_episodes
        best_validation_score = float(history[-1]["team_profit"])
    training_seconds = time.perf_counter() - train_started
    checkpoint_selection = {
        "selection_metric": "validation_team_profit_mean" if validation_enabled
                            else "final_model_fallback_validation_disabled",
        "best_episode": best_episode,
        "best_validation_score": best_validation_score,
        "best_checkpoint": "checkpoints/best_model.pt",
        "final_checkpoint": "checkpoints/final_model.pt",
    }
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
        "best_validation_profit_mean": best_validation_score,
        "checkpoint_selection": checkpoint_selection,
        "training_seconds": training_seconds,
        "seconds_per_episode_mean": training_seconds / training_episodes,
        **_training_diagnostic_summary(history),
    }
    write_json(run_dir / "training_summary.json", training_summary)
    metadata["end_time"] = datetime.now().astimezone().isoformat()
    write_json(run_dir / "run_metadata.json", metadata)

    test_result = evaluate(
        config_path, algorithm_path, best_checkpoint, split="test",
        run_milp=not args.no_milp, output_dir=run_dir,
        checkpoint_metadata=checkpoint_selection, test_seed=args.seed,
        max_days=test_episodes, device=args.device, algorithm_name=spec.slug,
        experiment_identity=contract,
    )
    summary = {
        **training_summary,
        "algorithm_slug": spec.slug,
        "algorithm_display_name": spec.display_name,
        "run_id": run_id, "run_directory": str(run_dir),
        "best_team_profit": float(np.max(profits)),
        "last_team_profit": float(profits[-1]),
        "checkpoint": str(final_checkpoint), "best_checkpoint": str(best_checkpoint),
        "test_days_count": test_result["test_days_count"],
        "comparison": test_result["comparison"],
        "algorithm_ess_terminal_soc": test_result["algorithm"]["ess"]["terminal_soc_mean"],
        "matd3_ess_terminal_soc": test_result["matd3"]["ess"]["terminal_soc_mean"],
        "test_evaluation": str(run_dir / "test_evaluation.json"),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(algorithm_root / "latest_run.json", {
        "algorithm_slug": spec.slug,
        "algorithm_display_name": spec.display_name,
        "run_id": run_id,
        "run_directory": str(run_dir),
    })

    try:
        figure_paths = generate_figures(run_dir)
        summary["plotting_succeeded"] = True
        summary["figures_generated"] = [str(path.relative_to(run_dir)) for path in figure_paths]
        summary["plotting_error"] = None
    except Exception as exc:  # test artifacts and checkpoints remain authoritative and untouched
        summary["plotting_succeeded"] = False
        summary["figures_generated"] = []
        summary["plotting_error"] = f"{type(exc).__name__}: {exc}"
        print(f"TRAINING/TEST SUCCEEDED, PLOTTING FAILED: {type(exc).__name__}: {exc}")
    write_json(run_dir / "summary.json", summary)

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
    print(f"Best validation episode: {best_episode}\nBest validation team profit: {best_validation_score:.4f}")
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
    print(f"Results: {run_dir}")
    print("=" * 60)
    return summary


def main(argv: list[str] | None = None) -> int:
    train(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
