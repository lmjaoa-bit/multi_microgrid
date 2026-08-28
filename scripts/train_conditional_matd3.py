from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.live import Live

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.conditional_matd3.networks import ConditionalActor  # noqa: E402
from algorithms.conditional_matd3.trainer import ConditionalMATD3Trainer  # noqa: E402
from algorithms.conditional_matd3.config import (  # noqa: E402
    ALGORITHM_NAME,
    DEFAULT_ALGORITHM_CONFIG,
    DEFAULT_ENV_CONFIG,
)
from envs.multi_microgrid_env import MultiMicrogridEnv  # noqa: E402
from scripts.training_dashboard import build_dashboard_data, render_dashboard  # noqa: E402
from utils.config import (  # noqa: E402
    compose_experiment_config,
    load_algorithm_config,
    runtime_config_snapshot,
)


DEFAULT_ALGO_CONFIG = DEFAULT_ALGORITHM_CONFIG
ALGORITHM_RUN_NAME = ALGORITHM_NAME
TRAINING_METRICS = (
    "critic_loss",
    "q1_mean",
    "q2_mean",
    "q_mean",
    "q_disagreement",
    "actor_loss",
    "actor_grad_norm",
    "commitment_loss",
    "common_loss",
    "generator_loss",
    "q_on",
    "q_off",
    "delta_q",
    "p_on",
    "commit_target",
    "q_scale",
)
METRIC_COLUMN_NAMES = {"q_scale": "q_scale_ema"}
TRAINING_LOG_METRICS = tuple(
    METRIC_COLUMN_NAMES.get(metric, metric) for metric in TRAINING_METRICS
)
LEGACY_LOG_COLUMN_NAMES = {
    "system_p2p_trade_kwh": "market_trade_kwh",
    "system_renewable_curtailment_kwh": "renewable_curtailment_kwh",
    "system_grid_import_kwh": "grid_import_kwh",
    "system_grid_export_kwh": "grid_export_kwh",
    **{f"MG{index}_p2p_trade_kwh": f"MG{index}_p2p_kwh" for index in range(1, 4)},
    **{f"MG{index}_q_scale": f"MG{index}_q_scale_ema" for index in range(1, 4)},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the standalone three-agent Conditional MATD3 baseline."
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override total episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override run seed.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. 'auto' selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/conditional_matd3_<timestamp>.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override the YAML replay warmup steps.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument(
        "--keep-periodic-checkpoints",
        action="store_true",
        default=None,
        help="Keep debug checkpoint_epXXXX.pt snapshots in addition to last_checkpoint.pt.",
    )
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Deprecated alias for --env-config.",
    )
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--algo-config", type=Path, default=DEFAULT_ALGO_CONFIG)
    parser.add_argument(
        "--no-live",
        action="store_true",
        default=None,
        help="Disable the live dashboard and use periodic plain-text logging.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=None,
        help="Episode window used only for dashboard rolling means.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume algorithm/optimizer counters from a checkpoint (not exact trajectory resume).",
    )
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_output(output_config: Mapping[str, Any] | None = None) -> Path:
    output_config = output_config or {}
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(str(output_config.get("root", "outputs")))
    if not root.is_absolute():
        root = ROOT / root
    run_name = str(output_config.get("run_name", ALGORITHM_RUN_NAME))
    return root / f"{run_name}_{stamp}"


def _resolve_run_arguments(
    args: argparse.Namespace,
    algorithm_document: Mapping[str, Any],
) -> argparse.Namespace:
    training = algorithm_document.get("training", {})
    algorithm = algorithm_document["algorithm"]
    defaults = {
        "episodes": 300,
        "seed": 42,
        "device": "auto",
        "checkpoint_interval": 50,
        "validation_interval": 20,
        "validation_episodes": 10,
        "log_interval": 10,
        "rolling_window": 20,
    }
    for name, fallback in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, training.get(name, fallback))
    if args.warmup_steps is None:
        args.warmup_steps = int(algorithm.get("warmup_steps", 5000))
    if args.keep_periodic_checkpoints is None:
        args.keep_periodic_checkpoints = bool(
            training.get("keep_periodic_checkpoints", False)
        )
    if args.no_live is None:
        args.no_live = not bool(training.get("live_dashboard", True))
    return args


def _log_columns(agent_ids: Sequence[str]) -> list[str]:
    columns = [
        "episode",
        "env_steps",
        "replay_size",
        "critic_update_count",
        "actor_update_count",
        "validation_available",
        "best_validation_cost",
        "best_episode",
        "system_total_cost",
        "system_total_reward",
        "market_trade_kwh",
        "renewable_curtailment_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
    ]
    for mg in agent_ids:
        label = mg.upper()
        columns.extend([f"{label}_reward", f"{label}_cost", f"{label}_p2p_kwh"])
    for mg in agent_ids:
        label = mg.upper()
        columns.extend(f"{label}_{metric}" for metric in TRAINING_LOG_METRICS)
    columns.append("policy_updated")
    return columns


def _assert_finite_mapping(values: Mapping[str, Any], context: str) -> None:
    for name, value in values.items():
        if isinstance(value, (bool, int, float, np.number)):
            if not math.isfinite(float(value)):
                raise RuntimeError(f"{context}: {name} is non-finite ({value!r}).")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_log(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_existing_log(path: Path, columns: Sequence[str], start_episode: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        normalized_fields = [
            LEGACY_LOG_COLUMN_NAMES.get(name, name) for name in (reader.fieldnames or [])
        ]
        optional_new_columns = {
            "critic_update_count",
            "actor_update_count",
            "validation_available",
            "best_validation_cost",
            "best_episode",
            *{
                f"{mg.upper()}_{metric}"
                for mg in ("mg1", "mg2", "mg3")
                for metric in (
                    "q1_mean",
                    "q2_mean",
                    "q_disagreement",
                    "actor_grad_norm",
                )
            },
        }
        expected_without_new = [
            name for name in columns if name not in optional_new_columns
        ]
        if normalized_fields not in (list(columns), expected_without_new):
            raise ValueError("Existing training_log.csv columns do not match this CLI version.")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            converted: dict[str, Any] = {}
            for name, value in raw.items():
                normalized_name = LEGACY_LOG_COLUMN_NAMES.get(name, name)
                converted[normalized_name] = (
                    int(value)
                    if normalized_name in {"episode", "env_steps", "replay_size"}
                    else float(value)
                )
            for name in optional_new_columns:
                converted.setdefault(name, 0.0)
            _assert_finite_mapping(converted, "existing training log")
            if int(converted["episode"]) < start_episode:
                rows.append(converted)
        return rows


def _torch_load(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.6
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint {path} is not a mapping.")
    return checkpoint


def _save_checkpoint(
    path: Path,
    trainer: ConditionalMATD3Trainer,
    episode: int,
    run_metadata: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
    exception: BaseException | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "checkpoint_type": "crash" if exception is not None else "training_state",
        "algorithm": trainer.algorithm.state_dict(),
        "episode": int(episode),
        "env_steps": int(trainer.env_steps),
        "trainer_episodes": int(trainer.episodes),
        "metadata": dict(run_metadata),
        "exact_resume_supported": False,
        "resume_limitation": "Replay buffer and RNG states are not restored.",
    }
    if training_state is not None:
        payload.update(dict(training_state))
    if exception is not None:
        payload["failure"] = {
            "type": type(exception).__name__,
            "message": str(exception),
        }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_best_model(
    path: Path,
    trainer: ConditionalMATD3Trainer,
    episode: int,
    validation_metrics: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> None:
    policy = trainer.algorithm.get_policy_state_dict()
    payload = {
        "format_version": 1,
        "checkpoint_type": "best_policy",
        "actors": policy["actors"],
        "algorithm_metadata": policy["metadata"],
        "episode": int(episode),
        "best_episode": int(episode),
        "env_steps": int(trainer.env_steps),
        "best_validation_cost": float(validation_metrics["validation_system_cost_mean"]),
        "validation_metrics": dict(validation_metrics),
        "metadata": dict(run_metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _maybe_update_best_model(
    path: Path,
    trainer: ConditionalMATD3Trainer,
    episode: int,
    validation_metrics: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    best_validation_cost: float,
    best_episode: int,
) -> tuple[float, int, bool]:
    current_cost = float(validation_metrics["validation_system_cost_mean"])
    if current_cost >= best_validation_cost:
        return best_validation_cost, best_episode, False
    _save_best_model(path, trainer, episode, validation_metrics, run_metadata)
    return current_cost, int(episode), True


def _validation_columns(agent_ids: Sequence[str]) -> list[str]:
    columns = [
        "episode",
        "env_steps",
        "validation_system_cost_mean",
        "validation_system_cost_std",
        "validation_system_reward_mean",
        "validation_p2p_trade_kwh_mean",
        "validation_curtailment_kwh_mean",
        "validation_grid_import_kwh_mean",
        "validation_grid_export_kwh_mean",
        "is_best",
        "best_validation_cost",
    ]
    for mg in agent_ids:
        label = mg.upper()
        columns.extend(
            [
                f"{label}_validation_cost_mean",
                f"{label}_validation_reward_mean",
                f"{label}_p2p_kwh_mean",
            ]
        )
    return columns


def _policy_stability_columns(agent_ids: Sequence[str]) -> list[str]:
    return [
        "episode",
        *[f"{mg.upper()}_policy_action_l2" for mg in agent_ids],
        "max_policy_action_l2",
    ]


def _load_policy_stability_log(
    path: Path,
    columns: Sequence[str],
    start_episode: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(columns):
            raise ValueError("Existing policy_stability_log.csv columns do not match this CLI version.")
        rows = []
        for raw in reader:
            converted = {
                name: int(value) if name == "episode" else float(value)
                for name, value in raw.items()
            }
            _assert_finite_mapping(converted, "existing policy stability log")
            if int(converted["episode"]) < start_episode:
                rows.append(converted)
        return rows


@torch.no_grad()
def _collect_policy_probe_observations(
    algorithm,
    validation_env: MultiMicrogridEnv,
    day: str,
    seed: int,
) -> dict[str, np.ndarray]:
    observations = {mg: [] for mg in validation_env.agent_ids}
    obs, _ = validation_env.reset(day=day, seed=seed, deterministic=True)
    done = False
    while not done:
        for mg in validation_env.agent_ids:
            observations[mg].append(np.asarray(obs[mg], dtype=np.float32).copy())
        actions, _ = algorithm.select_actions(obs, explore=False, warmup=False)
        obs, _, done, _ = validation_env.step(actions)
    return {
        mg: np.stack(values).astype(np.float32, copy=False)
        for mg, values in observations.items()
    }


@torch.no_grad()
def _policy_actions_on_probes(
    algorithm,
    probes: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    actions = {}
    for mg in algorithm.agent_ids:
        obs = torch.as_tensor(
            probes[mg],
            dtype=torch.float32,
            device=algorithm.device,
        )
        actions[mg] = ConditionalActor.hard_action(
            algorithm.actors[mg](obs)
        ).cpu().numpy()
    return actions


def _policy_stability_row(
    episode: int,
    agent_ids: Sequence[str],
    previous_actions: Mapping[str, np.ndarray],
    current_actions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    changes = {
        mg: float(
            np.linalg.norm(current_actions[mg] - previous_actions[mg], axis=1).mean()
        )
        for mg in agent_ids
    }
    row: dict[str, Any] = {"episode": int(episode)}
    for mg in agent_ids:
        row[f"{mg.upper()}_policy_action_l2"] = changes[mg]
    row["max_policy_action_l2"] = max(changes.values())
    _assert_finite_mapping(row, f"episode {episode} policy stability")
    return row


def _plot_policy_stability(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    agent_ids: Sequence[str],
) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    colors = ("#27628D", "#C97A2B", "#6B7D3A")
    episodes = [int(row["episode"]) for row in rows]
    for mg, color in zip(agent_ids, colors):
        axis.plot(
            episodes,
            [float(row[f"{mg.upper()}_policy_action_l2"]) for row in rows],
            label=mg.upper(),
            color=color,
            linewidth=1.3,
            marker="o",
            markersize=3,
        )
    axis.set(
        title="Policy action change on fixed validation probes",
        xlabel="Training episode",
        ylabel="Mean action L2 since previous validation",
    )
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", color="#D5DAE0", linewidth=0.7, alpha=0.8)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _load_validation_log(path: Path, columns: Sequence[str], start_episode: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(columns):
            raise ValueError("Existing validation_log.csv columns do not match this CLI version.")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            converted = {
                name: (
                    int(value)
                    if name in {"episode", "env_steps", "is_best"}
                    else float(value)
                )
                for name, value in raw.items()
            }
            _assert_finite_mapping(converted, "existing validation log")
            if int(converted["episode"]) < start_episode:
                rows.append(converted)
        return rows


def run_deterministic_validation(
    trainer: ConditionalMATD3Trainer,
    validation_env: MultiMicrogridEnv,
    validation_days: Sequence[str],
    seed: int,
) -> dict[str, float]:
    """Evaluate the current policy on fixed val days without changing training state."""
    if validation_env.split != "val":
        raise ValueError("Best-model validation must use an environment with split='val'.")
    if not validation_days:
        raise ValueError("At least one fixed validation day is required.")

    replay_size_before = len(trainer.replay)
    env_steps_before = trainer.env_steps
    update_counter_before = trainer.algorithm.update_counter
    q_scale_before = dict(trainer.algorithm.q_scale_ema)
    algorithm_rng_before = copy.deepcopy(trainer.algorithm.rng.bit_generator.state)
    evaluator = ConditionalMATD3Trainer(
        env=validation_env,
        algorithm=trainer.algorithm,
        replay=trainer.replay,
        batch_size=trainer.batch_size,
        warmup_steps=trainer.warmup_steps,
        updates_per_step=trainer.updates_per_step,
    )

    system_costs: list[float] = []
    system_rewards: list[float] = []
    trades: list[float] = []
    curtailments: list[float] = []
    imports: list[float] = []
    exports: list[float] = []
    mg_costs = {mg: [] for mg in validation_env.agent_ids}
    mg_rewards = {mg: [] for mg in validation_env.agent_ids}
    mg_p2p = {mg: [] for mg in validation_env.agent_ids}
    for index, day in enumerate(validation_days):
        result = evaluator.run_episode(
            day=day,
            training=False,
            deterministic_eval=True,
            seed=int(seed) + index,
        )
        system_costs.append(
            sum(float(result["by_microgrid"][mg]["cost"]) for mg in validation_env.agent_ids)
        )
        system_rewards.append(
            sum(float(result["by_microgrid"][mg]["reward"]) for mg in validation_env.agent_ids)
        )
        trades.append(float(result["market_trade_kwh"]))
        curtailments.append(float(result["renewable_curtailment_kwh"]))
        imports.append(float(result["grid_import_kwh"]))
        exports.append(float(result["grid_export_kwh"]))
        for mg in validation_env.agent_ids:
            mg_costs[mg].append(float(result["by_microgrid"][mg]["cost"]))
            mg_rewards[mg].append(float(result["by_microgrid"][mg]["reward"]))
            mg_p2p[mg].append(float(result["by_microgrid"][mg]["p2p_kwh"]))

    if (
        len(trainer.replay) != replay_size_before
        or trainer.env_steps != env_steps_before
        or trainer.algorithm.update_counter != update_counter_before
        or trainer.algorithm.q_scale_ema != q_scale_before
        or trainer.algorithm.rng.bit_generator.state != algorithm_rng_before
    ):
        raise RuntimeError("Deterministic validation mutated training state.")

    metrics = {
        "validation_system_cost_mean": float(np.mean(system_costs)),
        "validation_system_cost_std": float(np.std(system_costs)),
        "validation_system_reward_mean": float(np.mean(system_rewards)),
        "validation_p2p_trade_kwh_mean": float(np.mean(trades)),
        "validation_curtailment_kwh_mean": float(np.mean(curtailments)),
        "validation_grid_import_kwh_mean": float(np.mean(imports)),
        "validation_grid_export_kwh_mean": float(np.mean(exports)),
    }
    for mg in validation_env.agent_ids:
        label = mg.upper()
        metrics[f"{label}_validation_cost_mean"] = float(np.mean(mg_costs[mg]))
        metrics[f"{label}_validation_reward_mean"] = float(np.mean(mg_rewards[mg]))
        metrics[f"{label}_p2p_kwh_mean"] = float(np.mean(mg_p2p[mg]))
    _assert_finite_mapping(metrics, "deterministic validation metrics")
    return metrics


def _episode_row(
    episode: int,
    result: Mapping[str, Any],
    agent_ids: Sequence[str],
    metric_defaults: Mapping[str, float],
    available_metrics: set[str] | None = None,
) -> dict[str, Any]:
    updates = result.get("last_update", {})
    if not isinstance(updates, Mapping):
        raise RuntimeError("Trainer last_update must be a mapping.")
    _assert_finite_mapping(updates, f"episode {episode} update metrics")
    if available_metrics is not None:
        available_metrics.clear()
    episode_metrics = dict(metric_defaults)
    for mg in agent_ids:
        label = mg.upper()
        for metric in TRAINING_METRICS:
            source = f"{mg}/{metric}"
            if source in updates:
                destination = f"{label}_{METRIC_COLUMN_NAMES.get(metric, metric)}"
                episode_metrics[destination] = float(updates[source])
                if available_metrics is not None:
                    available_metrics.add(destination)
    if "policy_updated" in updates:
        episode_metrics["policy_updated"] = float(updates["policy_updated"])

    by_microgrid = result["by_microgrid"]
    row: dict[str, Any] = {
        "episode": int(episode),
        "env_steps": int(result["env_steps"]),
        "replay_size": int(result["replay_size"]),
        "critic_update_count": int(updates.get("critic_update_count", 0.0)),
        "actor_update_count": int(updates.get("actor_update_count", 0.0)),
        "system_total_cost": sum(float(by_microgrid[mg]["cost"]) for mg in agent_ids),
        "system_total_reward": sum(float(by_microgrid[mg]["reward"]) for mg in agent_ids),
        "market_trade_kwh": float(result["market_trade_kwh"]),
        "renewable_curtailment_kwh": float(result["renewable_curtailment_kwh"]),
        "grid_import_kwh": float(result["grid_import_kwh"]),
        "grid_export_kwh": float(result["grid_export_kwh"]),
    }
    for mg in agent_ids:
        label = mg.upper()
        row[f"{label}_reward"] = float(by_microgrid[mg]["reward"])
        row[f"{label}_cost"] = float(by_microgrid[mg]["cost"])
        row[f"{label}_p2p_kwh"] = float(by_microgrid[mg]["p2p_kwh"])
    row.update(episode_metrics)
    _assert_finite_mapping(row, f"episode {episode} log row")
    return row


def _print_progress(row: Mapping[str, Any]) -> None:
    critic_values = [float(value) for key, value in row.items() if key.endswith("_critic_loss")]
    actor_values = [float(value) for key, value in row.items() if key.endswith("_actor_loss")]
    print(
        "episode={episode} env_steps={env_steps} replay={replay_size} "
        "reward={system_total_reward:.4f} cost={system_total_cost:.2f} "
        "trade_kwh={market_trade_kwh:.2f} curtail_kwh={renewable_curtailment_kwh:.2f} "
        "critic_loss={critic:.6f} actor_loss={actor:.6f}".format(
            **row,
            critic=float(np.mean(critic_values)) if critic_values else 0.0,
            actor=float(np.mean(actor_values)) if actor_values else 0.0,
        ),
        flush=True,
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    algorithm_config_path = args.algo_config.resolve()
    if not algorithm_config_path.is_file():
        raise FileNotFoundError(
            f"Algorithm config file does not exist: {algorithm_config_path}"
        )
    algorithm_document = load_algorithm_config(algorithm_config_path)
    args = _resolve_run_arguments(args, algorithm_document)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive.")
    if args.validation_interval <= 0:
        raise ValueError("--validation-interval must be positive.")
    if args.validation_episodes <= 0:
        raise ValueError("--validation-episodes must be positive.")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")
    if args.rolling_window <= 0:
        raise ValueError("--rolling-window must be positive.")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")

    config_path = (args.config or args.env_config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Environment config file does not exist: {config_path}")
    resume_path = args.resume.resolve() if args.resume is not None else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

    device = _resolve_device(args.device)
    _seed_everything(args.seed)
    runtime_config, algorithm_document = compose_experiment_config(
        config_path,
        algorithm_config_path,
    )
    env = MultiMicrogridEnv(runtime_config, split="train")
    env.config.setdefault("env", {})["seed"] = int(args.seed)
    env.seed = int(args.seed)
    trainer = ConditionalMATD3Trainer.from_env(env, device=str(device))
    if args.warmup_steps is not None:
        trainer.warmup_steps = int(args.warmup_steps)
    validation_env = MultiMicrogridEnv(runtime_config, split="val")

    output = (
        args.output.resolve()
        if args.output is not None
        else (
            resume_path.parent
            if resume_path is not None
            else _default_output(algorithm_document.get("output", {})).resolve()
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training_log.csv"
    validation_log_path = output / "validation_log.csv"
    policy_stability_log_path = output / "policy_stability_log.csv"
    policy_stability_plot_path = output / "policy_action_change_vs_episode.png"
    config_output_path = output / "run_config.json"
    best_model_path = output / "best_model.pt"
    last_checkpoint_path = output / "last_checkpoint.pt"
    if log_path.exists() and resume_path is None:
        raise FileExistsError(
            f"Refusing to overwrite existing {log_path}; choose another --output or use --resume."
        )

    start_episode = 1
    resume_checkpoint: Mapping[str, Any] | None = None
    if resume_path is not None:
        checkpoint = _torch_load(resume_path, device)
        resume_checkpoint = checkpoint
        required = {"algorithm", "episode", "env_steps"}
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(f"Resume checkpoint is missing keys: {sorted(missing)}")
        trainer.algorithm.load_state_dict(checkpoint["algorithm"])
        trainer.env_steps = int(checkpoint["env_steps"])
        trainer.episodes = int(checkpoint.get("trainer_episodes", checkpoint["episode"]))
        start_episode = int(checkpoint["episode"]) + 1
        print(
            "WARNING: --resume restores networks, optimizers, counters, episode, and env_steps only; "
            "replay buffer and RNG states are not restored, so this is not an exact trajectory resume.",
            flush=True,
        )
    if start_episode > args.episodes:
        raise ValueError(
            f"Checkpoint already completed episode {start_episode - 1}, beyond --episodes={args.episodes}."
        )

    saved_configs = output / "configs"
    saved_configs.mkdir(parents=True, exist_ok=True)
    config_suffix = "" if resume_path is None else f"_resume_ep{start_episode - 1}"
    shutil.copy2(
        config_path,
        saved_configs / f"environment{config_suffix}.yaml",
    )
    shutil.copy2(
        algorithm_config_path,
        saved_configs / f"algorithm{config_suffix}.yaml",
    )
    with (saved_configs / f"effective_runtime{config_suffix}.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            runtime_config_snapshot(runtime_config),
            handle,
            sort_keys=False,
            allow_unicode=True,
        )

    columns = _log_columns(env.agent_ids)
    rows = _load_existing_log(log_path, columns, start_episode) if resume_path is not None else []
    metric_defaults = {
        column: 0.0
        for column in columns
        if any(column.endswith(f"_{metric}") for metric in TRAINING_LOG_METRICS)
    }
    metric_defaults["policy_updated"] = 0.0
    available_metrics: set[str] = set()
    validation_columns = _validation_columns(env.agent_ids)
    validation_rows = (
        _load_validation_log(validation_log_path, validation_columns, start_episode)
        if resume_path is not None
        else []
    )

    available_validation_days = [
        str(day.date()) for day in validation_env.data.valid_days
    ]
    checkpoint_days = (
        list(resume_checkpoint.get("validation_days", []))
        if resume_checkpoint is not None
        else []
    )
    validation_days = checkpoint_days or available_validation_days[: args.validation_episodes]
    invalid_days = sorted(set(validation_days) - set(available_validation_days))
    if invalid_days:
        raise ValueError(f"Checkpoint validation days are unavailable in val split: {invalid_days}")
    if not validation_days:
        raise ValueError("The val split has no valid deterministic validation days.")

    policy_stability_columns = _policy_stability_columns(env.agent_ids)
    policy_stability_rows = (
        _load_policy_stability_log(
            policy_stability_log_path,
            policy_stability_columns,
            start_episode,
        )
        if resume_path is not None
        else []
    )
    policy_probe_day = validation_days[0]
    checkpoint_policy_probes = (
        resume_checkpoint.get("policy_probe_observations")
        if resume_checkpoint is not None
        else None
    )
    probes_restored = isinstance(checkpoint_policy_probes, Mapping) and set(
        checkpoint_policy_probes
    ) == set(env.agent_ids)
    if probes_restored:
        policy_probes = {
            mg: np.asarray(checkpoint_policy_probes[mg], dtype=np.float32).copy()
            for mg in env.agent_ids
        }
    else:
        policy_probes = _collect_policy_probe_observations(
            trainer.algorithm,
            validation_env,
            policy_probe_day,
            seed=int(args.seed) + 2_000_000,
        )
    checkpoint_probe_actions = (
        resume_checkpoint.get("policy_probe_actions")
        if resume_checkpoint is not None
        else None
    )
    if probes_restored and isinstance(checkpoint_probe_actions, Mapping) and set(
        checkpoint_probe_actions
    ) == set(env.agent_ids):
        previous_probe_actions = {
            mg: np.asarray(checkpoint_probe_actions[mg], dtype=np.float32).copy()
            for mg in env.agent_ids
        }
    else:
        previous_probe_actions = _policy_actions_on_probes(trainer.algorithm, policy_probes)

    best_validation_cost = float(
        resume_checkpoint.get("best_validation_cost", math.inf)
        if resume_checkpoint is not None
        else math.inf
    )
    best_episode = int(
        resume_checkpoint.get("best_episode", 0)
        if resume_checkpoint is not None
        else 0
    )
    last_validation_metrics: dict[str, Any] | None = (
        dict(resume_checkpoint.get("validation_metrics", {}))
        if resume_checkpoint is not None and resume_checkpoint.get("validation_metrics")
        else None
    )
    if validation_rows and not math.isfinite(best_validation_cost):
        best_row = min(validation_rows, key=lambda item: item["validation_system_cost_mean"])
        best_validation_cost = float(best_row["validation_system_cost_mean"])
        best_episode = int(best_row["episode"])

    run_started = datetime.now().astimezone().isoformat(timespec="seconds")
    run_metadata = {
        "algorithm": "conditional_matd3",
        "seed": int(args.seed),
        "device": str(device),
        "config_path": str(config_path),
        "algorithm_config_path": str(algorithm_config_path),
        "output": str(output),
        "started_at": run_started,
    }
    run_config = {
        "command": [str(item) for item in sys.argv],
        "training": {
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "device_requested": str(args.device),
            "device_resolved": str(device),
            "warmup_steps": int(trainer.warmup_steps),
            "checkpoint_interval": int(args.checkpoint_interval),
            "keep_periodic_checkpoints": bool(args.keep_periodic_checkpoints),
            "validation_interval": int(args.validation_interval),
            "validation_episodes": int(len(validation_days)),
            "validation_days": list(validation_days),
            "validation_split": "val",
            "validation_deterministic": True,
            "best_criterion": "mean_validation_system_cost_min",
            "policy_stability_probe_day": str(policy_probe_day),
            "policy_stability_probe_count": int(len(policy_probes[env.agent_ids[0]])),
            "policy_stability_metric": "mean_action_l2_since_previous_validation",
            "log_interval": int(args.log_interval),
            "reward_scale": float(env.config["cost"]["reward_scale"]),
            "actor_update_mode": str(trainer.algorithm.actor_update_mode),
            "live_dashboard": not bool(args.no_live),
            "rolling_window": int(args.rolling_window),
            "start_episode": int(start_episode),
            "resume": str(resume_path) if resume_path is not None else None,
        },
        "checkpoint_scope": {
            "best_model_online_actors_only": True,
            "last_checkpoint_algorithm_and_optimizers": True,
            "last_checkpoint_episode_and_env_steps": True,
            "periodic_history_by_default": False,
            "replay_buffer": False,
            "rng_states": False,
            "exact_trajectory_resume": False,
        },
        "runtime": {
            "started_at": run_started,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "resolved_environment_config": env.config,
        "source_configs": {
            "environment": str(config_path),
            "algorithm": str(algorithm_config_path),
            "saved_directory": str(saved_configs),
        },
    }
    _write_json(config_output_path, run_config)
    if not validation_log_path.exists():
        _write_log(validation_log_path, [], validation_columns)
    if not policy_stability_log_path.exists():
        _write_log(policy_stability_log_path, [], policy_stability_columns)

    completed_episode = start_episode - 1
    session_started = time.monotonic()
    console = Console()
    live: Live | None = None
    new_best_for_display = False
    last_checkpoint_episode = 0

    def checkpoint_training_state() -> dict[str, Any]:
        return {
            "best_validation_cost": float(best_validation_cost),
            "best_episode": int(best_episode),
            "validation_metrics": dict(last_validation_metrics or {}),
            "validation_days": list(validation_days),
            "policy_probe_actions": {
                mg: np.asarray(actions, dtype=np.float32).copy()
                for mg, actions in previous_probe_actions.items()
            },
            "policy_probe_observations": {
                mg: np.asarray(observations, dtype=np.float32).copy()
                for mg, observations in policy_probes.items()
            },
        }

    def next_validation_episode() -> int | None:
        if completed_episode >= args.episodes:
            return None
        regular_next = (
            completed_episode // args.validation_interval + 1
        ) * args.validation_interval
        return min(int(args.episodes), regular_next)

    def dashboard(status: str, reason: str | None = None):
        current_row = rows[-1] if rows else None
        data = build_dashboard_data(
            current_row,
            rows,
            env.agent_ids,
            current_episode=completed_episode,
            total_episodes=int(args.episodes),
            start_episode=start_episode,
            device=str(device),
            warmup_steps=int(trainer.warmup_steps),
            update_count=int(trainer.algorithm.update_counter),
            elapsed_seconds=time.monotonic() - session_started,
            rolling_window=int(args.rolling_window),
            available_metrics=available_metrics,
            validation={
                "last_validation_cost": (
                    float(last_validation_metrics["validation_system_cost_mean"])
                    if last_validation_metrics is not None
                    else None
                ),
                "best_validation_cost": (
                    float(best_validation_cost)
                    if math.isfinite(best_validation_cost)
                    else None
                ),
                "best_episode": int(best_episode) if best_episode > 0 else None,
                "next_validation_episode": next_validation_episode(),
                "new_best": bool(new_best_for_display),
            },
            status=status,
            reason=reason,
        )
        if current_row is None:
            data["overview"]["env_steps"] = int(trainer.env_steps)
            data["overview"]["replay_size"] = len(trainer.replay)
        return render_dashboard(data)

    if not args.no_live:
        live = Live(
            dashboard("RUNNING"),
            console=console,
            refresh_per_second=4,
            transient=False,
        )
        live.start(refresh=True)
    try:
        for episode in range(start_episode, args.episodes + 1):
            new_best_for_display = False
            result = trainer.run_episode(training=True, seed=int(args.seed) + episode - 1)
            row = _episode_row(
                episode,
                result,
                env.agent_ids,
                metric_defaults,
                available_metrics,
            )
            validation_due = (
                episode % args.validation_interval == 0 or episode == args.episodes
            )
            if validation_due:
                validation_metrics = run_deterministic_validation(
                    trainer,
                    validation_env,
                    validation_days,
                    seed=int(args.seed) + 1_000_000,
                )
                validation_cost = float(
                    validation_metrics["validation_system_cost_mean"]
                )
                best_validation_cost, best_episode, is_best = _maybe_update_best_model(
                    best_model_path,
                    trainer,
                    episode,
                    validation_metrics,
                    run_metadata,
                    best_validation_cost,
                    best_episode,
                )
                new_best_for_display = is_best
                last_validation_metrics = dict(validation_metrics)
                validation_row = {
                    "episode": int(episode),
                    "env_steps": int(trainer.env_steps),
                    **validation_metrics,
                    "is_best": int(is_best),
                    "best_validation_cost": float(best_validation_cost),
                }
                _assert_finite_mapping(validation_row, f"episode {episode} validation row")
                validation_rows.append(validation_row)
                _write_log(validation_log_path, validation_rows, validation_columns)
                current_probe_actions = _policy_actions_on_probes(
                    trainer.algorithm,
                    policy_probes,
                )
                policy_stability_rows.append(
                    _policy_stability_row(
                        episode,
                        env.agent_ids,
                        previous_probe_actions,
                        current_probe_actions,
                    )
                )
                _write_log(
                    policy_stability_log_path,
                    policy_stability_rows,
                    policy_stability_columns,
                )
                previous_probe_actions = current_probe_actions
                if live is None:
                    suffix = " NEW BEST" if is_best else ""
                    print(
                        f"validation episode={episode} mean_system_cost={validation_cost:.4f} "
                        f"best={best_validation_cost:.4f}{suffix}",
                        flush=True,
                    )

            row["validation_available"] = int(math.isfinite(best_validation_cost))
            row["best_validation_cost"] = (
                float(best_validation_cost) if math.isfinite(best_validation_cost) else 0.0
            )
            row["best_episode"] = int(best_episode)
            rows.append(row)
            completed_episode = episode
            _write_log(log_path, rows, columns)

            if episode % args.checkpoint_interval == 0:
                _save_checkpoint(
                    last_checkpoint_path,
                    trainer,
                    episode,
                    run_metadata,
                    training_state=checkpoint_training_state(),
                )
                last_checkpoint_episode = episode
                if args.keep_periodic_checkpoints:
                    checkpoint_path = output / f"checkpoint_ep{episode:04d}.pt"
                    _save_checkpoint(
                        checkpoint_path,
                        trainer,
                        episode,
                        run_metadata,
                        training_state=checkpoint_training_state(),
                    )
                if live is None:
                    print(f"last_checkpoint={last_checkpoint_path}", flush=True)

            if live is not None:
                live.update(dashboard("RUNNING"), refresh=True)
            elif episode % args.log_interval == 0 or episode == start_episode or episode == args.episodes:
                _print_progress(row)

        if last_checkpoint_episode != completed_episode:
            _save_checkpoint(
                last_checkpoint_path,
                trainer,
                completed_episode,
                run_metadata,
                training_state=checkpoint_training_state(),
            )
        if live is not None:
            live.update(dashboard("TRAINING COMPLETED"), refresh=True)
            live.stop()
            live = None
        _plot_policy_stability(
            policy_stability_plot_path,
            policy_stability_rows,
            env.agent_ids,
        )
        print("Training finished.", flush=True)
        print(f"Episodes: {completed_episode} / {args.episodes}", flush=True)
        print(f"Environment steps: {trainer.env_steps}", flush=True)
        print(f"Replay size: {len(trainer.replay)}", flush=True)
        print(f"Best model: {best_model_path}", flush=True)
        print(f"Last checkpoint: {last_checkpoint_path}", flush=True)
        print(f"Training log: {log_path}", flush=True)
        print(f"Validation log: {validation_log_path}", flush=True)
        print(f"Policy stability log: {policy_stability_log_path}", flush=True)
        print(f"Policy stability plot: {policy_stability_plot_path}", flush=True)
        print(f"Run config: {config_output_path}", flush=True)
        print(f"Output directory: {output}", flush=True)
        print(
            f"training_complete output={output} best_model={best_model_path} "
            f"last_checkpoint={last_checkpoint_path}",
            flush=True,
        )
    except Exception as exc:
        crash_path = output / "crash_checkpoint.pt"
        if live is not None:
            live.update(
                dashboard("TRAINING ABORTED", f"{type(exc).__name__}: {exc}"),
                refresh=True,
            )
        try:
            _save_checkpoint(
                crash_path,
                trainer,
                completed_episode,
                run_metadata,
                training_state=checkpoint_training_state(),
                exception=exc,
            )
            if live is not None:
                live.stop()
                live = None
            print(
                "TRAINING ABORTED\n"
                f"Episode: {completed_episode + 1}\n"
                f"Env step: {trainer.env_steps}\n"
                f"Reason: {type(exc).__name__}: {exc}\n"
                f"Crash checkpoint: {crash_path}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as checkpoint_exc:
            print(
                f"training_failed error={type(exc).__name__}: {exc}; "
                f"crash checkpoint also failed: {checkpoint_exc}",
                file=sys.stderr,
                flush=True,
            )
        raise
    finally:
        if live is not None:
            live.stop()

    return {
        "output": output,
        "training_log": log_path,
        "validation_log": validation_log_path,
        "policy_stability_log": policy_stability_log_path,
        "policy_stability_plot": policy_stability_plot_path,
        "run_config": config_output_path,
        "best_model": best_model_path,
        "last_checkpoint": last_checkpoint_path,
        "episode": completed_episode,
        "env_steps": trainer.env_steps,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
