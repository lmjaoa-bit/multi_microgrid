from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.conditional_matd3 import (  # noqa: E402
    ConditionalMATD3,
    ConditionalMATD3Trainer,
    JointReplayBuffer,
)
from algorithms.conditional_matd3.config import (  # noqa: E402
    DEFAULT_ALGORITHM_CONFIG,
    DEFAULT_ENV_CONFIG,
)
from algorithms.conditional_matd3.networks import ConditionalActor  # noqa: E402
from envs.multi_microgrid_env import MultiMicrogridEnv  # noqa: E402
from utils.config import (  # noqa: E402
    compose_experiment_config,
    load_algorithm_config,
    runtime_config_snapshot,
)


DEFAULT_ALGO_CONFIG = DEFAULT_ALGORITHM_CONFIG
EPS = 1e-9


@dataclass
class BestResponseResult:
    focal_agent: str
    restart: int
    seed: int
    initialization: str
    best_validation_cost: float
    best_episode: int
    policy_state: dict[str, Any]
    training_rows: list[dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc empirical epsilon-Nash verification using unilateral "
            "best-response retraining."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Deprecated alias for --env-config."
    )
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--algo-config", type=Path, default=DEFAULT_ALGO_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--focal-agents", nargs="+", default=None)
    parser.add_argument("--br-episodes", type=int, default=None)
    parser.add_argument("--br-restarts", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--validation-days", type=int, default=None)
    parser.add_argument("--test-days", type=int, default=None)
    parser.add_argument("--epsilon-threshold-pct", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--restart-init",
        choices=("mixed", "equilibrium", "random"),
        default=None,
        help="mixed: restart 0 uses the learned focal actor; later restarts use random focal actors.",
    )
    parser.add_argument("--br-warmup-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--updates-per-step", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    return parser


def _resolve_nash_arguments(
    args: argparse.Namespace,
    algorithm_document: Mapping[str, Any],
) -> argparse.Namespace:
    verification = algorithm_document.get("nash_verification", {})
    training = algorithm_document.get("training", {})
    defaults = {
        "br_episodes": 1000,
        "br_restarts": 5,
        "validation_interval": 50,
        "validation_days": 10,
        "test_days": 30,
        "epsilon_threshold_pct": 1.0,
        "restart_init": "mixed",
    }
    for name, fallback in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, verification.get(name, fallback))
    if args.focal_agents is None:
        configured_agents = verification.get("focal_agents")
        args.focal_agents = list(configured_agents) if configured_agents else None
    if args.seed is None:
        args.seed = int(training.get("seed", 42))
    if args.device is None:
        args.device = str(training.get("device", "auto"))
    return args


def _resolve_device(requested: str) -> torch.device:
    if requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _torch_load(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.6
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def _extract_policy_state(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(checkpoint.get("actors"), Mapping):
        return {"actors": checkpoint["actors"]}
    algorithm = checkpoint.get("algorithm")
    if isinstance(algorithm, Mapping) and isinstance(algorithm.get("actors"), Mapping):
        return {"actors": algorithm["actors"]}
    raise ValueError(
        "Checkpoint contains neither top-level actors (best_model.pt) nor "
        "algorithm.actors (training checkpoint)."
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    columns = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _select_days(env: MultiMicrogridEnv, count: int) -> list[str]:
    if count <= 0:
        raise ValueError("Day counts must be positive.")
    available = env.data.valid_days
    if not available:
        raise ValueError(f"The {env.split} split has no valid days.")
    count = min(int(count), len(available))
    indices = np.linspace(0, len(available) - 1, num=count, dtype=int)
    return [str(available[index].date()) for index in np.unique(indices)]


def _build_algorithm(
    env: MultiMicrogridEnv,
    device: torch.device,
    seed: int,
) -> ConditionalMATD3:
    algorithm = ConditionalMATD3(
        env.agent_ids,
        env.observation_dims,
        env.global_state_dim,
        env.config,
        device=device,
    )
    algorithm.rng = np.random.default_rng(int(seed))
    return algorithm


def _build_trainer(
    env: MultiMicrogridEnv,
    algorithm: ConditionalMATD3,
    seed: int,
    *,
    batch_size: int | None = None,
    warmup_steps: int | None = None,
    updates_per_step: int | None = None,
    replay_capacity: int | None = None,
) -> ConditionalMATD3Trainer:
    algo_cfg = env.config["algorithm"]
    replay = JointReplayBuffer(
        capacity=int(replay_capacity or algo_cfg.get("replay_capacity", 500000)),
        agent_ids=env.agent_ids,
        observation_dims=env.observation_dims,
        state_dim=env.global_state_dim,
        action_dim=env.action_dim,
        seed=int(seed),
    )
    return ConditionalMATD3Trainer(
        env=env,
        algorithm=algorithm,
        replay=replay,
        batch_size=int(batch_size or algo_cfg.get("batch_size", 256)),
        warmup_steps=int(
            algo_cfg.get("warmup_steps", 5000)
            if warmup_steps is None else warmup_steps
        ),
        updates_per_step=int(
            algo_cfg.get("updates_per_step", 1)
            if updates_per_step is None else updates_per_step
        ),
    )


def _episode_metric_row(
    result: Mapping[str, Any],
    agent_ids: Sequence[str],
    *,
    split: str,
    day: str,
    seed: int,
    policy: str,
    focal_agent: str = "",
    restart: int = -1,
) -> dict[str, Any]:
    by = result["by_microgrid"]
    row: dict[str, Any] = {
        "policy": policy,
        "focal_agent": focal_agent,
        "restart": int(restart),
        "split": split,
        "day": day,
        "seed": int(seed),
        "system_cost": float(sum(float(by[mg]["cost"]) for mg in agent_ids)),
        "system_reward": float(sum(float(by[mg]["reward"]) for mg in agent_ids)),
        "p2p_kwh": float(result["market_trade_kwh"]),
        "grid_import_kwh": float(result["grid_import_kwh"]),
        "grid_export_kwh": float(result["grid_export_kwh"]),
        "carbon_kg": float(sum(float(by[mg]["carbon_kg"]) for mg in agent_ids)),
        "carbon_cost": float(sum(float(by[mg]["carbon_cost"]) for mg in agent_ids)),
        "renewable_curtailment_kwh": float(result["renewable_curtailment_kwh"]),
    }
    for mg in agent_ids:
        row[f"{mg}_cost"] = float(by[mg]["cost"])
        row[f"{mg}_reward"] = float(by[mg]["reward"])
        row[f"{mg}_p2p_kwh"] = float(by[mg]["p2p_kwh"])
        row[f"{mg}_grid_import_kwh"] = float(by[mg]["grid_import_kwh"])
        row[f"{mg}_grid_export_kwh"] = float(by[mg]["grid_export_kwh"])
        row[f"{mg}_carbon_kg"] = float(by[mg]["carbon_kg"])
        row[f"{mg}_carbon_cost"] = float(by[mg]["carbon_cost"])
        row[f"{mg}_generator_cost"] = float(by[mg]["generator_cost"])
        row[f"{mg}_battery_cost"] = float(by[mg]["battery_cost"])
    numeric = [float(value) for value in row.values() if isinstance(value, (int, float))]
    if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
        raise RuntimeError("Evaluation produced a non-finite metric.")
    return row


def evaluate_policy(
    algorithm: ConditionalMATD3,
    env: MultiMicrogridEnv,
    days: Sequence[str],
    seed: int,
    *,
    policy: str,
    focal_agent: str = "",
    restart: int = -1,
    environment_deterministic: bool = False,
) -> list[dict[str, Any]]:
    if not days:
        raise ValueError("Evaluation requires at least one fixed day.")
    trainer = _build_trainer(env, algorithm, seed, batch_size=1, replay_capacity=1)
    rows = []
    for index, day in enumerate(days):
        episode_seed = int(seed) + index
        result = trainer.run_episode(
            day=day,
            training=False,
            deterministic_eval=False,
            environment_deterministic=environment_deterministic,
            seed=episode_seed,
        )
        rows.append(
            _episode_metric_row(
                result,
                env.agent_ids,
                split=env.split,
                day=day,
                seed=episode_seed,
                policy=policy,
                focal_agent=focal_agent,
                restart=restart,
            )
        )
    return rows


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _std(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.std([float(row[key]) for row in rows]))


@torch.no_grad()
def _collect_probe_observations(
    algorithm: ConditionalMATD3,
    env: MultiMicrogridEnv,
    day: str,
    seed: int,
) -> dict[str, np.ndarray]:
    observations = {mg: [] for mg in env.agent_ids}
    obs, _ = env.reset(day=day, seed=seed, deterministic=True)
    done = False
    while not done:
        for mg in env.agent_ids:
            observations[mg].append(np.asarray(obs[mg], dtype=np.float32).copy())
        actions, _ = algorithm.select_actions(obs, explore=False, warmup=False)
        obs, _, done, _ = env.step(actions)
    return {
        mg: np.stack(values).astype(np.float32, copy=False)
        for mg, values in observations.items()
    }


@torch.no_grad()
def _probe_actions(
    algorithm: ConditionalMATD3,
    focal_agent: str,
    probe_observations: np.ndarray,
) -> np.ndarray:
    obs = torch.as_tensor(
        probe_observations,
        dtype=torch.float32,
        device=algorithm.device,
    )
    outputs = algorithm.actors[focal_agent](obs)
    return ConditionalActor.hard_action(outputs).cpu().numpy()


@torch.no_grad()
def _policy_action_l2(
    algorithm: ConditionalMATD3,
    focal_agent: str,
    probe_observations: np.ndarray,
    reference_actions: np.ndarray,
) -> float:
    actions = _probe_actions(algorithm, focal_agent, probe_observations)
    return float(np.linalg.norm(actions - reference_actions, axis=1).mean())


@torch.no_grad()
def _reference_probe_actions(
    algorithm: ConditionalMATD3,
    focal_agent: str,
    probe_observations: np.ndarray,
) -> np.ndarray:
    return _probe_actions(algorithm, focal_agent, probe_observations)


def _restart_initialization(mode: str, restart: int) -> str:
    if mode == "mixed":
        return "equilibrium" if restart == 0 else "random"
    return mode


def train_best_response(
    runtime_config: Mapping[str, Any],
    baseline_policy: Mapping[str, Any],
    focal_agent: str,
    restart: int,
    args: argparse.Namespace,
    device: torch.device,
    validation_days: Sequence[str],
    probe_observations: np.ndarray,
    reference_probe_actions: np.ndarray,
    baseline_validation_cost: float,
    validation_seed_base: int,
) -> BestResponseResult:
    restart_seed = int(args.seed) + 10_000 * (restart + 1) + 1_000 * (
        int(focal_agent.removeprefix("mg"))
        if focal_agent.removeprefix("mg").isdigit() else 1
    )
    _seed_everything(restart_seed)
    train_env = MultiMicrogridEnv(copy.deepcopy(runtime_config), split="train")
    val_env = MultiMicrogridEnv(copy.deepcopy(runtime_config), split="val")
    algorithm = _build_algorithm(train_env, device, restart_seed)
    random_focal_state = copy.deepcopy(algorithm.actors[focal_agent].state_dict())
    algorithm.load_policy_state_dict(baseline_policy, sync_targets=True)
    equilibrium_focal_state = copy.deepcopy(
        algorithm.actors[focal_agent].state_dict()
    )
    initialization = _restart_initialization(args.restart_init, restart)
    if initialization == "random":
        algorithm.actors[focal_agent].load_state_dict(random_focal_state)
        algorithm.target_actors[focal_agent].load_state_dict(random_focal_state)

    trainer = _build_trainer(
        train_env,
        algorithm,
        restart_seed,
        batch_size=args.batch_size,
        warmup_steps=args.br_warmup_steps,
        updates_per_step=args.updates_per_step,
        replay_capacity=args.replay_capacity,
    )
    train_days = [str(day.date()) for day in train_env.data.valid_days]
    day_rng = np.random.default_rng(restart_seed)
    day_rng.shuffle(train_days)
    rows: list[dict[str, Any]] = []
    best_validation_cost = float(baseline_validation_cost)
    best_episode = 0
    best_actor_state: dict[str, torch.Tensor] = copy.deepcopy(equilibrium_focal_state)

    previous_validation_probe_actions = _probe_actions(
        algorithm,
        focal_agent,
        probe_observations,
    )

    for episode in range(1, int(args.br_episodes) + 1):
        day = train_days[(episode - 1) % len(train_days)]
        result = trainer.run_episode(
            day=day,
            training=True,
            seed=restart_seed + episode,
            active_agents={focal_agent},
        )
        updates = result["last_update"]
        validation_available = (
            episode % int(args.validation_interval) == 0
            or episode == int(args.br_episodes)
        )
        validation_cost = 0.0
        validation_reward = 0.0
        validation_system_cost = 0.0
        br_policy_step_change_l2 = 0.0
        if validation_available:
            validation_rows = evaluate_policy(
                algorithm,
                val_env,
                validation_days,
                validation_seed_base,
                policy="br_validation",
                focal_agent=focal_agent,
                restart=restart,
                environment_deterministic=False,
            )
            validation_cost = _mean(validation_rows, f"{focal_agent}_cost")
            validation_reward = _mean(validation_rows, f"{focal_agent}_reward")
            validation_system_cost = _mean(validation_rows, "system_cost")

            current_probe_actions = _probe_actions(
                algorithm,
                focal_agent,
                probe_observations,
            )
            br_policy_step_change_l2 = float(
                np.linalg.norm(
                    current_probe_actions - previous_validation_probe_actions,
                    axis=1,
                ).mean()
            )
            previous_validation_probe_actions = current_probe_actions.copy()

            if validation_cost < best_validation_cost:
                best_validation_cost = validation_cost
                best_episode = episode
                best_actor_state = copy.deepcopy(
                    algorithm.actors[focal_agent].state_dict()
                )

        rows.append(
            {
                "agent": focal_agent,
                "restart": int(restart),
                "seed": int(restart_seed),
                "initialization": initialization,
                "episode": int(episode),
                "env_steps": int(result["env_steps"]),
                "train_agent_cost": float(result["by_microgrid"][focal_agent]["cost"]),
                "train_agent_reward": float(result["by_microgrid"][focal_agent]["reward"]),
                "validation_available": int(validation_available),
                "validation_agent_cost": float(validation_cost),
                "validation_agent_reward": float(validation_reward),
                "validation_system_cost": float(validation_system_cost),
                "system_cost": float(
                    sum(
                        float(result["by_microgrid"][mg]["cost"])
                        for mg in train_env.agent_ids
                    )
                ),
                "actor_loss": float(updates.get(f"{focal_agent}/actor_loss", 0.0)),
                "actor_grad_norm": float(
                    updates.get(f"{focal_agent}/actor_grad_norm", 0.0)
                ),
                "critic_loss": float(updates.get(f"{focal_agent}/critic_loss", 0.0)),
                "q_mean": float(updates.get(f"{focal_agent}/q_mean", 0.0)),
                "q_disagreement": float(
                    updates.get(f"{focal_agent}/q_disagreement", 0.0)
                ),
                "policy_deviation_from_baseline_l2": _policy_action_l2(
                    algorithm,
                    focal_agent,
                    probe_observations,
                    reference_probe_actions,
                ),
                "br_policy_step_change_l2": float(br_policy_step_change_l2),
            }
        )

    if best_actor_state is None or not math.isfinite(best_validation_cost):
        raise RuntimeError(f"No validation checkpoint was selected for {focal_agent} restart {restart}.")
    algorithm.actors[focal_agent].load_state_dict(best_actor_state)
    algorithm.target_actors[focal_agent].load_state_dict(best_actor_state)
    return BestResponseResult(
        focal_agent=focal_agent,
        restart=restart,
        seed=restart_seed,
        initialization=initialization,
        best_validation_cost=float(best_validation_cost),
        best_episode=int(best_episode),
        policy_state=algorithm.get_policy_state_dict(),
        training_rows=rows,
    )


def _save_br_policy(
    path: Path,
    result: BestResponseResult,
    source_checkpoint: Path,
) -> None:
    payload = {
        "format_version": 1,
        "checkpoint_type": "unilateral_best_response_policy",
        "actors": result.policy_state["actors"],
        "focal_agent": result.focal_agent,
        "restart": result.restart,
        "seed": result.seed,
        "initialization": result.initialization,
        "best_validation_cost": result.best_validation_cost,
        "best_episode": result.best_episode,
        "best_br_is_baseline_policy": bool(result.best_episode == 0),
        "source_checkpoint": str(source_checkpoint),
        "selection_split": "val",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _summary_row(
    focal_agent: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    br_rows: Sequence[Mapping[str, Any]],
    selected: BestResponseResult,
    restart_count: int,
    epsilon_threshold_pct: float,
) -> dict[str, Any]:
    baseline_cost = _mean(baseline_rows, f"{focal_agent}_cost")
    br_cost = _mean(br_rows, f"{focal_agent}_cost")
    signed_gain_abs = baseline_cost - br_cost
    signed_gain_pct = 100.0 * signed_gain_abs / (abs(baseline_cost) + EPS)
    profitable_gain_abs = max(0.0, signed_gain_abs)
    profitable_gain_pct = 100.0 * profitable_gain_abs / (abs(baseline_cost) + EPS)
    baseline_system = _mean(baseline_rows, "system_cost")
    br_system = _mean(br_rows, "system_cost")
    externality = br_system - baseline_system
    return {
        "agent": focal_agent,
        "baseline_cost_mean": baseline_cost,
        "baseline_cost_std": _std(baseline_rows, f"{focal_agent}_cost"),
        "br_cost_mean": br_cost,
        "br_cost_std": _std(br_rows, f"{focal_agent}_cost"),
        "unilateral_gain_signed_abs": signed_gain_abs,
        "unilateral_gain_signed_pct": signed_gain_pct,
        "unilateral_gain_abs": profitable_gain_abs,
        "unilateral_gain_pct": profitable_gain_pct,
        "baseline_reward_mean": _mean(baseline_rows, f"{focal_agent}_reward"),
        "baseline_reward_std": _std(baseline_rows, f"{focal_agent}_reward"),
        "br_reward_mean": _mean(br_rows, f"{focal_agent}_reward"),
        "br_reward_std": _std(br_rows, f"{focal_agent}_reward"),
        "baseline_system_cost_mean": baseline_system,
        "br_system_cost_mean": br_system,
        "system_externality_abs": externality,
        "system_externality_pct": 100.0 * externality / (abs(baseline_system) + EPS),
        "baseline_p2p_kwh_mean": _mean(baseline_rows, "p2p_kwh"),
        "br_p2p_kwh_mean": _mean(br_rows, "p2p_kwh"),
        "baseline_grid_import_kwh_mean": _mean(baseline_rows, "grid_import_kwh"),
        "br_grid_import_kwh_mean": _mean(br_rows, "grid_import_kwh"),
        "baseline_grid_export_kwh_mean": _mean(baseline_rows, "grid_export_kwh"),
        "br_grid_export_kwh_mean": _mean(br_rows, "grid_export_kwh"),
        "baseline_carbon_kg_mean": _mean(baseline_rows, "carbon_kg"),
        "br_carbon_kg_mean": _mean(br_rows, "carbon_kg"),
        "best_br_seed": int(selected.seed),
        "best_br_restart": int(selected.restart),
        "best_br_initialization": selected.initialization,
        "best_br_episode": int(selected.best_episode),
        "best_br_validation_cost": float(selected.best_validation_cost),
        "best_br_is_baseline_policy": int(selected.best_episode == 0),
        "br_restart_count": int(restart_count),
        "br_beats_or_matches_baseline": int(signed_gain_abs >= -1e-8),
        "nash_gap_pct": profitable_gain_pct,
        "is_epsilon_nash": int(profitable_gain_pct <= float(epsilon_threshold_pct)),
    }


def _generate_plots(
    output: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    baseline_validation_rows: Sequence[Mapping[str, Any]],
    epsilon_threshold_pct: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    agents = [str(row["agent"]) for row in summary_rows]
    gains = [float(row["unilateral_gain_pct"]) for row in summary_rows]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    bars = axis.bar(agents, gains, color="#27628D", edgecolor="#183B56", linewidth=0.8)
    axis.axhline(
        float(epsilon_threshold_pct),
        color="#5E6875",
        linestyle="--",
        linewidth=1.2,
        label=f"epsilon threshold = {epsilon_threshold_pct:.2f}%",
    )
    axis.set(
        title="Unilateral cost improvement / empirical Nash gap",
        xlabel="Focal microgrid",
        ylabel="Profitable unilateral cost improvement (%)",
    )
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", color="#D5DAE0", linewidth=0.7, alpha=0.8)
    axis.legend(frameon=False)
    for bar, value in zip(bars, gains):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output / "nash_unilateral_gain.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    for agent in agents:
        agent_summary = next((r for r in summary_rows if r["agent"] == agent), None)
        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        restarts = sorted(
            {int(row["restart"]) for row in training_rows if row["agent"] == agent}
        )
        for restart in restarts:
            selected_rows = [
                row
                for row in training_rows
                if row["agent"] == agent
                and int(row["restart"]) == restart
                and int(row["validation_available"]) == 1
            ]
            axis.plot(
                [int(row["episode"]) for row in selected_rows],
                [float(row["validation_agent_cost"]) for row in selected_rows],
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=f"restart {restart}",
            )
        baseline_cost = _mean(baseline_validation_rows, f"{agent}_cost")
        axis.axhline(
            baseline_cost,
            color="#20242A",
            linestyle="--",
            linewidth=1.2,
            label="baseline validation cost",
        )
        if agent_summary and int(agent_summary.get("best_br_is_baseline_policy", 0)) == 1:
            axis.text(
                0.03,
                0.06,
                "Baseline remains best response candidate",
                transform=axis.transAxes,
                fontsize=8.5,
                color="#1B3A4B",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#E8F1F5",
                    edgecolor="#B0C8D8",
                    alpha=0.9,
                ),
            )
        axis.set(
            title=f"{agent.upper()} unilateral best-response retraining",
            xlabel="BR training episode",
            ylabel="Focal-agent validation cost",
        )
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="y", color="#D5DAE0", linewidth=0.7, alpha=0.8)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(
            output / f"nash_best_response_{agent}.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.close(fig)


def run_nash_verification(args: argparse.Namespace) -> dict[str, Any]:
    algorithm_config_path = args.algo_config.resolve()
    if not algorithm_config_path.is_file():
        raise FileNotFoundError(
            f"Algorithm config file does not exist: {algorithm_config_path}"
        )
    algorithm_document = load_algorithm_config(algorithm_config_path)
    args = _resolve_nash_arguments(args, algorithm_document)
    config_path = (args.config or args.env_config).resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Environment config file does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    for name in (
        "br_episodes",
        "br_restarts",
        "validation_interval",
        "validation_days",
        "test_days",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if float(args.epsilon_threshold_pct) < 0.0:
        raise ValueError("--epsilon-threshold-pct must be non-negative.")
    if args.br_warmup_steps is not None and args.br_warmup_steps < 0:
        raise ValueError("--br-warmup-steps must be non-negative.")
    for name in ("batch_size", "updates_per_step", "replay_capacity"):
        value = getattr(args, name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")

    device = _resolve_device(args.device)
    _seed_everything(args.seed)
    runtime_config, algorithm_document = compose_experiment_config(
        config_path,
        algorithm_config_path,
    )
    train_env = MultiMicrogridEnv(copy.deepcopy(runtime_config), split="train")
    val_env = MultiMicrogridEnv(copy.deepcopy(runtime_config), split="val")
    focal_agents = list(args.focal_agents or train_env.agent_ids)
    unknown = set(focal_agents) - set(train_env.agent_ids)
    if unknown:
        raise ValueError(f"Unknown focal agents: {sorted(unknown)}")
    if len(set(focal_agents)) != len(focal_agents):
        raise ValueError("--focal-agents cannot contain duplicates.")

    output = (
        args.output.resolve()
        if args.output is not None
        else checkpoint_path.parent / "nash"
    )
    output.mkdir(parents=True, exist_ok=True)
    saved_configs = output / "configs"
    saved_configs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, saved_configs / "environment.yaml")
    shutil.copy2(algorithm_config_path, saved_configs / "algorithm.yaml")
    with (saved_configs / "effective_runtime.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            runtime_config_snapshot(runtime_config),
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    checkpoint = _torch_load(checkpoint_path, device)
    baseline_policy = _extract_policy_state(checkpoint)
    baseline_algorithm = _build_algorithm(train_env, device, args.seed)
    baseline_algorithm.load_policy_state_dict(baseline_policy, sync_targets=True)

    validation_seed_base = int(args.seed) + 2_000_000
    test_seed_base = int(args.seed) + 3_000_000

    validation_days = _select_days(val_env, args.validation_days)
    baseline_validation_rows = evaluate_policy(
        baseline_algorithm,
        val_env,
        validation_days,
        validation_seed_base,
        policy="baseline",
        environment_deterministic=False,
    )
    baseline_validation_costs = {
        mg: _mean(
            baseline_validation_rows,
            f"{mg}_cost",
        )
        for mg in focal_agents
    }
    probe_by_agent = _collect_probe_observations(
        baseline_algorithm,
        val_env,
        validation_days[0],
        int(args.seed) + 4_000_000,
    )
    reference_actions = {
        mg: _probe_actions(
            baseline_algorithm,
            mg,
            probe_by_agent[mg],
        )
        for mg in focal_agents
    }

    all_training_rows: list[dict[str, Any]] = []
    selected_results: dict[str, BestResponseResult] = {}
    for focal_agent in focal_agents:
        restart_results = []
        for restart in range(int(args.br_restarts)):
            result = train_best_response(
                runtime_config,
                baseline_policy,
                focal_agent,
                restart,
                args,
                device,
                validation_days,
                probe_by_agent[focal_agent],
                reference_actions[focal_agent],
                baseline_validation_cost=baseline_validation_costs[focal_agent],
                validation_seed_base=validation_seed_base,
            )
            restart_results.append(result)
            all_training_rows.extend(result.training_rows)
            print(
                f"BR {focal_agent} restart={restart} init={result.initialization} "
                f"best_val_cost={result.best_validation_cost:.6f} "
                f"episode={result.best_episode}",
                flush=True,
            )
        # Select the restart using validation only. Test remains a final holdout.
        selected = min(restart_results, key=lambda item: item.best_validation_cost)
        selected_results[focal_agent] = selected
        _save_br_policy(
            output / f"br_{focal_agent}_best.pt",
            selected,
            checkpoint_path,
        )

    # The test split is not opened until every BR checkpoint/restart has been
    # selected exclusively from validation performance.
    test_env = MultiMicrogridEnv(copy.deepcopy(runtime_config), split="test")
    test_days = _select_days(test_env, args.test_days)
    baseline_test_rows = evaluate_policy(
        baseline_algorithm,
        test_env,
        test_days,
        test_seed_base,
        policy="baseline",
        environment_deterministic=False,
    )
    _write_csv(output / "baseline_test_metrics.csv", baseline_test_rows)

    summary_rows: list[dict[str, Any]] = []
    all_br_test_rows: list[dict[str, Any]] = []
    for focal_agent in focal_agents:
        selected = selected_results[focal_agent]
        br_test_env = MultiMicrogridEnv(copy.deepcopy(runtime_config), split="test")
        br_algorithm = _build_algorithm(br_test_env, device, selected.seed)
        br_algorithm.load_policy_state_dict(selected.policy_state, sync_targets=True)
        br_test_rows = evaluate_policy(
            br_algorithm,
            br_test_env,
            test_days,
            test_seed_base,
            policy="best_response",
            focal_agent=focal_agent,
            restart=selected.restart,
            environment_deterministic=False,
        )
        all_br_test_rows.extend(br_test_rows)
        summary_rows.append(
            _summary_row(
                focal_agent,
                baseline_test_rows,
                br_test_rows,
                selected,
                int(args.br_restarts),
                float(args.epsilon_threshold_pct),
            )
        )

    _write_csv(output / "br_training_log.csv", all_training_rows)
    _write_csv(output / "br_test_metrics.csv", all_br_test_rows)
    _write_csv(output / "nash_summary.csv", summary_rows)
    epsilon_nash_pct = max(float(row["nash_gap_pct"]) for row in summary_rows)
    worst_agent = max(summary_rows, key=lambda row: float(row["nash_gap_pct"]))["agent"]
    summary = {
        "verification": "empirical_epsilon_nash_unilateral_best_response",
        "epsilon_nash_pct": epsilon_nash_pct,
        "worst_agent": worst_agent,
        "epsilon_threshold_pct": float(args.epsilon_threshold_pct),
        "passes_empirical_epsilon_nash": bool(
            epsilon_nash_pct <= float(args.epsilon_threshold_pct)
        ),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "algorithm_config": str(algorithm_config_path),
        "device": str(device),
        "seed": int(args.seed),
        "focal_agents": list(focal_agents),
        "br_episodes": int(args.br_episodes),
        "br_restarts": int(args.br_restarts),
        "restart_initialization": str(args.restart_init),
        "train_split": "train",
        "validation_split": "val",
        "test_split": "test",
        "validation_days": list(validation_days),
        "test_days": list(test_days),
        "restart_selection": "minimum focal-agent validation cost",
        "test_selection_used": False,
        "agent_results": summary_rows,
        "interpretation_caveat": (
            "This is an empirical epsilon-Nash verification under the tested "
            "scenarios and approximate neural best-response search, not a formal "
            "proof of an exact Nash equilibrium."
        ),
    }
    _write_json(output / "nash_summary.json", summary)
    _generate_plots(
        output,
        summary_rows,
        all_training_rows,
        baseline_validation_rows,
        float(args.epsilon_threshold_pct),
    )
    print(f"Empirical epsilon-Nash gap: {epsilon_nash_pct:.6f}% ({worst_agent})", flush=True)
    print(
        f"Pass threshold {float(args.epsilon_threshold_pct):.6f}%: "
        f"{summary['passes_empirical_epsilon_nash']}",
        flush=True,
    )
    print(f"Output: {output}", flush=True)
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_nash_verification(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
