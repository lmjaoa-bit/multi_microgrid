from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


def _rolling_mean(rows: Sequence[Mapping[str, Any]], column: str, window: int) -> float | None:
    values = [float(row[column]) for row in rows[-window:] if column in row]
    return float(np.mean(values)) if values else None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def build_dashboard_data(
    current_row: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
    agent_ids: Sequence[str],
    *,
    current_episode: int,
    total_episodes: int,
    start_episode: int,
    device: str,
    warmup_steps: int,
    update_count: int,
    elapsed_seconds: float,
    rolling_window: int,
    available_metrics: set[str] | None = None,
    validation: Mapping[str, Any] | None = None,
    status: str = "RUNNING",
    reason: str | None = None,
) -> dict[str, Any]:
    """Build display-only dashboard data from already produced training records."""

    available = available_metrics or set()
    session_completed = max(0, current_episode - start_episode + 1)
    remaining_episodes = max(0, total_episodes - current_episode)
    eta = (
        elapsed_seconds * remaining_episodes / session_completed
        if session_completed > 0
        else None
    )
    env_steps = int(current_row["env_steps"]) if current_row is not None else 0
    replay_size = int(current_row["replay_size"]) if current_row is not None else 0
    progress = 100.0 * current_episode / total_episodes if total_episodes else 0.0

    def current(name: str) -> float | None:
        return float(current_row[name]) if current_row is not None and name in current_row else None

    def learned(name: str) -> float | None:
        return current(name) if name in available else None

    critic_values = [
        learned(f"{mg.upper()}_critic_loss")
        for mg in agent_ids
    ]
    actor_values = [
        learned(f"{mg.upper()}_actor_loss")
        for mg in agent_ids
    ]
    mean_critic = [value for value in critic_values if value is not None]
    mean_actor = [value for value in actor_values if value is not None]

    system_columns = {
        "total_reward": "system_total_reward",
        "total_cost": "system_total_cost",
        "market_trade_kwh": "market_trade_kwh",
        "renewable_curtailment_kwh": "renewable_curtailment_kwh",
        "grid_import_kwh": "grid_import_kwh",
        "grid_export_kwh": "grid_export_kwh",
    }
    system = {
        name: {
            "current": current(column),
            "rolling_mean": _rolling_mean(rows, column, rolling_window),
        }
        for name, column in system_columns.items()
    }
    system["mean_critic_loss"] = {
        "current": float(np.mean(mean_critic)) if mean_critic else None,
        "rolling_mean": None,
    }
    system["mean_actor_loss"] = {
        "current": float(np.mean(mean_actor)) if mean_actor else None,
        "rolling_mean": None,
    }

    microgrids: dict[str, dict[str, Any]] = {}
    for mg in agent_ids:
        label = mg.upper()
        microgrids[label] = {
            "reward": {
                "current": current(f"{label}_reward"),
                "rolling_mean": _rolling_mean(rows, f"{label}_reward", rolling_window),
            },
            "cost": current(f"{label}_cost"),
            "p2p_kwh": current(f"{label}_p2p_kwh"),
            "p_on": learned(f"{label}_p_on"),
            "q_on": learned(f"{label}_q_on"),
            "q_off": learned(f"{label}_q_off"),
            "delta_q": learned(f"{label}_delta_q"),
            "commitment_loss": learned(f"{label}_commitment_loss"),
            "common_loss": learned(f"{label}_common_loss"),
            "generator_loss": learned(f"{label}_generator_loss"),
            "critic_loss": learned(f"{label}_critic_loss"),
        }

    return {
        "status": status,
        "reason": reason,
        "overview": {
            "current_episode": int(current_episode),
            "total_episodes": int(total_episodes),
            "progress_percent": progress,
            "env_steps": env_steps,
            "replay_size": replay_size,
            "device": str(device).upper(),
            "elapsed": _format_duration(elapsed_seconds),
            "eta": _format_duration(eta),
            "warmup": (
                "finished"
                if env_steps >= warmup_steps
                else f"{warmup_steps - env_steps} steps remaining"
            ),
            "update_count": int(update_count),
        },
        "system": system,
        "microgrids": microgrids,
        "validation": dict(validation or {}),
        "rolling_window": int(rolling_window),
    }


def render_dashboard(data: Mapping[str, Any]) -> RenderableType:
    overview = data["overview"]
    status = str(data["status"])
    status_style = {
        "RUNNING": "bold cyan",
        "TRAINING COMPLETED": "bold green",
        "TRAINING ABORTED": "bold red",
    }.get(status, "bold yellow")

    training = Table.grid(expand=True, padding=(0, 1))
    training.add_column(style="bold")
    training.add_column()
    training.add_column(style="bold")
    training.add_column()
    training.add_row("Status", Text(status, style=status_style), "Device", overview["device"])
    training.add_row(
        "Episode",
        f"{overview['current_episode']} / {overview['total_episodes']}",
        "Progress",
        f"{overview['progress_percent']:.1f}%",
    )
    training.add_row(
        "Progress bar",
        ProgressBar(
            total=max(1, overview["total_episodes"]),
            completed=overview["current_episode"],
            width=28,
        ),
        "Updates",
        str(overview["update_count"]),
    )
    training.add_row(
        "Env steps",
        str(overview["env_steps"]),
        "Replay",
        str(overview["replay_size"]),
    )
    training.add_row("Elapsed", overview["elapsed"], "ETA", overview["eta"])
    training.add_row("Warmup", overview["warmup"], "", "")
    if data.get("reason"):
        training.add_row("Reason", Text(str(data["reason"]), style="red"), "", "")

    system = data["system"]
    window = data["rolling_window"]
    system_table = Table(expand=True, box=None, show_header=True)
    system_table.add_column("Metric", style="bold")
    system_table.add_column("Current", justify="right")
    system_table.add_column(f"MA{window}", justify="right")
    labels = {
        "total_reward": "Total reward",
        "total_cost": "Total cost",
        "market_trade_kwh": "P2P trade (kWh)",
        "renewable_curtailment_kwh": "Curtailment (kWh)",
        "grid_import_kwh": "Grid import (kWh)",
        "grid_export_kwh": "Grid export (kWh)",
        "mean_critic_loss": "Mean critic loss",
        "mean_actor_loss": "Mean actor loss",
    }
    for key, label in labels.items():
        metric = system[key]
        system_table.add_row(
            label,
            _format_number(metric["current"]),
            _format_number(metric["rolling_mean"]),
        )

    mg_table = Table(expand=True, show_lines=False)
    for heading in ("MG", "Reward", f"Reward MA{window}", "Cost", "P2P kWh", "p_on", "DeltaQ", "Q_ON", "Q_OFF"):
        mg_table.add_column(heading, justify="right" if heading != "MG" else "left")
    for label, metrics in data["microgrids"].items():
        mg_table.add_row(
            label,
            _format_number(metrics["reward"]["current"]),
            _format_number(metrics["reward"]["rolling_mean"]),
            _format_number(metrics["cost"], 2),
            _format_number(metrics["p2p_kwh"], 2),
            _format_number(metrics["p_on"]),
            _format_number(metrics["delta_q"]),
            _format_number(metrics["q_on"]),
            _format_number(metrics["q_off"]),
        )

    validation = data.get("validation", {})
    validation_table = Table.grid(expand=True, padding=(0, 1))
    validation_table.add_column(style="bold")
    validation_table.add_column(justify="right")
    validation_table.add_column(style="bold")
    validation_table.add_column(justify="right")
    validation_table.add_row(
        "Last val cost",
        _format_number(validation.get("last_validation_cost"), 2),
        "Best val cost",
        _format_number(validation.get("best_validation_cost"), 2),
    )
    validation_table.add_row(
        "Best episode",
        str(validation.get("best_episode") or "-"),
        "Next validation",
        (
            f"EP {validation['next_validation_episode']}"
            if validation.get("next_validation_episode") is not None
            else "-"
        ),
    )
    if validation.get("new_best"):
        validation_table.add_row(
            "State",
            Text("NEW BEST", style="bold green"),
            "",
            "",
        )

    loss_table = Table(expand=True, show_lines=False)
    for heading in ("MG", "Critic", "Commitment", "Common", "Generator"):
        loss_table.add_column(heading, justify="right" if heading != "MG" else "left")
    for label, metrics in data["microgrids"].items():
        loss_table.add_row(
            label,
            _format_number(metrics["critic_loss"]),
            _format_number(metrics["commitment_loss"]),
            _format_number(metrics["common_loss"]),
            _format_number(metrics["generator_loss"]),
        )

    return Group(
        Panel(training, title="Conditional MATD3 Training", border_style="cyan"),
        Panel(system_table, title="System", border_style="blue"),
        Panel(validation_table, title="Validation", border_style="green"),
        Panel(mg_table, title="Microgrids", border_style="magenta"),
        Panel(loss_table, title="Learning losses", border_style="yellow"),
    )
