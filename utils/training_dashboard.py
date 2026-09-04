from __future__ import annotations

import math
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _format_time(seconds: float) -> str:
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return "calculating..."
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


class AlgorithmTrainingDashboard:
    """Rich terminal dashboard shared by registered multi-agent algorithms."""

    def __init__(
        self,
        total_episodes: int,
        warmup_steps: int = 1000,
        replay_capacity: int = 200_000,
        rolling_windows: Sequence[int] = (10, 50),
        refresh_hz: float = 4.0,
        enabled: bool = True,
        agent_ids: Sequence[str] = ("mg1", "mg2", "mg3"),
        initial_soc: Mapping[str, float] | None = None,
        soc_min: Mapping[str, float] | None = None,
        soc_tolerance: float = 0.03,
        algorithm_name: str = "MATD3",
    ) -> None:
        self.total_episodes = max(1, int(total_episodes))
        self.warmup_steps = int(warmup_steps)
        self.replay_capacity = int(replay_capacity)
        self.rolling_windows = list(rolling_windows)
        self.refresh_hz = float(refresh_hz)
        self.enabled = bool(enabled and sys.stdout.isatty())
        self.agent_ids = list(agent_ids)
        self.initial_soc = dict(initial_soc or {"mg1": 0.55, "mg2": 0.70, "mg3": 0.35})
        self.soc_min = dict(soc_min or {"mg1": 0.10, "mg2": 0.10, "mg3": 0.10})
        self.soc_tolerance = float(soc_tolerance)
        self.algorithm_name = str(algorithm_name)

        self.console = Console()
        self.live: Live | None = None
        self.start_time = time.time()

        # Metrics tracking
        self.current_episode = 0
        self.total_steps = 0
        self.profits: list[float] = []
        self.best_profit = -float("inf")
        self.best_episode = 0
        self.best_validation_profit: float | None = None

        self.last_critic_loss: float | None = None
        self.last_actor_loss: float | None = None
        self.exploration_noise = 0.0
        self.replay_size = 0

        self.status = "WARMUP"
        self.status_detail = f"Learning starts at step {self.warmup_steps}"
        self.checkpoint_msg = "No checkpoint saved yet"
        self.has_error = False
        self.error_reason: str | None = None
        self.last_metrics: dict[str, Any] = {}

    def start(self) -> None:
        self.start_time = time.time()
        if self.enabled:
            self.live = Live(
                self._render_view(),
                console=self.console,
                refresh_per_second=self.refresh_hz,
                transient=False,
                auto_refresh=False,
            )
            self.live.start()

    def set_status(self, status: str, detail: str = "") -> None:
        self.status = str(status)
        if detail:
            self.status_detail = str(detail)
        if self.enabled and self.live is not None:
            self.live.update(self._render_view(), refresh=True)

    def notify_checkpoint(self, episode: int, profit: float, path: str = "") -> None:
        self.checkpoint_msg = f"[bold green][OK] Checkpoint saved[/] @ episode {episode} (Profit: {profit:.2f})"
        if self.enabled and self.live is not None:
            self.live.update(self._render_view(), refresh=True)

    def set_error(self, reason: str) -> None:
        self.has_error = True
        self.status = "ERROR"
        self.error_reason = str(reason)
        if self.enabled and self.live is not None:
            self.live.update(self._render_view(), refresh=True)

    def update(self, ep_metrics: dict[str, Any]) -> None:
        self.last_metrics = ep_metrics
        self.current_episode = int(ep_metrics.get("episode", self.current_episode + 1))
        self.total_steps = int(ep_metrics.get("steps", self.total_steps))

        profit = float(ep_metrics.get("team_profit", 0.0))
        if math.isnan(profit) or math.isinf(profit):
            self.set_error(f"Invalid team profit: {profit}")
            return

        self.profits.append(profit)
        if profit > self.best_profit:
            self.best_profit = profit
            self.best_episode = self.current_episode

        # Track losses
        crit_loss = ep_metrics.get("critic_loss")
        if crit_loss is not None and not (isinstance(crit_loss, float) and math.isnan(crit_loss)):
            self.last_critic_loss = float(crit_loss)

        act_loss = ep_metrics.get("actor_loss")
        if act_loss is not None and not (isinstance(act_loss, float) and math.isnan(act_loss)):
            self.last_actor_loss = float(act_loss)

        self.replay_size = int(ep_metrics.get("replay_size", min(self.total_steps, self.replay_capacity)))
        self.exploration_noise = float(ep_metrics.get("exploration_noise", 0.0))
        validation_profit = ep_metrics.get("best_validation_profit")
        if validation_profit is not None and np.isfinite(float(validation_profit)):
            self.best_validation_profit = float(validation_profit)

        if self.total_steps >= self.warmup_steps and self.status == "WARMUP":
            self.status = "TRAINING"
            self.status_detail = f"Active {self.algorithm_name} actor-critic updates"

        if self.enabled and self.live is not None:
            self.live.update(self._render_view(), refresh=True)

    def _render_view(self) -> Panel:
        elapsed = max(1e-5, time.time() - self.start_time)
        ep_per_sec = self.current_episode / elapsed if self.current_episode > 0 else 0.0
        ep_per_min = ep_per_sec * 60.0
        remaining_ep = max(0, self.total_episodes - self.current_episode)
        eta_sec = remaining_ep / ep_per_sec if ep_per_sec > 0 else float("nan")
        progress_pct = (self.current_episode / self.total_episodes) * 100.0

        # Header Progress Table
        top_table = Table.grid(expand=True, padding=(0, 2))
        top_table.add_column(justify="left", ratio=1)
        top_table.add_column(justify="right", ratio=1)
        top_table.add_row(
            f"Episode:   [bold cyan]{self.current_episode:d}[/] / {self.total_episodes:d}",
            f"Progress:  [bold green]{progress_pct:6.2f}%[/]",
        )
        top_table.add_row(
            f"Env Steps: [bold white]{self.total_steps:,d}[/]",
            f"Elapsed:   [yellow]{_format_time(elapsed)}[/]",
        )
        top_table.add_row(
            f"Speed:     {ep_per_min:5.1f} ep/min",
            f"ETA:       {_format_time(eta_sec)}",
        )

        # Team Profit Table
        profit_table = Table.grid(expand=True, padding=(0, 2))
        profit_table.add_column(justify="left", ratio=1)
        profit_table.add_column(justify="right", ratio=1)

        cur_profit_str = f"{self.profits[-1]:8.2f}" if self.profits else "N/A"
        avg10 = np.mean(self.profits[-10:]) if self.profits else float("nan")
        avg50 = np.mean(self.profits[-50:]) if self.profits else float("nan")
        best_str = f"{self.best_profit:8.2f}" if self.best_episode > 0 else "N/A"

        profit_table.add_row(
            f"Current:  [bold]{cur_profit_str}[/]",
            f"Best:     [bold green]{best_str}[/] (Ep {self.best_episode})",
        )
        profit_table.add_row(
            f"Avg(10):  {avg10:8.2f}" if not math.isnan(avg10) else "Avg(10):  N/A",
            f"Avg(50):  {avg50:8.2f}" if not math.isnan(avg50) else "Avg(50):  N/A",
        )
        profit_table.add_row(
            f"MA(100): {float(self.last_metrics.get('team_profit_ma_100', avg50)):8.2f}",
            f"Best validation: {self.best_validation_profit:8.2f}"
            if self.best_validation_profit is not None else "Best validation: N/A",
        )

        # MATD3 Algorithm State Table
        matd3_table = Table.grid(expand=True, padding=(0, 2))
        matd3_table.add_column(justify="left", ratio=1)
        matd3_table.add_column(justify="right", ratio=1)

        crit_str = f"{self.last_critic_loss:.4f}" if self.last_critic_loss is not None else "N/A"
        act_str = f"{self.last_actor_loss:.4f}" if self.last_actor_loss is not None else "N/A"
        status_color = "red" if self.status == "ERROR" else "yellow" if self.status == "WARMUP" else "green"

        matd3_table.add_row(
            f"Critic Loss: [cyan]{crit_str}[/]",
            f"Replay:      {self.replay_size:,d} / {self.replay_capacity:,d}",
        )
        matd3_table.add_row(
            f"Actor Loss:  [cyan]{act_str}[/]",
            f"Exploration: {self.exploration_noise:.3f}",
        )
        matd3_table.add_row(
            f"Status:      [bold {status_color}]{self.status}[/]",
            f"Detail:      {self.status_detail}",
        )

        # Energy & Coordination Table
        energy_table = Table.grid(expand=True, padding=(0, 2))
        energy_table.add_column(justify="left", ratio=1)
        energy_table.add_column(justify="right", ratio=1)

        r_decomp = self.last_metrics.get("reward_decomposition", {})

        grid_cost = (
            float(r_decomp.get("grid_import_cost", 0.0))
            - float(r_decomp.get("grid_export_revenue", 0.0))
        )
        p2p_energy = float(self.last_metrics.get("p2p_total_energy_kwh", 0.0))
        renewable_utilization = 100.0 * float(
            self.last_metrics.get("renewable_utilization_rate", 0.0)
        )
        degradation_cost = float(r_decomp.get("degradation_cost", 0.0))

        energy_table.add_row(
            f"Grid Cost:    {grid_cost:8.2f}",
            f"RES Use:      {renewable_utilization:6.1f}%",
        )
        energy_table.add_row(
            f"P2P Energy:   {p2p_energy:8.1f} kWh",
            f"ESS Degrade:  {degradation_cost:8.2f}",
        )

        # ESS Table
        ess_table = Table(expand=True, show_header=True, header_style="bold magenta", padding=(0, 1))
        ess_table.add_column("Metric", justify="left")
        for mg in self.agent_ids:
            ess_table.add_column(mg.upper(), justify="center")

        soc_stats = self.last_metrics.get("soc_statistics", {})
        term_soc = soc_stats.get("terminal_soc", {})
        throughputs = self.last_metrics.get("ess_throughput_kwh", {})

        row_soc = ["Final SOC"]
        row_delta = ["SOC Delta (|T-0|)"]
        row_tp = ["Throughput (kWh)"]
        row_status = ["ESS Status"]

        for mg in self.agent_ids:
            soc_val = term_soc.get(mg, float("nan"))
            init = self.initial_soc.get(mg, 0.5)
            s_min = self.soc_min.get(mg, 0.10)
            delta = abs(soc_val - init) if not math.isnan(soc_val) else float("nan")
            tp = throughputs.get(mg, 0.0)

            # Check warning conditions
            if not math.isnan(soc_val):
                if soc_val <= s_min + self.soc_tolerance + 1e-4:
                    soc_disp = f"[bold yellow]{soc_val:.2f}[/]"
                    stat_disp = "[bold yellow]LOW SOC[/]"
                elif delta > self.soc_tolerance * 2:
                    soc_disp = f"[bold red]{soc_val:.2f}[/]"
                    stat_disp = "[bold red]WARNING[/]"
                else:
                    soc_disp = f"[green]{soc_val:.2f}[/]"
                    stat_disp = "[green]OK[/]"
            else:
                soc_disp = "N/A"
                stat_disp = "N/A"

            row_soc.append(soc_disp)
            row_delta.append(f"{delta:.2f}" if not math.isnan(delta) else "N/A")
            row_tp.append(f"{tp:.1f}")
            row_status.append(stat_disp)

        ess_table.add_row(*row_soc)
        ess_table.add_row(*row_delta)
        ess_table.add_row(*row_tp)
        ess_table.add_row(*row_status)

        # Algorithm-neutral, per-agent Actor diagnostics (not just a team mean).
        single_dispatch = any("dispatch_saturation_rate" in d for d in
                              self.last_metrics.get("actor_diagnostics", {}).values())
        grad_table = Table(expand=True, title=("Single dispatch: feature gradients" if single_dispatch
                                             else "Actor branches (episode batch means)"))
        grad_table.add_column("Agent")
        columns = (("Grad max", "Local features", "Gate grad", "Dispatch grad", "Clipped", "Dispatch sat.", "Local override")
                   if single_dispatch else
                   ("Grad max", "Local grad", "Offer grad", "Offer share", "Clipped", "Local sat.", "Masked grad max"))
        for name in columns:
            grad_table.add_column(name, justify="right")
        for mg in self.agent_ids:
            diag = self.last_metrics.get("actor_diagnostics", {}).get(mg, {})
            branch = self.last_metrics.get("actor_branch_grad_norms", {}).get(mg, {})
            def fmt(value, percent=False):
                if value is None:
                    return "N/A"
                return f"{100 * value:.1f}%" if percent else f"{value:.4g}"
            masked = self.last_metrics.get("actor_diagnostic_statistics", {}).get(mg, {}).get(
                "masked_offer_action_grad_max", {}).get("max")
            maximum = self.last_metrics.get("actor_diagnostic_statistics", {}).get(mg, {}).get(
                "total_grad_norm", {}).get("max")
            if single_dispatch:
                override = self.last_metrics.get("actor_diagnostic_statistics", {}).get(mg, {}).get(
                    "local_override_kw_max", {}).get("max")
                grad_table.add_row(mg.upper(), fmt(maximum), fmt(branch.get("local_features")),
                    fmt(branch.get("gates")), fmt(branch.get("dispatch")),
                    fmt(diag.get("clip_applied"), True), fmt(diag.get("dispatch_saturation_rate"), True), fmt(override))
                continue
            share = diag.get("offer_squared_grad_share")
            share_text = fmt(share, True)
            if share is not None and share > .85:
                share_text = f"[yellow]{share_text}[/]"
            grad_table.add_row(mg.upper(), fmt(maximum), fmt(branch.get("battery")), fmt(branch.get("offer")),
                share_text, fmt(diag.get("clip_applied"), True),
                fmt(diag.get("battery_saturation_rate"), True), fmt(masked))

        # Progress Bar visual
        bar_len = 36
        filled_len = int(round(bar_len * (progress_pct / 100.0)))
        bar = "=" * filled_len + "-" * (bar_len - filled_len)
        progress_text = Text(f"[{bar}]  {progress_pct:6.2f}%", style="cyan")

        # Footer Checkpoint Info
        footer = Text.from_markup(f"Checkpoint: {self.checkpoint_msg}")

        # Assemble Master Layout
        body = Group(
            top_table,
            Text("-" * 60, style="dim"),
            Text("Team Profit", style="bold underline yellow"),
            profit_table,
            Text("-" * 60, style="dim"),
            Text(f"{self.algorithm_name} Learning", style="bold underline blue"),
            matd3_table,
            grad_table,
            Text("-" * 60, style="dim"),
            Text("Energy & Coordination", style="bold underline cyan"),
            energy_table,
            Text("-" * 60, style="dim"),
            Text("Energy Storage System (ESS)", style="bold underline green"),
            ess_table,
            Text("-" * 60, style="dim"),
            footer,
            progress_text,
        )

        return Panel(
            body,
            title=f"[bold white]* Multi-Microgrid {self.algorithm_name} Training *[/]",
            border_style="bright_blue",
            padding=(1, 2),
        )

    def stop(self) -> None:
        if self.enabled and self.live is not None:
            self.live.stop()

    def render_final_summary(
        self,
        checkpoint_path: str = "",
        metrics_path: str = "",
        evaluation_path: str = "",
    ) -> None:
        elapsed = max(1e-5, time.time() - self.start_time)
        avg50 = np.mean(self.profits[-50:]) if self.profits else float("nan")

        summary_table = Table.grid(expand=True, padding=(0, 2))
        summary_table.add_column(justify="left", style="bold")
        summary_table.add_column(justify="right")

        summary_table.add_row("Training Status:", "[bold green]Completed Successfully[/]")
        summary_table.add_row("Total Episodes:", f"{self.current_episode:d} / {self.total_episodes:d}")
        summary_table.add_row("Total Steps:", f"{self.total_steps:,d}")
        summary_table.add_row("Elapsed Time:", _format_time(elapsed))
        summary_table.add_row("Best Team Profit:", f"[bold green]{self.best_profit:.2f}[/] (Episode {self.best_episode})")
        summary_table.add_row("Final Avg(50) Profit:", f"{avg50:.2f}" if not math.isnan(avg50) else "N/A")
        summary_table.add_row("Final Critic Loss:", f"{self.last_critic_loss:.4f}" if self.last_critic_loss is not None else "N/A")
        summary_table.add_row("Final Actor Loss:", f"{self.last_actor_loss:.4f}" if self.last_actor_loss is not None else "N/A")
        if checkpoint_path:
            summary_table.add_row("Saved Checkpoint:", str(checkpoint_path))
        if metrics_path:
            summary_table.add_row("Training Metrics:", str(metrics_path))
        if evaluation_path:
            summary_table.add_row("Test Evaluation:", str(evaluation_path))

        self.console.print()
        self.console.print(
            Panel(
                summary_table,
                title=f"[bold green][OK] {self.algorithm_name} Training Summary[/]",
                border_style="green",
                padding=(1, 2),
            )
        )


# Backward-compatible name for external code that imported the old dashboard class.
MATD3TrainingDashboard = AlgorithmTrainingDashboard
