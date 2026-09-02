from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


BLUE = "#2F6B9A"
GOLD = "#D5A021"
ORANGE = "#D97841"
OLIVE = "#7A8B45"
PINK = "#B75D75"
INK = "#263238"
GREY = "#AAB2B7"
LIGHT_GREY = "#E6E9EB"
PALETTE = [BLUE, GOLD, ORANGE, OLIVE, PINK]


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").dropna()


def _mean_std(frame: pd.DataFrame, field: str) -> tuple[float, float]:
    values = _numeric(frame, field)
    return (
        float(values.mean()) if len(values) else np.nan,
        float(values.std(ddof=0)) if len(values) else 0.0,
    )


def _rate(frame: pd.DataFrame, field: str) -> float:
    if field not in frame:
        return np.nan
    values = frame[field].dropna()
    if values.empty:
        return np.nan
    normalized = values.map(
        lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes"}
    )
    return float(normalized.astype(float).mean() * 100.0)


def _apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 10,
        "legend.frameon": False,
        "grid.color": LIGHT_GREY,
        "grid.linewidth": 0.7,
        "savefig.bbox": "tight",
        "savefig.dpi": 180,
    })


def _save(fig: Any, directory: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for extension in formats:
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, format=extension)
        paths.append(path)
    plt.close(fig)
    return paths


def _label_bars(ax: Any, bars: Iterable[Any], *, suffix: str = "", digits: int = 1) -> None:
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        va = "bottom" if height >= 0 else "top"
        offset = 3 if height >= 0 else -3
        ax.annotate(
            f"{height:.{digits}f}{suffix}",
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color=INK,
        )


def _no_data(ax: Any, title: str, message: str = "Required saved metrics are unavailable") -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color=INK)
    ax.set_axis_off()


def _bar_figure(
    labels: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    *,
    title: str,
    ylabel: str,
    colors: Sequence[str] | None = None,
    zero_line: bool = False,
    percent: bool = False,
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    values = np.asarray(means, dtype=float)
    if not np.isfinite(values).any():
        _no_data(ax, title)
        return fig
    errors = np.nan_to_num(np.asarray(stds, dtype=float), nan=0.0)
    bars = ax.bar(labels, values, yerr=errors, capsize=4, color=list(colors or PALETTE[:len(labels)]), edgecolor=INK, linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    if zero_line:
        ax.axhline(0.0, color=INK, linewidth=1.0)
    elif np.nanmin(values - errors) >= 0.0:
        upper = float(np.nanmax(values + errors))
        ax.set_ylim(bottom=0.0, top=1.0 if upper <= 0.0 else upper * 1.15)
    if percent and np.nanmin(values) >= 0.0:
        ax.set_ylim(bottom=0.0, top=max(105.0, float(np.nanmax(values + errors)) * 1.15))
    _label_bars(ax, bars, suffix="%" if percent else "")
    fig.tight_layout()
    return fig


def _grouped_bars(
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], Sequence[float], str]],
    *,
    title: str,
    ylabel: str,
    zero_line: bool = False,
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    finite = any(np.isfinite(np.asarray(values, dtype=float)).any() for _, values, _, _ in series)
    if not finite:
        _no_data(ax, title)
        return fig
    x = np.arange(len(labels), dtype=float)
    width = 0.78 / max(1, len(series))
    for index, (name, values, errors, color) in enumerate(series):
        positions = x + (index - (len(series) - 1) / 2.0) * width
        ax.bar(
            positions,
            values,
            width,
            yerr=np.nan_to_num(np.asarray(errors, dtype=float), nan=0.0),
            capsize=3,
            label=name,
            color=color,
            edgecolor=INK,
            linewidth=0.5,
        )
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    if zero_line:
        ax.axhline(0.0, color=INK, linewidth=1.0)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    return fig


def generate_figures(
    run_dir: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Regenerate all figures exclusively from files already saved in a run directory."""
    import matplotlib.pyplot as plt

    _apply_style()
    root = Path(run_dir).resolve()
    daily_path = root / "test_daily_metrics.csv"
    summary_path = root / "test_summary.json"
    if not daily_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Plotting requires completed test_daily_metrics.csv and test_summary.json")
    daily = pd.read_csv(daily_path)
    if daily.empty:
        raise ValueError("Plotting requires at least one completed test day")
    summary: Mapping[str, Any] = _read_json(summary_path, {})
    metadata: Mapping[str, Any] = _read_json(root / "run_metadata.json", {})
    training_summary: Mapping[str, Any] = _read_json(root / "training_summary.json", {})
    training = _read_json(root / "training_history.json", []) or []
    validation = _read_json(root / "validation_history.json", []) or []
    algorithm_label = str(
        summary.get("algorithm_display_name")
        or metadata.get("algorithm_display_name")
        or metadata.get("algorithm")
        or "MATD3"
    )
    joint_field = "algorithm_joint_profit" if "algorithm_joint_profit" in daily else "matd3_joint_profit"
    surplus_field = (
        "algorithm_cooperation_surplus"
        if "algorithm_cooperation_surplus" in daily else "matd3_cooperation_surplus"
    )
    gain_field = (
        "algorithm_cooperation_gain_pct"
        if "algorithm_cooperation_gain_pct" in daily else "matd3_cooperation_gain_pct"
    )
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 00: compact experiment dashboard (summary JSON is intentionally the source here).
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    ax.set_axis_off()
    ax.set_title("Experiment Summary", loc="left", fontsize=18, pad=16)
    vs_idle = summary.get("comparisons", {}).get("vs_idle", {})
    dashboard_rows = [
        ("Algorithm", algorithm_label),
        ("Run ID", metadata.get("run_id", root.name)),
        ("Seed", metadata.get("seed", summary.get("test_seed"))),
        ("Checkpoint", Path(str(summary.get("checkpoint_used", ""))).name),
        ("Checkpoint Episode", summary.get("checkpoint_episode")),
        ("Joint Team Profit", summary.get("joint_team_profit_mean")),
        ("Gain vs Idle", vs_idle.get("gain_mean_abs")),
        ("Win Rate vs Idle", None if vs_idle.get("win_rate") is None else f"{100*vs_idle['win_rate']:.1f}%"),
        ("Cooperation Surplus", summary.get("cooperation_surplus_mean")),
        ("Cooperation Gain", None if summary.get("cooperation_gain_pct_mean") is None else f"{summary['cooperation_gain_pct_mean']:.1f}%"),
        ("CVCR", summary.get("cooperative_value_capture_ratio")),
        ("RES Utilization", None if summary.get("res_utilization_mean_pct") is None else f"{summary['res_utilization_mean_pct']:.1f}%"),
        ("Curtailment Reduction", summary.get("curtailment_reduction_mean_kwh")),
        ("P2P Energy", summary.get("p2p_total_energy_kwh_mean")),
        ("Feasible-Day Rate", None if summary.get("feasible_day_rate") is None else f"{100*summary['feasible_day_rate']:.1f}%"),
        ("Equal All-IR Rate", None if summary.get("equal_all_ir_satisfaction_rate") is None else f"{100*summary['equal_all_ir_satisfaction_rate']:.1f}%"),
        ("Shapley All-IR Rate", None if summary.get("shapley_all_ir_satisfaction_rate") is None else f"{100*summary['shapley_all_ir_satisfaction_rate']:.1f}%"),
    ]
    for index, (label, value) in enumerate(dashboard_rows):
        column = index // 9
        row_index = index % 9
        x = 0.04 + column * 0.50
        y = 0.91 - row_index * 0.095
        ax.text(x, y, label, transform=ax.transAxes, color="#59636A", fontsize=10)
        if isinstance(value, (float, np.floating)):
            text = f"{float(value):,.3f}"
        else:
            text = "N/A" if value is None or value == "" else str(value)
        ax.text(x, y - 0.035, text, transform=ax.transAxes, color=INK, fontsize=12, fontweight="bold")
    written += _save(fig, figure_dir, "00_experiment_summary", formats)

    # 01: training convergence.
    fig, ax = plt.subplots(figsize=(9.3, 5.4))
    if training:
        episodes = np.asarray([item.get("episode", index + 1) for index, item in enumerate(training)], dtype=float)
        raw = np.asarray([item.get("team_profit", np.nan) for item in training], dtype=float)
        moving = pd.Series(raw).rolling(100, min_periods=1).mean().to_numpy()
        ax.plot(episodes, raw, color=BLUE, alpha=0.18, linewidth=0.8, label="Training Team Profit (raw)")
        ax.plot(episodes, moving, color=BLUE, linewidth=2.3, label="Team Profit MA100")
        if validation:
            val_x = [item["episode"] for item in validation]
            val_y = [item["team_profit_mean"] for item in validation]
            ax.plot(val_x, val_y, color=GOLD, marker="o", linewidth=1.8, label="Validation Mean Team Profit")
            best = max(validation, key=lambda item: float(item["team_profit_mean"]))
            ax.scatter([best["episode"]], [best["team_profit_mean"]], color=ORANGE, zorder=5)
            ax.annotate(
                f"Best: ep {best['episode']}\n{best['team_profit_mean']:.1f}",
                (best["episode"], best["team_profit_mean"]),
                xytext=(8, 12), textcoords="offset points", fontsize=8,
            )
        ax.axvline(episodes[-1], color=INK, linestyle="--", linewidth=0.9, label=f"Final Episode {int(episodes[-1])}")
        ax.set_title("Training Convergence")
        ax.set_xlabel("Training Episode")
        ax.set_ylabel("Team Profit")
        ax.grid(axis="y")
        ax.legend(loc="best")
    else:
        _no_data(ax, "Training Convergence", "training_history.json is unavailable")
    fig.tight_layout()
    written += _save(fig, figure_dir, "01_training_convergence", formats)

    labels = ["Idle", "Independent MILP", f"{algorithm_label} Joint", "Cooperative MILP"]
    fields = ["idle_team_profit", "independent_total_profit", joint_field, "cooperative_milp_profit"]
    mean_std = [_mean_std(daily, field) for field in fields]
    written += _save(
        _bar_figure(labels, [item[0] for item in mean_std], [item[1] for item in mean_std],
                    title="Joint Team Profit", ylabel="Average Team Profit (Mean ± 1 Std)"),
        figure_dir, "02_joint_team_profit", formats,
    )

    labels = [algorithm_label, "Cooperative MILP"]
    mean_std = [_mean_std(daily, field) for field in (surplus_field, "cooperative_milp_cooperation_surplus")]
    written += _save(
        _bar_figure(labels, [item[0] for item in mean_std], [item[1] for item in mean_std],
                    title="Cooperation Surplus", ylabel="Average Cooperation Surplus (Mean ± 1 Std)", zero_line=True),
        figure_dir, "03_cooperation_surplus", formats,
    )

    mean_std = [_mean_std(daily, field) for field in (gain_field, "cooperative_milp_cooperation_gain_pct")]
    written += _save(
        _bar_figure(labels, [item[0] for item in mean_std], [item[1] for item in mean_std],
                    title="Cooperation Gain", ylabel="Cooperation Gain (%) (Mean ± 1 Std)", zero_line=True),
        figure_dir, "04_cooperation_gain_pct", formats,
    )

    mg_labels = ["MG1", "MG2", "MG3"]
    profit_series = []
    for name, prefix, color in (
        ("Independent Profit", "v_", GREY),
        ("Joint Physical Profit", "physical_profit_", BLUE),
        ("Equal Settled Profit", "equal_settled_profit_", GOLD),
        ("Shapley Settled Profit", "shapley_settled_profit_", ORANGE),
    ):
        fields = [f"v_mg{index}" if prefix == "v_" else f"{prefix}mg{index}" for index in (1, 2, 3)]
        values = [_mean_std(daily, field) for field in fields]
        profit_series.append((name, [item[0] for item in values], [item[1] for item in values], color))
    written += _save(
        _grouped_bars(mg_labels, profit_series, title="Microgrid Profit and Settlement",
                      ylabel="Average Profit (Mean ± 1 Std)", zero_line=True),
        figure_dir, "05_microgrid_profit_settlement", formats,
    )

    res_fields = ["independent_renewable_utilization_pct", "res_utilization_pct", "cooperative_milp_res_utilization_pct"]
    mean_std = [_mean_std(daily, field) for field in res_fields]
    written += _save(
        _bar_figure(["Independent", f"{algorithm_label} Joint", "Cooperative MILP"],
                    [item[0] for item in mean_std], [item[1] for item in mean_std],
                    title="RES Utilization", ylabel="RES Utilization (%)", percent=True),
        figure_dir, "06_res_utilization", formats,
    )

    curtail_fields = ["independent_curtailment_kwh", "renewable_curtailment_kwh", "cooperative_milp_curtailment_kwh"]
    mean_std = [_mean_std(daily, field) for field in curtail_fields]
    fig = _bar_figure(
        ["Independent", f"{algorithm_label} Joint", "Cooperative MILP"],
        [item[0] for item in mean_std], [item[1] for item in mean_std],
        title="Renewable Curtailment Comparison",
        ylabel="Average Renewable Curtailment (kWh/day)",
    )
    if fig.axes and np.isfinite(_mean_std(daily, "curtailment_reduction_kwh")[0]):
        reduction = _mean_std(daily, "curtailment_reduction_kwh")[0]
        reduction_pct = _mean_std(daily, "curtailment_reduction_pct")[0]
        fig.axes[0].text(
            0.98, 0.96,
            f"{algorithm_label} reduction: {reduction:.1f} kWh/day\nReduction: {reduction_pct:.1f}%" if np.isfinite(reduction_pct) else f"{algorithm_label} reduction: {reduction:.1f} kWh/day",
            transform=fig.axes[0].transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": LIGHT_GREY, "pad": 5},
        )
    written += _save(fig, figure_dir, "07_curtailment_comparison", formats)

    # 08: external-grid interaction and overall physical feasibility.
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.8), gridspec_kw={"width_ratios": [1.6, 1.0]})
    grid_fields = ["grid_import_kwh", "grid_export_kwh"]
    grid_stats = [_mean_std(daily, field) for field in grid_fields]
    bars = axes[0].bar(
        ["Grid Import", "Grid Export"],
        [item[0] for item in grid_stats],
        yerr=[item[1] for item in grid_stats], capsize=4,
        color=[BLUE, ORANGE], edgecolor=INK, linewidth=0.6,
    )
    axes[0].set_ylabel("Average Grid Energy (kWh/day, Mean ± 1 Std)")
    axes[0].grid(axis="y")
    _label_bars(axes[0], bars)
    axes[1].set_axis_off()
    feasible = _rate(daily, "feasible_day")
    axes[1].text(0.05, 0.58, "Feasible-Day Rate", color="#59636A", transform=axes[1].transAxes)
    axes[1].text(0.05, 0.42, f"{feasible:.1f}%", color=INK, fontsize=24, fontweight="bold", transform=axes[1].transAxes)
    fig.suptitle("External Grid Interaction", fontsize=14)
    fig.tight_layout()
    written += _save(fig, figure_dir, "08_grid_interaction", formats)

    equal_rates = [_rate(daily, f"equal_ir_satisfied_mg{index}") for index in (1, 2, 3)] + [_rate(daily, "equal_all_ir_satisfied")]
    shapley_rates = [_rate(daily, f"shapley_ir_satisfied_mg{index}") for index in (1, 2, 3)] + [_rate(daily, "shapley_all_ir_satisfied")]
    fig = _grouped_bars(
        ["MG1", "MG2", "MG3", "All-MG"],
        [("Equal Sharing", equal_rates, [0.0] * 4, GOLD), ("Shapley", shapley_rates, [0.0] * 4, BLUE)],
        title="Individual Rationality", ylabel="IR Satisfaction Rate (%)",
    )
    if fig.axes:
        fig.axes[0].set_ylim(0.0, 105.0)
    written += _save(fig, figure_dir, "09_individual_rationality", formats)

    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    matrix = np.zeros((3, 3), dtype=float)
    for seller in range(1, 4):
        for buyer in range(1, 4):
            if seller != buyer:
                matrix[seller - 1, buyer - 1] = _mean_std(daily, f"p2p_energy_mg{seller}_to_mg{buyer}_kwh")[0]
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0)
    ax.set_xticks(range(3), mg_labels)
    ax.set_yticks(range(3), mg_labels)
    ax.set_xlabel("Buyer (column)")
    ax.set_ylabel("Seller (row)")
    ax.set_title("P2P Trading Matrix")
    for row_index in range(3):
        for column_index in range(3):
            label = "—" if row_index == column_index else f"{matrix[row_index, column_index]:.1f}"
            text_color = "white" if matrix[row_index, column_index] > float(np.nanmax(matrix)) * 0.55 else INK
            ax.text(column_index, row_index, label, ha="center", va="center", color=text_color)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Average P2P Energy (kWh/day)")
    fig.tight_layout()
    written += _save(fig, figure_dir, "10_p2p_trading_matrix", formats)

    ess_series = []
    for name, prefix, color in (
        ("Battery Charge", "battery_charge_", BLUE),
        ("Battery Discharge", "battery_discharge_", ORANGE),
        ("Battery Throughput", "battery_throughput_", GOLD),
    ):
        values = [_mean_std(daily, f"{prefix}mg{index}_kwh") for index in (1, 2, 3)]
        ess_series.append((name, [item[0] for item in values], [item[1] for item in values], color))
    written += _save(
        _grouped_bars(mg_labels, ess_series, title="ESS Performance",
                      ylabel="Average Energy (kWh/day, Mean ± 1 Std)"),
        figure_dir, "11_ess_performance", formats,
    )

    constraint_labels = ["Feasible Day", "SOC Violation", "Power Balance Violation", "P2P Balance Violation"]
    constraint_values = [
        _rate(daily, "feasible_day"),
        _rate(daily, "soc_violation_day"),
        _rate(daily, "power_balance_violation_day"),
        _rate(daily, "p2p_balance_violation_day"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    positions = np.arange(len(constraint_labels))
    bars = ax.barh(positions, constraint_values, color=[BLUE, ORANGE, OLIVE, PINK],
                   edgecolor=INK, linewidth=0.6)
    ax.set_yticks(positions, constraint_labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 105.0)
    ax.set_xlabel("Test-Day Rate (%)")
    ax.set_title("Constraint and Feasibility")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for bar in bars:
        width_value = float(bar.get_width())
        ax.text(min(width_value + 1.0, 101.0), bar.get_y() + bar.get_height() / 2.0,
                f"{width_value:.1f}%", va="center", ha="left" if width_value < 100 else "right", fontsize=8)
    fig.tight_layout()
    written += _save(fig, figure_dir, "12_constraint_feasibility", formats)

    allocation_series = []
    for name, prefix, color in (
        ("Equal Allocated Surplus", "equal_allocated_surplus_", GOLD),
        ("Shapley Allocated Surplus", "shapley_allocated_surplus_", BLUE),
    ):
        values = [_mean_std(daily, f"{prefix}mg{index}") for index in (1, 2, 3)]
        allocation_series.append((name, [item[0] for item in values], [item[1] for item in values], color))
    written += _save(
        _grouped_bars(mg_labels, allocation_series, title="Equal vs Shapley Surplus Allocation",
                      ylabel="Average Allocated Surplus (Mean ± 1 Std)", zero_line=True),
        figure_dir, "13_equal_vs_shapley_surplus_allocation", formats,
    )

    transfer_series = []
    for name, prefix, color in (
        ("Equal Transfer Payment", "equal_transfer_", GOLD),
        ("Shapley Transfer Payment", "shapley_transfer_payment_", BLUE),
    ):
        values = [_mean_std(daily, f"{prefix}mg{index}") for index in (1, 2, 3)]
        transfer_series.append((name, [item[0] for item in values], [item[1] for item in values], color))
    fig = _grouped_bars(mg_labels, transfer_series, title="Transfer Payments",
                        ylabel="Average Transfer Payment (Mean ± 1 Std)", zero_line=True)
    if fig.axes:
        fig.axes[0].text(0.01, 0.02, "Positive = receives payment; negative = pays into coalition",
                         transform=fig.axes[0].transAxes, fontsize=8, color="#59636A")
    written += _save(fig, figure_dir, "14_transfer_payments", formats)
    return written
