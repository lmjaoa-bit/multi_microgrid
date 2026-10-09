from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from evaluation.paper_artifacts import generate_paper_artifacts
from evaluation.dispatch_overview import dispatch_overview_figure, seasonal_selection, select_seasonal_days
from experiments.layout import artifact_directory, figure_directory, run_root


BLUE = "#2F6B9A"
GOLD = "#D5A021"
ORANGE = "#D97841"
OLIVE = "#7A8B45"
PINK = "#B75D75"
INK = "#263238"
GREY = "#AAB2B7"
LIGHT_GREY = "#E6E9EB"
PALETTE = [BLUE, GOLD, ORANGE, OLIVE, PINK]


def _plot_safe_label(value: Any) -> str:
    """Return a font-independent display label for the ordinary Transformer.

    Stored run IDs and metadata remain unchanged.  CI plots cannot assume a
    CJK-capable font is installed, so the known current-method label gets an
    ASCII rendering at presentation time only.
    """
    text = str(value)
    if text in {"普通 T", "普通T"}:
        return "Ordinary Transformer"
    for prefix in ("普通T_", "普通 T_"):
        if text.startswith(prefix):
            return "OrdinaryT_" + text[len(prefix):]
    return text


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
        "font.family": ["Noto Sans SC", "Microsoft YaHei", "DejaVu Sans"],
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


def _warmup_episode_position(
    training: Sequence[Mapping[str, Any]],
    warmup_steps: int | float | None,
) -> float | None:
    """Map an environment-step warm-up boundary onto the episode axis."""
    if not training or warmup_steps is None:
        return None
    target = float(warmup_steps)
    if not np.isfinite(target) or target <= 0.0:
        return None

    previous_episode = 0.0
    previous_steps = 0.0
    for index, item in enumerate(training):
        episode = float(item.get("episode", index + 1))
        step_value = item.get("total_steps", item.get("steps"))
        if step_value is None:
            continue
        steps = float(step_value)
        if not np.isfinite(episode) or not np.isfinite(steps):
            continue
        if steps >= target:
            delta_steps = steps - previous_steps
            if delta_steps <= 0.0:
                return episode
            fraction = float(np.clip((target - previous_steps) / delta_steps, 0.0, 1.0))
            return previous_episode + fraction * (episode - previous_episode)
        previous_episode = episode
        previous_steps = steps
    return None


def _training_convergence_figure(
    training: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    *,
    warmup_steps: int | float | None,
    reward_semantics_version: str = "legacy",
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.3, 5.4))
    if not training:
        _no_data(ax, "Training Convergence", "training_history.json is unavailable")
        return fig

    episodes = np.asarray(
        [item.get("episode", index + 1) for index, item in enumerate(training)],
        dtype=float,
    )
    is_v2 = reward_semantics_version == "generalized_system_cost_v2"
    metric_key = "team_reward" if is_v2 else "team_profit"
    validation_key = "team_reward_mean" if is_v2 else "team_profit_mean"
    raw = np.asarray(
        [item.get(metric_key, np.nan) for item in training], dtype=float
    )
    moving = pd.Series(raw).rolling(100, min_periods=1).mean().to_numpy()
    raw_label = (
        "Training Generalized Return (raw)"
        if is_v2 else "Training Team Profit (raw)"
    )
    ma_label = "Generalized Return MA100" if is_v2 else "Team Profit MA100"
    validation_label = (
        "Validation Mean Generalized Return"
        if is_v2 else "Validation Mean Team Profit"
    )
    ax.plot(
        episodes, raw, color=BLUE, alpha=0.18, linewidth=0.8,
        label=raw_label,
    )
    ax.plot(episodes, moving, color=BLUE, linewidth=2.3, label=ma_label)

    valid_validation = [
        item for item in validation
        if item.get(validation_key) is not None
        and np.isfinite(float(item[validation_key]))
    ]
    if valid_validation:
        val_x = [item["episode"] for item in valid_validation]
        val_y = [item[validation_key] for item in valid_validation]
        ax.plot(
            val_x, val_y, color=GOLD, marker="o", linewidth=1.8,
            label=validation_label,
        )
        best = max(
            valid_validation, key=lambda item: float(item[validation_key])
        )
        ax.scatter(
            [best["episode"]], [best[validation_key]], color=ORANGE, zorder=5
        )
        ax.annotate(
            f"Best: ep {best['episode']}\n{float(best[validation_key]):.3f}",
            (best["episode"], best[validation_key]),
            xytext=(8, 12), textcoords="offset points", fontsize=8,
        )

    if is_v2:
        accounting = np.asarray(
            [item.get("team_profit", np.nan) for item in training], dtype=float
        )
        accounting_ma = (
            pd.Series(accounting).rolling(100, min_periods=1).mean().to_numpy()
        )
        ax_profit = ax.twinx()
        ax_profit.plot(
            episodes,
            accounting_ma,
            color=OLIVE,
            linewidth=1.5,
            linestyle="--",
            label="Accounting Profit MA100",
        )
        val_profit = [
            item for item in validation
            if item.get("team_profit_mean") is not None
            and np.isfinite(float(item["team_profit_mean"]))
        ]
        if val_profit:
            ax_profit.plot(
                [item["episode"] for item in val_profit],
                [item["team_profit_mean"] for item in val_profit],
                color=PINK,
                linewidth=1.2,
                marker=".",
                label="Validation Mean Accounting Profit",
            )
        ax_profit.set_ylabel("Accounting Profit")

    warmup_episode = _warmup_episode_position(training, warmup_steps)
    if warmup_episode is not None:
        steps_label = int(float(warmup_steps))
        ax.axvline(
            warmup_episode, color=INK, linestyle="--", linewidth=1.0,
            label=f"Warm-up ends ({steps_label} steps)",
        )

    ax.set_title("Training Convergence")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel(
        "Generalized Return (-generalized cost / scale)"
        if is_v2 else "Team Profit"
    )
    ax.grid(axis="y")
    if len(episodes) > 1:
        ax.set_xlim(left=1.0, right=float(episodes[-1]))
    if is_v2 and len(fig.axes) > 1:
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = fig.axes[1].get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc="best")
    else:
        ax.legend(loc="best")
    return fig


def generate_figures(
    run_dir: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Regenerate all figures exclusively from files already saved in a run directory."""
    import matplotlib.pyplot as plt

    _apply_style()
    public_root = run_root(run_dir)
    if (public_root / "release_manifest.json").exists():
        raise ValueError("Retained releases are immutable; regenerate a working run instead")
    root = artifact_directory(run_dir)
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
    config_snapshot: Mapping[str, Any] = {}
    config_snapshot_path = root / "config_snapshot.yaml"
    if config_snapshot_path.exists():
        loaded_snapshot = yaml.safe_load(
            config_snapshot_path.read_text(encoding="utf-8")
        ) or {}
        if isinstance(loaded_snapshot, Mapping):
            config_snapshot = loaded_snapshot
    reward_semantics_version = str(
        summary.get("reward_semantics_version")
        or metadata.get("reward_semantics_version")
        or config_snapshot.get("environment", {}).get("reward", {}).get(
            "semantics_version", "legacy"
        )
    )
    algorithm_label = _plot_safe_label(
        summary.get("algorithm_display_name")
        or metadata.get("algorithm_display_name")
        or metadata.get("algorithm")
        or "MATD3"
    )
    trajectory_paths = sorted((root / "test_hourly_trajectories").glob("*.csv"))
    if not trajectory_paths:
        raise FileNotFoundError("Dispatch overviews require saved test_hourly_trajectories/*.csv")
    hourly = pd.concat([pd.read_csv(path) for path in trajectory_paths], ignore_index=True)
    if "date" in daily and set(daily.date.astype(str)) != set(hourly.date.astype(str)):
        raise ValueError("Saved hourly trajectories and test metrics must contain identical dates")
    selection = seasonal_selection(hourly)
    figure_dir = figure_directory(run_dir)
    if figure_dir.is_symlink():
        raise ValueError("Generated figure directory must not be a symbolic link")
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 00: compact experiment dashboard (summary JSON is intentionally the source here).
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    ax.set_axis_off()
    ax.set_title("Experiment Summary", loc="left", fontsize=18, pad=16)
    vs_idle = summary.get("comparisons", {}).get("vs_idle", {})
    dashboard_rows = [
        ("Algorithm", algorithm_label),
        ("Run ID", _plot_safe_label(metadata.get("run_id", root.name))),
        ("Seed", metadata.get("seed", summary.get("test_seed"))),
        ("Checkpoint", Path(str(summary.get("checkpoint_used", ""))).name),
        ("Checkpoint Episode", summary.get("checkpoint_episode")),
        (
            "Accounting Joint Team Profit"
            if reward_semantics_version == "generalized_system_cost_v2"
            else "Joint Team Profit",
            summary.get("joint_team_profit_mean"),
        ),
        ("Generalized Cost", summary.get("generalized_system_cost_mean"))
        if reward_semantics_version == "generalized_system_cost_v2"
        else ("Gain vs Idle", vs_idle.get("gain_mean_abs")),
        ("Win Rate vs Idle", None if vs_idle.get("win_rate") is None else f"{100*vs_idle['win_rate']:.1f}%"),
        ("Cooperation Surplus", summary.get("cooperation_surplus_mean")),
        ("Cooperation Gain", None if summary.get("cooperation_gain_pct_mean") is None else f"{summary['cooperation_gain_pct_mean']:.1f}%"),
        ("CVCR", summary.get("cooperative_value_capture_ratio")),
        ("RES Non-Curtailment", None if summary.get("res_utilization_mean_pct") is None else f"{summary['res_utilization_mean_pct']:.1f}%"),
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

    # 01: complete training convergence from Episode 1 with raw, MA100, validation, and warm-up boundary.
    warmup_steps = config_snapshot.get("algorithm", {}).get("algorithm", {}).get("warmup_steps")
    fig = _training_convergence_figure(
        training,
        validation,
        warmup_steps=warmup_steps,
        reward_semantics_version=reward_semantics_version,
    )
    fig.tight_layout()
    written += _save(fig, figure_dir, "01_training_convergence", formats)

    environment = config_snapshot.get("environment", config_snapshot)
    reference_paths = []
    for season, date in selection["seasonal_days"].items():
        day = hourly.loc[hourly["date"].astype(str) == date].sort_values("hour")
        reference_path = root / "dr_reference_hourly_trajectories" / f"{date}.csv"
        reference = pd.read_csv(reference_path) if reference_path.is_file() else None
        if reference is not None:
            reference_paths.append(reference_path)
        fig = dispatch_overview_figure(day, environment, algorithm_label, date, season, reference)
        written += _save(fig, figure_dir, f"02_dispatch_overview_{season}_{date}", formats)

    # Cleanup is limited to generated image formats inside this run's figures folder.
    # CSVs, checkpoints, custom notes and immutable releases are never removed.
    keep = {path.resolve() for path in written}
    for old in figure_dir.iterdir():
        if old.is_file() and old.suffix.lower() in {".png", ".pdf"} and old.resolve() not in keep:
            old.unlink()
    sources = [*trajectory_paths, *reference_paths]
    for name in ("test_daily_metrics.csv", "test_summary.json", "config_snapshot.yaml",
                 "training_history.json", "validation_history.json",
                 "run_metadata.json", "training_summary.json"):
        path = root / name
        if path.is_file():
            sources.append(path)
    manifest = {
        **selection, "test_day_count": int(hourly["date"].nunique()),
        "figures": [path.name for path in written],
        "source_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "layout": "00 summary; 01 training; one dispatch overview per available season",
    }
    (root / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return written


def generate_plan_diagnostic_figures(run_dir: str | Path, formats=('png','pdf')) -> list[Path]:
    """Render auxiliary evidence exclusively from saved observations/trajectories."""
    import matplotlib.pyplot as plt
    root = artifact_directory(run_dir)
    source = root/'auxiliary_plan/formal_test_plan_attention.jsonl'
    if not source.exists():
        return []
    manifest = json.loads((root/'auxiliary_plan/manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line]
    expected_days = manifest['formal_test']['days']
    expected_seeds = manifest['formal_test']['seeds']
    daily_path = root/'test_daily_metrics.csv'
    if not daily_path.exists():
        raise ValueError('Formal-test day/seed identity file is missing')
    daily = pd.read_csv(daily_path)
    actual_pairs = list(zip(daily['date'].astype(str), daily['test_seed'].astype(int)))
    if actual_pairs != list(zip(expected_days,expected_seeds)):
        raise ValueError('Plan diagnostic day/seed identity disagrees with saved formal test')
    agents = tuple(manifest['actor_contract']['observation_schemas'])
    expected_keys = {(day,mg,t) for day in expected_days for mg in agents for t in range(24)}
    keys = [(r['day'],r['agent'],r['t']) for r in rows]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError('Incomplete or duplicate plan diagnostic coverage')
    seed_by_day = dict(zip(expected_days,expected_seeds))
    if any(r['seed'] != seed_by_day[r['day']] for r in rows):
        raise ValueError('Plan diagnostic seed identity disagrees with saved formal test')
    checkpoint = manifest['metadata'].get('checkpoint')
    if any(r['day'] not in expected_days or r['checkpoint'] != checkpoint or
           r['checkpoint_sha256'] != manifest.get('checkpoint_sha256') for r in rows):
        raise ValueError('Saved plan diagnostic checkpoint/day mismatch')
    summary_path = root/'test_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('checkpoint_used', summary.get('checkpoint_path', checkpoint)) != checkpoint:
            raise ValueError('Plan diagnostics disagree with formal test checkpoint')
        if 'checkpoint_sha256' in summary and summary['checkpoint_sha256'] != manifest.get('checkpoint_sha256'):
            raise ValueError('Plan diagnostics disagree with checkpoint hash')
        if 'test_days' in summary and summary['test_days'] != expected_days:
            raise ValueError('Plan diagnostics disagree with test days')
        if 'test_days_count' in summary and summary['test_days_count'] != len(expected_days):
            raise ValueError('Plan diagnostic day coverage disagrees with formal test count')
    actual_by_day = {}
    for day in expected_days:
        trajectory_path = root/'test_hourly_trajectories'/f'{day}.csv'
        if not trajectory_path.exists():
            raise FileNotFoundError(f'Actual trajectory missing: {trajectory_path}')
        actual = pd.read_csv(trajectory_path)
        if len(actual)!=24 or set(actual['timestep'])!=set(range(24)):
            raise ValueError('Actual formal-test trajectory coverage must contain 24 unique hours')
        if 'date' in actual and set(actual['date'].astype(str))!={day}:
            raise ValueError('Actual trajectory day identity mismatch')
        actual_by_day[day] = actual.sort_values('timestep')
    directory = root/'figures/plan_diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    groups = sorted({(r['day'], r['agent']) for r in rows})
    for day, mg in groups:
        samples = sorted([r for r in rows if r['day']==day and r['agent']==mg], key=lambda r:r['t'])
        heads = len(samples[0]['attention'][0])
        fig, axes = plt.subplots(2, heads, figsize=(4*heads,7), squeeze=False, constrained_layout=True)
        for layer in range(2):
            for head in range(heads):
                matrix = np.full((24,26), np.nan)
                for r in samples:
                    weights = np.asarray(r['attention'][layer][head], dtype=float)
                    valid = np.r_[r['forecast_mask'], [True,True]]
                    matrix[r['t']] = np.where(valid, weights, np.nan)
                ax = axes[layer,head]
                artist = ax.imshow(np.ma.masked_invalid(matrix), cmap='viridis', vmin=0, vmax=np.nanmax(matrix), aspect='auto')
                ax.set_xticks([0,5,11,17,23,24,25], ['+1','+6','+12','+18','+24',*samples[0]['peer_order']],
                              rotation=60, ha='right', fontsize=8)
                ax.set_yticks([0,6,12,18,23])
                ax.set_xlabel('Forecast relative hour / peer')
                ax.set_ylabel('Current hour t (forecast hour = t + k)')
                ax.set_title(f'Layer {layer+1}, head {head+1}')
                fig.colorbar(artist, ax=ax, shrink=.7)
        fig.suptitle(f'{mg.upper()} {day}: NOW cross-attention (masked cells blank)')
        written += _save(fig, directory, f'{day}_{mg}_attention', formats)
        actual = actual_by_day[day]
        fig, axes = plt.subplots(2,1,figsize=(11,7),sharex=True)
        for r in samples:
            valid = np.asarray(r['plan_mask'], dtype=bool)
            hours = r['t'] + np.arange(24)
            axes[0].plot(hours[valid]+1, np.asarray(r['soc_after'])[valid], alpha=.22, color=BLUE)
            axes[1].plot(hours[valid], np.asarray(r['ess_net_kw'])[valid], alpha=.22, color=BLUE)
        first = samples[0]
        valid = np.asarray(first['plan_mask'], dtype=bool)
        axes[0].plot((first['t']+np.arange(24)+1)[valid], np.asarray(first['soc_after'])[valid], color=BLUE, label=f"Auxiliary plan at t={first['t']}", marker='.', markersize=3)
        axes[1].plot((first['t']+np.arange(24))[valid], np.asarray(first['ess_net_kw'])[valid], color=BLUE, label='Auxiliary ESS plan', marker='.', markersize=3)
        axes[0].plot(actual['timestep']+1, actual[f'{mg}_soc'], color=ORANGE, label='Actual SOC after dispatch', marker='.', markersize=3)
        axes[1].plot(actual['timestep'], actual[f'{mg}_battery_discharge_kw']-actual[f'{mg}_battery_charge_kw'], color=ORANGE, label='Actual ESS discharge - charge', marker='.', markersize=3)
        axes[0].set_ylabel('SOC at resulting boundary')
        axes[1].set_ylabel('ESS net power (kW)')
        axes[1].set_xlabel('Hour (SOC boundary t+1; ESS interval t)')
        axes[0].set_title(f'{mg.upper()} {day}: auxiliary plans and actual dispatch')
        for ax in axes:
            ax.legend()
            ax.grid(alpha=.25)
        fig.tight_layout()
        written += _save(fig,directory,f'{day}_{mg}_plans',formats)
    return written
