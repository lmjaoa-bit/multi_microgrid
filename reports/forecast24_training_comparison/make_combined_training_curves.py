from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent

RUNS = {
    "MATD3": [
        "outputs/matd3_dispatch/runs/MATD3_2026-09-08_20-08-38_seed42",
    ],
    "PNN": [
        "outputs/pnn/runs/PNN_2026-09-08_21-20-58_seed43",
        "outputs/pnn/runs/PNN_2026-09-09_04-03-49_seed44",
    ],
    "MATD3-Transformer": [
        "outputs/matd3_transformer/runs/MATD3-TRANSFORMER_2026-09-08_22-33-06_seed43",
        "outputs/matd3_transformer/runs/MATD3-TRANSFORMER_2026-09-09_05-02-46_seed44",
    ],
    "PNN-Transformer": [
        "outputs/pnn_transformer/runs/PNN-TRANSFORMER_2026-09-09_01-09-51_seed43",
        "outputs/pnn_transformer/runs/PNN-TRANSFORMER_2026-09-09_07-39-42_seed44",
    ],
}

COLORS = {
    "MATD3": "#2F6B9A",
    "PNN": "#D5A021",
    "MATD3-Transformer": "#D97841",
    "PNN-Transformer": "#7A8B45",
}

LINESTYLES = {
    "MATD3": "--",
    "PNN": ":",
    "MATD3-Transformer": "-.",
    "PNN-Transformer": "-",
}


def read_json(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def training_curve(run_directory: Path) -> pd.Series:
    history = read_json(run_directory / "training_history.json")
    episodes = [int(item.get("episode", index + 1)) for index, item in enumerate(history)]
    values = pd.Series(
        [float(item["team_profit"]) for item in history],
        index=pd.Index(episodes, name="episode"),
        dtype=float,
    )
    return values.rolling(window=500, min_periods=500).mean()


def validation_curve(run_directory: Path) -> pd.Series:
    history = read_json(run_directory / "validation_history.json")
    return pd.Series(
        [float(item["team_profit_mean"]) for item in history],
        index=pd.Index([int(item["episode"]) for item in history], name="episode"),
        dtype=float,
    )


def combined_frame(algorithm: str, filename: str) -> pd.DataFrame:
    curves = []
    for relative_path in RUNS[algorithm]:
        run_directory = PROJECT_ROOT / relative_path
        curve = training_curve(run_directory) if filename == "training" else validation_curve(run_directory)
        curves.append(curve)
    return pd.concat(curves, axis=1)


def draw_curve(ax: plt.Axes, algorithm: str, frame: pd.DataFrame, *, linewidth: float) -> None:
    x = frame.index.to_numpy(dtype=float)
    mean = frame.mean(axis=1).to_numpy(dtype=float)
    color = COLORS[algorithm]
    ax.plot(
        x,
        mean,
        color=color,
        linestyle=LINESTYLES[algorithm],
        linewidth=linewidth,
        label=algorithm,
    )
    if frame.shape[1] > 1:
        low = frame.min(axis=1).to_numpy(dtype=float)
        high = frame.max(axis=1).to_numpy(dtype=float)
        ax.fill_between(x, low, high, color=color, alpha=0.11, linewidth=0)


def main() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#263238",
            "axes.labelcolor": "#263238",
            "axes.titlecolor": "#263238",
            "xtick.color": "#263238",
            "ytick.color": "#263238",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "legend.frameon": False,
            "grid.color": "#E6E9EB",
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )

    figure, axes = plt.subplots(2, 1, figsize=(11.2, 8.1), sharex=True)
    figure.suptitle("Training-Curve Comparison with a 24-Hour Forecast Window", fontsize=16, y=0.985)
    figure.text(
        0.5,
        0.952,
        "Team profit per 24-hour episode; higher values indicate better economic performance",
        ha="center",
        va="top",
        color="#59636A",
        fontsize=10.5,
    )

    for algorithm in RUNS:
        draw_curve(axes[0], algorithm, combined_frame(algorithm, "training"), linewidth=2.15)
        draw_curve(axes[1], algorithm, combined_frame(algorithm, "validation"), linewidth=2.0)

    axes[0].set_title("(a) Training team profit — 500-episode moving average", loc="left", fontsize=12)
    axes[1].set_title("(b) Validation mean team profit — evaluated every 100 episodes", loc="left", fontsize=12)
    axes[0].set_ylabel("Team profit")
    axes[1].set_ylabel("Team profit")
    axes[1].set_xlabel("Training episode")

    for ax in axes:
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0)
        ax.ticklabel_format(axis="x", style="plain")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        columnspacing=1.6,
        handlelength=3.2,
    )
    figure.subplots_adjust(top=0.89, bottom=0.11, left=0.095, right=0.98, hspace=0.28)

    png_path = OUTPUT_DIR / "combined_training_curves.png"
    pdf_path = OUTPUT_DIR / "combined_training_curves.pdf"
    figure.savefig(png_path)
    figure.savefig(pdf_path)
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
