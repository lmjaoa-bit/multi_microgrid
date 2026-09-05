from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PNN_HISTORY = (
    ROOT
    / "outputs"
    / "pnn"
    / "runs"
    / "PNN_2026-09-05_18-11-21_seed42"
    / "validation_history.json"
)
MATD3_HISTORY = (
    ROOT
    / "outputs"
    / "matd3_plain_single_dispatch"
    / "runs"
    / "MATD3PLAINA3_2026-09-04_16-19-49_seed42"
    / "validation_history.json"
)


def load_validation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    episodes = np.asarray([int(row["episode"]) for row in rows])
    profit = np.asarray([float(row["team_profit_mean"]) for row in rows])
    if episodes.size != 50 or not np.isfinite(profit).all():
        raise ValueError(f"Unexpected validation history in {path}")
    return episodes, profit


def main() -> None:
    pnn_ep, pnn = load_validation(PNN_HISTORY)
    matd3_ep, matd3 = load_validation(MATD3_HISTORY)
    if not np.array_equal(pnn_ep, matd3_ep):
        raise ValueError("Validation checkpoints are not aligned")

    pnn_best = np.maximum.accumulate(pnn)
    matd3_best = np.maximum.accumulate(matd3)
    best_gap = pnn_best - matd3_best
    pnn_best_idx = int(np.argmax(pnn))
    matd3_best_idx = int(np.argmax(matd3))

    blue = "#2563A6"
    orange = "#D97706"
    ink = "#20242A"
    grid = "#D9DEE5"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.edgecolor": "#9AA3AE",
            "axes.labelcolor": ink,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )

    fig = plt.figure(figsize=(11.5, 7.3), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=(3.1, 1.25), hspace=0.12)
    ax = fig.add_subplot(gs[0])
    gap_ax = fig.add_subplot(gs[1], sharex=ax)

    # The checkpoint values show real validation fluctuation; the cumulative-best
    # traces show the learning frontier used by best-checkpoint selection.
    ax.plot(
        matd3_ep,
        matd3,
        color=orange,
        linewidth=1.0,
        alpha=0.30,
        marker="o",
        markersize=2.5,
        label="MATD3 checkpoint score",
    )
    ax.plot(
        pnn_ep,
        pnn,
        color=blue,
        linewidth=1.0,
        alpha=0.30,
        marker="o",
        markersize=2.5,
        label="PNN checkpoint score",
    )
    ax.step(
        matd3_ep,
        matd3_best,
        where="post",
        color=orange,
        linewidth=2.35,
        label="MATD3 best so far",
    )
    ax.step(
        pnn_ep,
        pnn_best,
        where="post",
        color=blue,
        linewidth=2.5,
        label="PNN best so far",
    )
    ax.scatter(
        [matd3_ep[matd3_best_idx]],
        [matd3[matd3_best_idx]],
        s=60,
        facecolor="white",
        edgecolor=orange,
        linewidth=2,
        zorder=5,
    )
    ax.scatter(
        [pnn_ep[pnn_best_idx]],
        [pnn[pnn_best_idx]],
        s=60,
        facecolor="white",
        edgecolor=blue,
        linewidth=2,
        zorder=5,
    )
    ax.annotate(
        f"PNN best: {pnn[pnn_best_idx]:.2f}\nEpisode {pnn_ep[pnn_best_idx]:,}",
        xy=(pnn_ep[pnn_best_idx], pnn[pnn_best_idx]),
        xytext=(-78, 28),
        textcoords="offset points",
        color=blue,
        arrowprops={"arrowstyle": "-", "color": blue, "lw": 1},
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CBD5E1"},
    )
    ax.annotate(
        f"MATD3 best: {matd3[matd3_best_idx]:.2f}\nEpisode {matd3_ep[matd3_best_idx]:,}",
        xy=(matd3_ep[matd3_best_idx], matd3[matd3_best_idx]),
        xytext=(-116, -52),
        textcoords="offset points",
        color=orange,
        arrowprops={"arrowstyle": "-", "color": orange, "lw": 1},
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CBD5E1"},
    )
    ax.set_ylabel("Mean team profit on fixed validation days")
    ax.grid(axis="y", color=grid, linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=8.9)

    gap_ax.axhline(0, color="#475569", linewidth=1.0)
    gap_ax.plot(pnn_ep, best_gap, color=blue, linewidth=2.0)
    gap_ax.fill_between(
        pnn_ep,
        0,
        best_gap,
        where=best_gap >= 0,
        color=blue,
        alpha=0.18,
        step="post",
    )
    gap_ax.fill_between(
        pnn_ep,
        0,
        best_gap,
        where=best_gap < 0,
        color=orange,
        alpha=0.20,
        step="post",
    )
    gap_ax.annotate(
        f"Final best-score gap: {best_gap[-1]:+.2f}",
        xy=(pnn_ep[-1], best_gap[-1]),
        xytext=(-12, 12),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=ink,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CBD5E1"},
    )
    gap_ax.set_xlabel("Training episode")
    gap_ax.set_ylabel("Best-score gap\nPNN - MATD3")
    gap_ax.grid(axis="y", color=grid, linewidth=0.7, alpha=0.8)
    gap_ax.spines[["top", "right"]].set_visible(False)
    gap_ax.set_xlim(100, 5000)

    fig.suptitle(
        "PNN vs MATD3 Learning Curve",
        x=0.08,
        y=0.975,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=ink,
    )
    fig.text(
        0.08,
        0.932,
        "Fixed validation scenarios every 100 episodes | higher profit is better | focused y-axis",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#5B6470",
    )
    fig.text(
        0.08,
        0.012,
        "Thin lines: score at each checkpoint. Bold step lines: best score observed so far.",
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#6B7280",
    )
    fig.subplots_adjust(left=0.105, right=0.975, top=0.895, bottom=0.105)

    png_path = OUT / "pnn_vs_matd3_validation_learning_curve.png"
    pdf_path = OUT / "pnn_vs_matd3_validation_learning_curve.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(
        json.dumps(
            {
                "png": str(png_path),
                "pdf": str(pdf_path),
                "pnn_best": float(pnn[pnn_best_idx]),
                "pnn_best_episode": int(pnn_ep[pnn_best_idx]),
                "matd3_best": float(matd3[matd3_best_idx]),
                "matd3_best_episode": int(matd3_ep[matd3_best_idx]),
                "final_best_gap": float(best_gap[-1]),
                "pnn_gain_from_first_checkpoint": float(pnn_best[-1] - pnn_best[0]),
                "matd3_gain_from_first_checkpoint": float(matd3_best[-1] - matd3_best[0]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
