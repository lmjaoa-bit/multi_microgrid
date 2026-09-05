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
    / "training_history.json"
)
MATD3_HISTORY = (
    ROOT
    / "outputs"
    / "matd3_plain_single_dispatch"
    / "runs"
    / "MATD3PLAINA3_2026-09-04_16-19-49_seed42"
    / "training_history.json"
)


def load_reward(path: Path) -> np.ndarray:
    records = json.loads(path.read_text(encoding="utf-8"))
    reward = np.asarray([float(row["team_reward"]) for row in records], dtype=float)
    if reward.size != 5000 or not np.isfinite(reward).all():
        raise ValueError(f"Unexpected reward history in {path}")
    return reward


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.full(values.shape, np.nan, dtype=float)
    result[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return result


def main() -> None:
    pnn = load_reward(PNN_HISTORY)
    matd3 = load_reward(MATD3_HISTORY)
    episodes = np.arange(1, pnn.size + 1)
    smooth_window = 200
    gap_window = 500
    pnn_smooth = trailing_mean(pnn, smooth_window)
    matd3_smooth = trailing_mean(matd3, smooth_window)
    gap = trailing_mean(pnn - matd3, gap_window)

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
    gs = fig.add_gridspec(2, 1, height_ratios=(3.15, 1.25), hspace=0.12)
    ax = fig.add_subplot(gs[0])
    gap_ax = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(episodes, matd3, color=orange, alpha=0.055, linewidth=0.55)
    ax.plot(episodes, pnn, color=blue, alpha=0.055, linewidth=0.55)
    ax.plot(
        episodes,
        matd3_smooth,
        color=orange,
        linewidth=2.15,
        label=f"MATD3 ({smooth_window}-episode mean)",
    )
    ax.plot(
        episodes,
        pnn_smooth,
        color=blue,
        linewidth=2.3,
        label=f"PNN ({smooth_window}-episode mean)",
    )
    ax.set_ylabel("Team training reward")
    ax.grid(axis="y", color=grid, linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=9.5)

    valid = np.isfinite(gap)
    gap_ax.axhline(0, color="#475569", linewidth=1.0)
    gap_ax.plot(episodes, gap, color=blue, linewidth=1.8)
    gap_ax.fill_between(
        episodes,
        0,
        gap,
        where=valid & (gap >= 0),
        color=blue,
        alpha=0.18,
        interpolate=True,
        label="PNN higher",
    )
    gap_ax.fill_between(
        episodes,
        0,
        gap,
        where=valid & (gap < 0),
        color=orange,
        alpha=0.20,
        interpolate=True,
        label="MATD3 higher",
    )
    gap_ax.set_xlabel("Training episode")
    gap_ax.set_ylabel("Reward gap\nPNN - MATD3")
    gap_ax.grid(axis="y", color=grid, linewidth=0.7, alpha=0.75)
    gap_ax.spines[["top", "right"]].set_visible(False)
    gap_ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9.2)
    gap_ax.set_xlim(1, 5000)

    final_500_pnn = float(np.mean(pnn[-500:]))
    final_500_matd3 = float(np.mean(matd3[-500:]))
    final_gap = final_500_pnn - final_500_matd3
    gap_ax.annotate(
        f"Final 500 episodes: {final_gap:+.3f}",
        xy=(5000, gap[-1]),
        xytext=(-12, 12),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=ink,
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CBD5E1"},
    )

    fig.suptitle(
        "PNN vs MATD3 Training Reward",
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
        "Seed 42 | 5,000 episodes | higher reward is better | raw traces shown faintly",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#5B6470",
    )
    fig.text(
        0.08,
        0.012,
        "Source: saved training_history.json files. The lower panel uses a 500-episode trailing mean.",
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#6B7280",
    )
    fig.subplots_adjust(left=0.09, right=0.975, top=0.895, bottom=0.105)

    png_path = OUT / "pnn_vs_matd3_training_reward.png"
    pdf_path = OUT / "pnn_vs_matd3_training_reward.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        json.dumps(
            {
                "png": str(png_path),
                "pdf": str(pdf_path),
                "final_500_pnn_reward": final_500_pnn,
                "final_500_matd3_reward": final_500_matd3,
                "final_500_gap": final_gap,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
