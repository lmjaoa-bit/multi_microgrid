"""Generate 01-style training convergence comparison plot between MATD3 and PNN without validation info."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PNN_RUN = ROOT / "outputs" / "pnn" / "runs" / "PNN_2026-09-05_18-11-21_seed42"
MATD3_RUN = (
    ROOT
    / "outputs"
    / "matd3_plain_single_dispatch"
    / "runs"
    / "MATD3PLAINA3_2026-09-04_16-19-49_seed42"
)
REPORT_DIR = ROOT / "reports" / "pnn_matd3_training_curve_20260905"

INK = "#263238"
LIGHT_GREY = "#E6E9EB"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": LIGHT_GREY,
        "grid.color": LIGHT_GREY,
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def load_run_data(run_dir: Path):
    train_file = run_dir / "training_history.json"
    train_data = json.loads(train_file.read_text(encoding="utf-8"))
    return train_data


def plot_combined_convergence():
    apply_style()

    pnn_train = load_run_data(PNN_RUN)
    matd3_train = load_run_data(MATD3_RUN)

    episodes = np.asarray(
        [item.get("episode", idx + 1) for idx, item in enumerate(pnn_train)],
        dtype=float,
    )

    # MATD3 series
    m_raw = np.asarray([item.get("team_profit", np.nan) for item in matd3_train], dtype=float)
    m_ma = pd.Series(m_raw).rolling(100, min_periods=1).mean().to_numpy()

    # PNN series
    p_raw = np.asarray([item.get("team_profit", np.nan) for item in pnn_train], dtype=float)
    p_ma = pd.Series(p_raw).rolling(100, min_periods=1).mean().to_numpy()

    # Color definitions
    C_MATD3_RAW = "#85C1E9"
    C_MATD3_MA = "#1B4F72"

    C_PNN_RAW = "#F8C471"
    C_PNN_MA = "#D35400"

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    # Raw training clouds
    ax.plot(episodes, m_raw, color=C_MATD3_RAW, alpha=0.12, linewidth=0.4, label="MATD3 Training (raw)")
    ax.plot(episodes, p_raw, color=C_PNN_RAW, alpha=0.12, linewidth=0.4, label="PNN Training (raw)")

    # MA100 lines with 0.5 pt line width
    ax.plot(episodes, m_ma, color=C_MATD3_MA, linewidth=0.5, linestyle="--", label="MATD3 Team Profit MA100")
    ax.plot(episodes, p_ma, color=C_PNN_MA, linewidth=0.5, linestyle="-", label="PNN Team Profit MA100")

    # Final episode line
    ax.axvline(
        episodes[-1],
        color=INK,
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label=f"Final Episode {int(episodes[-1])}",
    )

    ax.set_title("Training Convergence: MATD3 vs. PNN", pad=10)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Team Profit")
    ax.grid(axis="y", linestyle="-", alpha=0.7)

    # Original broad Y-axis range
    ax.set_ylim(-3800, 1500)
    ax.set_xlim(0, 5000)

    ax.legend(loc="lower right", ncol=2, fontsize=9.0)
    fig.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = REPORT_DIR / "pnn_vs_matd3_training_convergence.png"
    out_pdf = REPORT_DIR / "pnn_vs_matd3_training_convergence.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved convergence plot to:\n  {out_png}\n  {out_pdf}")
    return out_png, out_pdf


if __name__ == "__main__":
    plot_combined_convergence()
