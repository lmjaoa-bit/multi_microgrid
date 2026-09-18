from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_results(run_dir: str | Path) -> Path:
    """Generate all post-training figures and summaries for one experiment.

    The function is intentionally separated from training so every algorithm
    (MATD3, Transformer, MILP) can share the same reporting pipeline.
    """
    run_dir = Path(run_dir)
    result_dir = run_dir / "results"
    result_dir.mkdir(exist_ok=True)

    # Plot modules are imported lazily so training does not pay plotting cost.
    try:
        from tools.plotting.plot_all import plot_all
        plot_all(run_dir, result_dir)
    except ImportError:
        # Keep training/evaluation usable before optional plotting modules land.
        (result_dir / "PLOTTING_PENDING.txt").write_text(
            "Plotting modules are not installed yet.\n", encoding="utf-8"
        )

    return result_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    generate_results(args.run_dir)
