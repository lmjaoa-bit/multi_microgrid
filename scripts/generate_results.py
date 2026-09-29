from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.plotting import generate_figures
from evaluation.service_reliability import augment_run_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_results(run_dir: str | Path) -> Path:
    """Regenerate saved-result reports without executing environment or models."""
    run_dir = Path(run_dir).resolve()
    result_dir = run_dir / "results"
    result_dir.mkdir(exist_ok=True)

    # Both consumers operate only on already persisted JSON/CSV artifacts.
    # Missing required evidence is an error; PLOTTING_PENDING is not a valid
    # substitute for the canonical 15-figure contract.
    augment_run_outputs(run_dir, generate_plot=False)
    paths = generate_figures(run_dir)
    manifest = result_dir / "generated_figures.txt"
    manifest.write_text(
        "\n".join(str(path.relative_to(run_dir)) for path in paths) + "\n",
        encoding="utf-8",
    )
    return result_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    generate_results(args.run_dir)
