from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.layout import artifact_directory, figure_directory, run_root

from evaluation.dr_reference import evaluate_and_save_dr_reference
from evaluation.plotting import generate_figures


def build_reference(run_dir: str | Path, *, regenerate_figures: bool = True) -> list[Path]:
    """Backfill the deterministic DR reference for an existing completed run."""

    root = artifact_directory(run_dir)
    snapshot_path = root / "config_snapshot.yaml"
    daily_path = root / "test_daily_metrics.csv"
    if not snapshot_path.exists() or not daily_path.exists():
        raise FileNotFoundError(
            "Backfill requires config_snapshot.yaml and test_daily_metrics.csv"
        )
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
    env_config = dict(snapshot.get("environment", snapshot))
    env_config["_project_root"] = str(PROJECT_ROOT)
    daily = pd.read_csv(daily_path)
    if daily.empty or "date" not in daily:
        raise ValueError("test_daily_metrics.csv must contain at least one date")
    days = daily["date"].astype(str).tolist()
    seed = int(daily["test_seed"].iloc[0]) if "test_seed" in daily else 42
    forecast_noise = bool(
        env_config.get("observation", {}).get("evaluation_forecast_noise", False)
    )
    evaluate_and_save_dr_reference(
        env_config,
        split="test",
        days=days,
        seed=seed,
        forecast_noise=forecast_noise,
        output_dir=root,
    )
    if regenerate_figures:
        return generate_figures(root)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the saved deterministic DR reference for an existing run"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    written = build_reference(args.run, regenerate_figures=not args.no_plot)
    print(f"Saved deterministic DR reference under {args.run.resolve()}")
    if written:
        print(f"Regenerated {len(written)} figure files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
