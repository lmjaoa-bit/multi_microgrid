from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.reward_calibration import collect_calibration


def _select_days(all_days: list[str], count: int) -> list[str]:
    if count <= 0:
        raise ValueError("--days must be positive")
    if not all_days:
        raise ValueError("No valid natural days are available")
    if count >= len(all_days):
        return list(all_days)
    indices = np.linspace(0, len(all_days) - 1, num=count, dtype=int)
    return [all_days[int(index)] for index in indices]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect Reward V2 calibration evidence from train/validation days"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--days", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("idle", "service_priority"),
        default="service_priority",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_project_root"] = str(config_path.parent.parent)

    output_dir = args.output.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(
            "Calibration --output must be an existing diagnostics directory"
        )
    if output_dir.name != "diagnostics":
        raise ValueError(
            "Calibration --output must point to the run's diagnostics directory"
        )
    run_dir = output_dir.parent
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Calibration diagnostics must belong to an existing working run "
            "with run_metadata.json"
        )

    env = MultiMicrogridEnv(config, split=args.split)
    all_days = [str(day.date()) for day in env.data.valid_days]
    selected = _select_days(all_days, int(args.days))
    report = collect_calibration(
        config,
        split=args.split,
        days=selected,
        policy=args.policy,
        seed=int(args.seed),
    )
    report_path = output_dir / "reward_calibration.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["run_purpose"] = "reward_calibration"
    metadata["formal_test_completed"] = False
    metadata["reward_calibration_report"] = str(
        report_path.relative_to(run_dir)
    )
    metadata["reward_calibration_split"] = args.split
    metadata["reward_calibration_days"] = selected
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
