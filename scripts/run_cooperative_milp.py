from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimization.cooperative_dispatch import solve_cooperative_dispatch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cooperative social-planner oracle")
    parser.add_argument("--config", type=Path, default=Path("configs/cooperative_v1.yaml"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--day")
    parser.add_argument("--output", type=Path, default=Path("outputs/cooperative_milp"))
    args = parser.parse_args(argv)
    result = solve_cooperative_dispatch(args.config, args.split, args.day)
    args.output.mkdir(parents=True, exist_ok=True)
    result.trajectories.to_csv(args.output / "trajectories.csv", index=False)
    summary = {**result.summary(), "pair_flows_kwh": result.pair_flows_kwh}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
