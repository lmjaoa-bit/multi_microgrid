from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.service_reliability import augment_run_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add unmet-load, DR supply reliability, and strict-feasibility metrics to a completed run."
    )
    parser.add_argument("run_dir", type=Path, help="Completed experiment run directory")
    parser.add_argument("--no-plot", action="store_true", help="Do not refresh figure 12")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = augment_run_outputs(args.run_dir, generate_plot=not args.no_plot)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
