from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.paper_artifacts import write_comparison_tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build MATD3/candidate paper metric tables from two completed saved runs",
    )
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = write_comparison_tables(
        args.baseline_run,
        args.candidate_run,
        args.output_dir,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
