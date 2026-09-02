from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.plotting import generate_figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate algorithm figures from saved result files without running the environment",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--png-only", action="store_true")
    args = parser.parse_args(argv)
    formats = ("png",) if args.png_only else ("png", "pdf")
    paths = generate_figures(args.run_dir, formats=formats)
    print(f"Generated {len(paths)} files in {Path(args.run_dir).resolve() / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
