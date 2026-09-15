from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.result_lifecycle import PromotionError, promote_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage working and retained experiment results")
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote = subparsers.add_parser("promote", help="Copy a completed working run into a named release")
    promote.add_argument("--run", type=Path, required=True, help="Completed working run directory")
    promote.add_argument("--release", required=True, help="Retained release group name")
    promote.add_argument("--releases-root", type=Path, default=Path("outputs") / "releases")
    promote.add_argument("--allow-dirty", action="store_true",
                         help="Allow promotion when recorded Git state is dirty or unknown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = promote_run(
            args.run,
            args.releases_root,
            args.release,
            allow_dirty=args.allow_dirty,
        )
    except PromotionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
