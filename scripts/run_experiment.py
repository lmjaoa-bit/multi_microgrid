from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.layout import DEFAULT_WORKING_OUTPUT_ROOT
from experiments.result_lifecycle import capture_git_identity, update_run_metadata_git_identity
from experiments.training_pipeline import build_parser, train


def build_managed_parser():
    parser = build_parser()
    parser.set_defaults(output_root=DEFAULT_WORKING_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    identity = capture_git_identity(PROJECT_ROOT)
    summary = train(build_managed_parser().parse_args(argv))
    update_run_metadata_git_identity(Path(str(summary["run_directory"])), identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
