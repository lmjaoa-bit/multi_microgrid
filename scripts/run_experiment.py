from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.training_pipeline import build_parser, train
from utils.run_finalizer import collect_git_provenance, finalize_completed_run


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provenance = collect_git_provenance(PROJECT_ROOT)
    summary = train(args)
    finalize_completed_run(
        summary,
        project_root=PROJECT_ROOT,
        provenance=provenance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
