from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.plotting import generate_figures
from experiments.layout import artifact_directory, run_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate algorithm figures from saved result files without running the environment",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--png-only", action="store_true")
    args = parser.parse_args(argv)
    formats = ("png",) if args.png_only else ("png", "pdf")
    paths = generate_figures(args.run_dir, formats=formats)
    summary_path = artifact_directory(args.run_dir) / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(plotting_succeeded=True, plotting_error=None,
                       figures_generated=[path.relative_to(run_root(args.run_dir)).as_posix() for path in paths])
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_index = artifact_directory(args.run_dir) / "results" / "generated_figures.txt"
    if figure_index.is_file():
        figure_index.write_text("\n".join(path.relative_to(run_root(args.run_dir)).as_posix()
                                         for path in paths) + "\n", encoding="utf-8")
    print(f"Generated {len(paths)} files in {Path(args.run_dir).resolve() / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
