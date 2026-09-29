from __future__ import annotations

import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.registry import get_algorithm_spec
from evaluation.service_reliability import augment_run_outputs
from experiments.layout import DEFAULT_WORKING_OUTPUT_ROOT
from experiments.result_lifecycle import capture_git_identity, update_run_metadata_git_identity
from experiments.training_pipeline import build_parser, train
from scripts.generate_results import generate_results


def build_managed_parser():
    parser = build_parser()
    parser.set_defaults(output_root=DEFAULT_WORKING_OUTPUT_ROOT, device=None)
    return parser


def _apply_algorithm_runtime_defaults(args):
    if args.device is not None:
        return args
    if not args.algorithm:
        args.device = "cpu"
        return args

    spec = get_algorithm_spec(args.algorithm)
    algorithm_path = Path(args.algorithm_config or spec.default_config_path).resolve()
    algorithm_cfg = yaml.safe_load(algorithm_path.read_text(encoding="utf-8")) or {}
    runtime_cfg = algorithm_cfg.get("runtime", {})
    args.device = str(runtime_cfg.get("device", "cpu"))
    return args


def main(argv: list[str] | None = None) -> int:
    identity = capture_git_identity(PROJECT_ROOT)
    args = _apply_algorithm_runtime_defaults(build_managed_parser().parse_args(argv))
    summary = train(args)
    run_dir = Path(str(summary["run_directory"]))
    if bool(summary.get("formal_test_completed", True)):
        augment_run_outputs(run_dir)
        generate_results(run_dir)
    update_run_metadata_git_identity(run_dir, identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
