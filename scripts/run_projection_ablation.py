from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluation.evaluate_matd3 as evaluate_module
import experiments.training_pipeline as training_module
from envs.instantaneous_projection_env import InstantaneousProjectionEnv
from evaluation.service_reliability import augment_run_outputs
from experiments.result_lifecycle import capture_git_identity, update_run_metadata_git_identity
from scripts import run_experiment
from utils.metrics_collector import write_json


ABLATION_CONFIG = PROJECT_ROOT / "configs" / "cooperative_instantaneous_projection.yaml"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "working_runs" / "ablations" / "instantaneous_projection"
)


def _activate_instantaneous_environment() -> None:
    # The baseline command imports/uses MultiMicrogridEnv normally. Only this
    # dedicated process swaps the constructors used by training, validation and
    # formal test, so strong and instantaneous runs cannot leak into one another.
    training_module.MultiMicrogridEnv = InstantaneousProjectionEnv
    evaluate_module.MultiMicrogridEnv = InstantaneousProjectionEnv


def _record_projection_mode(run_dir: Path) -> None:
    for name in ("run_metadata.json", "training_summary.json", "summary.json"):
        path = run_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["constraint_projection_mode"] = "instantaneous"
            write_json(path, payload)


def main(argv: list[str] | None = None) -> int:
    _activate_instantaneous_environment()
    args = list(sys.argv[1:] if argv is None else argv)
    if "--algorithm" not in args:
        args.extend(["--algorithm", "matd3_dispatch"])
    if "--config" not in args:
        args.extend(["--config", str(ABLATION_CONFIG)])
    if "--output-root" not in args and "--output" not in args:
        args.extend(["--output-root", str(DEFAULT_OUTPUT_ROOT)])

    identity = capture_git_identity(PROJECT_ROOT)
    parsed = run_experiment._apply_algorithm_runtime_defaults(
        run_experiment.build_managed_parser().parse_args(args)
    )
    summary = training_module.train(parsed)
    run_dir = Path(str(summary["run_directory"]))
    augment_run_outputs(run_dir)
    update_run_metadata_git_identity(run_dir, identity)
    _record_projection_mode(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
