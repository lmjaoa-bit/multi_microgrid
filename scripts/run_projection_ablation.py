from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.environment_factory import build_environment
import evaluation.evaluate_matd3 as evaluate_module
import experiments.training_pipeline as training_module


ABLATION_CONFIG = PROJECT_ROOT / "configs" / "ablations" / "cooperative_instantaneous_projection.yaml"


class _ProjectionAwareEnvironment:
    """Constructor-compatible proxy used only by this isolated ablation runner."""

    def __new__(cls, config, split: str = "train"):
        return build_environment(config, split=split)


def _activate_projection_factory() -> None:
    # The shared pipeline/evaluator intentionally keep the canonical strong
    # MultiMicrogridEnv import. This runner swaps only their constructor globals
    # for this process, so the baseline command and code path stay untouched.
    training_module.MultiMicrogridEnv = _ProjectionAwareEnvironment
    evaluate_module.MultiMicrogridEnv = _ProjectionAwareEnvironment


def main(argv: list[str] | None = None) -> int:
    _activate_projection_factory()
    from scripts import run_experiment

    args = list(sys.argv[1:] if argv is None else argv)
    if "--algorithm" not in args:
        args.extend(["--algorithm", "matd3_dispatch"])
    if "--config" not in args:
        args.extend(["--config", str(ABLATION_CONFIG)])
    return run_experiment.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
