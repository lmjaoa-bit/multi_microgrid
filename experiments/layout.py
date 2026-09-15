from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_WORKING_OUTPUT_ROOT = Path("outputs") / "working_runs"


@dataclass(frozen=True)
class ExperimentLayout:
    """Stable per-algorithm output layout shared by every experiment."""

    algorithm_root: Path
    runs_root: Path
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    figures_dir: Path
    hourly_dir: Path

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        algorithm_slug: str,
        display_name: str,
        seed: int,
        *,
        algorithm_root_override: str | Path | None = None,
    ) -> "ExperimentLayout":
        algorithm_root = (
            Path(algorithm_root_override).resolve()
            if algorithm_root_override is not None
            else Path(output_root).resolve() / algorithm_slug
        )
        runs_root = algorithm_root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        normalized = "".join(
            character for character in display_name.upper()
            if character.isalnum() or character in {"-", "_"}
        ) or algorithm_slug.upper()
        base = f"{normalized}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_seed{int(seed)}"
        run_dir = runs_root / base
        suffix = 1
        while run_dir.exists():
            run_dir = runs_root / f"{base}_{suffix}"
            suffix += 1
        run_dir.mkdir()
        checkpoints = run_dir / "checkpoints"
        figures = run_dir / "figures"
        hourly = run_dir / "test_hourly_trajectories"
        checkpoints.mkdir()
        figures.mkdir()
        return cls(
            algorithm_root=algorithm_root,
            runs_root=runs_root,
            run_id=run_dir.name,
            run_dir=run_dir.resolve(),
            checkpoints_dir=checkpoints.resolve(),
            figures_dir=figures.resolve(),
            hourly_dir=hourly.resolve(),
        )
