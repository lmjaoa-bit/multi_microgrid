from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.training_pipeline import build_parser, main, train

__all__ = ["build_parser", "main", "train"]


if __name__ == "__main__":
    raise SystemExit(main())
