from __future__ import annotations

# Canonical algorithm-neutral entry point. The old module remains import-compatible.
from evaluation.evaluate_matd3 import evaluate, evaluate_policy_on_days, main

__all__ = ["evaluate", "evaluate_policy_on_days", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
