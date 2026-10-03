"""Run the compact read-only critic diagnostic from the repository root."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnostics.critic_probe import SLUGS, run_probe


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare current Critic rankings with frozen-policy realized returns; no training/MILP")
    parser.add_argument("--matd3-run", type=Path, help="Explicit completed MATD3-Q run directory; default: newest eligible full run")
    parser.add_argument("--tq-run", type=Path, help="Explicit completed T-Q readout run directory; default: newest eligible full run")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--states", type=int, help="Total states; default: six per selected day and forecast seed")
    parser.add_argument("--days", nargs="+", help="Explicit validation dates; default: one per season")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--days-per-season", type=int, choices=(1, 3, 6), default=1,
                        help="Calendar sampling: 1 per quarter, 1 per month, or 2 per month")
    parser.add_argument("--seeds", nargs="+", type=int, help="Forecast base seeds; actual reset seed = base + day index")
    parser.add_argument("--tie-cost", type=float, default=1., help="Ignore pair ordering when realized cost difference is <= this value")
    parser.add_argument("--output", type=Path, help="New, separate diagnostic directory")
    parser.add_argument("--check-only", action="store_true", help="Validate and print sources/scenarios without writing results or rolling candidates")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "outputs/diagnostics/critic_probe" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_probe(root=ROOT, output=output, state_limit=args.states, days=args.days, device=args.device,
              seed=args.seed, forecast_seeds=args.seeds, days_per_season=args.days_per_season,
              tie_cost=args.tie_cost, check_only=args.check_only,
              run_paths={SLUGS[0]: args.matd3_run, SLUGS[1]: args.tq_run})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Diagnostic stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
