"""Diagnose local Critic/projection/Actor directions without training."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.learning_chain_probe import find_expanded_probe, run_chain_probe


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only local learning-chain diagnosis from a completed expanded Critic probe')
    parser.add_argument('--probe', type=Path, help='Completed expanded phase-one directory; default: newest eligible expanded probe')
    parser.add_argument('--states', type=int, help='Subset for a small smoke check; default: one case per day and forecast seed')
    parser.add_argument('--radii', nargs='+', type=float, default=[.02,.1])
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--output', type=Path, help='New separate diagnostic directory')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    probe = args.probe or find_expanded_probe(ROOT)
    output = args.output or ROOT/'outputs/diagnostics/learning_chain_probe'/datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    run_chain_probe(root=ROOT,probe=probe,output=output,state_limit=args.states,radii=args.radii,
                    device=args.device,check_only=args.check_only)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError,FileNotFoundError) as exc:
        print(f'Learning-chain diagnosis stopped: {exc}',file=sys.stderr)
        raise SystemExit(2)
