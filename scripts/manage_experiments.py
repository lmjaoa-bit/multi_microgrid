"""List experiment contracts or intentionally change defaults for future runs."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep the repository's numpy-before-torch import order on Windows.
import numpy as np
from experiments.profiles import catalog, default_profile, set_default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--set-default", metavar="PROFILE")
    group.add_argument("--clear-default", action="store_true")
    group.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.set_default:
        set_default(args.set_default)
    elif args.clear_default:
        set_default(None)
    print("Default experiment:", default_profile() or "matd3 (default fallback)")
    print("PROFILE                       POLICY      ACTION             ENVIRONMENT")
    for name, item in catalog("experiment_profiles").items():
        print(f"{name:29} {item['policy']:11} {item['action_design']:18} {item['environment']}")
    print("Explicit --algorithm and --experiment always override the global default.")
    print("No existing checkpoint, result, or physical setting is modified.")


if __name__ == "__main__":
    main()
