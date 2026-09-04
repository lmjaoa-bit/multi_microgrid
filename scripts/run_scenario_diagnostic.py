from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnostics.scenario_diagnostic import diagnose_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose microgrid complementarity")
    parser.add_argument("--config", type=Path, default=Path("configs/cooperative.yaml"))
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path, default=Path("outputs/scenario_diagnostic.json"))
    args = parser.parse_args(argv)
    result = diagnose_scenario(args.config, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
