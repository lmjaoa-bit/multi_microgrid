"""Recover a standard history snapshot from a run's incremental journal."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.training_history import recover_training_history
from experiments.layout import artifact_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path,
                        help="Defaults to training_history.recovered.json; never overwrites by default")
    args = parser.parse_args()
    data_dir = artifact_directory(args.run_directory)
    output = args.output or data_dir / "training_history.recovered.json"
    if output.exists():
        raise FileExistsError(output)
    count = recover_training_history(data_dir / "training_history.jsonl", output)
    print(f"Recovered {count} episodes to {output}")
