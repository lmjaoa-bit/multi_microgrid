"""Append-only episode journal, with compatible JSON snapshots on demand."""
import json
import warnings
from pathlib import Path

from utils.metrics_collector import json_ready, write_json


class TrainingHistoryJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A run owns its journal. Never silently append another run to it.
        with self.path.open("x", encoding="utf-8"):
            pass

    def append(self, record):
        clean, invalid = json_ready(record)
        if invalid:
            warnings.warn(f"Non-finite history fields replaced with null: {invalid}",
                          RuntimeWarning, stacklevel=2)
        payload = json.dumps(clean, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        # Close/flush every episode; no persistent buffered handle on interruption.
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")


def recover_training_history(journal_path, output_path):
    """Rebuild a JSON snapshot without rerunning any policy or training step.

    Only an unterminated final line may be discarded after an interrupted write;
    malformed complete records raise instead of silently hiding corruption.
    This recovers metrics, not optimizer/replay state or resumable training.
    """
    records = []
    with Path(journal_path).open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if not line.endswith(b"\n"):
                warnings.warn(f"Ignoring incomplete final journal record {number}",
                              RuntimeWarning, stacklevel=2)
                break
            records.append(json.loads(line))
    write_json(output_path, records)
    return len(records)
