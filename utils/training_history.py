"""Append-only episode journal, with compatible JSON snapshots on demand."""
import json
import warnings
from pathlib import Path

from utils.metrics_collector import json_ready, write_json


class TrainingHistoryJournal:
    def __init__(self, path, *, resume=False, completed_records=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            if not self.path.is_file():
                raise FileNotFoundError(f"Training history journal not found: {self.path}")
            if completed_records is not None:
                count = int(completed_records)
                if count < 0:
                    raise ValueError("completed_records must be non-negative")
                lines = self.path.read_bytes().splitlines(keepends=True)
                if len(lines) < count or any(not line.endswith(b"\n") for line in lines[:count]):
                    raise ValueError("Training history journal is shorter than the durable cursor")
                for line in lines[:count]:
                    json.loads(line)
                temporary = self.path.with_name(f".{self.path.name}.tmp")
                try:
                    temporary.write_bytes(b"".join(lines[:count]))
                    temporary.replace(self.path)
                finally:
                    temporary.unlink(missing_ok=True)
        else:
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
