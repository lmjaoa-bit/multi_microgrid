import json

import numpy as np
import pytest
import torch

from utils.training_history import TrainingHistoryJournal, recover_training_history
from utils.torch_runtime import configure_torch_runtime


def test_history_journal_appends_only_new_records_and_recovers(tmp_path):
    path = tmp_path / "history.jsonl"
    journal = TrainingHistoryJournal(path)
    record = {"episode": 1, "value": np.float32(1.5), "mg3": {"offer": 4.}}
    journal.append(record)
    first = path.read_bytes()
    record["episode"] = 2
    journal.append(record)
    assert path.read_bytes().startswith(first)
    output = tmp_path / "history.json"
    assert recover_training_history(path, output) == 2
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert [row["episode"] for row in rows] == [1, 2]
    assert rows[0]["value"] == 1.5
    assert rows[1]["mg3"]["offer"] == 4.
    with pytest.raises(FileExistsError):
        TrainingHistoryJournal(path)


def test_history_recovery_ignores_only_incomplete_tail(tmp_path):
    path = tmp_path / "history.jsonl"
    journal = TrainingHistoryJournal(path)
    journal.append({"episode": 1})
    with path.open("ab") as handle:
        handle.write(b'{"episode": 2, "partial": "\xe4')
    with pytest.warns(RuntimeWarning, match="incomplete final"):
        assert recover_training_history(path, tmp_path / "recovered.json") == 1
    with path.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises((ValueError, UnicodeDecodeError)):
        recover_training_history(path, tmp_path / "corrupt.json")


def test_history_nonfinite_values_are_sanitized(tmp_path):
    path = tmp_path / "history.jsonl"
    journal = TrainingHistoryJournal(path)
    with pytest.warns(RuntimeWarning, match="Non-finite"):
        journal.append({"episode": 1, "bad": float("nan")})
    assert json.loads(path.read_text())["bad"] is None


def test_cpu_threads_are_opt_in_and_validated():
    original = torch.get_num_threads()
    try:
        assert configure_torch_runtime({}, "cpu")["cpu_threads"] == original
        assert configure_torch_runtime({"runtime": {"cpu_threads": 1}}, "cpu")["cpu_threads"] == 1
        for invalid in (0, -1, True, 1.5, "1"):
            with pytest.raises(ValueError, match="positive integer"):
                configure_torch_runtime({"runtime": {"cpu_threads": invalid}}, "cpu")
        configure_torch_runtime({"runtime": {"cpu_threads": 2}}, "cuda:0")
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)
