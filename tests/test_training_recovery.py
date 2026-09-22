from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.base import RecoverableAlgorithmProtocol
from algorithms.runtime_helpers import FastReplayBuffer
from algorithms.training_recovery import (
    _atomic_torch_save,
    _atomic_write_json,
    read_transition_chunk,
    sha256_file,
    validate_fingerprints,
    write_transition_chunk,
)


AGENTS = ["mg1", "mg2", "mg3"]


def _add_numbered_transition(replay: FastReplayBuffer, number: int) -> None:
    observations = {mg: np.array([number, i], dtype=np.float32) for i, mg in enumerate(AGENTS)}
    actions = {mg: np.array([number, -number], dtype=np.float32) for mg in AGENTS}
    replay.add(
        observations,
        np.array([number], dtype=np.float32),
        actions,
        float(number),
        {mg: value + 1 for mg, value in observations.items()},
        np.array([number + 1], dtype=np.float32),
        number % 4 == 3,
    )


def test_fast_replay_restore_preserves_ring_order_position_and_rng() -> None:
    original = FastReplayBuffer(capacity=5, seed=17)
    for number in range(8):
        _add_numbered_transition(original, number)

    logical = original.logical_transitions()
    metadata = original.state_metadata()
    assert [int(row.global_state[0]) for row in logical] == [3, 4, 5, 6, 7]

    restored = FastReplayBuffer(capacity=5, seed=999)
    restored.restore_logical(logical, metadata)

    assert restored._next_index == original._next_index
    assert restored._total_added == 8
    assert [int(row.global_state[0]) for row in restored.logical_transitions()] == [3, 4, 5, 6, 7]
    for _ in range(3):
        left = original.sample(3, AGENTS, torch.device("cpu"))["global_state"]
        right = restored.sample(3, AGENTS, torch.device("cpu"))["global_state"]
        assert torch.equal(left, right)


@pytest.mark.parametrize(
    "metadata_patch,match",
    [
        ({"capacity": 6}, "capacity"),
        ({"size": 4}, "size"),
        ({"next_index": 5}, "next_index"),
        ({"total_added": 4}, "total_added"),
    ],
)
def test_fast_replay_restore_rejects_incompatible_metadata(metadata_patch, match) -> None:
    replay = FastReplayBuffer(capacity=5, seed=1)
    for number in range(5):
        _add_numbered_transition(replay, number)
    metadata = replay.state_metadata()
    metadata.update(metadata_patch)
    with pytest.raises(ValueError, match=match):
        FastReplayBuffer(capacity=5).restore_logical(replay.logical_transitions(), metadata)


def test_optional_recovery_protocol_does_not_change_base_algorithm_contract() -> None:
    class Recoverable:
        def save_training_state(self, directory: Path, progress):
            return {"directory": str(directory), "progress": dict(progress)}

        def load_training_state(self, directory: Path):
            return {"directory": str(directory)}

    class Ordinary:
        pass

    assert isinstance(Recoverable(), RecoverableAlgorithmProtocol)
    assert not isinstance(Ordinary(), RecoverableAlgorithmProtocol)


def test_atomic_writers_and_transition_chunk_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "state.json"
    tensor_path = tmp_path / "state.pt"
    _atomic_write_json({"episode": 7}, json_path)
    _atomic_torch_save({"weight": torch.tensor([2.0])}, tensor_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"episode": 7}
    assert torch.equal(torch.load(tensor_path, weights_only=False)["weight"], torch.tensor([2.0]))
    assert len(sha256_file(tensor_path)) == 64
    assert not list(tmp_path.glob("*.tmp"))

    replay = FastReplayBuffer(capacity=5, seed=3)
    for number in range(3):
        _add_numbered_transition(replay, number)
    chunk = tmp_path / "chunk.pt"
    digest = write_transition_chunk(chunk, replay.logical_transitions())
    restored = read_transition_chunk(chunk, expected_sha256=digest)
    assert [int(row.global_state[0]) for row in restored] == [0, 1, 2]


def test_fingerprint_validation_names_the_mismatch() -> None:
    expected = {"algorithm_slug": "peak", "schema": ["a", "b"]}
    validate_fingerprints(expected, dict(expected))
    with pytest.raises(ValueError, match="schema"):
        validate_fingerprints(expected, {"algorithm_slug": "peak", "schema": ["a"]})
