from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from algorithms.matd3.replay_buffer import Transition


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.{uuid4().hex}.tmp")


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_transition_chunk(
    path: str | Path,
    transitions: Sequence[Transition],
) -> str:
    target = Path(path)
    _atomic_torch_save(list(transitions), target)
    return sha256_file(target)


def read_transition_chunk(
    path: str | Path,
    *,
    expected_sha256: str,
) -> list[Transition]:
    source = Path(path)
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError(
            f"Replay chunk hash mismatch for {source.name}: expected {expected_sha256}, got {actual}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, list) or not all(isinstance(row, Transition) for row in payload):
        raise ValueError(f"Invalid replay transition chunk: {source}")
    return payload


def validate_fingerprints(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise ValueError(f"Training-state fingerprint is missing {key}")
        if actual[key] != expected_value:
            raise ValueError(
                f"Training-state fingerprint mismatch for {key}: "
                f"expected {expected_value!r}, got {actual[key]!r}"
            )


def save_incremental_training_state(
    directory: str | Path,
    *,
    replay: Any,
    state_payload: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    previous: dict[str, Any] | None = None
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_fingerprints(fingerprints, previous.get("fingerprints", {}))

    metadata = replay.state_metadata()
    total_added = int(metadata["total_added"])
    size = int(metadata["size"])
    earliest = total_added - size
    logical = replay.logical_transitions()

    chunks: list[dict[str, Any]] = []
    start = earliest
    if previous is not None:
        prior_replay = previous.get("replay", {})
        prior_last = int(prior_replay.get("last_sequence_id", -1))
        if prior_last > total_added:
            raise ValueError(
                f"Recovery replay moved backwards: prior={prior_last}, current={total_added}"
            )
        if prior_last >= earliest:
            chunks = [
                dict(item)
                for item in prior_replay.get("chunks", [])
                if int(item["last_sequence_id"]) > earliest
            ]
            start = prior_last

    if start < total_added:
        offset = start - earliest
        if offset < 0 or offset > len(logical):
            raise ValueError("Incremental replay range is not available in memory")
        chunk_name = f"replay-{start:012d}-{total_added:012d}.pt"
        digest = write_transition_chunk(target / chunk_name, logical[offset:])
        chunks.append({
            "path": chunk_name,
            "sha256": digest,
            "first_sequence_id": start,
            "last_sequence_id": total_added,
        })

    state_name = f"state-{total_added:012d}-{uuid4().hex}.pt"
    state_path = target / state_name
    _atomic_torch_save(dict(state_payload), state_path)
    manifest = {
        "format_version": 1,
        "fingerprints": dict(fingerprints),
        "progress": dict(progress),
        "state": {"path": state_name, "sha256": sha256_file(state_path)},
        "replay": {
            "metadata": metadata,
            "first_sequence_id": earliest,
            "last_sequence_id": total_added,
            "chunks": chunks,
        },
    }
    _atomic_write_json(manifest, manifest_path)
    return manifest


def load_incremental_training_state(
    directory: str | Path,
    *,
    expected_fingerprints: Mapping[str, Any],
    map_location: Any = "cpu",
) -> tuple[dict[str, Any], list[Transition], dict[str, Any], dict[str, Any]]:
    source = Path(directory)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training-state manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported training-state format: {manifest.get('format_version')}")
    validate_fingerprints(expected_fingerprints, manifest.get("fingerprints", {}))

    state_info = manifest.get("state", {})
    state_path = source / str(state_info.get("path", ""))
    expected_state_hash = str(state_info.get("sha256", ""))
    if not state_path.is_file() or sha256_file(state_path) != expected_state_hash:
        raise ValueError(f"Training-state payload hash mismatch: {state_path}")
    state_payload = torch.load(state_path, map_location=map_location, weights_only=False)
    if not isinstance(state_payload, dict):
        raise ValueError("Training-state payload must be a mapping")

    replay_info = manifest.get("replay", {})
    first = int(replay_info.get("first_sequence_id", 0))
    last = int(replay_info.get("last_sequence_id", 0))
    transitions: list[Transition] = []
    expected_start: int | None = None
    for chunk_info in replay_info.get("chunks", []):
        chunk_first = int(chunk_info["first_sequence_id"])
        chunk_last = int(chunk_info["last_sequence_id"])
        if expected_start is not None and chunk_first != expected_start:
            raise ValueError(
                f"Non-contiguous replay chunks: expected {expected_start}, got {chunk_first}"
            )
        chunk = read_transition_chunk(
            source / str(chunk_info["path"]),
            expected_sha256=str(chunk_info["sha256"]),
        )
        if len(chunk) != chunk_last - chunk_first:
            raise ValueError(f"Replay chunk range/length mismatch: {chunk_info['path']}")
        transitions.extend(chunk)
        expected_start = chunk_last

    metadata = dict(replay_info.get("metadata", {}))
    size = int(metadata.get("size", -1))
    total_added = int(metadata.get("total_added", -1))
    if total_added != last or first != last - size:
        raise ValueError("Replay manifest range does not match replay metadata")
    if size:
        if not transitions or expected_start != last or len(transitions) < size:
            raise ValueError("Replay chunks do not contain the committed logical buffer")
        transitions = transitions[-size:]
    else:
        transitions = []
    progress = dict(manifest.get("progress", {}))
    return state_payload, transitions, metadata, progress
