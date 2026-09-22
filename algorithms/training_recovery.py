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
