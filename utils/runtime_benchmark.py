from __future__ import annotations

from collections.abc import Mapping, Sequence


def cpu_thread_candidates(logical_cpus: int | None) -> list[int]:
    """Return a compact thread sweep from 1 up to the machine logical-CPU limit."""
    limit = max(1, int(logical_cpus or 1))
    return sorted({value for value in (1, 2, 4, 8, limit) if value <= limit})


def select_best_result(results: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    """Pick the fastest successful eager candidate by steady-state seconds/episode.

    torch.compile rows stay in the report as an experimental comparison, but are not selected
    as the production recommendation because the normal training entry point intentionally
    remains eager/checkpoint-compatible.
    """
    valid = [
        row for row in results
        if row.get("status") == "ok"
        and row.get("seconds_per_episode") is not None
        and not bool(row.get("compile_actors", False))
    ]
    if not valid:
        return None
    return min(valid, key=lambda row: float(row["seconds_per_episode"]))


def estimate_training_seconds(seconds_per_episode: float, episodes: int) -> float:
    return float(seconds_per_episode) * int(episodes)
