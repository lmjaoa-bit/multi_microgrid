from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_dr_physics import (
    DayCandidate,
    build_stress_actions,
    require_clean_audit,
    select_monthly_representatives,
)


def _candidates() -> list[DayCandidate]:
    result: list[DayCandidate] = []
    for month in range(1, 13):
        start = pd.Timestamp(year=2018, month=month, day=1)
        weekdays = [start + pd.Timedelta(days=i) for i in range(27) if (start + pd.Timedelta(days=i)).dayofweek < 5]
        weekends = [start + pd.Timedelta(days=i) for i in range(27) if (start + pd.Timedelta(days=i)).dayofweek >= 5]
        result.append(DayCandidate(split="train", day=weekdays[len(weekdays) // 2]))
        result.append(DayCandidate(split="val", day=weekends[len(weekends) // 2]))
    return result


def test_select_monthly_representatives_covers_year_and_day_types() -> None:
    selected = select_monthly_representatives(_candidates(), year=2018)
    assert len(selected) == 12
    assert [item.day.month for item in selected] == list(range(1, 13))
    assert any(item.day.dayofweek < 5 for item in selected)
    assert any(item.day.dayofweek >= 5 for item in selected)


def test_select_monthly_representatives_requires_every_month() -> None:
    candidates = [item for item in _candidates() if item.day.month != 8]
    with pytest.raises(ValueError, match="month 8"):
        select_monthly_representatives(candidates, year=2018)


def test_stress_actions_are_reproducible_bounded_and_hit_corners() -> None:
    ids = ["mg1", "mg2", "mg3"]
    first = build_stress_actions(ids, action_dim=2, seed=1234, step=6)
    second = build_stress_actions(ids, action_dim=2, seed=1234, step=6)
    assert all(np.array_equal(first[mg], second[mg]) for mg in ids)
    assert all(value.shape == (2,) for value in first.values())
    assert all(np.all(value >= -1.0) and np.all(value <= 1.0) for value in first.values())
    assert any(np.isclose(np.abs(value), 1.0).any() for value in first.values())


def test_require_clean_audit_rejects_any_hard_violation() -> None:
    clean = {"hard_violation_count": 0, "data_quality_flags": [], "projection_exercised": True}
    require_clean_audit(clean)
    with pytest.raises(RuntimeError, match="hard violations"):
        require_clean_audit({**clean, "hard_violation_count": 1})
    with pytest.raises(RuntimeError, match="data quality"):
        require_clean_audit({**clean, "data_quality_flags": ["fallback"]})
    with pytest.raises(RuntimeError, match="projection"):
        require_clean_audit({**clean, "projection_exercised": False})
