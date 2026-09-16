from __future__ import annotations

import numpy as np

from evaluation.plotting import _training_convergence_figure, _warmup_episode_position


def _history(count: int = 50):
    return [
        {
            "episode": episode,
            "total_steps": episode * 24,
            "team_profit": float(episode),
        }
        for episode in range(1, count + 1)
    ]


def test_warmup_episode_position_interpolates_inside_episode() -> None:
    position = _warmup_episode_position(_history(), warmup_steps=1000)
    assert np.isclose(position, 41.0 + 16.0 / 24.0)


def test_training_convergence_keeps_full_history_and_marks_warmup() -> None:
    validation = [
        {"episode": 25, "team_profit_mean": 20.0},
        {"episode": 50, "team_profit_mean": 35.0},
    ]

    fig = _training_convergence_figure(_history(), validation, warmup_steps=1000)
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.lines]

    assert "Training Team Profit (raw)" in labels
    assert "Team Profit MA100" in labels
    assert "Validation Mean Team Profit" in labels
    assert "Warm-up ends (1000 steps)" in labels
    assert not any(label.startswith("Final Episode") for label in labels)

    raw_line = next(line for line in ax.lines if line.get_label() == "Training Team Profit (raw)")
    assert raw_line.get_xdata()[0] == 1
    assert raw_line.get_xdata()[-1] == 50
