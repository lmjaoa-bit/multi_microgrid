from __future__ import annotations

__all__ = ["MATD3TrainingDashboard"]


def __getattr__(name: str):
    if name == "MATD3TrainingDashboard":
        from utils.training_dashboard import MATD3TrainingDashboard
        return MATD3TrainingDashboard
    raise AttributeError(name)
