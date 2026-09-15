from __future__ import annotations

from typing import Sequence

import numpy as np

from core.types import DecodedAction


class ActionCodec:
    """Decode the selected raw actor action contract."""

    def __init__(self, action_design: str = "single_dispatch") -> None:
        self.action_design = str(action_design)
        if self.action_design == "single_dispatch":
            self.action_dim = 1
        elif self.action_design == "single_dispatch_dr":
            self.action_dim = 2
        else:
            raise ValueError(f"Unsupported action design: {self.action_design}")

    def decode(self, raw_action: Sequence[float]) -> DecodedAction:
        a = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if a.size != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} actions, got {a.size}.")
        if not np.isfinite(a).all():
            raise ValueError("Actions must contain only finite values.")
        decoded: DecodedAction = {"battery": float(np.clip(a[0], -1.0, 1.0))}
        if self.action_design == "single_dispatch_dr":
            decoded["dr"] = float(np.clip(a[1], -1.0, 1.0))
        return decoded
