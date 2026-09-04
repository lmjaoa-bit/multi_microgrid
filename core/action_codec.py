from __future__ import annotations

from typing import Sequence

import numpy as np

from core.types import DecodedAction


class ActionCodec:
    """Decode the single signed battery dispatch intent."""

    def __init__(self, control_mode: str = "single_dispatch") -> None:
        self.control_mode = str(control_mode)
        if self.control_mode != "single_dispatch":
            raise ValueError(f"Unsupported ESS sharing control mode: {self.control_mode}")
        self.action_dim = 1

    def decode(self, raw_action: Sequence[float]) -> DecodedAction:
        a = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if a.size != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} actions, got {a.size}.")
        if not np.isfinite(a).all():
            raise ValueError("Actions must contain only finite values.")
        return {"battery": float(np.clip(a[0], -1.0, 1.0))}
