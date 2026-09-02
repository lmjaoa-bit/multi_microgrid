from __future__ import annotations

from typing import Sequence

import numpy as np

from core.types import DecodedAction


class ActionCodec:
    """Decode the battery action and optional ESS sharing-offer action."""

    def __init__(self, control_mode: str = "residual_auto") -> None:
        self.control_mode = str(control_mode)
        if self.control_mode not in {"residual_auto", "fixed_full_offer", "explicit_offer"}:
            raise ValueError(f"Unsupported ESS sharing control mode: {self.control_mode}")
        self.action_dim = 2 if self.control_mode == "explicit_offer" else 1

    def decode(self, raw_action: Sequence[float]) -> DecodedAction:
        a = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if a.size != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} actions, got {a.size}.")
        if not np.isfinite(a).all():
            raise ValueError("Actions must contain only finite values.")
        a = np.clip(a, -1.0, 1.0)
        decoded: DecodedAction = {"battery": float(a[0])}
        if self.control_mode == "explicit_offer":
            decoded["sharing_offer_ratio"] = float((a[1] + 1.0) / 2.0)
        elif self.control_mode == "fixed_full_offer":
            decoded["sharing_offer_ratio"] = 1.0
        return decoded
