from __future__ import annotations

from typing import Sequence

import numpy as np

from core.types import DecodedAction


class ActionCodec:
    """Fixed multi-microgrid action definition.

    [battery, generator_commitment, generator_setpoint, p2p_bid]
    All policy outputs are in [-1, 1].
    """

    action_dim = 4
    hard_action_tolerance = 1e-6

    @staticmethod
    def decode(raw_action: Sequence[float], sell_price: float, buy_price: float) -> DecodedAction:
        a = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if a.size != 4:
            raise ValueError(f"Expected 4 actions, got {a.size}.")
        if not np.isfinite(a).all():
            raise ValueError("Actions must contain only finite values.")
        if not np.isfinite([sell_price, buy_price]).all():
            raise ValueError("Grid prices used for bid decoding must be finite.")
        if sell_price < 0.0 or buy_price < sell_price:
            raise ValueError("Grid prices must satisfy 0 <= sell_price <= buy_price.")
        tolerance = ActionCodec.hard_action_tolerance
        if np.isclose(a[1], -1.0, rtol=0.0, atol=tolerance):
            commitment = -1.0
        elif np.isclose(a[1], 1.0, rtol=0.0, atol=tolerance):
            commitment = 1.0
        else:
            raise ValueError("Generator commitment must be hard encoded as -1 (OFF) or +1 (ON).")
        if commitment < 0.0 and not np.isclose(a[2], -1.0, rtol=0.0, atol=tolerance):
            raise ValueError("Generator setpoint must be -1 when commitment is OFF.")

        a = np.clip(a, -1.0, 1.0)
        generator_setpoint = -1.0 if commitment < 0.0 else float(a[2])
        bid = float(sell_price + 0.5 * (a[3] + 1.0) * (buy_price - sell_price))
        return {
            "battery": float(a[0]),
            "generator_commitment": commitment,
            "generator_setpoint": generator_setpoint,
            "p2p_bid_action": float(a[3]),
            "p2p_bid_price": bid,
        }
