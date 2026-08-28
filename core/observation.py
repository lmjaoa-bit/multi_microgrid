from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from core.forecast import SharedForecastProvider
from core.profiles import HeterogeneousProfileModel


class ObservationBuilder:
    """Build decentralized observations with shared public forecasts."""

    def __init__(
        self,
        config: Dict[str, Any],
        profiles: HeterogeneousProfileModel,
        forecasts: SharedForecastProvider,
        delta_t: float,
    ) -> None:
        self.config = config
        self.agent_ids = [str(m["id"]) for m in config["microgrids"]]
        self.mg_cfg = {str(m["id"]): m for m in config["microgrids"]}
        self.future_h = int(config.get("observation", {}).get("future_horizon", 4))
        self.profiles = profiles
        self.forecasts = forecasts
        self.delta_t = float(delta_t)
        self._norm = {mg: profiles.normalizers(mg, delta_t=self.delta_t) for mg in self.agent_ids}
        self._schemas = {mg: self._schema_for(mg) for mg in self.agent_ids}

    def _schema_for(self, mg_id: str) -> List[str]:
        names = [
            "hour_sin", "hour_cos", "load", "pv", "wind", "grid_buy_price",
            "grid_sell_price", "soc", "generator_on", "generator_power", "grid_carbon",
        ]
        for k in range(1, self.future_h + 1):
            names.extend(
                [
                    f"load_t+{k}",
                    f"pv_t+{k}",
                    f"wind_t+{k}",
                    f"grid_price_t+{k}",
                    f"ghi_t+{k}",
                    f"wind_speed_t+{k}",
                ]
            )
        names.extend(
            [
                "market_prev_avg_price",
                "market_prev_total_trade",
                "market_prev_submitted_buy",
                "market_prev_submitted_sell",
                "self_prev_bid",
                "self_prev_trade",
                "self_prev_role",
            ]
        )
        for other in self.agent_ids:
            if other != mg_id:
                names.extend([f"{other}_prev_bid", f"{other}_prev_trade", f"{other}_prev_role"])
        return names

    def schema(self, mg_id: str) -> List[str]:
        return list(self._schemas[mg_id])

    def obs_dim(self, mg_id: str) -> int:
        return len(self._schemas[mg_id])

    def build(
        self,
        mg_id: str,
        row: pd.Series,
        t: int,
        battery_soc: float,
        generator_power_kw: float,
        market_history: Mapping[str, Any],
    ) -> np.ndarray:
        n = self._norm[mg_id]
        ex = self.profiles.exogenous(mg_id, row)
        hour = int(pd.Timestamp(row["timestamp"]).hour)
        values: List[float] = [
            float(np.sin(2.0 * np.pi * hour / 24.0)),
            float(np.cos(2.0 * np.pi * hour / 24.0)),
            ex["load_kw"] / n["load"],
            ex["pv_kw"] / n["pv"],
            ex["wind_kw"] / n["wind"],
            float(row["buy_price_per_kwh"]) / n["price"],
            float(row["sell_price_per_kwh"]) / n["price"],
            float(battery_soc),
            1.0 if generator_power_kw > 1e-9 else 0.0,
            float(generator_power_kw) / n["generator"],
            float(row.get("grid_carbon_factor_kg_per_kwh", 0.0)) / n["carbon"],
        ]
        for k in range(1, self.future_h + 1):
            f = self.forecasts.combined(mg_id, t, k)
            values.extend(
                [
                    f["load_kw"] / n["load"],
                    f["pv_kw"] / n["pv"],
                    f["wind_kw"] / n["wind"],
                    f["grid_price_per_kwh"] / n["price"],
                    f["ghi_w_per_m2"] / n["ghi"],
                    f["wind_speed_mps"] / n["wind_speed"],
                ]
            )

        prev = market_history
        by_mg = prev["by_microgrid"]
        self_prev = by_mg[mg_id]
        values.extend(
            [
                float(prev["weighted_average_price"]) / n["price"],
                float(prev["total_trade_kwh"]) / n["trade"],
                float(prev["submitted_buy_kwh"]) / n["trade"],
                float(prev["submitted_sell_kwh"]) / n["trade"],
                float(self_prev["bid_price"]) / n["price"],
                float(self_prev["trade_kwh"]) / n["trade"],
                float(self_prev["role"]),
            ]
        )
        for other in self.agent_ids:
            if other == mg_id:
                continue
            o = by_mg[other]
            values.extend(
                [
                    float(o["bid_price"]) / n["price"],
                    float(o["trade_kwh"]) / n["trade"],
                    float(o["role"]),
                ]
            )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.obs_dim(mg_id),) or not np.isfinite(observation).all():
            raise RuntimeError(f"{mg_id}: observation construction produced an invalid vector.")
        return observation
