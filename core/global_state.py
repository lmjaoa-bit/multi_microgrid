from __future__ import annotations

from typing import Any, List, Mapping

import numpy as np
import pandas as pd

from core.forecast import SharedForecastProvider
from core.profiles import HeterogeneousProfileModel


class GlobalStateBuilder:
    """Build the centralized critic state from legal current information.

    The critic sees current truth for every MG plus the same forecasts that are
    available at the current decision time. It never receives future realized
    load/weather values directly.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        profiles: HeterogeneousProfileModel,
        forecasts: SharedForecastProvider,
        delta_t: float,
    ) -> None:
        self.agent_ids = [str(mg["id"]) for mg in config["microgrids"]]
        self.future_h = int(config.get("observation", {}).get("future_horizon", 4))
        self.profiles = profiles
        self.forecasts = forecasts
        self.delta_t = float(delta_t)
        self.norms = {mg: profiles.normalizers(mg, delta_t=self.delta_t) for mg in self.agent_ids}
        self.price_norm = profiles.price_max
        self.carbon_norm = profiles.carbon_max
        self.market_energy_norm = max(1.0, sum(self.norms[mg]["trade"] for mg in self.agent_ids))
        self._schema = self._build_schema()

    def _build_schema(self) -> List[str]:
        names: List[str] = []
        for mg_id in self.agent_ids:
            names.extend(
                [
                    f"{mg_id}_load",
                    f"{mg_id}_pv",
                    f"{mg_id}_wind",
                    f"{mg_id}_soc",
                    f"{mg_id}_generator_on",
                    f"{mg_id}_generator_power",
                ]
            )
        names.extend(
            [
                "hour_sin",
                "hour_cos",
                "grid_buy_price",
                "grid_sell_price",
                "grid_carbon",
                "market_prev_avg_price",
                "market_prev_total_trade",
                "market_prev_submitted_buy",
                "market_prev_submitted_sell",
            ]
        )
        for mg_id in self.agent_ids:
            names.extend(
                [
                    f"{mg_id}_prev_bid",
                    f"{mg_id}_prev_trade",
                    f"{mg_id}_prev_role",
                ]
            )
        for k in range(1, self.future_h + 1):
            names.extend(
                [
                    f"grid_price_t+{k}",
                    f"ghi_t+{k}",
                    f"wind_speed_t+{k}",
                ]
            )
            for mg_id in self.agent_ids:
                names.extend(
                    [
                        f"{mg_id}_load_t+{k}",
                        f"{mg_id}_pv_t+{k}",
                        f"{mg_id}_wind_t+{k}",
                    ]
                )
        return names

    @property
    def dim(self) -> int:
        return len(self._schema)

    def schema(self) -> List[str]:
        return list(self._schema)

    def build(
        self,
        row: pd.Series,
        t: int,
        cores: Mapping[str, Any],
        market_history: Mapping[str, Any],
    ) -> np.ndarray:
        values: List[float] = []
        for mg_id in self.agent_ids:
            core = cores[mg_id]
            exogenous = core.exogenous(row)
            norm = self.norms[mg_id]
            generator_power = float(core.generator.last_power_kw)
            values.extend(
                [
                    float(exogenous["load_kw"]) / norm["load"],
                    float(exogenous["pv_kw"]) / norm["pv"],
                    float(exogenous["wind_kw"]) / norm["wind"],
                    float(core.battery.soc),
                    1.0 if generator_power > 1e-9 else 0.0,
                    generator_power / norm["generator"],
                ]
            )

        hour = int(pd.Timestamp(row["timestamp"]).hour)
        values.extend(
            [
                float(np.sin(2.0 * np.pi * hour / 24.0)),
                float(np.cos(2.0 * np.pi * hour / 24.0)),
                float(row["buy_price_per_kwh"]) / self.price_norm,
                float(row["sell_price_per_kwh"]) / self.price_norm,
                float(row.get("grid_carbon_factor_kg_per_kwh", 0.0)) / self.carbon_norm,
                float(market_history["weighted_average_price"]) / self.price_norm,
                float(market_history["total_trade_kwh"]) / self.market_energy_norm,
                float(market_history["submitted_buy_kwh"]) / self.market_energy_norm,
                float(market_history["submitted_sell_kwh"]) / self.market_energy_norm,
            ]
        )
        by_microgrid = market_history["by_microgrid"]
        for mg_id in self.agent_ids:
            previous = by_microgrid[mg_id]
            values.extend(
                [
                    float(previous["bid_price"]) / self.price_norm,
                    float(previous["trade_kwh"]) / self.norms[mg_id]["trade"],
                    float(previous["role"]),
                ]
            )

        for k in range(1, self.future_h + 1):
            public = self.forecasts.public(t, k)
            values.extend(
                [
                    public["grid_price_per_kwh"] / self.price_norm,
                    public["ghi_w_per_m2"] / self.profiles.ghi_max,
                    public["wind_speed_mps"] / self.profiles.wind_speed_max,
                ]
            )
            for mg_id in self.agent_ids:
                local = self.forecasts.local(mg_id, t, k)
                norm = self.norms[mg_id]
                values.extend(
                    [
                        local["load_kw"] / norm["load"],
                        local["pv_kw"] / norm["pv"],
                        local["wind_kw"] / norm["wind"],
                    ]
                )

        state = np.asarray(values, dtype=np.float32)
        if state.shape != (self.dim,) or not np.isfinite(state).all():
            raise RuntimeError("Global state construction produced an invalid vector.")
        return state
