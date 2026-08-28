from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from core.profiles import HeterogeneousProfileModel


class SharedForecastProvider:
    """Episode-scoped forecasts with shared public weather and grid-price views.

    Regional GHI, wind-speed, and grid-price forecasts are generated once per
    (time, horizon) and are therefore identical for every microgrid. Each MG has
    its own load forecast noise, while PV/wind power forecasts are derived from
    the shared weather forecast and that MG's installed capacity.
    """

    def __init__(self, config: Mapping[str, Any], profiles: HeterogeneousProfileModel, seed: int = 42) -> None:
        self.config = config
        self.profiles = profiles
        self.agent_ids = list(profiles.agent_ids)
        self.future_h = int(config.get("observation", {}).get("future_horizon", 4))
        self.noisy = bool(config.get("observation", {}).get("noisy_forecast", True))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._public: Dict[tuple[int, int], Dict[str, float]] = {}
        self._local: Dict[tuple[str, int, int], Dict[str, float]] = {}
        self.context: pd.DataFrame | None = None
        self.deterministic = False

    def begin_episode(self, context: pd.DataFrame, seed: int | None = None, deterministic: bool = False) -> None:
        self.context = context.reset_index(drop=True).copy()
        self.deterministic = bool(deterministic)
        rng = self.rng if seed is None else np.random.default_rng(int(seed))
        self._public.clear()
        self._local.clear()
        sigma_cfg = self.config.get("observation", {}).get("forecast_sigma_1h", {})
        episode_length = int(self.config.get("time", {}).get("episode_length", 24))
        for t in range(episode_length + 1):
            for k in range(1, self.future_h + 1):
                idx = min(t + k, len(self.context) - 1)
                row = self.context.iloc[idx]
                growth = float(np.sqrt(k))
                ghi = float(row["ghi_w_per_m2"])
                wind_speed = float(row["wind_speed_mps"])
                price = float(row["buy_price_per_kwh"])
                if self.noisy and not self.deterministic:
                    ghi *= 1.0 + float(rng.normal(0.0, float(sigma_cfg.get("ghi", 0.0)) * growth))
                    wind_speed *= 1.0 + float(rng.normal(0.0, float(sigma_cfg.get("wind_speed", 0.0)) * growth))
                    price *= 1.0 + float(rng.normal(0.0, float(sigma_cfg.get("price", 0.0)) * growth))
                public = {
                    "ghi_w_per_m2": max(0.0, ghi),
                    "wind_speed_mps": max(0.0, wind_speed),
                    "grid_price_per_kwh": max(0.0, price),
                }
                self._public[(t, k)] = public
                for mg_id in self.agent_ids:
                    load = self.profiles.load_kw(mg_id, row)
                    if self.noisy and not self.deterministic:
                        load *= 1.0 + float(rng.normal(0.0, float(sigma_cfg.get("load", 0.0)) * growth))
                    generation = self.profiles.generation_from_weather(
                        mg_id,
                        public["ghi_w_per_m2"],
                        public["wind_speed_mps"],
                    )
                    self._local[(mg_id, t, k)] = {
                        "load_kw": max(0.0, load),
                        "pv_kw": float(generation["pv_kw"]),
                        "wind_kw": float(generation["wind_kw"]),
                    }

    def public(self, t: int, k: int) -> Dict[str, float]:
        if self.context is None:
            raise RuntimeError("Forecast provider has not been initialized for an episode.")
        return dict(self._public[(int(t), int(k))])

    def local(self, mg_id: str, t: int, k: int) -> Dict[str, float]:
        if self.context is None:
            raise RuntimeError("Forecast provider has not been initialized for an episode.")
        return dict(self._local[(str(mg_id), int(t), int(k))])

    def combined(self, mg_id: str, t: int, k: int) -> Dict[str, float]:
        return {**self.local(mg_id, t, k), **self.public(t, k)}
