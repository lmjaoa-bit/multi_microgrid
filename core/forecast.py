from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.profiles import HeterogeneousProfileModel


class SharedForecastProvider:
    """Vectorized episode forecasts with shared public weather and prices."""

    def __init__(self, config: Mapping[str, Any], profiles: HeterogeneousProfileModel,
                 seed: int = 42) -> None:
        self.config = config
        self.profiles = profiles
        self.agent_ids = list(profiles.agent_ids)
        self.future_h = int(config.get("observation", {}).get("future_horizon", 4))
        self.noisy = bool(config.get("observation", {}).get("noisy_forecast", True))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._public_arrays: dict[str, np.ndarray] = {}
        self._local_arrays: dict[str, dict[str, np.ndarray]] = {}
        self.context: pd.DataFrame | None = None
        self.deterministic = False

    def begin_episode(
        self,
        context: pd.DataFrame,
        seed: int | None = None,
        deterministic: bool = False,
        exogenous_by_microgrid: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        self.context = context.reset_index(drop=True)
        self.deterministic = bool(deterministic)
        rng = self.rng if seed is None else np.random.default_rng(int(seed))
        sigma = self.config.get("observation", {}).get("forecast_sigma_1h", {})
        episode_length = int(self.config.get("time", {}).get("episode_length", 24))
        times = np.arange(episode_length + 1, dtype=np.int64)[:, None]
        horizons = np.arange(1, self.future_h + 1, dtype=np.int64)[None, :]
        indices = np.minimum(times + horizons, len(self.context) - 1)
        growth = np.sqrt(horizons.astype(np.float64))
        shape = indices.shape

        ghi_source = self.context["ghi_w_per_m2"].to_numpy(dtype=np.float64)
        wind_source = self.context["wind_speed_mps"].to_numpy(dtype=np.float64)
        price_source = self.context["buy_price_per_kwh"].to_numpy(dtype=np.float64)
        ghi = ghi_source[indices].copy()
        wind_speed = wind_source[indices].copy()
        price = price_source[indices].copy()
        if self.noisy and not self.deterministic:
            ghi *= 1.0 + rng.normal(size=shape) * float(sigma.get("ghi", 0.0)) * growth
            wind_speed *= 1.0 + rng.normal(size=shape) * float(sigma.get("wind_speed", 0.0)) * growth
            price *= 1.0 + rng.normal(size=shape) * float(sigma.get("price", 0.0)) * growth
        ghi = np.maximum(0.0, ghi)
        wind_speed = np.maximum(0.0, wind_speed)
        price = np.maximum(0.0, price)
        self._public_arrays = {
            "ghi_w_per_m2": ghi,
            "wind_speed_mps": wind_speed,
            "grid_price_per_kwh": price,
        }

        self._local_arrays = {}
        for mg_id in self.agent_ids:
            if exogenous_by_microgrid is None:
                exogenous = self.profiles.exogenous_series(mg_id, self.context)
            else:
                exogenous = exogenous_by_microgrid[mg_id]
            load = np.asarray(exogenous["load_kw"], dtype=np.float64)[indices].copy()
            if self.noisy and not self.deterministic:
                load *= 1.0 + rng.normal(size=shape) * float(sigma.get("load", 0.0)) * growth
            renewable_cfg = self.profiles.mg_cfg[mg_id]["renewables"]
            pv = float(renewable_cfg["pv_capacity_kw"]) * self.profiles.pv_capacity_factor_series(ghi)
            wind = float(renewable_cfg["wind_capacity_kw"]) * self.profiles.wind_capacity_factor_series(wind_speed)
            self._local_arrays[mg_id] = {
                "load_kw": np.maximum(0.0, load),
                "pv_kw": pv,
                "wind_kw": wind,
            }

    def public(self, t: int, k: int) -> dict[str, float]:
        if self.context is None:
            raise RuntimeError("Forecast provider has not been initialized for an episode.")
        row, horizon = int(t), int(k) - 1
        return {name: float(values[row, horizon]) for name, values in self._public_arrays.items()}

    def local(self, mg_id: str, t: int, k: int) -> dict[str, float]:
        if self.context is None:
            raise RuntimeError("Forecast provider has not been initialized for an episode.")
        row, horizon = int(t), int(k) - 1
        return {
            name: float(values[row, horizon])
            for name, values in self._local_arrays[str(mg_id)].items()
        }

    def combined(self, mg_id: str, t: int, k: int) -> dict[str, float]:
        return {**self.local(mg_id, t, k), **self.public(t, k)}
