from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.profiles import HeterogeneousProfileModel
from core.tariffs import buy_price_column, sell_price_column
from core.tariffs import has_heterogeneous_tariffs


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
        self.heterogeneous_tariffs = has_heterogeneous_tariffs(config)

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
        ghi = ghi_source[indices].copy()
        wind_speed = wind_source[indices].copy()
        shared_price = None
        if not self.heterogeneous_tariffs:
            shared_price = self.context["buy_price_per_kwh"].to_numpy(dtype=np.float64)[indices].copy()
        if self.noisy and not self.deterministic:
            ghi *= 1.0 + rng.normal(size=shape) * float(sigma.get("ghi", 0.0)) * growth
            wind_speed *= 1.0 + rng.normal(size=shape) * float(sigma.get("wind_speed", 0.0)) * growth
            if shared_price is not None:
                shared_price *= 1.0 + rng.normal(size=shape) * float(sigma.get("price", 0.0)) * growth
        ghi = np.maximum(0.0, ghi)
        wind_speed = np.maximum(0.0, wind_speed)
        self._public_arrays = {
            "ghi_w_per_m2": ghi,
            "wind_speed_mps": wind_speed,
        }
        if shared_price is not None:
            self._public_arrays["grid_price_per_kwh"] = np.maximum(0.0, shared_price)

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
            has_local_weather = (
                "ghi_column" in renewable_cfg
                or "wind_speed_column" in renewable_cfg
                or abs(float(renewable_cfg.get("weather_time_shift_hours", 0.0))) > 1e-12
            )
            if has_local_weather:
                # Shifted-weather scenarios use their cached local generation
                # path so the deterministic forecast matches the environment.
                pv = np.asarray(exogenous["pv_kw"], dtype=np.float64)[indices].copy()
                wind = np.asarray(exogenous["wind_kw"], dtype=np.float64)[indices].copy()
                if self.noisy and not self.deterministic:
                    pv *= 1.0 + rng.normal(size=shape) * float(sigma.get("ghi", 0.0)) * growth
                    wind *= 1.0 + rng.normal(size=shape) * float(sigma.get("wind_speed", 0.0)) * growth
                    pv = np.maximum(0.0, pv)
                    wind = np.maximum(0.0, wind)
            else:
                pv = float(renewable_cfg["pv_capacity_kw"]) * self.profiles.pv_capacity_factor_series(ghi)
                wind = float(renewable_cfg["wind_capacity_kw"]) * self.profiles.wind_capacity_factor_series(wind_speed)
            self._local_arrays[mg_id] = {
                "load_kw": np.maximum(0.0, load),
                "pv_kw": pv,
                "wind_kw": wind,
            }
            if self.heterogeneous_tariffs:
                buy_source = self.context[buy_price_column(mg_id)].to_numpy(dtype=np.float64)
                sell_source = self.context[sell_price_column(mg_id)].to_numpy(dtype=np.float64)
                buy_price = buy_source[indices].copy()
                sell_price = sell_source[indices].copy()
                price_sigma = float(sigma.get("price", 0.0))
                if self.noisy and not self.deterministic and price_sigma > 0.0:
                    price_noise = 1.0 + rng.normal(size=shape) * price_sigma * growth
                    buy_price *= price_noise
                    sell_price *= price_noise
                self._local_arrays[mg_id].update({
                    "grid_price_per_kwh": np.maximum(0.0, buy_price),
                    "grid_sell_price_per_kwh": np.maximum(0.0, sell_price),
                })
            else:
                sell_ratio = float(self.config.get("price", {}).get("sell_ratio", 0.5))
                assert shared_price is not None
                self._local_arrays[mg_id]["grid_sell_price_per_kwh"] = (
                    np.maximum(0.0, shared_price) * sell_ratio
                )

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
