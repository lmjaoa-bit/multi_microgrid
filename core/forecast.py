from __future__ import annotations

from typing import Any, Mapping

import hashlib
import numpy as np
import pandas as pd

from core.profiles import HeterogeneousProfileModel
from core.tariffs import buy_price_column, sell_price_column
from core.tariffs import has_heterogeneous_tariffs


class SharedForecastProvider:
    """Vectorized episode forecasts for exogenous/base-load quantities only."""

    def __init__(self, config: Mapping[str, Any], profiles: HeterogeneousProfileModel,
                 seed: int = 42) -> None:
        self.config = config
        self.profiles = profiles
        self.agent_ids = list(profiles.agent_ids)
        self.future_h = int(config.get("observation", {}).get("future_horizon", 4))
        self.forecast_horizon = max(
            self.future_h, int(config.get("time", {}).get("episode_length", 24))
        )
        self.noisy = bool(config.get("observation", {}).get("noisy_forecast", True))
        observation = config.get("observation", {})
        self.temporal_correlation = float(
            observation.get("forecast_temporal_correlation", 0.0)
        )
        self.error_max = dict(observation.get("forecast_error_max", {}))
        self.mask_beyond_episode = bool(
            observation.get("mask_forecast_beyond_episode", False)
        )
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._public_arrays: dict[str, np.ndarray] = {}
        self._local_arrays: dict[str, dict[str, np.ndarray]] = {}
        self._valid_array = np.empty((0, self.forecast_horizon), dtype=np.float64)
        self.context: pd.DataFrame | None = None
        self.deterministic = False
        self.heterogeneous_tariffs = has_heterogeneous_tariffs(config)

    def begin_episode(
        self,
        context: pd.DataFrame,
        seed: int | None = None,
        deterministic: bool = False,
        forecast_noise: bool | None = None,
        exogenous_by_microgrid: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        self.context = context.reset_index(drop=True)
        self.deterministic = bool(deterministic)
        rng = self.rng if seed is None else np.random.default_rng(int(seed))
        sigma = self.config.get("observation", {}).get("forecast_sigma_1h", {})
        episode_length = int(self.config.get("time", {}).get("episode_length", 24))
        times = np.arange(episode_length + 1, dtype=np.int64)[:, None]
        horizons = np.arange(1, self.forecast_horizon + 1, dtype=np.int64)[None, :]
        raw_indices = times + horizons
        context_valid = raw_indices < len(self.context)
        if self.mask_beyond_episode:
            valid = context_valid & (raw_indices < episode_length)
        else:
            valid = context_valid
        indices = np.minimum(raw_indices, len(self.context) - 1)
        self._valid_array = valid.astype(np.float64)
        growth = np.sqrt(horizons.astype(np.float64))
        shape = indices.shape
        apply_noise = self.noisy and (
            bool(forecast_noise) if forecast_noise is not None else not self.deterministic
        )

        def correlated_error(name: str) -> np.ndarray:
            sigma_1h = float(sigma.get(name, 0.0))
            if not apply_noise or sigma_1h <= 0.0:
                return np.zeros(shape, dtype=np.float64)
            innovations = rng.normal(size=shape)
            correlated = np.empty(shape, dtype=np.float64)
            correlated[:, 0] = innovations[:, 0]
            rho = self.temporal_correlation
            innovation_scale = np.sqrt(max(0.0, 1.0 - rho * rho))
            for k in range(1, self.forecast_horizon):
                correlated[:, k] = (
                    rho * correlated[:, k - 1]
                    + innovation_scale * innovations[:, k]
                )
            error = correlated * sigma_1h * growth
            maximum = self.error_max.get(name)
            if maximum is not None:
                error = np.clip(error, -float(maximum), float(maximum))
            return error

        def multiplier(name: str) -> np.ndarray:
            return 1.0 + correlated_error(name)

        def finish_nonnegative(values: np.ndarray) -> np.ndarray:
            return np.where(valid, np.maximum(0.0, values), 0.0)

        def finish_signed(values: np.ndarray) -> np.ndarray:
            return np.where(valid, values, 0.0)

        ghi_source = self.context["ghi_w_per_m2"].to_numpy(dtype=np.float64)
        wind_source = self.context["wind_speed_mps"].to_numpy(dtype=np.float64)
        ghi_multiplier = multiplier("ghi")
        wind_multiplier = multiplier("wind_speed")
        ghi = ghi_source[indices].copy() * ghi_multiplier
        wind_speed = wind_source[indices].copy() * wind_multiplier
        shared_price = None
        if not self.heterogeneous_tariffs:
            shared_price = (
                self.context["buy_price_per_kwh"].to_numpy(dtype=np.float64)[indices].copy()
                * multiplier("price")
            )
        ghi = finish_nonnegative(ghi)
        wind_speed = finish_nonnegative(wind_speed)
        self._public_arrays = {
            "ghi_w_per_m2": ghi,
            "wind_speed_mps": wind_speed,
        }
        if shared_price is not None:
            self._public_arrays["grid_price_per_kwh"] = finish_nonnegative(shared_price)

        self._local_arrays = {}
        for mg_id in self.agent_ids:
            if exogenous_by_microgrid is None:
                exogenous = self.profiles.exogenous_series(mg_id, self.context)
            else:
                exogenous = exogenous_by_microgrid[mg_id]
            load = np.asarray(exogenous["load_kw"], dtype=np.float64)[indices].copy()
            load *= multiplier("load")
            renewable_cfg = self.profiles.mg_cfg[mg_id]["renewables"]
            has_local_weather = (
                "ghi_column" in renewable_cfg
                or "wind_speed_column" in renewable_cfg
                or abs(float(renewable_cfg.get("weather_time_shift_hours", 0.0))) > 1e-12
            )
            if has_local_weather:
                pv = np.asarray(exogenous["pv_kw"], dtype=np.float64)[indices].copy()
                wind = np.asarray(exogenous["wind_kw"], dtype=np.float64)[indices].copy()
                pv *= ghi_multiplier
                wind *= wind_multiplier
                pv = finish_nonnegative(pv)
                wind = finish_nonnegative(wind)
            else:
                pv = float(renewable_cfg["pv_capacity_kw"]) * self.profiles.pv_capacity_factor_series(ghi)
                wind = float(renewable_cfg["wind_capacity_kw"]) * self.profiles.wind_capacity_factor_series(wind_speed)
            self._local_arrays[mg_id] = {
                # This is intentionally the exogenous/base load. Future flexible
                # load is endogenous to the policy and is never forecast here.
                "load_kw": finish_nonnegative(load),
                "pv_kw": np.where(valid, pv, 0.0),
                "wind_kw": np.where(valid, wind, 0.0),
            }

            dr_cfg = self.profiles.mg_cfg[mg_id].get("dr", {})
            if str(dr_cfg.get("type", "")) == "commercial_hvac":
                column = str(dr_cfg.get("temperature_column", "temperature_c"))
                temperature = self.context[column].to_numpy(dtype=np.float64)[indices].copy()
                temperature += correlated_error("temperature")
                self._local_arrays[mg_id]["temperature_c"] = finish_signed(temperature)
                ghi_column = str(dr_cfg.get("ghi_column", "ghi_w_per_m2"))
                local_ghi = (
                    self.context[ghi_column].to_numpy(dtype=np.float64)[indices].copy()
                    * ghi_multiplier
                )
                self._local_arrays[mg_id]["ghi_w_per_m2"] = finish_nonnegative(local_ghi)

            if self.heterogeneous_tariffs:
                buy_source = self.context[buy_price_column(mg_id)].to_numpy(dtype=np.float64)
                sell_source = self.context[sell_price_column(mg_id)].to_numpy(dtype=np.float64)
                buy_price = buy_source[indices].copy()
                sell_price = sell_source[indices].copy()
                price_sigma = float(sigma.get("price", 0.0))
                if price_sigma > 0.0:
                    price_noise = multiplier("price")
                    buy_price *= price_noise
                    sell_price *= price_noise
                self._local_arrays[mg_id].update({
                    "grid_price_per_kwh": finish_nonnegative(buy_price),
                    "grid_sell_price_per_kwh": finish_nonnegative(sell_price),
                })
            else:
                sell_ratio = float(self.config.get("price", {}).get("sell_ratio", 0.5))
                assert shared_price is not None
                self._local_arrays[mg_id]["grid_sell_price_per_kwh"] = (
                    finish_nonnegative(shared_price) * sell_ratio
                )

    def valid(self, t: int, k: int) -> float:
        if self.context is None:
            raise RuntimeError("Forecast provider has not been initialized for an episode.")
        return float(self._valid_array[int(t), int(k) - 1])

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


    def hvac_viability_inputs(
        self, mg_id: str, t: int, episode_length: int, delta_t: float
    ) -> dict[str, Any]:
        """Return one immutable-cache view for intervals t..episode_end-1.

        Interval t uses current measured exogenous data.  Later intervals use
        forecasts frozen at time t; no hidden future truth is consulted after t.
        """
        if self.context is None:
            raise RuntimeError("Forecast provider has not been initialized")
        mg_id = str(mg_id)
        start = int(t)
        end = int(episode_length)
        if start >= end:
            return {
                "snapshot_id": f"{mg_id}:{start}:terminal",
                "boundary_timestamps": tuple(),
                "temperature_c": np.empty(0),
                "ghi_w_per_m2": np.empty(0),
                "base_load_kw": np.empty(0),
            }
        h_count = end - start
        if h_count - 1 > self.forecast_horizon:
            raise RuntimeError(
                "HVAC viability horizon exceeds forecast cache; refusing future truth fallback"
            )
        dr_cfg = self.profiles.mg_cfg[mg_id].get("dr", {})
        temp_column = str(dr_cfg["temperature_column"])
        ghi_column = str(dr_cfg["ghi_column"])
        temperatures = np.empty(h_count, dtype=np.float64)
        ghi = np.empty(h_count, dtype=np.float64)
        loads = np.empty(h_count, dtype=np.float64)
        current_ex = self.profiles.exogenous(mg_id, self.context.iloc[start])
        temperatures[0] = float(self.context.iloc[start][temp_column])
        ghi[0] = max(0.0, float(self.context.iloc[start][ghi_column]))
        loads[0] = float(current_ex["load_kw"])
        for offset in range(1, h_count):
            if self.valid(start, offset) <= 0.0:
                raise RuntimeError(
                    f"HVAC viability forecast missing at t={start}, horizon={offset}"
                )
            local = self.local(mg_id, start, offset)
            temperatures[offset] = float(local["temperature_c"])
            ghi[offset] = float(local["ghi_w_per_m2"])
            loads[offset] = float(local["load_kw"])

        boundaries = [
            str(pd.Timestamp(self.context.iloc[start]["timestamp"]))
        ]
        for offset in range(1, h_count + 1):
            absolute = start + offset
            if absolute < len(self.context):
                boundaries.append(
                    str(pd.Timestamp(self.context.iloc[absolute]["timestamp"]))
                )
            else:
                boundaries.append(str(
                    pd.Timestamp(boundaries[-1])
                    + pd.to_timedelta(float(delta_t), unit="h")
                ))
        digest = hashlib.sha256()
        digest.update("|".join(boundaries).encode())
        for array in (temperatures, ghi, loads):
            digest.update(np.asarray(array, dtype=np.float64).tobytes())
        return {
            "snapshot_id": digest.hexdigest(),
            "boundary_timestamps": tuple(boundaries),
            "temperature_c": temperatures,
            "ghi_w_per_m2": ghi,
            "base_load_kw": loads,
        }
