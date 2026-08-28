from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd


class HeterogeneousProfileModel:
    """Generate heterogeneous load and renewable power from shared regional data.

    Loads use microgrid-specific daily templates plus the base data's hour-specific
    deviation factor. PV and wind use one shared regional weather realization but
    different installed capacities, so heterogeneity is structural rather than a
    simple multiplier applied to every original power series.
    """

    def __init__(self, config: Mapping[str, Any], train_frame: pd.DataFrame) -> None:
        self.config = config
        self.agent_ids = [str(mg["id"]) for mg in config["microgrids"]]
        self.mg_cfg = {str(mg["id"]): mg for mg in config["microgrids"]}
        self.train_frame = train_frame.copy()
        self.timestamp_col = str(config.get("data", {}).get("timestamp_col", "timestamp"))
        if not pd.api.types.is_datetime64_any_dtype(self.train_frame[self.timestamp_col]):
            self.train_frame[self.timestamp_col] = pd.to_datetime(self.train_frame[self.timestamp_col])

        weather_cfg = config.get("weather_to_power", {})
        pv_cfg = weather_cfg.get("pv", {})
        wind_cfg = weather_cfg.get("wind", {})
        self.ghi_reference = float(pv_cfg.get("irradiance_reference_w_per_m2", 1000.0))
        self.wind_cut_in = float(wind_cfg.get("cut_in_mps", 3.0))
        self.wind_rated = float(wind_cfg.get("rated_mps", 12.0))
        self.wind_cut_out = float(wind_cfg.get("cut_out_mps", 25.0))

        self._hourly_base_load = (
            self.train_frame.groupby(self.train_frame[self.timestamp_col].dt.hour)["load_kw"].mean()
        )
        self._load_normalizers: Dict[str, float] = {}
        self._load_max: Dict[str, float] = {}
        for mg_id in self.agent_ids:
            raw = self._load_raw_series(mg_id, self.train_frame)
            mean_raw = float(raw.mean())
            if mean_raw <= 1e-9:
                raise ValueError(f"{mg_id}: load profile has non-positive normalization mean.")
            self._load_normalizers[mg_id] = mean_raw
            transformed = self._target_mean(mg_id) * raw / mean_raw
            self._load_max[mg_id] = max(1.0, float(transformed.max()))

        self.price_max = max(1e-6, float(self.train_frame["buy_price_per_kwh"].max()))
        self.carbon_max = max(
            1e-6,
            float(self.train_frame.get("grid_carbon_factor_kg_per_kwh", pd.Series([1.0])).max()),
        )
        self.ghi_max = max(self.ghi_reference, float(self.train_frame["ghi_w_per_m2"].max()), 1.0)
        self.wind_speed_max = max(self.wind_cut_out, float(self.train_frame["wind_speed_mps"].max()), 1.0)

    def _target_mean(self, mg_id: str) -> float:
        return float(self.mg_cfg[mg_id]["load_profile"]["target_mean_kw"])

    def _hourly_shape(self, mg_id: str) -> np.ndarray:
        values = np.asarray(self.mg_cfg[mg_id]["load_profile"]["hourly_shape"], dtype=np.float64)
        if values.shape != (24,):
            raise ValueError(f"{mg_id}: load_profile.hourly_shape must contain 24 values.")
        return values

    def _base_variation(self, frame: pd.DataFrame, mg_id: str) -> np.ndarray:
        hours = frame[self.timestamp_col].dt.hour.to_numpy()
        base_hour_mean = np.asarray([float(self._hourly_base_load.loc[int(h)]) for h in hours])
        base_load = frame["load_kw"].to_numpy(dtype=np.float64)
        ratio = base_load / np.maximum(base_hour_mean, 1e-9)
        clip_cfg = self.mg_cfg[mg_id]["load_profile"].get("variability_clip", [0.65, 1.45])
        low, high = float(clip_cfg[0]), float(clip_cfg[1])
        return np.clip(ratio, low, high)

    def _load_raw_series(self, mg_id: str, frame: pd.DataFrame) -> np.ndarray:
        shape = self._hourly_shape(mg_id)
        hours = frame[self.timestamp_col].dt.hour.to_numpy()
        dayofweek = frame[self.timestamp_col].dt.dayofweek.to_numpy()
        weekend_multiplier = float(self.mg_cfg[mg_id]["load_profile"].get("weekend_multiplier", 1.0))
        calendar = shape[hours] * np.where(dayofweek >= 5, weekend_multiplier, 1.0)
        return calendar * self._base_variation(frame, mg_id)

    def load_series(self, mg_id: str, frame: pd.DataFrame) -> np.ndarray:
        work = frame.copy()
        if not pd.api.types.is_datetime64_any_dtype(work[self.timestamp_col]):
            work[self.timestamp_col] = pd.to_datetime(work[self.timestamp_col])
        raw = self._load_raw_series(mg_id, work)
        return self._target_mean(mg_id) * raw / self._load_normalizers[mg_id]

    def load_kw(self, mg_id: str, row: pd.Series) -> float:
        frame = pd.DataFrame([row])
        if not pd.api.types.is_datetime64_any_dtype(frame[self.timestamp_col]):
            frame[self.timestamp_col] = pd.to_datetime(frame[self.timestamp_col])
        raw = float(self._load_raw_series(mg_id, frame)[0])
        return max(0.0, self._target_mean(mg_id) * raw / self._load_normalizers[mg_id])

    def pv_capacity_factor(self, ghi_w_per_m2: float) -> float:
        return float(np.clip(float(ghi_w_per_m2) / max(self.ghi_reference, 1e-9), 0.0, 1.0))

    def wind_capacity_factor(self, wind_speed_mps: float) -> float:
        v = max(0.0, float(wind_speed_mps))
        if v < self.wind_cut_in or v >= self.wind_cut_out:
            return 0.0
        if v >= self.wind_rated:
            return 1.0
        numerator = v**3 - self.wind_cut_in**3
        denominator = self.wind_rated**3 - self.wind_cut_in**3
        return float(np.clip(numerator / max(denominator, 1e-9), 0.0, 1.0))

    def generation_from_weather(self, mg_id: str, ghi_w_per_m2: float, wind_speed_mps: float) -> Dict[str, float]:
        ren = self.mg_cfg[mg_id]["renewables"]
        pv = float(ren["pv_capacity_kw"]) * self.pv_capacity_factor(ghi_w_per_m2)
        wind = float(ren["wind_capacity_kw"]) * self.wind_capacity_factor(wind_speed_mps)
        return {"pv_kw": pv, "wind_kw": wind, "renewable_kw": pv + wind}

    def exogenous(self, mg_id: str, row: pd.Series) -> Dict[str, float]:
        generation = self.generation_from_weather(
            mg_id,
            float(row["ghi_w_per_m2"]),
            float(row["wind_speed_mps"]),
        )
        return {
            "load_kw": self.load_kw(mg_id, row),
            **generation,
        }

    def normalizers(self, mg_id: str, delta_t: float = 1.0) -> Dict[str, float]:
        cfg = self.mg_cfg[mg_id]
        pv_capacity = max(1.0, float(cfg["renewables"]["pv_capacity_kw"]))
        wind_capacity = max(1.0, float(cfg["renewables"]["wind_capacity_kw"]))
        generator = max(1.0, float(cfg["generator"]["max_power_kw"]))
        trade = max(self._load_max[mg_id], pv_capacity + wind_capacity, generator) * float(delta_t)
        return {
            "load": self._load_max[mg_id],
            "pv": pv_capacity,
            "wind": wind_capacity,
            "generator": generator,
            "trade": max(1.0, trade),
            "price": self.price_max,
            "carbon": self.carbon_max,
            "ghi": self.ghi_max,
            "wind_speed": self.wind_speed_max,
        }
