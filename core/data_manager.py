from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.tariffs import (
    build_buy_price,
    build_sell_price,
    buy_price_column,
    sell_price_column,
    prices_from_row,
)


@dataclass
class DataManager:
    config: Dict[str, Any]
    split: str = "train"
    seed: int = 42

    def __post_init__(self) -> None:
        data_cfg = self.config["data"]
        path = Path(data_cfg[f"{self.split}_path"])
        if not path.is_absolute():
            root = Path(self.config.get("_project_root", "."))
            path = root / path
        self.df = pd.read_csv(path)
        ts_col = data_cfg.get("timestamp_col", "timestamp")
        required = {ts_col, "load_kw", "buy_price_per_kwh", "ghi_w_per_m2", "wind_speed_mps"}
        for microgrid in self.config["microgrids"]:
            renewable = microgrid.get("renewables", {})
            required.add(str(renewable.get("ghi_column", "ghi_w_per_m2")))
            required.add(str(renewable.get("wind_speed_column", "wind_speed_mps")))
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Data split {self.split!r} is missing columns: {sorted(missing)}")
        self.df[ts_col] = pd.to_datetime(self.df[ts_col], errors="raise")
        self.df = self.df.sort_values(ts_col).reset_index(drop=True)
        self.ts_col = ts_col
        self.rng = np.random.default_rng(self.seed)
        self._apply_prices()
        self._validate_frame()
        self._timestamp_indexed = self.df.set_index(self.ts_col, drop=False)
        self.valid_days = self._find_valid_days()
        if not self.valid_days:
            raise ValueError(f"Data split {self.split!r} contains no valid natural-day episodes.")

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    @staticmethod
    def prices_at(row: pd.Series, microgrid_id: str) -> tuple[float, float]:
        return prices_from_row(row, microgrid_id)

    def _apply_prices(self) -> None:
        price_cfg = self.config.get("price", {})
        sell_ratio = float(price_cfg.get("sell_ratio", 0.5))
        if price_cfg.get("override_buy_price_with_tou", False):
            valley = float(price_cfg["valley_price"])
            flat = float(price_cfg["flat_price"])
            peak = float(price_cfg["peak_price"])
            valley_h = set(int(h) for h in price_cfg.get("valley_hours", []))
            peak_h = set(int(h) for h in price_cfg.get("peak_hours", []))
            hours = self.df[self.ts_col].dt.hour
            self.df["buy_price_per_kwh"] = [
                valley if h in valley_h else peak if h in peak_h else flat for h in hours
            ]
        self.df["sell_price_per_kwh"] = self.df["buy_price_per_kwh"] * sell_ratio
        for microgrid in self.config["microgrids"]:
            microgrid_id = str(microgrid["id"])
            tariff = microgrid.get("tariff", {"mode": "inherited", "sell_ratio": sell_ratio})
            buy = build_buy_price(self.df, tariff)
            sell = build_sell_price(buy, tariff)
            self.df[buy_price_column(microgrid_id)] = buy
            self.df[sell_price_column(microgrid_id)] = sell

    def _validate_frame(self) -> None:
        numeric = ["load_kw", "buy_price_per_kwh", "sell_price_per_kwh", "ghi_w_per_m2", "wind_speed_mps"]
        for microgrid in self.config["microgrids"]:
            microgrid_id = str(microgrid["id"])
            renewable = microgrid.get("renewables", {})
            numeric.extend([
                str(renewable.get("ghi_column", "ghi_w_per_m2")),
                str(renewable.get("wind_speed_column", "wind_speed_mps")),
                buy_price_column(microgrid_id),
                sell_price_column(microgrid_id),
            ])
        numeric = list(dict.fromkeys(numeric))
        if "grid_carbon_factor_kg_per_kwh" in self.df:
            numeric.append("grid_carbon_factor_kg_per_kwh")
        values = self.df[numeric].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Data split {self.split!r} contains NaN or infinite physical/price values.")
        for column in numeric:
            if (values[column] < 0.0).any():
                raise ValueError(f"Data split {self.split!r} contains negative values in {column}.")
        if (values["sell_price_per_kwh"] > values["buy_price_per_kwh"] + 1e-12).any():
            raise ValueError("Grid sell price cannot exceed grid buy price.")
        for microgrid in self.config["microgrids"]:
            microgrid_id = str(microgrid["id"])
            buy = values[buy_price_column(microgrid_id)]
            sell = values[sell_price_column(microgrid_id)]
            if (sell > buy + 1e-12).any():
                raise ValueError(
                    f"{microgrid_id}: grid sell price cannot exceed grid buy price"
                )
        if self.df[self.ts_col].isna().any() or self.df[self.ts_col].duplicated().any():
            raise ValueError(f"Data split {self.split!r} has missing or duplicate timestamps.")

    def _find_valid_days(self) -> List[pd.Timestamp]:
        episode_length = int(self.config["time"].get("episode_length", 24))
        valid: List[pd.Timestamp] = []
        for day, frame in self.df.groupby(self.df[self.ts_col].dt.normalize(), sort=True):
            timestamps = frame[self.ts_col].iloc[:episode_length].reset_index(drop=True)
            if len(timestamps) < episode_length:
                continue
            expected = pd.date_range(start=day, periods=episode_length, freq="h")
            if timestamps.equals(pd.Series(expected)):
                valid.append(pd.Timestamp(day))
        return valid

    def get_day(self, day: str | pd.Timestamp | None = None) -> pd.DataFrame:
        length = int(self.config["time"].get("episode_length", 24))
        if day is None:
            selected = self.valid_days[int(self.rng.integers(0, len(self.valid_days)))]
        else:
            selected = pd.Timestamp(day).normalize()
            if selected not in self.valid_days:
                raise ValueError(f"Day {selected.date()} is not a valid episode day.")
        mask = self.df[self.ts_col].dt.normalize() == selected
        return self.df.loc[mask].iloc[:length].reset_index(drop=True).copy()

    def get_episode_context(
        self,
        day: str | pd.Timestamp | None = None,
        future_horizon: int = 0,
    ) -> pd.DataFrame:
        """Return the 24-hour episode plus contiguous next-day forecast context.

        At a discontinuous split boundary the available context may be shorter; the
        observation builder then applies its explicit last-value fallback.
        """

        episode = self.get_day(day)
        horizon = max(0, int(future_horizon))
        if horizon == 0:
            return episode
        last_timestamp = pd.Timestamp(episode.iloc[-1][self.ts_col])
        future_rows = []
        for offset in range(1, horizon + 1):
            expected = last_timestamp + pd.Timedelta(hours=offset)
            if expected not in self._timestamp_indexed.index:
                break
            future_rows.append(self._timestamp_indexed.loc[[expected]].iloc[0])
        if future_rows:
            episode = pd.concat([episode, pd.DataFrame(future_rows)], ignore_index=True)
        return episode.reset_index(drop=True).copy()
