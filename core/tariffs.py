from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def buy_price_column(microgrid_id: str) -> str:
    return f"{microgrid_id}_buy_price_per_kwh"


def sell_price_column(microgrid_id: str) -> str:
    return f"{microgrid_id}_sell_price_per_kwh"


def has_heterogeneous_tariffs(config: Mapping[str, Any]) -> bool:
    return any("tariff" in microgrid for microgrid in config.get("microgrids", []))


def prices_from_row(row: pd.Series, microgrid_id: str) -> tuple[float, float]:
    buy_column = buy_price_column(microgrid_id)
    sell_column = sell_price_column(microgrid_id)
    buy = float(row[buy_column] if buy_column in row.index else row["buy_price_per_kwh"])
    sell = float(row[sell_column] if sell_column in row.index else row["sell_price_per_kwh"])
    return buy, sell


def build_buy_price(frame: pd.DataFrame, tariff: Mapping[str, Any]) -> np.ndarray:
    mode = str(tariff.get("mode", "inherited"))
    if mode == "inherited":
        values = frame["buy_price_per_kwh"].to_numpy(dtype=np.float64)
    elif mode == "flat":
        values = np.full(len(frame), float(tariff["buy_price_per_kwh"]), dtype=np.float64)
    elif mode == "indexed":
        source_column = str(tariff.get("source_column", "buy_price_per_kwh"))
        if source_column not in frame:
            raise ValueError(f"Indexed tariff source column is missing: {source_column}")
        values = (
            frame[source_column].to_numpy(dtype=np.float64)
            * float(tariff.get("multiplier", 1.0))
            + float(tariff.get("adder_per_kwh", 0.0))
        )
        values = np.clip(
            values,
            float(tariff.get("minimum_buy_price_per_kwh", 0.0)),
            float(tariff.get("maximum_buy_price_per_kwh", np.inf)),
        )
    elif mode == "tou":
        hours = frame["timestamp"].dt.hour.to_numpy(dtype=np.int64)
        peak_hours = {int(hour) for hour in tariff.get("peak_hours", [])}
        valley_hours = {int(hour) for hour in tariff.get("valley_hours", [])}
        overlap = peak_hours & valley_hours
        if overlap:
            raise ValueError(f"TOU peak and valley hours overlap: {sorted(overlap)}")
        flat = float(tariff["flat_price_per_kwh"])
        peak = float(tariff["peak_price_per_kwh"])
        valley = float(tariff["valley_price_per_kwh"])
        values = np.asarray(
            [peak if hour in peak_hours else valley if hour in valley_hours else flat
             for hour in hours],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported tariff mode: {mode}")
    return values


def build_sell_price(buy_price: np.ndarray, tariff: Mapping[str, Any]) -> np.ndarray:
    if "sell_price_per_kwh" in tariff:
        return np.full(
            len(buy_price), float(tariff["sell_price_per_kwh"]), dtype=np.float64
        )
    return np.asarray(buy_price, dtype=np.float64) * float(tariff.get("sell_ratio", 0.5))
