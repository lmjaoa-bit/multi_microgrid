from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


class GlobalStateBuilder:
    """Centralized critic state containing all current physical states."""

    def __init__(self, config: Mapping[str, Any], profiles: Any, forecasts: Any, delta_t: float) -> None:
        self.agent_ids = [str(m["id"]) for m in config["microgrids"]]
        self.profiles = profiles
        self.norm = {mg: profiles.normalizers(mg, delta_t) for mg in self.agent_ids}
        self.price_norm = profiles.price_max
        self.forecasts = forecasts
        self.future_h = int(config.get("observation", {}).get("future_horizon", 4))
        self.control_mode = str(config.get("ess_sharing", {}).get("control_mode", "residual_auto"))

    def schema(self) -> list[str]:
        names: list[str] = []
        for mg in self.agent_ids:
            names.extend([f"{mg}_load", f"{mg}_pv", f"{mg}_wind", f"{mg}_soc",
                          f"{mg}_previous_p2p_net"])
        names.extend(["hour_sin", "hour_cos", "grid_buy_price", "grid_sell_price",
                      "previous_system_p2p"])
        if self.control_mode == "explicit_offer":
            for mg in self.agent_ids:
                names.extend([f"{mg}_available_charge", f"{mg}_available_discharge"])
            for mg in self.agent_ids:
                names.extend([
                    f"{mg}_previous_ess_p2p_offer",
                    f"{mg}_previous_ess_p2p_accepted",
                    f"{mg}_previous_ess_p2p_unmatched",
                ])
            for k in range(1, self.future_h + 1):
                for mg in self.agent_ids:
                    names.append(f"{mg}_net_load_t+{k}")
            names.extend([
                "previous_system_buy_demand", "previous_system_sell_offer",
                "previous_clearing_ratio",
            ])
        return names

    @property
    def dim(self) -> int:
        return len(self.schema())

    def build(self, row: pd.Series, t: int, cores: Mapping[str, Any], history: Mapping[str, Any],
              exogenous_by_microgrid: Mapping[str, Mapping[str, float]] | None = None) -> np.ndarray:
        values: list[float] = []
        system_norm = max(1.0, sum(self.norm[x]["trade"] for x in self.agent_ids))
        for mg in self.agent_ids:
            ex = (dict(exogenous_by_microgrid[mg]) if exogenous_by_microgrid is not None
                  else cores[mg].exogenous(row))
            n = self.norm[mg]
            previous = history["by_microgrid"][mg]
            values.extend([ex["load_kw"] / n["load"], ex["pv_kw"] / n["pv"],
                           ex["wind_kw"] / n["wind"], float(cores[mg].battery.soc),
                           previous["p2p_net_kwh"] / n["trade"]])
        hour = int(pd.Timestamp(row["timestamp"]).hour)
        values.extend([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                       float(row["buy_price_per_kwh"]) / self.price_norm,
                       float(row["sell_price_per_kwh"]) / self.price_norm,
                       history["total_p2p_kwh"] / system_norm])
        if self.control_mode == "explicit_offer":
            for mg in self.agent_ids:
                battery = cores[mg].battery
                values.extend([
                    battery.available_charge_kw() / max(1.0, battery.max_charge_kw),
                    battery.available_discharge_kw() / max(1.0, battery.max_discharge_kw),
                ])
            for mg in self.agent_ids:
                previous = history["by_microgrid"][mg]
                values.extend([
                    previous["ess_p2p_offer_kwh"] / self.norm[mg]["trade"],
                    previous["ess_p2p_accepted_kwh"] / self.norm[mg]["trade"],
                    previous["ess_p2p_unmatched_kwh"] / self.norm[mg]["trade"],
                ])
            for k in range(1, self.future_h + 1):
                for mg in self.agent_ids:
                    forecast = self.forecasts.combined(mg, t, k)
                    net_load = forecast["load_kw"] - forecast["pv_kw"] - forecast["wind_kw"]
                    values.append(net_load / self.norm[mg]["power"])
            values.extend([
                history["total_buy_demand_kwh"] / system_norm,
                history["total_sell_offer_kwh"] / system_norm,
                float(history["clearing_ratio"]),
            ])
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.dim,) or not np.isfinite(result).all():
            raise RuntimeError("Invalid centralized state")
        return result
