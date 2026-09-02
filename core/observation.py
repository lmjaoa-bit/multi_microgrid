from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.forecast import SharedForecastProvider
from core.profiles import HeterogeneousProfileModel


class ObservationBuilder:
    """Local observation with public forecasts and previous energy-sharing outcomes."""

    def __init__(self, config: Mapping[str, Any], profiles: HeterogeneousProfileModel,
                 forecasts: SharedForecastProvider, delta_t: float) -> None:
        self.config = config
        self.agent_ids = [str(m["id"]) for m in config["microgrids"]]
        self.future_h = int(config.get("observation", {}).get("future_horizon", 4))
        sharing_cfg = config.get("ess_sharing", {})
        self.control_mode = str(sharing_cfg.get("control_mode", "residual_auto"))
        self.observation_mode = str(sharing_cfg.get("observation_mode", "local"))
        self.profiles = profiles
        self.forecasts = forecasts
        self.norm = {mg: profiles.normalizers(mg, delta_t) for mg in self.agent_ids}
        self.initial_soc = {
            str(mg["id"]): float(mg["battery"]["initial_soc"])
            for mg in config["microgrids"]
        }

    def schema(self, mg_id: str) -> list[str]:
        names = ["hour_sin", "hour_cos", "load", "pv", "wind", "grid_buy_price",
                 "grid_sell_price", "soc"]
        for k in range(1, self.future_h + 1):
            names.extend([f"load_t+{k}", f"pv_t+{k}", f"wind_t+{k}", f"price_t+{k}"])
        names.extend(["previous_p2p_net", "previous_grid_net", "previous_system_p2p"])
        if self.control_mode == "explicit_offer":
            names.extend([
                "available_charge_ratio", "available_discharge_ratio", "terminal_soc_gap",
                "previous_ess_p2p_offer", "previous_ess_p2p_accepted",
                "previous_ess_p2p_unmatched", "previous_system_buy_demand",
                "previous_system_sell_offer", "previous_clearing_ratio",
            ])
            if self.observation_mode == "cross_microgrid":
                for peer in self.agent_ids:
                    if peer != mg_id:
                        names.extend([f"{peer}_net_load", f"{peer}_soc"])
                for k in range(1, self.future_h + 1):
                    for peer in self.agent_ids:
                        if peer != mg_id:
                            names.append(f"{peer}_net_load_t+{k}")
        return names

    def obs_dim(self, mg_id: str) -> int:
        return len(self.schema(mg_id))

    def build(self, mg_id: str, row: pd.Series, t: int, battery_soc: float,
              history: Mapping[str, Any],
              exogenous: Mapping[str, float] | None = None,
              soc_by_microgrid: Mapping[str, float] | None = None,
              battery_flex_by_microgrid: Mapping[str, Mapping[str, float]] | None = None) -> np.ndarray:
        n = self.norm[mg_id]
        ex = dict(exogenous) if exogenous is not None else self.profiles.exogenous(mg_id, row)
        hour = int(pd.Timestamp(row["timestamp"]).hour)
        system_norm = max(1.0, sum(self.norm[x]["trade"] for x in self.agent_ids))
        values = [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            ex["load_kw"] / n["load"],
            ex["pv_kw"] / n["pv"],
            ex["wind_kw"] / n["wind"],
            float(row["buy_price_per_kwh"]) / n["price"],
            float(row["sell_price_per_kwh"]) / n["price"],
            float(battery_soc),
        ]
        for k in range(1, self.future_h + 1):
            f = self.forecasts.combined(mg_id, t, k)
            values.extend([
                f["load_kw"] / n["load"],
                f["pv_kw"] / n["pv"],
                f["wind_kw"] / n["wind"],
                f["grid_price_per_kwh"] / n["price"],
            ])
        previous = history["by_microgrid"][mg_id]
        values.extend([
            previous["p2p_net_kwh"] / n["trade"],
            previous["grid_net_kwh"] / n["trade"],
            history["total_p2p_kwh"] / system_norm,
        ])
        if self.control_mode == "explicit_offer":
            flex = (battery_flex_by_microgrid or {}).get(mg_id, {})
            values.extend([
                float(flex.get("available_charge_ratio", 0.0)),
                float(flex.get("available_discharge_ratio", 0.0)),
                float(battery_soc) - self.initial_soc[mg_id],
                previous["ess_p2p_offer_kwh"] / n["trade"],
                previous["ess_p2p_accepted_kwh"] / n["trade"],
                previous["ess_p2p_unmatched_kwh"] / n["trade"],
                history["total_buy_demand_kwh"] / system_norm,
                history["total_sell_offer_kwh"] / system_norm,
                float(history["clearing_ratio"]),
            ])
            if self.observation_mode == "cross_microgrid":
                peer_soc = soc_by_microgrid or {}
                for peer in self.agent_ids:
                    if peer == mg_id:
                        continue
                    peer_ex = self.profiles.exogenous(peer, row)
                    values.extend([
                        (peer_ex["load_kw"] - peer_ex["renewable_kw"]) / self.norm[peer]["power"],
                        float(peer_soc.get(peer, self.initial_soc[peer])),
                    ])
                for k in range(1, self.future_h + 1):
                    for peer in self.agent_ids:
                        if peer == mg_id:
                            continue
                        forecast = self.forecasts.combined(peer, t, k)
                        peer_net = forecast["load_kw"] - forecast["pv_kw"] - forecast["wind_kw"]
                        values.append(peer_net / self.norm[peer]["power"])
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.obs_dim(mg_id),) or not np.isfinite(result).all():
            raise RuntimeError(f"{mg_id}: invalid observation")
        return result
