from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.data_manager import DataManager
from core.profiles import HeterogeneousProfileModel
from optimization.common_constraints import load_config


def diagnose_scenario(config: str | Path | Mapping[str, Any], split: str = "train") -> dict[str, Any]:
    """Quantify heterogeneity and coordination opportunities before training."""
    cfg = load_config(config)
    manager = DataManager(cfg, split=split, seed=int(cfg.get("env", {}).get("seed", 42)))
    train = manager if split == "train" else DataManager(cfg, split="train")
    profiles = HeterogeneousProfileModel(cfg, train.df)
    frame = manager.df
    agent_ids = [str(item["id"]) for item in cfg["microgrids"]]
    exogenous = {mg: profiles.exogenous_series(mg, frame) for mg in agent_ids}
    load_by_mg = {mg: exogenous[mg]["load_kw"] for mg in agent_ids}
    renewable_by_mg = {mg: exogenous[mg]["renewable_kw"] for mg in agent_ids}
    buy_price = frame["buy_price_per_kwh"].to_numpy(float)
    sell_price = frame["sell_price_per_kwh"].to_numpy(float)
    by_cfg = {str(item["id"]): item for item in cfg["microgrids"]}
    microgrids: dict[str, dict[str, float]] = {}
    surplus_by_mg: dict[str, np.ndarray] = {}
    deficit_by_mg: dict[str, np.ndarray] = {}
    for mg in agent_ids:
        load, renewable = load_by_mg[mg], renewable_by_mg[mg]
        surplus, deficit = np.maximum(renewable - load, 0.0), np.maximum(load - renewable, 0.0)
        surplus_by_mg[mg], deficit_by_mg[mg] = surplus, deficit
    for mg in agent_ids:
        load, renewable = load_by_mg[mg], renewable_by_mg[mg]
        surplus, deficit = surplus_by_mg[mg], deficit_by_mg[mg]
        battery = by_cfg[mg]["battery"]
        microgrids[mg] = {
            "res_load_ratio": float(renewable.sum() / max(load.sum(), 1e-9)),
            "surplus_hour_ratio": float(np.mean(surplus > 1e-9)),
            "deficit_hour_ratio": float(np.mean(deficit > 1e-9)),
            "mean_surplus_kw": float(surplus.mean()), "mean_deficit_kw": float(deficit.mean()),
            "grid_import_dependency": float(deficit.sum() / max(load.sum(), 1e-9)),
            "grid_export_ratio": float(surplus.sum() / max(renewable.sum(), 1e-9)),
            "ess_utilization_potential": float(min(1.0, (surplus.sum() + deficit.sum()) /
                                                       max(2 * float(battery["capacity_kwh"]) * len(load), 1e-9))),
            "initial_soc": float(battery["initial_soc"]),
            "soc_operating_range": float(battery["soc_max"]) - float(battery["soc_min"]),
            "potential_p2p_energy_kwh": float(min(surplus.sum(), sum(deficit_by_mg[x].sum()
                                                                      for x in agent_ids if x != mg))),
        }
    total_surplus = np.sum([surplus_by_mg[mg] for mg in agent_ids], axis=0)
    total_deficit = np.sum([deficit_by_mg[mg] for mg in agent_ids], axis=0)
    p2p_potential = np.minimum(total_surplus, total_deficit)
    net_load_by_mg = {
        mg: load_by_mg[mg] - renewable_by_mg[mg] for mg in agent_ids
    }
    pair_net_load_correlation = {
        f"{left}-{right}": float(np.corrcoef(net_load_by_mg[left], net_load_by_mg[right])[0, 1])
        for index, left in enumerate(agent_ids)
        for right in agent_ids[index + 1:]
    }

    future_horizon = int(cfg.get("observation", {}).get("future_horizon", 4))
    pair_capacity_kw = float(cfg["p2p"]["pair_capacity_kw"])
    opportunity_windows: set[tuple[str, int]] = set()
    opportunity_days: set[str] = set()
    window_deliverable_kwh: dict[tuple[str, int], float] = {}
    for day in manager.valid_days:
        day_frame = manager.get_day(day)
        day_exogenous = {
            mg: profiles.exogenous_series(mg, day_frame) for mg in agent_ids
        }
        day_net_load = {
            mg: day_exogenous[mg]["load_kw"] - day_exogenous[mg]["renewable_kw"]
            for mg in agent_ids
        }
        day_surplus = {mg: np.maximum(-day_net_load[mg], 0.0) for mg in agent_ids}
        day_deficit = {mg: np.maximum(day_net_load[mg], 0.0) for mg in agent_ids}
        day_key = str(day.date())
        for t in range(len(day_frame) - 1):
            end = min(len(day_frame), t + future_horizon + 1)
            for source in agent_ids:
                for destination in agent_ids:
                    if source == destination:
                        continue
                    current_deficit = float(day_deficit[destination][t])
                    source_after_load = max(
                        0.0, float(day_surplus[source][t]) - current_deficit,
                    )
                    future_deficit = float(day_deficit[destination][t + 1:end].sum())
                    battery = by_cfg[destination]["battery"]
                    eta_charge = float(battery["eta_charge"])
                    eta_discharge = float(battery["eta_discharge"])
                    charge_headroom_input = (
                        (float(battery["soc_max"]) - float(battery["initial_soc"]))
                        * float(battery["capacity_kwh"]) / eta_charge
                    )
                    transfer_input = min(
                        source_after_load,
                        pair_capacity_kw,
                        float(battery["max_charge_kw"]),
                        charge_headroom_input,
                        future_deficit / (eta_charge * eta_discharge)
                        if future_deficit > 0.0 else 0.0,
                    )
                    if current_deficit <= 1e-9 or transfer_input <= 1e-9:
                        continue
                    key = (day_key, t)
                    deliverable = transfer_input * eta_charge * eta_discharge
                    opportunity_windows.add(key)
                    opportunity_days.add(day_key)
                    window_deliverable_kwh[key] = max(
                        window_deliverable_kwh.get(key, 0.0), float(deliverable),
                    )
    system = {
        "total_p2p_potential_kwh": float(p2p_potential.sum()),
        "simultaneous_surplus_deficit_hour_ratio": float(np.mean(p2p_potential > 1e-9)),
        "pair_net_load_correlation": pair_net_load_correlation,
        "spatiotemporal_horizon_hours": future_horizon,
        "spatiotemporal_opportunity_windows": len(opportunity_windows),
        "spatiotemporal_opportunity_window_ratio": float(
            len(opportunity_windows) / max(len(manager.valid_days) * int(cfg["time"]["episode_length"]), 1)
        ),
        "spatiotemporal_opportunity_days": len(opportunity_days),
        "spatiotemporal_opportunity_day_ratio": float(
            len(opportunity_days) / max(len(manager.valid_days), 1)
        ),
        "spatiotemporal_window_max_deliverable_kwh_sum": float(
            sum(window_deliverable_kwh.values())
        ),
        "p2p_pair_capacity_kw": pair_capacity_kw,
        "allow_grid_resale": bool(cfg["p2p"]["allow_grid_resale"]),
        "res_curtailment_potential_kwh": float(np.maximum(total_surplus - total_deficit, 0.0).sum()),
        "grid_cost_baseline": float(sum((deficit_by_mg[mg] * buy_price
                                         - surplus_by_mg[mg] * sell_price).sum()
                                        for mg in agent_ids)),
    }
    return {"split": split, "hours": len(frame), "microgrids": microgrids, "system": system}
