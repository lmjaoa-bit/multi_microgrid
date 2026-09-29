from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from core.data_manager import DataManager
from core.profiles import HeterogeneousProfileModel
from core.reward_v2 import (
    calculate_reward_v2,
    resolve_carbon_factors,
    resolve_reward_v2_config,
)
from core.tariffs import buy_price_column, sell_price_column


@dataclass(frozen=True)
class DispatchData:
    timestamps: list[str]
    agent_ids: list[str]
    load_kw: dict[str, np.ndarray]
    renewable_kw: dict[str, np.ndarray]
    buy_price: np.ndarray | dict[str, np.ndarray]
    sell_price: np.ndarray | dict[str, np.ndarray]
    frame: pd.DataFrame | None = None

    @property
    def horizon(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True)
class DispatchResult:
    mode: str
    status: str
    total_cost: float
    total_profit: float
    profit_by_microgrid: dict[str, float]
    trajectories: pd.DataFrame
    pair_flows_kwh: dict[str, float]
    cost_breakdown: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {"mode": self.mode, "status": self.status, "total_cost": self.total_cost,
                "total_profit": self.total_profit, "profit_by_microgrid": self.profit_by_microgrid,
                "total_p2p_energy_kwh": float(sum(self.pair_flows_kwh.values())),
                **({"cost_breakdown": self.cost_breakdown} if self.cost_breakdown is not None else {}),
                **({"metadata": self.metadata} if self.metadata is not None else {})}


def load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        result = yaml.safe_load(path.read_text(encoding="utf-8"))
        result["_project_root"] = str(path.parent.parent)
        return result
    return dict(config)


def build_dispatch_data(config: str | Path | Mapping[str, Any], split: str = "test",
                        day: str | None = None) -> tuple[dict[str, Any], DispatchData]:
    cfg = load_config(config)
    manager = DataManager(cfg, split=split, seed=int(cfg.get("env", {}).get("seed", 42)))
    frame = manager.get_day(day)
    train = manager if split == "train" else DataManager(cfg, split="train")
    profiles = HeterogeneousProfileModel(cfg, train.df)
    ids = [str(mg["id"]) for mg in cfg["microgrids"]]
    exogenous = {mg: profiles.exogenous_series(mg, frame) for mg in ids}
    load = {mg: exogenous[mg]["load_kw"] for mg in ids}
    renewable = {mg: exogenous[mg]["renewable_kw"] for mg in ids}
    buy_price = {
        mg: frame[
            buy_price_column(mg) if buy_price_column(mg) in frame else "buy_price_per_kwh"
        ].to_numpy(float)
        for mg in ids
    }
    sell_price = {
        mg: frame[
            sell_price_column(mg) if sell_price_column(mg) in frame else "sell_price_per_kwh"
        ].to_numpy(float)
        for mg in ids
    }
    data = DispatchData(
        [str(x) for x in frame["timestamp"]], ids, load, renewable, buy_price, sell_price,
        frame=frame.reset_index(drop=True).copy(),
    )
    return cfg, data


def _dispatch_fingerprint(payload: Mapping[str, Any]) -> str:
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _array_values(value: np.ndarray | Mapping[str, np.ndarray], ids: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(mg): np.asarray(value[mg], dtype=np.float64).tolist()
            for mg in ids
        }
    return np.asarray(value, dtype=np.float64).tolist()


def _add_shared_pair_capacity_constraints(
    model: Any,
    flow_variables: Mapping[tuple[Any, ...], Any],
    agent_ids: Sequence[str],
    horizon: int,
    pair_capacity_kw: float,
) -> None:
    """Enforce one physical bidirectional capacity for every unordered P2P pair."""
    ids = list(agent_ids)
    for i, first in enumerate(ids):
        for second in ids[i + 1:]:
            for t in range(int(horizon)):
                model.addConstr(
                    flow_variables[(first, second, t)]
                    + flow_variables[(second, first, t)]
                    <= float(pair_capacity_kw)
                )


def solve_dispatch(
    config: Mapping[str, Any],
    data: DispatchData,
    agent_ids: Sequence[str],
    cooperative: bool,
    solver_output: bool = False,
    *,
    initial_soc_by_microgrid: Mapping[str, float] | None = None,
    terminal_target_soc_by_microgrid: Mapping[str, float] | None = None,
    apply_terminal_condition: bool = True,
    p2p_export_source: str = "unrestricted",
    include_dr: bool = False,
    objective_kind: str = "accounting_profit",
) -> DispatchResult:
    try:
        import highspy
    except ImportError as exc:  # pragma: no cover - exercised only on missing optional dependency
        raise ImportError("Install highspy to use the dispatch oracle") from exc

    ids = list(agent_ids)
    if objective_kind not in {"accounting_profit", "generalized_cost"}:
        raise ValueError(f"Unsupported dispatch objective_kind: {objective_kind!r}")
    generalized = objective_kind == "generalized_cost"
    reward_v2 = resolve_reward_v2_config(config) if generalized else None
    if generalized:
        if reward_v2 is None:
            raise ValueError(
                "generalized_cost objective requires reward.semantics_version="
                "generalized_system_cost_v2"
            )
        all_ids = [str(mg["id"]) for mg in config["microgrids"]]
        if ids != all_ids:
            raise ValueError(
                "generalized_cost reference is defined only for the full coalition"
            )
        if not include_dr:
            raise ValueError("generalized_cost reference requires include_dr=True")
    if p2p_export_source not in {"unrestricted", "renewable_only"}:
        raise ValueError(f"Unsupported P2P export source: {p2p_export_source}")
    cfg_by_id = {str(mg["id"]): mg for mg in config["microgrids"]}
    scenario_fingerprint = _dispatch_fingerprint({
        "timestamps": list(data.timestamps),
        "agent_ids": ids,
        "load_kw": {
            mg: np.asarray(data.load_kw[mg], dtype=np.float64).tolist()
            for mg in ids
        },
        "renewable_kw": {
            mg: np.asarray(data.renewable_kw[mg], dtype=np.float64).tolist()
            for mg in ids
        },
        "buy_price": _array_values(data.buy_price, ids),
        "sell_price": _array_values(data.sell_price, ids),
    })
    dt = float(config["time"].get("delta_t", 1.0))
    cost_cfg = config["cost"]
    unmet_price = float(cost_cfg["unmet_load_cost_per_kwh"])
    accounting_curtail_price = float(
        cost_cfg["renewable_curtailment_cost_per_kwh"]
    )
    curtail_price = (
        float(reward_v2["curtailment_price_per_kwh"])
        if reward_v2 is not None else accounting_curtail_price
    )
    carbon_factors = (
        resolve_carbon_factors(config, data.frame)
        if generalized and data.frame is not None else None
    )
    terminal_cfg = config["terminal_soc"]
    rate = float(terminal_cfg["penalty_per_kwh"])
    p2p_cfg = config["p2p"]
    p2p_enabled = bool(p2p_cfg.get("enabled", True))
    cooperative_requested = bool(cooperative)
    cooperative = bool(cooperative_requested and p2p_enabled)
    dispatch_mode = str(
        config.get("dispatch", {}).get("priority_mode", "legacy_local_first")
    )
    renewable_first = dispatch_mode == "renewable_load_first"
    # Bound unlimited interfaces by total physically available source power.
    physical_supply = sum(float(np.max(data.renewable_kw[m])) +
                          float(cfg_by_id[m]["battery"]["max_discharge_kw"]) for m in ids)
    pair_capacity_kw = (physical_supply if p2p_cfg["pair_capacity_kw"] is None
                        else float(p2p_cfg["pair_capacity_kw"]))
    allow_grid_resale = bool(p2p_cfg["allow_grid_resale"])
    model = highspy.Highs()
    model.setOptionValue("output_flag", bool(solver_output))
    continuous, integer, inf = highspy.HighsVarType.kContinuous, highspy.HighsVarType.kInteger, highspy.kHighsInf
    variables: dict[str, dict[tuple[Any, ...], Any]] = {name: {} for name in (
        "energy", "charge", "discharge", "battery_mode", "grid_import", "grid_export",
        "grid_mode", "curtailment", "unmet", "flow", "renewable_export_mode",
        "terminal_positive", "terminal_negative", "res_load", "res_charge", "ess_load",
        "grid_load", "grid_charge", "res_peer_load", "res_peer_charge", "ess_peer_load",
        "ess_export_mode", "team_load_peak")}

    def price_at(values: np.ndarray | Mapping[str, np.ndarray], mg: str, t: int) -> float:
        series = values[mg] if isinstance(values, Mapping) else values
        return float(series[t])

    def add(name: str, key: tuple[Any, ...], **kwargs: Any) -> Any:
        variable = model.addVariable(name=f"{name}[{','.join(map(str, key))}]", **kwargs)
        variables[name][key] = variable
        return variable

    # The original ESS-only benchmark keeps include_dr=False for backward
    # compatibility. The new full-information entry point enables this block.
    dr: dict[tuple[str, str, int], Any] = {}
    dr_power: dict[tuple[str, int], Any] = {}
    dr_terminal: dict[str, list[Any]] = {mg: [] for mg in ids}
    dr_devices: dict[str, Any] = {}
    def dvar(name: str, mg: str, t: int, **kwargs: Any) -> Any:
        variable = model.addVariable(name=f"dr_{name}[{mg},{t}]", type=continuous, **kwargs)
        dr[(mg, name, t)] = variable
        return variable

    if include_dr:
        if data.frame is None or len(data.frame) != data.horizon:
            raise ValueError("Full-information DR oracle requires the complete 24-hour episode frame")
        from components.industrial_mill import IndustrialMillDR
        from components.commercial_hvac import CommercialHVACDR
        from components.ev_charging import EVChargingDR
        for mg in ids:
            device_cfg = cfg_by_id[mg].get("dr")
            if device_cfg is None:
                raise ValueError(f"Missing physical DR configuration for {mg}")
            kind = str(device_cfg["type"])
            cls = {"industrial_mill": IndustrialMillDR,
                   "commercial_hvac": CommercialHVACDR,
                   "ev_charging": EVChargingDR}.get(kind)
            if cls is None:
                raise ValueError(f"Unsupported full-information DR type: {kind}")
            device = cls(device_cfg, delta_t=dt, episode_length=data.horizon)
            device.reset(data.frame, np.asarray(data.load_kw[mg], dtype=np.float64))
            dr_devices[mg] = device
            if kind == "industrial_mill":
                for t in range(data.horizon + 1):
                    initial = device.initial_silo_t
                    dvar("silo", mg, t,
                         lb=initial if t == 0 else device.silo_min_t,
                         ub=initial if t == 0 else device.silo_max_t)
                for t in range(data.horizon):
                    duty = dvar("duty", mg, t, lb=0.0, ub=1.0)
                    dr_power[(mg, t)] = device.rated_power_kw * duty
                    model.addConstr(dr[(mg, "silo", t+1)] ==
                                    dr[(mg, "silo", t)] +
                                    (device.production_tph * duty - device.downstream_tph) * dt)
                gap = dvar("silo_terminal_excess", mg, data.horizon, lb=0.0, ub=inf,
                           obj=(0.0 if device_cfg.get("terminal_mode") == "none" else rate) * device.rated_power_kw / device.production_tph)
                model.addConstr(gap >= dr[(mg, "silo", data.horizon)] -
                                device.initial_silo_t - device.terminal_tolerance_t)
                model.addConstr(gap >= device.initial_silo_t -
                                dr[(mg, "silo", data.horizon)] - device.terminal_tolerance_t)
                dr_terminal[mg].append(gap)
            elif kind == "commercial_hvac":
                for t in range(data.horizon+1):
                    dvar("air", mg, t, lb=device.initial_t_air if t == 0 else -inf,
                         ub=device.initial_t_air if t == 0 else inf)
                    dvar("mass", mg, t, lb=device.initial_t_mass if t == 0 else -inf,
                         ub=device.initial_t_mass if t == 0 else inf)
                for t in range(data.horizon):
                    heat = dvar("heat", mg, t, lb=0.0, ub=device.heat_max_kw)
                    cool = dvar("cool", mg, t, lb=0.0, ub=device.cool_max_kw)
                    mode = model.addVariable(name=f"dr_hvac_mode[{mg},{t}]",
                                             lb=0.0, ub=1.0, type=integer)
                    model.addConstr(heat <= device.heat_max_kw * mode)
                    model.addConstr(cool <= device.cool_max_kw * (1.0-mode))
                    dr_power[(mg,t)] = heat + cool
                    t_out, q_solar, q_internal = device._exogenous(t)
                    air, mass = dr[(mg,"air",t)], dr[(mg,"mass",t)]
                    model.addConstr(dr[(mg,"air",t+1)] ==
                        air + dt / device.c_air * ((t_out-air)/device.r_oa +
                        (mass-air)/device.r_am + q_internal +
                        device.solar_air_fraction*q_solar +
                        device.cop_heat*heat - device.cop_cool*cool))
                    model.addConstr(dr[(mg,"mass",t+1)] ==
                        mass + dt/device.c_mass * ((air-mass)/device.r_am +
                        (1.0-device.solar_air_fraction)*q_solar))
                    low, high = device._state_comfort_bounds(t+1)
                    if generalized:
                        low_slack = dvar(
                            "comfort_low_slack", mg, t, lb=0.0, ub=inf,
                            obj=float(reward_v2["comfort_price_per_degree_hour"]) * dt,
                        )
                        high_slack = dvar(
                            "comfort_high_slack", mg, t, lb=0.0, ub=inf,
                            obj=float(reward_v2["comfort_price_per_degree_hour"]) * dt,
                        )
                        model.addConstr(dr[(mg,"air",t+1)] + low_slack >= low)
                        model.addConstr(dr[(mg,"air",t+1)] - high_slack <= high)
                    else:
                        model.addConstr(dr[(mg,"air",t+1)] >= low)
                        model.addConstr(dr[(mg,"air",t+1)] <= high)
                # The environment converts positive final temperature deviation
                # with cooling COP, negative deviation with heating COP.
                for key, initial, tolerance, heat_capacity in (
                    ("air",device.initial_t_air,device.terminal_air_tolerance,device.c_air),
                    ("mass",device.initial_t_mass,device.terminal_mass_tolerance,device.c_mass),
                ):
                    state = dr[(mg,key,data.horizon)]
                    hot = dvar(key+"_terminal_hot",mg,data.horizon,lb=0.0,ub=inf,
                               obj=(0.0 if device_cfg.get("terminal_mode") == "none" else rate)*heat_capacity/device.cop_cool)
                    cold = dvar(key+"_terminal_cold",mg,data.horizon,lb=0.0,ub=inf,
                                obj=(0.0 if device_cfg.get("terminal_mode") == "none" else rate)*heat_capacity/device.cop_heat)
                    model.addConstr(hot >= state-initial-tolerance)
                    model.addConstr(cold >= initial-state-tolerance)
                    dr_terminal[mg].extend([hot,cold])
            elif kind == "ev_charging":
                if device.cohort_mode:
                    departure_rate = (
                        float(reward_v2["ev_shortfall_price_per_kwh"])
                        if generalized
                        else float(device_cfg.get("departure_penalty_per_kwh", rate))
                    )
                    group_energy_names: list[tuple[str, dict[str, Any]]] = []
                    for session in device.sessions:
                        group = str(session["group_id"])
                        energy_name = f"ev_{group}_energy"
                        charge_name = f"ev_{group}_charge"
                        group_energy_names.append((energy_name, session))
                        initial = float(session["capacity_kwh"]) * float(session["initial_soc"])
                        maximum = float(session["capacity_kwh"]) * float(session["target_soc"])
                        for t in range(data.horizon + 1):
                            dvar(
                                energy_name, mg, t,
                                lb=initial if t == 0 else 0.0,
                                ub=initial if t == 0 else maximum,
                            )
                        for t in range(data.horizon):
                            hour = pd.Timestamp(data.timestamps[t]).hour
                            available = int(session["arrival_hour"]) <= hour < int(session["departure_hour"])
                            charge = dvar(
                                charge_name, mg, t, lb=0.0,
                                ub=float(session["max_charge_kw"]) if available else 0.0,
                            )
                            model.addConstr(
                                dr[(mg, energy_name, t + 1)]
                                == dr[(mg, energy_name, t)] + device.eta * charge * dt
                            )
                        departure_index = min(int(session["departure_hour"]), data.horizon)
                        shortfall = dvar(
                            f"ev_{group}_departure_shortfall", mg, departure_index,
                            lb=0.0, ub=inf, obj=departure_rate,
                        )
                        model.addConstr(
                            shortfall >= maximum - dr[(mg, energy_name, departure_index)]
                        )
                        dr_terminal[mg].append(shortfall)

                    for t in range(data.horizon + 1):
                        aggregate_energy = dvar(
                            "ev_energy", mg, t, lb=0.0, ub=device.capacity_kwh,
                        )
                        model.addConstr(
                            aggregate_energy
                            == sum(dr[(mg, energy_name, t)]
                                   for energy_name, _ in group_energy_names)
                        )
                    for t in range(data.horizon):
                        charge = dvar("ev_charge", mg, t, lb=0.0, ub=device.max_charge_kw)
                        model.addConstr(
                            charge
                            == sum(
                                dr[(mg, f"ev_{session['group_id']}_charge", t)]
                                for _, session in group_energy_names
                            )
                        )
                        dr_power[(mg, t)] = charge
                else:
                    capacity = device.capacity_kwh
                    initial = capacity * device.initial_soc
                    maximum = capacity * device.max_soc
                    for t in range(data.horizon+1):
                        dvar("ev_energy",mg,t,lb=initial if t==0 else 0.0,
                             ub=initial if t==0 else maximum)
                    for t in range(data.horizon):
                        hour = pd.Timestamp(data.timestamps[t]).hour
                        available = device.available_start_hour <= hour < device.departure_hour
                        charge = dvar("ev_charge",mg,t,lb=0.0,
                                      ub=device.max_charge_kw if available else 0.0)
                        dr_power[(mg,t)] = charge
                        model.addConstr(dr[(mg,"ev_energy",t+1)] ==
                                        dr[(mg,"ev_energy",t)] + device.eta*charge*dt)
                    shortfall = dvar(
                        "ev_terminal_shortfall", mg, data.horizon,
                        lb=0.0, ub=inf,
                        obj=(
                            float(reward_v2["ev_shortfall_price_per_kwh"])
                            if generalized else rate
                        ),
                    )
                    model.addConstr(shortfall >= capacity*device.target_soc -
                                    dr[(mg,"ev_energy",data.horizon)])
                    dr_terminal[mg].append(shortfall)

    initial_state_payload: dict[str, Any] = {}
    for mg in ids:
        battery_cfg = cfg_by_id[mg]["battery"]
        initial_soc = (
            float(initial_soc_by_microgrid[mg])
            if initial_soc_by_microgrid is not None and mg in initial_soc_by_microgrid
            else float(battery_cfg["initial_soc"])
        )
        entry: dict[str, Any] = {"battery_initial_soc": initial_soc}
        device = dr_devices.get(mg)
        if device is not None:
            if hasattr(device, "initial_silo_t"):
                entry["mill_initial_silo_t"] = float(device.initial_silo_t)
            elif hasattr(device, "initial_t_air"):
                entry["hvac_initial_t_air_c"] = float(device.initial_t_air)
                entry["hvac_initial_t_mass_c"] = float(device.initial_t_mass)
            elif getattr(device, "cohort_mode", False):
                entry["ev_sessions"] = [
                    {
                        "group_id": str(session["group_id"]),
                        "arrival_hour": int(session["arrival_hour"]),
                        "departure_hour": int(session["departure_hour"]),
                        "demand_kwh": float(session["demand_kwh"]),
                        "initial_soc": float(session["initial_soc"]),
                    }
                    for session in device.sessions
                ]
            else:
                entry["ev_initial_soc"] = float(device.initial_soc)
        initial_state_payload[mg] = entry
    initial_state_fingerprint = _dispatch_fingerprint(initial_state_payload)

    import_bounds = {}
    for mg in ids:
        device = dr_devices.get(mg)
        dr_bound = (float(device.rated_power_kw) if hasattr(device, "initial_silo_t") else
                    max(float(device.heat_max_kw), float(device.cool_max_kw))
                    if hasattr(device, "initial_t_air") else
                    float(device.max_charge_kw) if device is not None else 0.0)
        physical_demand = float(np.max(data.load_kw[mg])) + dr_bound + float(cfg_by_id[mg]["battery"]["max_charge_kw"])
        configured = cfg_by_id[mg]["grid"]["import_limit_kw"]
        import_bounds[mg] = physical_demand if configured is None else float(configured)

    for mg in ids:
        battery = cfg_by_id[mg]["battery"]
        capacity = float(battery["capacity_kwh"])
        minimum = capacity * float(battery["soc_min"])
        maximum = capacity * float(battery["soc_max"])
        initial_soc = (
            float(initial_soc_by_microgrid[mg])
            if initial_soc_by_microgrid is not None and mg in initial_soc_by_microgrid
            else float(battery["initial_soc"])
        )
        initial = capacity * initial_soc
        add("energy", (mg, 0), lb=initial, ub=initial, type=continuous)
        for t in range(1, data.horizon + 1):
            add("energy", (mg, t), lb=minimum, ub=maximum, type=continuous)
        for t in range(data.horizon):
            charge = add("charge", (mg, t), lb=0.0, ub=float(battery["max_charge_kw"]),
                         obj=float(battery["degradation_cost_per_kwh"]) * dt, type=continuous)
            discharge = add("discharge", (mg, t), lb=0.0, ub=float(battery["max_discharge_kw"]),
                            obj=float(battery["degradation_cost_per_kwh"]) * dt, type=continuous)
            battery_mode = add("battery_mode", (mg, t), lb=0.0, ub=1.0, type=integer)
            grid_import_objective = price_at(data.buy_price, mg, t) * dt
            if generalized:
                if carbon_factors is None:
                    raise ValueError("generalized_cost requires carbon factors")
                grid_import_objective += (
                    float(reward_v2["carbon_price_per_kg"])
                    * float(carbon_factors[t])
                    * dt
                )
            grid_import = add(
                "grid_import", (mg, t), lb=0.0,
                ub=import_bounds[mg],
                obj=grid_import_objective, type=continuous,
            )
            grid_export = add(
                "grid_export", (mg, t), lb=0.0,
                ub=0.0 if generalized else float(cfg_by_id[mg]["grid"]["export_limit_kw"]),
                obj=0.0 if generalized else -price_at(data.sell_price, mg, t) * dt,
                type=continuous,
            )
            grid_mode = add("grid_mode", (mg, t), lb=0.0, ub=1.0, type=integer)
            if cooperative and p2p_export_source == "renewable_only":
                add("renewable_export_mode", (mg, t), lb=0.0, ub=1.0, type=integer)
            curtailment = add("curtailment", (mg, t), lb=0.0, ub=inf,
                              obj=curtail_price * dt, type=continuous)
            unmet = add("unmet", (mg, t), lb=0.0, ub=inf, obj=unmet_price * dt, type=continuous)
            model.addConstr(variables["energy"][(mg, t + 1)] == variables["energy"][(mg, t)]
                            + float(battery["eta_charge"]) * charge * dt
                            - discharge * dt / float(battery["eta_discharge"]))
            model.addConstr(charge <= float(battery["max_charge_kw"]) * battery_mode)
            model.addConstr(discharge <= float(battery["max_discharge_kw"]) * (1.0 - battery_mode))
            model.addConstr(discharge * dt / float(battery["eta_discharge"])
                            <= variables["energy"][(mg, t)] - minimum)
            big_m = max(import_bounds[mg],
                        float(cfg_by_id[mg]["grid"]["export_limit_kw"]))
            model.addConstr(grid_import <= big_m * grid_mode)
            model.addConstr(grid_export <= big_m * (1.0 - grid_mode))

    if cooperative:
        for t in range(data.horizon):
            for source in ids:
                for destination in ids:
                    if source != destination:
                        add("flow", (source, destination, t), lb=0.0,
                            ub=pair_capacity_kw, obj=1e-8, type=continuous)
        _add_shared_pair_capacity_constraints(
            model, variables["flow"], ids, data.horizon, pair_capacity_kw
        )

    # Source/use ledgers forbid storage-to-storage, storage-to-grid and resale.
    # Imported renewable power terminates at a load or charger; it cannot be relayed.
    for t in range(data.horizon):
        if cooperative:
            for source in ids:
                for destination in ids:
                    if source == destination:
                        continue
                    key = (source, destination, t)
                    for name in ("res_peer_load", "res_peer_charge", "ess_peer_load"):
                        add(name, key, lb=0, ub=pair_capacity_kw, type=continuous)
                    model.addConstr(variables["flow"][key] == sum(
                        variables[name][key] for name in
                        ("res_peer_load", "res_peer_charge", "ess_peer_load")))
                    if p2p_export_source == "renewable_only":
                        model.addConstr(variables["ess_peer_load"][key] == 0)
        for mg in ids:
            key = (mg, t)
            for name in ("res_load", "res_charge", "ess_load", "grid_load", "grid_charge"):
                add(name, key, lb=0, ub=inf, type=continuous)
            def peer_sum(name: str, incoming: bool = False) -> Any:
                return sum((variables[name][(other,mg,t) if incoming else (mg,other,t)]
                            for other in ids if other != mg), 0.0) if cooperative else 0.0
            res_out = peer_sum("res_peer_load") + peer_sum("res_peer_charge")
            ess_out = peer_sum("ess_peer_load")
            peer_load = peer_sum("res_peer_load", True) + peer_sum("ess_peer_load", True)
            model.addConstr(float(data.renewable_kw[mg][t]) ==
                variables["res_load"][key] + variables["res_charge"][key] + res_out +
                variables["grid_export"][key] + variables["curtailment"][key])
            model.addConstr(variables["discharge"][key] == variables["ess_load"][key] + ess_out)
            model.addConstr(variables["grid_import"][key] ==
                            variables["grid_load"][key] + variables["grid_charge"][key])
            model.addConstr(variables["charge"][key] == variables["res_charge"][key] +
                            variables["grid_charge"][key] + peer_sum("res_peer_charge", True))
            model.addConstr(float(data.load_kw[mg][t]) + (dr_power[key] if include_dr else 0) ==
                variables["res_load"][key] + variables["ess_load"][key] +
                variables["grid_load"][key] + peer_load + variables["unmet"][key])
            if cooperative:
                if renewable_first:
                    # In the renewable-first model, imported grid energy has no
                    # P2P/export path. ESS export is allowed only after the
                    # seller's post-renewable local residual load is not being
                    # carried by grid, unmet load, or another ESS. Incoming
                    # renewable may serve that local load without blocking the
                    # seller's own ESS from serving a different microgrid.
                    export_mode = add(
                        "ess_export_mode", key, lb=0, ub=1, type=integer
                    )
                    model.addConstr(
                        ess_out
                        <= float(cfg_by_id[mg]["battery"]["max_discharge_kw"])
                        * export_mode
                    )
                    incoming_ess = peer_sum("ess_peer_load", True)
                    model.addConstr(
                        variables["grid_load"][key]
                        + incoming_ess
                        + variables["unmet"][key]
                        <= import_bounds[mg] * (1 - export_mode)
                    )
                else:
                    # Preserve the historical finite-capacity/local-first
                    # oracle semantics for legacy configs.
                    if not allow_grid_resale:
                        model.addConstr(
                            res_out + ess_out
                            <= pair_capacity_kw * max(0, len(ids) - 1)
                            * (1 - variables["grid_mode"][key])
                        )
                    export_mode = add(
                        "ess_export_mode", key, lb=0, ub=1, type=integer
                    )
                    model.addConstr(
                        ess_out
                        <= float(cfg_by_id[mg]["battery"]["max_discharge_kw"])
                        * export_mode
                    )
                    model.addConstr(
                        variables["grid_load"][key]
                        + peer_load
                        + variables["unmet"][key]
                        <= import_bounds[mg] * (1 - export_mode)
                    )

    if renewable_first:
        for t in range(data.horizon):
            # Renewable-first priority is imposed on physical source totals while
            # preserving the oracle's deliberate within-coalition free allocation.
            # For cooperative solves with the current unlimited pair-capacity
            # configuration, the whole coalition is one pool. Independent solves
            # use one single-member pool per microgrid, so no virtual cross-MG
            # renewable transfer is introduced.
            priority_groups = [ids] if cooperative else [[mg] for mg in ids]
            for group_index, group in enumerate(priority_groups):
                renewable_total = float(sum(
                    float(data.renewable_kw[mg][t]) for mg in group
                ))
                demand_total = sum(
                    (
                        float(data.load_kw[mg][t])
                        + (dr_power[(mg, t)] if include_dr else 0.0)
                        for mg in group
                    ),
                    0.0,
                )
                res_to_load = sum(
                    (variables["res_load"][(mg, t)] for mg in group), 0.0
                )
                if cooperative:
                    res_to_load += sum(
                        (
                            variables["res_peer_load"][(src, dst, t)]
                            for src in group for dst in group if src != dst
                        ),
                        0.0,
                    )

                # Finite M proof: demand cannot exceed the sum of each MG's
                # import_bound, and renewable_total is known. import_bound itself
                # is max base demand + DR nameplate + battery charge nameplate.
                # Therefore M >= max(renewable_total, max feasible demand).
                priority_m = max(
                    1.0,
                    renewable_total,
                    float(sum(import_bounds[mg] for mg in group)),
                )
                load_mode = model.addVariable(
                    name=f"renewable_load_priority[{group_index},{t}]",
                    lb=0.0, ub=1.0, type=integer,
                )
                model.addConstr(res_to_load <= renewable_total)
                model.addConstr(res_to_load <= demand_total)
                model.addConstr(
                    res_to_load >= renewable_total - priority_m * load_mode
                )
                model.addConstr(
                    res_to_load >= demand_total - priority_m * (1.0 - load_mode)
                )

                res_to_charge = sum(
                    (variables["res_charge"][(mg, t)] for mg in group), 0.0
                )
                if cooperative:
                    res_to_charge += sum(
                        (
                            variables["res_peer_charge"][(src, dst, t)]
                            for src in group for dst in group if src != dst
                        ),
                        0.0,
                    )
                charge_total = sum(
                    (variables["charge"][(mg, t)] for mg in group), 0.0
                )
                renewable_after_load = renewable_total - res_to_load
                charge_mode = model.addVariable(
                    name=f"renewable_charge_priority[{group_index},{t}]",
                    lb=0.0, ub=1.0, type=integer,
                )
                model.addConstr(res_to_charge <= renewable_after_load)
                model.addConstr(res_to_charge <= charge_total)
                model.addConstr(
                    res_to_charge >= renewable_after_load - priority_m * charge_mode
                )
                model.addConstr(
                    res_to_charge >= charge_total - priority_m * (1.0 - charge_mode)
                )

    if generalized:
        peak = add(
            "team_load_peak", ("team",), lb=0.0, ub=inf,
            obj=float(reward_v2["peak_price_per_kw"]), type=continuous,
        )
        for t in range(data.horizon):
            model.addConstr(
                peak >= sum(
                    (
                        float(data.load_kw[mg][t])
                        + (dr_power[(mg, t)] if include_dr else 0.0)
                        for mg in ids
                    ),
                    0.0,
                )
            )

    tolerance = float(terminal_cfg["tolerance"])
    rate = float(terminal_cfg["penalty_per_kwh"])
    if apply_terminal_condition and terminal_cfg["mode"] != "none":
        for mg in ids:
            battery = cfg_by_id[mg]["battery"]
            target_soc = (
                float(terminal_target_soc_by_microgrid[mg])
                if terminal_target_soc_by_microgrid is not None
                and mg in terminal_target_soc_by_microgrid
                else float(battery["initial_soc"])
            )
            target = float(battery["capacity_kwh"]) * target_soc
            tolerance_kwh = tolerance * float(battery["capacity_kwh"])
            terminal_energy = variables["energy"][(mg, data.horizon)]
            if terminal_cfg["mode"] == "hard":
                model.addConstr(terminal_energy >= target - tolerance_kwh)
                model.addConstr(terminal_energy <= target + tolerance_kwh)
            else:
                positive = add("terminal_positive", (mg,), lb=0.0, ub=inf, obj=rate, type=continuous)
                negative = add("terminal_negative", (mg,), lb=0.0, ub=inf, obj=rate, type=continuous)
                model.addConstr(positive >= terminal_energy - target - tolerance_kwh)
                model.addConstr(negative >= target - terminal_energy - tolerance_kwh)

    model.minimize()
    status = str(model.getModelStatus()).replace("HighsModelStatus.", "")
    solution_valid = bool(model.getSolution().value_valid)
    if include_dr and not generalized and status != "kOptimal":
        raise RuntimeError(f"Full-information oracle not proven optimal: {status}")
    if not solution_valid:
        raise RuntimeError(f"Dispatch oracle failed: no feasible incumbent ({status})")
    solver_info = model.getInfo()
    solver_objective_value = float(model.getObjectiveValue())
    solver_bound_raw = getattr(solver_info, "mip_dual_bound", None)
    solver_gap_raw = getattr(solver_info, "mip_gap", None)
    solver_bound = (
        float(solver_bound_raw)
        if solver_bound_raw is not None and np.isfinite(float(solver_bound_raw))
        else None
    )
    solver_gap = (
        float(solver_gap_raw)
        if solver_gap_raw is not None and np.isfinite(float(solver_gap_raw))
        else None
    )

    def value(name: str, key: tuple[Any, ...]) -> float:
        result = float(model.val(variables[name][key]))
        return 0.0 if abs(result) < 1e-8 else result

    rows: list[dict[str, Any]] = []
    pair_totals: dict[str, float] = {}
    profit_by = {mg: 0.0 for mg in ids}
    cost_breakdown = {key: 0.0 for key in (
        "grid_import_cost", "grid_export_revenue",
        "battery_degradation_cost", "unmet_load_cost",
        "renewable_curtailment_cost", "ess_terminal_penalty",
        "industrial_terminal_penalty", "hvac_terminal_penalty",
        "ev_terminal_penalty")}
    for t in range(data.horizon):
        for mg in ids:
            incoming = outgoing = 0.0
            if cooperative:
                for other in ids:
                    if other == mg:
                        continue
                    received = value("flow", (other, mg, t))
                    sent = value("flow", (mg, other, t))
                    incoming += received
                    outgoing += sent
                    if sent > 0.0:
                        key = f"{mg}->{other}"
                        pair_totals[key] = pair_totals.get(key, 0.0) + sent * dt
            battery = cfg_by_id[mg]["battery"]
            charge, discharge = value("charge", (mg, t)), value("discharge", (mg, t))
            grid_import, grid_export = value("grid_import", (mg, t)), value("grid_export", (mg, t))
            curtailment, unmet = value("curtailment", (mg, t)), value("unmet", (mg, t))
            operating_profit = (
                grid_export * price_at(data.sell_price, mg, t)
                - grid_import * price_at(data.buy_price, mg, t)
            ) * dt
            operating_profit -= (charge + discharge) * dt * float(battery["degradation_cost_per_kwh"])
            operating_profit -= (
                curtailment * dt * accounting_curtail_price
                + unmet * dt * unmet_price
            )
            profit_by[mg] += float(operating_profit)
            cost_breakdown["grid_import_cost"] += grid_import*price_at(data.buy_price,mg,t)*dt
            cost_breakdown["grid_export_revenue"] += grid_export*price_at(data.sell_price,mg,t)*dt
            cost_breakdown["battery_degradation_cost"] += (
                charge+discharge)*dt*float(battery["degradation_cost_per_kwh"])
            cost_breakdown["unmet_load_cost"] += unmet*dt*unmet_price
            cost_breakdown["renewable_curtailment_cost"] += (
                curtailment * dt * accounting_curtail_price
            )
            flexible = 0.0
            dr_row: dict[str, float] = {}
            if include_dr:
                device = dr_devices[mg]
                if hasattr(device, "initial_silo_t"):
                    flexible = device.rated_power_kw * float(model.val(dr[(mg,"duty",t)]))
                    dr_row = {"mill_duty":float(model.val(dr[(mg,"duty",t)])),
                              "silo_after_t":float(model.val(dr[(mg,"silo",t+1)]))}
                elif hasattr(device, "initial_t_air"):
                    heat = float(model.val(dr[(mg,"heat",t)]))
                    cool = float(model.val(dr[(mg,"cool",t)]))
                    flexible = heat + cool
                    dr_row = {
                        "hvac_heating_kw": heat,
                        "hvac_cooling_kw": cool,
                        "hvac_air_after_c": float(model.val(dr[(mg,"air",t+1)])),
                        "hvac_mass_after_c": float(model.val(dr[(mg,"mass",t+1)])),
                    }
                    if generalized:
                        low_slack = float(model.val(dr[(mg, "comfort_low_slack", t)]))
                        high_slack = float(model.val(dr[(mg, "comfort_high_slack", t)]))
                        dr_row.update({
                            "comfort_low_slack_c": low_slack,
                            "comfort_high_slack_c": high_slack,
                            "comfort_violation_c": low_slack + high_slack,
                            "comfort_degree_hours": (low_slack + high_slack) * dt,
                        })
                else:
                    flexible = float(model.val(dr[(mg,"ev_charge",t)]))
                    dr_row = {
                        "ev_soc_after": float(model.val(dr[(mg,"ev_energy",t+1)]))
                        / device.capacity_kwh
                    }
                    if getattr(device, "cohort_mode", False):
                        for session in device.sessions:
                            group = str(session["group_id"])
                            charge_name = f"ev_{group}_charge"
                            energy_name = f"ev_{group}_energy"
                            dr_row[f"ev_{group}_charge_kw"] = float(
                                model.val(dr[(mg, charge_name, t)])
                            )
                            dr_row[f"ev_{group}_energy_kwh"] = float(
                                model.val(dr[(mg, energy_name, t + 1)])
                            )
                            departure_index = min(
                                int(session["departure_hour"]), data.horizon
                            )
                            dr_row[f"ev_{group}_departure_shortfall_kwh"] = (
                                float(model.val(dr[
                                    (mg, f"ev_{group}_departure_shortfall", departure_index)
                                ]))
                                if t + 1 == departure_index else 0.0
                            )

            rows.append({"timestamp": data.timestamps[t], "microgrid": mg,
                         "base_load_kw": float(data.load_kw[mg][t]),
                         "flexible_load_kw": flexible,
                         "load_kw": float(data.load_kw[mg][t]) + flexible, "renewable_kw": float(data.renewable_kw[mg][t]),
                         "battery_charge_kw": charge, "battery_discharge_kw": discharge,
                         "soc": value("energy", (mg, t + 1)) / float(battery["capacity_kwh"]),
                         "grid_import_kw": grid_import, "grid_export_kw": grid_export,
                         "p2p_import_kw": incoming, "p2p_export_kw": outgoing,
                         "renewable_to_local_load_kw": value("res_load", (mg, t)),
                         "renewable_to_local_charge_kw": value("res_charge", (mg, t)),
                         "ess_to_local_load_kw": value("ess_load", (mg, t)),
                         "grid_to_load_kw": value("grid_load", (mg, t)),
                         "grid_to_charge_kw": value("grid_charge", (mg, t)),
                         "renewable_peer_load_export_kw": (
                             sum(value("res_peer_load", (mg, other, t))
                                 for other in ids if other != mg)
                             if cooperative else 0.0
                         ),
                         "renewable_peer_charge_export_kw": (
                             sum(value("res_peer_charge", (mg, other, t))
                                 for other in ids if other != mg)
                             if cooperative else 0.0
                         ),
                         "ess_peer_load_export_kw": (
                             sum(value("ess_peer_load", (mg, other, t))
                                 for other in ids if other != mg)
                             if cooperative else 0.0
                         ),
                         "curtailment_kw": curtailment, "unmet_load_kw": unmet,
                         **dr_row})
    if apply_terminal_condition and terminal_cfg["mode"] == "penalty":
        for mg in ids:
            penalty = rate * (value("terminal_positive", (mg,)) + value("terminal_negative", (mg,)))
            profit_by[mg] -= penalty
            cost_breakdown["ess_terminal_penalty"] += penalty
    if include_dr:
        for mg in ids:
            device = dr_devices[mg]
            if cfg_by_id[mg]["dr"].get("terminal_mode") == "none" and not getattr(device, "cohort_mode", False):
                continue
            if hasattr(device, "initial_silo_t"):
                excess = max(0.0, abs(float(model.val(dr[(mg,"silo",data.horizon)])) -
                                     device.initial_silo_t) - device.terminal_tolerance_t)
                dr_penalty = rate * excess * device.rated_power_kw / device.production_tph
                profit_by[mg] -= dr_penalty
                cost_breakdown["industrial_terminal_penalty"] += dr_penalty
            elif hasattr(device, "initial_t_air"):
                for key,initial,tolerance,heat_capacity in (
                    ("air",device.initial_t_air,device.terminal_air_tolerance,device.c_air),
                    ("mass",device.initial_t_mass,device.terminal_mass_tolerance,device.c_mass),
                ):
                    final = float(model.val(dr[(mg,key,data.horizon)]))
                    dr_penalty = rate*heat_capacity*(
                        max(0.0,final-initial-tolerance)/device.cop_cool +
                        max(0.0,initial-final-tolerance)/device.cop_heat)
                    profit_by[mg] -= dr_penalty
                    cost_breakdown["hvac_terminal_penalty"] += dr_penalty
            else:
                if getattr(device, "cohort_mode", False):
                    departure_rate = float(
                        cfg_by_id[mg]["dr"].get("departure_penalty_per_kwh", rate)
                    )
                    dr_penalty = departure_rate * sum(
                        float(model.val(variable)) for variable in dr_terminal[mg]
                    )
                else:
                    final = float(model.val(dr[(mg,"ev_energy",data.horizon)]))
                    dr_penalty = rate*max(0.0,device.capacity_kwh*device.target_soc-final)
                profit_by[mg] -= dr_penalty
                cost_breakdown["ev_terminal_penalty"] += dr_penalty
    total_profit = float(sum(profit_by.values()))
    result_frame = pd.DataFrame(rows)

    def _row_finite_or_zero(row: Mapping[str, Any], key: str) -> float:
        value = row.get(key, 0.0)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return parsed if np.isfinite(parsed) else 0.0

    generalized_breakdown = None
    replayed_generalized_cost = None
    if generalized:
        if data.frame is None or carbon_factors is None:
            raise ValueError("generalized_cost replay requires the complete episode frame")
        raw = {
            "grid_import_cost": 0.0,
            "battery_degradation_cost": 0.0,
            "curtailment_kwh": 0.0,
            "carbon_kg": 0.0,
            "delta_peak_kw": 0.0,
            "mill_unserved_t": 0.0,
            "comfort_degree_hours": 0.0,
            "ev_departure_shortfall_kwh": 0.0,
            "unmet_kwh": 0.0,
        }
        for t in range(data.horizon):
            load_total = 0.0
            for mg in ids:
                row = result_frame[
                    (result_frame["timestamp"] == data.timestamps[t])
                    & (result_frame["microgrid"] == mg)
                ].iloc[0]
                grid_import = float(row["grid_import_kw"])
                raw["grid_import_cost"] += (
                    grid_import * price_at(data.buy_price, mg, t) * dt
                )
                battery = cfg_by_id[mg]["battery"]
                raw["battery_degradation_cost"] += (
                    (float(row["battery_charge_kw"]) + float(row["battery_discharge_kw"]))
                    * dt
                    * float(battery["degradation_cost_per_kwh"])
                )
                raw["curtailment_kwh"] += float(row["curtailment_kw"]) * dt
                raw["carbon_kg"] += (
                    grid_import * float(carbon_factors[t]) * dt
                )
                raw["unmet_kwh"] += float(row["unmet_load_kw"]) * dt
                raw["comfort_degree_hours"] += _row_finite_or_zero(
                    row, "comfort_degree_hours"
                )
                raw["ev_departure_shortfall_kwh"] += sum(
                    _row_finite_or_zero(row, str(key))
                    for key in row.index
                    if str(key).startswith("ev_")
                    and str(key).endswith("_departure_shortfall_kwh")
                )
                load_total += float(row["load_kw"])
            raw["delta_peak_kw"] = max(raw["delta_peak_kw"], load_total)
        generalized_breakdown = calculate_reward_v2(
            raw=raw, rates=reward_v2
        )
        replayed_generalized_cost = float(
            generalized_breakdown["generalized_system_cost"]
        )
        tolerance_obj = 1e-6 * max(1.0, abs(replayed_generalized_cost))
        if abs(replayed_generalized_cost - solver_objective_value) > tolerance_obj:
            raise RuntimeError(
                "Generalized solver objective does not match independently "
                f"recomputed trajectory cost: solver={solver_objective_value}, "
                f"replay={replayed_generalized_cost}"
            )

    result_cost = (
        replayed_generalized_cost if generalized else -total_profit
    )
    result_breakdown = (
        {key: float(value) for key, value in generalized_breakdown.items()
         if key.startswith("cost_")}
        if generalized and generalized_breakdown is not None
        else cost_breakdown if include_dr else None
    )
    return DispatchResult(("cooperative_full_information_relaxed" if cooperative else
                           "independent_full_information") if include_dr else
                          ("cooperative" if cooperative else "independent"),
                          status, float(result_cost),
                          total_profit, profit_by, result_frame, pair_totals,
                          result_breakdown,
                          {
                              "source_priority": (
                                  "renewable_load_first"
                                  if renewable_first else "legacy_local_first"
                              ),
                              "allocation": (
                                  "source_priority_free_allocation"
                                  if renewable_first else "legacy_oracle_allocation"
                              ),
                              "perfect_foresight": True,
                              "full_information_oracle": bool(include_dr),
                              "p2p_export_source": p2p_export_source,
                              "grid_resale_allowed": False,
                              "ess_to_ess_allowed": False,
                              "ess_to_external_grid_allowed": False,
                              "priority_scope": (
                                  "coalition" if cooperative else "single_microgrid"
                              ),
                              "cooperative_requested": cooperative_requested,
                              "p2p_enabled": p2p_enabled,
                              "cooperative_effective": cooperative,
                              "objective_kind": objective_kind,
                              "solver_objective_value": solver_objective_value,
                              "solver_objective_bound": solver_bound,
                              "solver_mip_gap": solver_gap,
                              "proven_optimal": status == "kOptimal",
                              "feasible_incumbent": solution_valid,
                              "replayed_trajectory_cost": replayed_generalized_cost,
                              "scenario_fingerprint": scenario_fingerprint,
                              "initial_state_fingerprint": initial_state_fingerprint,
                          })
