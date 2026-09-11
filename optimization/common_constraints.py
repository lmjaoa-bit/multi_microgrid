from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from core.data_manager import DataManager
from core.profiles import HeterogeneousProfileModel
from core.tariffs import buy_price_column, sell_price_column


@dataclass(frozen=True)
class DispatchData:
    timestamps: list[str]
    agent_ids: list[str]
    load_kw: dict[str, np.ndarray]
    renewable_kw: dict[str, np.ndarray]
    buy_price: np.ndarray | dict[str, np.ndarray]
    sell_price: np.ndarray | dict[str, np.ndarray]

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

    def summary(self) -> dict[str, Any]:
        return {"mode": self.mode, "status": self.status, "total_cost": self.total_cost,
                "total_profit": self.total_profit, "profit_by_microgrid": self.profit_by_microgrid,
                "total_p2p_energy_kwh": float(sum(self.pair_flows_kwh.values()))}


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
        [str(x) for x in frame["timestamp"]], ids, load, renewable, buy_price, sell_price
    )
    return cfg, data


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
) -> DispatchResult:
    try:
        import highspy
    except ImportError as exc:  # pragma: no cover - exercised only on missing optional dependency
        raise ImportError("Install highspy to use the dispatch oracle") from exc

    ids = list(agent_ids)
    if p2p_export_source not in {"unrestricted", "renewable_only"}:
        raise ValueError(f"Unsupported P2P export source: {p2p_export_source}")
    cfg_by_id = {str(mg["id"]): mg for mg in config["microgrids"]}
    dt = float(config["time"].get("delta_t", 1.0))
    cost_cfg = config["cost"]
    unmet_price = float(cost_cfg["unmet_load_cost_per_kwh"])
    curtail_price = float(cost_cfg["renewable_curtailment_cost_per_kwh"])
    terminal_cfg = config["terminal_soc"]
    p2p_cfg = config["p2p"]
    pair_capacity_kw = float(p2p_cfg["pair_capacity_kw"])
    allow_grid_resale = bool(p2p_cfg["allow_grid_resale"])
    model = highspy.Highs()
    model.setOptionValue("output_flag", bool(solver_output))
    continuous, integer, inf = highspy.HighsVarType.kContinuous, highspy.HighsVarType.kInteger, highspy.kHighsInf
    variables: dict[str, dict[tuple[Any, ...], Any]] = {name: {} for name in (
        "energy", "charge", "discharge", "battery_mode", "grid_import", "grid_export",
        "grid_mode", "curtailment", "unmet", "flow", "renewable_export_mode",
        "terminal_positive", "terminal_negative")}

    def price_at(values: np.ndarray | Mapping[str, np.ndarray], mg: str, t: int) -> float:
        series = values[mg] if isinstance(values, Mapping) else values
        return float(series[t])

    def add(name: str, key: tuple[Any, ...], **kwargs: Any) -> Any:
        variable = model.addVariable(name=f"{name}[{','.join(map(str, key))}]", **kwargs)
        variables[name][key] = variable
        return variable

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
            grid_import = add("grid_import", (mg, t), lb=0.0,
                              ub=float(cfg_by_id[mg]["grid"]["import_limit_kw"]),
                              obj=price_at(data.buy_price, mg, t) * dt, type=continuous)
            grid_export = add("grid_export", (mg, t), lb=0.0,
                              ub=float(cfg_by_id[mg]["grid"]["export_limit_kw"]),
                              obj=-price_at(data.sell_price, mg, t) * dt, type=continuous)
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
            big_m = max(float(cfg_by_id[mg]["grid"]["import_limit_kw"]),
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

    for t in range(data.horizon):
        for mg in ids:
            incoming = sum((variables["flow"][(other, mg, t)] for other in ids if other != mg), 0.0) if cooperative else 0.0
            outgoing = sum((variables["flow"][(mg, other, t)] for other in ids if other != mg), 0.0) if cooperative else 0.0
            if cooperative and not allow_grid_resale:
                model.addConstr(
                    outgoing <= pair_capacity_kw * (len(ids) - 1)
                    * (1.0 - variables["grid_mode"][(mg, t)])
                )
            if cooperative and p2p_export_source == "renewable_only":
                renewable_surplus = max(
                    0.0,
                    float(data.renewable_kw[mg][t]) - float(data.load_kw[mg][t]),
                )
                local_load_deficit = max(
                    0.0,
                    float(data.load_kw[mg][t]) - float(data.renewable_kw[mg][t]),
                )
                model.addConstr(variables["discharge"][(mg, t)] <= local_load_deficit)
                if renewable_surplus <= 1e-12:
                    model.addConstr(outgoing == 0.0)
                else:
                    battery = cfg_by_id[mg]["battery"]
                    export_mode = variables["renewable_export_mode"][(mg, t)]
                    maximum_outgoing = pair_capacity_kw * (len(ids) - 1)
                    model.addConstr(outgoing <= maximum_outgoing * (1.0 - export_mode))
                    model.addConstr(
                        outgoing + variables["charge"][(mg, t)]
                        <= renewable_surplus
                        + float(battery["max_charge_kw"]) * export_mode
                    )
            model.addConstr(float(data.renewable_kw[mg][t]) + variables["discharge"][(mg, t)]
                            + variables["grid_import"][(mg, t)] + incoming + variables["unmet"][(mg, t)]
                            == float(data.load_kw[mg][t]) + variables["charge"][(mg, t)]
                            + variables["grid_export"][(mg, t)] + outgoing
                            + variables["curtailment"][(mg, t)])
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
    if not model.getSolution().value_valid:
        raise RuntimeError(f"Dispatch oracle failed: {status}")

    def value(name: str, key: tuple[Any, ...]) -> float:
        result = float(model.val(variables[name][key]))
        return 0.0 if abs(result) < 1e-8 else result

    rows: list[dict[str, Any]] = []
    pair_totals: dict[str, float] = {}
    profit_by = {mg: 0.0 for mg in ids}
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
            operating_profit -= curtailment * dt * curtail_price + unmet * dt * unmet_price
            profit_by[mg] += float(operating_profit)
            rows.append({"timestamp": data.timestamps[t], "microgrid": mg,
                         "load_kw": float(data.load_kw[mg][t]), "renewable_kw": float(data.renewable_kw[mg][t]),
                         "battery_charge_kw": charge, "battery_discharge_kw": discharge,
                         "soc": value("energy", (mg, t + 1)) / float(battery["capacity_kwh"]),
                         "grid_import_kw": grid_import, "grid_export_kw": grid_export,
                         "p2p_import_kw": incoming, "p2p_export_kw": outgoing,
                         "curtailment_kw": curtailment, "unmet_load_kw": unmet})
    if apply_terminal_condition and terminal_cfg["mode"] == "penalty":
        for mg in ids:
            penalty = rate * (value("terminal_positive", (mg,)) + value("terminal_negative", (mg,)))
            profit_by[mg] -= penalty
    total_profit = float(sum(profit_by.values()))
    return DispatchResult("cooperative" if cooperative else "independent", status, -total_profit,
                          total_profit, profit_by, pd.DataFrame(rows), pair_totals)
