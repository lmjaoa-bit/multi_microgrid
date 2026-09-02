from __future__ import annotations

import csv
import json
import math
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPSILON = 1e-9
DEFAULT_TOLERANCE = 1e-7


def safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    return 0.0 if abs(denominator) <= EPSILON else float(numerator) / denominator


def distribution_statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0, "mean": None, "std": None, "median": None,
            "min": None, "max": None, "p05": None, "p25": None,
            "p75": None, "p95": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
    }


def _native(value: Any, path: str, invalid: list[str]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item, f"{path}.{key}" if path else str(key), invalid)
                for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_native(item, f"{path}[{index}]", invalid) for index, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        return _native(value.tolist(), path, invalid)
    if isinstance(value, np.generic):
        return _native(value.item(), path, invalid)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        invalid.append(path or "<root>")
        return None
    return value


def json_ready(value: Any) -> tuple[Any, list[str]]:
    invalid: list[str] = []
    result = _native(value, "", invalid)
    return result, invalid


def write_json(path: str | Path, value: Any) -> list[str]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean, invalid = json_ready(value)
    if invalid:
        warnings.warn(
            f"Non-finite values were replaced with null in {target}: {', '.join(invalid)}",
            RuntimeWarning,
            stacklevel=2,
        )
        if isinstance(clean, dict):
            clean.setdefault("serialization_warnings", []).extend(
                {"field": field, "reason": "non_finite_replaced_with_null"} for field in invalid
            )
    payload = json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = target.with_suffix(target.suffix + ".tmp")
    last_error: OSError | None = None
    for attempt in range(4):
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, target)
            last_error = None
            break
        except OSError as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return invalid


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean, invalid = json_ready(dict(row))
            if invalid:
                warnings.warn(
                    f"Non-finite CSV values were written as empty cells in {target}: {invalid}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            writer.writerow({key: "" if clean.get(key) is None else clean.get(key) for key in columns})


class EpisodeMetricsCollector:
    """One source of truth for train, validation, and test episode metrics."""

    def __init__(self, env: Any, capture_hourly: bool = False,
                 tolerance: float = DEFAULT_TOLERANCE) -> None:
        self.env = env
        self.agent_ids = list(env.agent_ids)
        self.capture_hourly = bool(capture_hourly)
        self.tolerance = float(tolerance)
        self.reset()

    def reset(self) -> None:
        ids = self.agent_ids
        self.steps = 0
        self.team_profit = 0.0
        self.team_reward = 0.0
        self.shaping_reward = 0.0
        self.profit = {mg: 0.0 for mg in ids}
        self.rewards = {mg: 0.0 for mg in ids}
        self.economics = {mg: defaultdict(float) for mg in ids}
        self.energy = {mg: defaultdict(float) for mg in ids}
        self.soc = {mg: [float(self.env.cores[mg].battery.soc)] for mg in ids}
        self.action_values = {mg: defaultdict(list) for mg in ids}
        self.action_clip_count = {mg: 0 for mg in ids}
        self.action_saturation_count = {mg: 0 for mg in ids}
        self.action_value_count = {mg: 0 for mg in ids}
        self.pair_energy = {
            f"{seller}_to_{buyer}": 0.0
            for seller in ids for buyer in ids if seller != buyer
        }
        self.p2p_prices: list[float] = []
        self.p2p_total = 0.0
        self.p2p_payment = {mg: 0.0 for mg in ids}
        self.violations: dict[str, list[float]] = {
            "soc": [], "power_balance": [], "p2p_balance": [],
            "grid_power": [], "battery_power": [],
        }
        self.hourly_rows: list[dict[str, Any]] = []

    def _economic_step(self, mg: str, info: Mapping[str, Any]) -> dict[str, float]:
        cost = info["costs"][mg]
        payment = float(info["p2p"]["by_microgrid"][mg].get("p2p_transfer_payment", 0.0))
        purchase = max(0.0, -payment)
        sale = max(0.0, payment)
        result = {
            "grid_import_cost": float(cost.get("grid_import_cost", 0.0)),
            "grid_export_revenue": float(cost.get("grid_export_revenue", 0.0)),
            "p2p_purchase_cost": purchase,
            "p2p_sale_revenue": sale,
            "p2p_net_revenue": payment,
            "battery_degradation_cost": float(cost.get("battery_degradation_cost", 0.0)),
            "unmet_load_cost": float(cost.get("unmet_load_cost", 0.0)),
            "renewable_curtailment_cost": float(cost.get("curtailment_cost", 0.0)),
            "terminal_soc_penalty": float(cost.get("terminal_soc_penalty", 0.0)),
            # P2P is an internal transfer: it changes MG allocation but sums to zero for the team.
            "physical_profit": float(cost.get("physical_profit", 0.0)) + payment,
        }
        for key, value in result.items():
            self.economics[mg][key] += value
        return result

    def _energy_step(self, mg: str, info: Mapping[str, Any]) -> dict[str, float]:
        plan = info["plans"][mg]
        settlement = info["settlements"][mg]
        device = info["devices"][mg]
        p2p = info["p2p"]["by_microgrid"][mg]
        dt = float(self.env.delta_t)
        renewable_available = float(plan.get("renewable_kw", 0.0)) * dt
        curtailed = float(settlement.get("renewable_curtailment_kw", 0.0)) * dt
        result = {
            "load_kwh": float(plan.get("load_kw", 0.0)) * dt,
            "renewable_available_kwh": renewable_available,
            "renewable_used_kwh": max(0.0, renewable_available - curtailed),
            "renewable_curtailment_kwh": curtailed,
            "grid_import_kwh": float(settlement.get("grid_import_kw", 0.0)) * dt,
            "grid_export_kwh": float(settlement.get("grid_export_kw", 0.0)) * dt,
            "p2p_buy_kwh": float(p2p.get("p2p_buy_kw", 0.0)) * dt,
            "p2p_sell_kwh": float(p2p.get("p2p_sell_kw", 0.0)) * dt,
            "renewable_p2p_sell_kwh": float(p2p.get("renewable_p2p_sell_kw", 0.0)) * dt,
            "ess_p2p_sell_kwh": float(p2p.get("ess_p2p_sell_kw", 0.0)) * dt,
            "p2p_to_load_kwh": float(p2p.get("p2p_to_load_kw", 0.0)) * dt,
            "p2p_to_battery_kwh": float(p2p.get("p2p_to_battery_kw", 0.0)) * dt,
            "renewable_p2p_offer_kwh": float(p2p.get("renewable_p2p_offer_kw", 0.0)) * dt,
            "ess_p2p_offer_kwh": float(p2p.get("ess_p2p_offer_kw", 0.0)) * dt,
            "ess_p2p_unmatched_kwh": float(p2p.get("ess_p2p_unmatched_kw", 0.0)) * dt,
            "ess_to_local_load_kwh": float(plan.get("ess_to_local_kw", 0.0)) * dt,
            "battery_charge_kwh": float(device.get("charge_kw", 0.0)) * dt,
            "battery_discharge_kwh": float(device.get("discharge_kw", 0.0)) * dt,
            "battery_throughput_kwh": float(device.get("throughput_kwh", 0.0)),
            "unmet_load_kwh": float(settlement.get("unmet_load_kw", 0.0)) * dt,
        }
        for key, value in result.items():
            self.energy[mg][key] += value
        return result

    def _constraint_step(self, info: Mapping[str, Any]) -> dict[str, Any]:
        ids = self.agent_ids
        soc_by_mg: dict[str, float] = {}
        battery_by_mg: dict[str, float] = {}
        grid_by_mg: dict[str, float] = {}
        for mg in ids:
            battery = self.env.cores[mg].battery
            soc = float(info["devices"][mg]["soc"])
            soc_violation = max(0.0, battery.soc_min - soc, soc - battery.soc_max)
            charge = float(info["devices"][mg]["charge_kw"])
            discharge = float(info["devices"][mg]["discharge_kw"])
            battery_violation = max(
                0.0, charge - battery.max_charge_kw, discharge - battery.max_discharge_kw,
                min(charge, discharge),
            )
            settlement = info["settlements"][mg]
            grid = self.env.grids[mg]
            grid_violation = max(
                0.0,
                float(settlement["grid_import_kw"]) - grid.import_limit_kw,
                float(settlement["grid_export_kw"]) - grid.export_limit_kw,
            )
            soc_by_mg[mg] = soc_violation
            battery_by_mg[mg] = battery_violation
            grid_by_mg[mg] = grid_violation
            self.violations["soc"].append(soc_violation)
            self.violations["battery_power"].append(battery_violation)
            self.violations["grid_power"].append(grid_violation)

        power_balance = sum(abs(float(info["settlements"][mg]["power_balance_residual_kw"])) for mg in ids)
        p2p_sell = sum(float(info["p2p"]["by_microgrid"][mg]["p2p_sell_kw"]) for mg in ids)
        p2p_buy = sum(float(info["p2p"]["by_microgrid"][mg]["p2p_buy_kw"]) for mg in ids)
        p2p_balance = abs(p2p_sell - p2p_buy)
        self.violations["power_balance"].append(power_balance)
        self.violations["p2p_balance"].append(p2p_balance)
        return {
            "power_balance": power_balance,
            "p2p_balance": p2p_balance,
            "soc_by_microgrid": soc_by_mg,
            "battery_by_microgrid": battery_by_mg,
            "grid_by_microgrid": grid_by_mg,
        }

    def record_step(self, raw_actions: Mapping[str, Sequence[float]], info: Mapping[str, Any],
                    rewards: Mapping[str, float] | None = None) -> None:
        self.steps += 1
        self.team_profit += float(info["team_profit"])
        self.team_reward += float(info.get("team_reward", 0.0))
        self.shaping_reward += float(info.get("shaping_reward", 0.0))
        economics: dict[str, dict[str, float]] = {}
        energies: dict[str, dict[str, float]] = {}

        for mg in self.agent_ids:
            economics[mg] = self._economic_step(mg, info)
            energies[mg] = self._energy_step(mg, info)
            self.profit[mg] += economics[mg]["physical_profit"]
            self.rewards[mg] += float((rewards or {}).get(mg, info.get("team_reward", 0.0)))
            self.soc[mg].append(float(info["devices"][mg]["soc"]))

            action = np.asarray(raw_actions[mg], dtype=np.float64).reshape(-1)
            action_names = ["battery"]
            if action.size >= 2:
                action_names.append("sharing_offer_ratio_raw")
            action_names.extend(f"action_{i}" for i in range(2, action.size))
            for index, value in enumerate(action):
                self.action_values[mg][action_names[index]].append(float(value))
                self.action_value_count[mg] += 1
                if value < -1.0 or value > 1.0:
                    self.action_clip_count[mg] += 1
                if abs(value) >= 1.0 - 1e-6:
                    self.action_saturation_count[mg] += 1

        p2p = info["p2p"]
        self.p2p_total += float(p2p.get("total_p2p_kwh", 0.0))
        for key, energy in p2p.get("pair_flows_kwh", {}).items():
            seller, buyer = str(key).split("->", 1)
            self.pair_energy[f"{seller}_to_{buyer}"] += float(energy)
        price = p2p.get("transfer_price_per_kwh")
        if price is not None and math.isfinite(float(price)):
            self.p2p_prices.append(float(price))
        for mg in self.agent_ids:
            payment = float(p2p["by_microgrid"][mg].get("p2p_transfer_payment", 0.0))
            self.p2p_payment[mg] += payment

        constraints = self._constraint_step(info)
        if self.capture_hourly:
            self.hourly_rows.append(self._hourly_row(raw_actions, info, economics, constraints))

    def _hourly_row(self, raw_actions: Mapping[str, Sequence[float]], info: Mapping[str, Any],
                    economics: Mapping[str, Mapping[str, float]], constraints: Mapping[str, Any]) -> dict[str, Any]:
        timestamp = str(info["timestamp"])
        date = timestamp[:10]
        try:
            hour = int(timestamp[11:13])
        except (TypeError, ValueError):
            hour = self.steps - 1
        row: dict[str, Any] = {"date": date, "hour": hour, "timestep": self.steps - 1}
        for mg in self.agent_ids:
            plan = info["plans"][mg]
            settlement = info["settlements"][mg]
            device = info["devices"][mg]
            p2p = info["p2p"]["by_microgrid"][mg]
            available = float(plan.get("renewable_kw", 0.0))
            curtailment = float(settlement.get("renewable_curtailment_kw", 0.0))
            row.update({
                f"{mg}_load_kw": float(plan.get("load_kw", 0.0)),
                f"{mg}_renewable_available_kw": available,
                f"{mg}_renewable_used_kw": max(0.0, available - curtailment),
                f"{mg}_renewable_curtailment_kw": curtailment,
                f"{mg}_grid_import_kw": float(settlement["grid_import_kw"]),
                f"{mg}_grid_export_kw": float(settlement["grid_export_kw"]),
                f"{mg}_battery_charge_kw": float(device["charge_kw"]),
                f"{mg}_battery_discharge_kw": float(device["discharge_kw"]),
                f"{mg}_soc": float(device["soc"]),
                f"{mg}_p2p_buy_kw": float(p2p["p2p_buy_kw"]),
                f"{mg}_p2p_sell_kw": float(p2p["p2p_sell_kw"]),
                f"{mg}_renewable_p2p_sell_kw": float(p2p.get("renewable_p2p_sell_kw", 0.0)),
                f"{mg}_ess_p2p_sell_kw": float(p2p.get("ess_p2p_sell_kw", 0.0)),
                f"{mg}_p2p_to_load_kw": float(p2p.get("p2p_to_load_kw", 0.0)),
                f"{mg}_p2p_to_battery_kw": float(p2p.get("p2p_to_battery_kw", 0.0)),
                f"{mg}_ess_p2p_offer_kw": float(p2p.get("ess_p2p_offer_kw", 0.0)),
                f"{mg}_ess_p2p_unmatched_kw": float(p2p.get("ess_p2p_unmatched_kw", 0.0)),
                f"{mg}_ess_to_local_load_kw": float(plan.get("ess_to_local_kw", 0.0)),
                f"{mg}_sharing_offer_ratio": float(plan.get("sharing_offer_ratio", 1.0)),
                f"{mg}_sharing_action_masked": int(bool(plan.get("sharing_action_masked", False))),
                f"{mg}_grid_import_cost": economics[mg]["grid_import_cost"],
                f"{mg}_grid_export_revenue": economics[mg]["grid_export_revenue"],
                f"{mg}_p2p_purchase_cost": economics[mg]["p2p_purchase_cost"],
                f"{mg}_p2p_sale_revenue": economics[mg]["p2p_sale_revenue"],
                f"{mg}_battery_degradation_cost": economics[mg]["battery_degradation_cost"],
                f"{mg}_hourly_profit": economics[mg]["physical_profit"],
                f"{mg}_soc_violation": constraints["soc_by_microgrid"][mg],
                f"{mg}_battery_power_violation": constraints["battery_by_microgrid"][mg],
            })
            raw = np.asarray(raw_actions[mg], dtype=np.float64).reshape(-1)
            for index, value in enumerate(raw):
                row[f"{mg}_raw_action_{index}"] = float(value)
            # Executed action fields are physical execution, after scaling/projection/clipping.
            row[f"{mg}_executed_action_0"] = float(device["discharge_kw"] - device["charge_kw"])
            if raw.size >= 2:
                row[f"{mg}_executed_action_1"] = float(plan.get("sharing_offer_ratio", 0.0))

        for seller in self.agent_ids:
            for buyer in self.agent_ids:
                if seller == buyer:
                    continue
                key = f"{seller}->{buyer}"
                energy = float(info["p2p"].get("pair_flows_kwh", {}).get(key, 0.0))
                row[f"p2p_{seller}_to_{buyer}_kw"] = energy / max(float(self.env.delta_t), EPSILON)
        row.update({
            "p2p_price": info["p2p"].get("transfer_price_per_kwh"),
            "p2p_total_energy_kwh": float(info["p2p"]["total_p2p_kwh"]),
            "power_balance_violation_kw": constraints["power_balance"],
            "p2p_balance_violation_kw": constraints["p2p_balance"],
        })
        return row

    def _violation_summary(self, values: Sequence[float], mode: str) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        active = array[array > self.tolerance]
        if mode == "mean_abs":
            return {
                "count": int(active.size),
                "mean_abs_kw": float(np.mean(np.abs(array))) if array.size else 0.0,
                "max_abs_kw": float(np.max(np.abs(array))) if array.size else 0.0,
            }
        return {
            "count": int(active.size),
            "sum": float(np.sum(array)) if array.size else 0.0,
            "max": float(np.max(array)) if array.size else 0.0,
        }

    def finalize_episode(self) -> dict[str, Any]:
        energy = {}
        economics = {}
        for mg in self.agent_ids:
            economics[mg] = {key: float(value) for key, value in self.economics[mg].items()}
            # Preserve the complete cross-algorithm schema even when the environment has no term.
            for key in (
                "grid_import_cost", "grid_export_revenue", "p2p_purchase_cost",
                "p2p_sale_revenue", "p2p_net_revenue",
                "battery_degradation_cost", "unmet_load_cost",
                "renewable_curtailment_cost",
                "terminal_soc_penalty", "physical_profit",
            ):
                economics[mg].setdefault(key, 0.0)
            energy[mg] = {key: float(value) for key, value in self.energy[mg].items()}
            for key in (
                "load_kwh", "renewable_available_kwh", "renewable_used_kwh",
                "renewable_curtailment_kwh", "grid_import_kwh", "grid_export_kwh",
                "p2p_buy_kwh", "p2p_sell_kwh", "renewable_p2p_sell_kwh",
                "ess_p2p_sell_kwh", "p2p_to_load_kwh", "p2p_to_battery_kwh",
                "renewable_p2p_offer_kwh", "ess_p2p_offer_kwh",
                "ess_p2p_unmatched_kwh", "ess_to_local_load_kwh", "battery_charge_kwh",
                "battery_discharge_kwh", "battery_throughput_kwh", "unmet_load_kwh",
            ):
                energy[mg].setdefault(key, 0.0)
            energy[mg]["renewable_utilization_rate"] = safe_ratio(
                energy[mg]["renewable_used_kwh"], energy[mg]["renewable_available_kwh"]
            )

        p2p_buy = {mg: energy[mg]["p2p_buy_kwh"] for mg in self.agent_ids}
        p2p_sell = {mg: energy[mg]["p2p_sell_kwh"] for mg in self.agent_ids}
        p2p_stats = {
            "total_energy_kwh": float(self.p2p_total),
            "pair_energy_kwh": dict(self.pair_energy),
            "buy_energy_by_microgrid_kwh": p2p_buy,
            "sell_energy_by_microgrid_kwh": p2p_sell,
            "p2p_price_mean": float(np.mean(self.p2p_prices)) if self.p2p_prices else None,
            "p2p_price_std": float(np.std(self.p2p_prices)) if self.p2p_prices else None,
            "p2p_payment_by_microgrid": dict(self.p2p_payment),
            "p2p_net_revenue_by_microgrid": dict(self.p2p_payment),
            "payment_sign_convention": "positive=sale revenue, negative=purchase payment",
            "renewable_sell_energy_by_microgrid_kwh": {
                mg: energy[mg]["renewable_p2p_sell_kwh"] for mg in self.agent_ids
            },
            "ess_sell_energy_by_microgrid_kwh": {
                mg: energy[mg]["ess_p2p_sell_kwh"] for mg in self.agent_ids
            },
            "energy_to_load_by_microgrid_kwh": {
                mg: energy[mg]["p2p_to_load_kwh"] for mg in self.agent_ids
            },
            "energy_to_battery_by_microgrid_kwh": {
                mg: energy[mg]["p2p_to_battery_kwh"] for mg in self.agent_ids
            },
            "ess_offer_energy_by_microgrid_kwh": {
                mg: energy[mg]["ess_p2p_offer_kwh"] for mg in self.agent_ids
            },
            "ess_unmatched_energy_by_microgrid_kwh": {
                mg: energy[mg]["ess_p2p_unmatched_kwh"] for mg in self.agent_ids
            },
            "ess_offer_acceptance_rate_by_microgrid": {
                mg: safe_ratio(
                    energy[mg]["ess_p2p_sell_kwh"], energy[mg]["ess_p2p_offer_kwh"]
                ) for mg in self.agent_ids
            },
        }
        soc_stats = {}
        for mg in self.agent_ids:
            battery = self.env.cores[mg].battery
            values = self.soc[mg]
            terminal = float(values[-1])
            target = float(battery.initial_soc)
            soc_stats[mg] = {
                "initial_soc": float(values[0]),
                "min_soc": float(np.min(values)),
                "max_soc": float(np.max(values)),
                "terminal_soc": terminal,
                "terminal_soc_target": target,
                "terminal_soc_deviation": float(terminal - target),
            }

        action_by_mg: dict[str, Any] = {}
        for mg in self.agent_ids:
            dimensions = {}
            for name, values in self.action_values[mg].items():
                array = np.asarray(values, dtype=np.float64)
                dimensions[name] = {
                    "mean": float(np.mean(array)), "std": float(np.std(array)),
                    "min": float(np.min(array)), "max": float(np.max(array)),
                }
            count = max(1, self.action_value_count[mg])
            action_by_mg[mg] = {
                "dimensions": dimensions,
                "action_clip_count": int(self.action_clip_count[mg]),
                "action_clip_rate": self.action_clip_count[mg] / count,
                "action_saturation_rate": self.action_saturation_count[mg] / count,
            }
        total_action_values = max(1, sum(self.action_value_count.values()))
        total_clips = sum(self.action_clip_count.values())
        total_saturations = sum(self.action_saturation_count.values())

        soc_v = self._violation_summary(self.violations["soc"], "sum")
        power_v = self._violation_summary(self.violations["power_balance"], "mean_abs")
        p2p_v = self._violation_summary(self.violations["p2p_balance"], "mean_abs")
        grid_v_raw = self._violation_summary(self.violations["grid_power"], "sum")
        battery_v_raw = self._violation_summary(self.violations["battery_power"], "sum")
        constraints = {
            "feasible_episode": all(
                max(values, default=0.0) <= self.tolerance for values in self.violations.values()
            ),
            "soc_violation": soc_v,
            "power_balance_violation": power_v,
            "p2p_balance_violation": p2p_v,
            "grid_power_violation": {"count": grid_v_raw["count"], "max": grid_v_raw["max"]},
            "battery_power_violation": {"count": battery_v_raw["count"], "max": battery_v_raw["max"]},
        }

        total_available = sum(x["renewable_available_kwh"] for x in energy.values())
        total_used = sum(x["renewable_used_kwh"] for x in energy.values())
        profit_error = self.team_profit - sum(self.profit.values())
        p2p_error = sum(p2p_sell.values()) - sum(p2p_buy.values())
        renewable_error = sum(
            x["renewable_used_kwh"] + x["renewable_curtailment_kwh"]
            - x["renewable_available_kwh"] for x in energy.values()
        )
        checks = {
            "tolerance": self.tolerance,
            "profit_balance_error": float(profit_error),
            "p2p_balance_error": float(p2p_error),
            "renewable_balance_error": float(renewable_error),
            "passed": all(abs(value) <= self.tolerance for value in (
                profit_error, p2p_error, renewable_error,
            )),
        }
        if not checks["passed"]:
            warnings.warn(f"Episode metric consistency check failed: {checks}", RuntimeWarning, stacklevel=2)

        return {
            "team_profit": float(self.team_profit),
            "team_reward": float(self.team_reward),
            "shaping_reward": float(self.shaping_reward),
            "profit_by_microgrid": {mg: float(self.profit[mg]) for mg in self.agent_ids},
            "reward_by_microgrid": {mg: float(self.rewards[mg]) for mg in self.agent_ids},
            "reward_by_microgrid_semantics": "shared_team_reward_repeated_per_agent",
            "system_level_costs": {},
            "system_level_profit": 0.0,
            "economic_breakdown_by_microgrid": economics,
            "energy_statistics_by_microgrid": energy,
            "renewable_utilization_rate": safe_ratio(total_used, total_available),
            "p2p_statistics": p2p_stats,
            "p2p_total_energy_kwh": float(self.p2p_total),
            "constraint_statistics": constraints,
            "soc_statistics_by_microgrid": soc_stats,
            "soc_statistics": {
                "terminal_soc": {mg: soc_stats[mg]["terminal_soc"] for mg in self.agent_ids},
                "min_soc": {mg: soc_stats[mg]["min_soc"] for mg in self.agent_ids},
                "max_soc": {mg: soc_stats[mg]["max_soc"] for mg in self.agent_ids},
            },
            "ess_throughput_kwh": {
                mg: energy[mg]["battery_throughput_kwh"] for mg in self.agent_ids
            },
            "action_statistics": {
                "by_microgrid": action_by_mg,
                "action_clip_count": int(total_clips),
                "action_clip_rate": total_clips / total_action_values,
                "action_saturation_rate": total_saturations / total_action_values,
            },
            "sanity_checks": checks,
        }


def aggregate_episode_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    profits = [float(item["team_profit"]) for item in episodes]
    ids = list(episodes[0]["profit_by_microgrid"]) if episodes else []
    return {
        "days_count": len(episodes),
        "team_profit_statistics": distribution_statistics(profits),
        "team_profit_mean": float(np.mean(profits)) if profits else 0.0,
        "team_profit_total": float(np.sum(profits)) if profits else 0.0,
        "profit_by_microgrid_mean": {
            mg: float(np.mean([day["profit_by_microgrid"][mg] for day in episodes]))
            for mg in ids
        },
        "renewable_utilization_rate": float(np.mean([
            day["renewable_utilization_rate"] for day in episodes
        ])) if episodes else 0.0,
        "p2p_total_energy_kwh": float(np.mean([
            day["p2p_statistics"]["total_energy_kwh"] for day in episodes
        ])) if episodes else 0.0,
        "feasible_rate": float(np.mean([
            bool(day["constraint_statistics"]["feasible_episode"]) for day in episodes
        ])) if episodes else 0.0,
    }
