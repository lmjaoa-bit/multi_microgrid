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

from core.tariffs import prices_from_row


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
        return {
            str(key): _native(item, f"{path}.{key}" if path else str(key), invalid)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_native(item, f"{path}[{index}]", invalid)
                for index, item in enumerate(value)]
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
    return _native(value, "", invalid), invalid


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
                {"field": field, "reason": "non_finite_replaced_with_null"}
                for field in invalid
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
            if str(key) not in columns:
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
            writer.writerow({key: "" if clean.get(key) is None else clean.get(key)
                             for key in columns})


class EpisodeMetricsCollector:
    """Shared train/validation/test episode metrics with DR-aware diagnostics."""

    def __init__(self, env: Any, capture_hourly: bool = False,
                 tolerance: float = DEFAULT_TOLERANCE) -> None:
        self.env = env
        self.agent_ids = list(env.agent_ids)
        self.capture_hourly = bool(capture_hourly)
        self.tolerance = float(tolerance)
        self.action_design = str(getattr(env, "action_design", "single_dispatch"))
        self.dr_enabled = self.action_design == "single_dispatch_dr"
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
        self.dr_totals = {mg: defaultdict(float) for mg in ids}
        self.dr_last: dict[str, dict[str, float]] = {mg: {} for mg in ids}
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
        self.p2p_unconstrained_total = 0.0
        self.p2p_blocked_by_capacity = 0.0
        self.physical_pairs = {
            f"{ids[i]}_{ids[j]}": (ids[i], ids[j])
            for i in range(len(ids)) for j in range(i + 1, len(ids))
        }
        self.pair_saturation_steps = {key: 0 for key in self.physical_pairs}
        self._last_p2p_capacity_diag: dict[str, Any] = {
            "unconstrained_total_energy_kwh": 0.0,
            "blocked_by_pair_capacity_kwh": 0.0,
            "pair_usage_kw": {key: 0.0 for key in self.physical_pairs},
            "pair_saturated": {key: False for key in self.physical_pairs},
        }
        self.p2p_payment = {mg: 0.0 for mg in ids}
        self.sharing_action_masked_steps = {mg: 0 for mg in ids}
        self.ess_offer_steps = {mg: 0 for mg in ids}
        self.ess_offer_any_fill_steps = {mg: 0 for mg in ids}
        self.ess_offer_full_fill_steps = {mg: 0 for mg in ids}
        self.violations: dict[str, list[float]] = {
            "soc": [], "power_balance": [], "p2p_balance": [],
            "grid_power": [], "battery_power": [],
        }
        self.hourly_rows: list[dict[str, Any]] = []
        self.ev_sessions: dict[str, list[dict[str, Any]]] = {}
        self.ev_departure_events: dict[str, list[dict[str, Any]]] = {}

    def _economic_step(self, mg: str, info: Mapping[str, Any]) -> dict[str, float]:
        cost = info["costs"][mg]
        payment = float(info["p2p"]["by_microgrid"][mg].get("p2p_transfer_payment", 0.0))
        result = {
            "grid_import_cost": float(cost.get("grid_import_cost", 0.0)),
            "grid_export_revenue": float(cost.get("grid_export_revenue", 0.0)),
            "p2p_purchase_cost": max(0.0, -payment),
            "p2p_sale_revenue": max(0.0, payment),
            "p2p_net_revenue": payment,
            "battery_degradation_cost": float(cost.get("battery_degradation_cost", 0.0)),
            "unmet_load_cost": float(cost.get("unmet_load_cost", 0.0)),
            "renewable_curtailment_cost": float(cost.get("curtailment_cost", 0.0)),
            "terminal_soc_penalty": float(cost.get("terminal_soc_penalty", 0.0)),
            "ev_departure_penalty": float(cost.get("ev_departure_penalty", 0.0)),
            # Individual accounting includes the internal P2P transfer. Transfers
            # sum to zero, so the sum still equals team physical profit.
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
        actual_load = float(plan.get("actual_load_kw", plan.get("load_kw", 0.0)))
        base_load = float(plan.get("base_load_kw", actual_load))
        flexible_load = float(plan.get("flexible_load_kw", max(0.0, actual_load - base_load)))
        result = {
            "load_kwh": actual_load * dt,
            "base_load_kwh": base_load * dt,
            "flexible_load_kwh": flexible_load * dt,
            "actual_load_kwh": actual_load * dt,
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
            "ess_p2p_offer_uncapped_kwh": float(
                plan.get("ess_p2p_offer_uncapped_kw", p2p.get("ess_p2p_offer_kw", 0.0))
            ) * dt,
            "ess_p2p_offer_projection_reduction_kwh": float(
                plan.get("ess_p2p_offer_projection_reduction_kw", 0.0)
            ) * dt,
            "ess_to_local_load_kwh": float(plan.get("ess_to_local_kw", 0.0)) * dt,
            "battery_charge_kwh": float(device.get("charge_kw", 0.0)) * dt,
            "battery_discharge_kwh": float(device.get("discharge_kw", 0.0)) * dt,
            "battery_throughput_kwh": float(device.get("throughput_kwh", 0.0)),
            "battery_charge_curtailed_kwh": float(plan.get("battery_charge_curtailed_kw", 0.0)) * dt,
            "unmet_load_kwh": float(settlement.get("unmet_load_kw", 0.0)) * dt,
        }
        for key, value in result.items():
            self.energy[mg][key] += value
        return result

    def _record_dr_step(self, mg: str, info: Mapping[str, Any]) -> None:
        if not self.dr_enabled:
            return
        plan = info["plans"][mg]
        dr_plan = info.get("dr_plans", {}).get(mg, {})
        state_after = info.get("dr_state_after", {}).get(mg, {})
        if mg in info.get("ev_sessions", {}):
            self.ev_sessions[mg] = [dict(item) for item in info["ev_sessions"][mg]]
        if mg in info.get("ev_departure_events", {}):
            self.ev_departure_events[mg] = [dict(item) for item in info["ev_departure_events"][mg]]
        dt = float(self.env.delta_t)
        totals = self.dr_totals[mg]
        totals["flexible_load_energy_kwh"] += float(plan.get("flexible_load_kw", 0.0)) * dt
        totals["projection_steps"] += float(bool(plan.get("dr_projected", False)))
        totals["ess_projection_steps"] += float(bool(plan.get("ess_projected", False)))
        totals["supply_infeasible_steps"] += float(bool(plan.get("dr_supply_infeasible", False)))
        kind = str(self.env.cfg_by_id[mg]["dr"]["type"])
        if kind == "industrial_mill":
            totals["mill_energy_kwh"] += float(dr_plan.get("flexible_load_kw", 0.0)) * dt
            totals["mill_effective_duty_sum"] += float(dr_plan.get("effective_duty", 0.0))
            totals["mill_production_t"] += float(dr_plan.get("production_tph", 0.0)) * dt
            totals["terminal_infeasible_steps"] += float(bool(dr_plan.get("terminal_infeasible", False)))
        elif kind == "commercial_hvac":
            totals["hvac_energy_kwh"] += float(dr_plan.get("flexible_load_kw", 0.0)) * dt
            totals["heating_energy_kwh"] += float(dr_plan.get("heating_power_kw", 0.0)) * dt
            totals["cooling_energy_kwh"] += float(dr_plan.get("cooling_power_kw", 0.0)) * dt
            totals["comfort_infeasible_steps"] += float(bool(dr_plan.get("comfort_infeasible", False)))
            totals["terminal_infeasible_steps"] += float(bool(dr_plan.get("terminal_thermal_infeasible", False)))
        elif kind == "ev_charging":
            totals["ev_charge_energy_kwh"] += float(dr_plan.get("effective_power_kw", 0.0)) * dt
            totals["target_missed_steps"] += float(bool(dr_plan.get("target_missed", False)))
        self.dr_last[mg] = {key: float(value) for key, value in state_after.items()
                            if isinstance(value, (int, float, np.number, bool))}

    def _p2p_capacity_diagnostics(self, info: Mapping[str, Any]) -> dict[str, Any]:
        """Re-clear the same DR offers without pair limits for diagnostics only."""
        p2p = info["p2p"]
        actual_total = float(p2p.get("total_p2p_kwh", 0.0))
        raw_capacity = getattr(self.env.energy_pool, "pair_capacity_kw", None)
        pair_capacity = float(raw_capacity) if raw_capacity is not None else math.inf
        pair_usage = {key: 0.0 for key in self.physical_pairs}
        flows = p2p.get("pair_flows_kwh", {})
        dt = max(float(self.env.delta_t), EPSILON)
        for key, (first, second) in self.physical_pairs.items():
            energy = float(flows.get(f"{first}->{second}", 0.0)) + float(
                flows.get(f"{second}->{first}", 0.0)
            )
            pair_usage[key] = energy / dt
        pair_saturated = {
            key: bool(
                math.isfinite(pair_capacity)
                and pair_usage[key] >= pair_capacity - self.tolerance
            )
            for key in self.physical_pairs
        }

        if not math.isfinite(pair_capacity) or not self.dr_enabled or not hasattr(
            self.env.energy_pool, "allocate_structured_unconstrained"
        ):
            return {
                "unconstrained_total_energy_kwh": actual_total,
                "blocked_by_pair_capacity_kwh": 0.0,
                "pair_usage_kw": pair_usage,
                "pair_saturated": pair_saturated,
            }

        plans = info["plans"]
        required = (
            "renewable_p2p_offer_kw", "ess_p2p_offer_kw",
            "p2p_load_demand_kw", "p2p_charge_demand_kw",
        )
        if any(any(field not in plans[mg] for field in required) for mg in self.agent_ids):
            return {
                "unconstrained_total_energy_kwh": actual_total,
                "blocked_by_pair_capacity_kwh": 0.0,
                "pair_usage_kw": pair_usage,
                "pair_saturated": pair_saturated,
            }

        if self.env.episode_df is None:
            raise RuntimeError("P2P capacity diagnostics require an active episode")
        row_index = min(max(self.steps - 1, 0), len(self.env.episode_df) - 1)
        row = self.env.episode_df.iloc[row_index]
        prices = {mg: prices_from_row(row, mg) for mg in self.agent_ids}
        unconstrained = self.env.energy_pool.allocate_structured_unconstrained(
            {mg: float(plans[mg]["renewable_p2p_offer_kw"]) for mg in self.agent_ids},
            {mg: float(plans[mg]["ess_p2p_offer_kw"]) for mg in self.agent_ids},
            {mg: float(plans[mg]["p2p_load_demand_kw"]) for mg in self.agent_ids},
            {mg: float(plans[mg]["p2p_charge_demand_kw"]) for mg in self.agent_ids},
            {mg: prices[mg][1] for mg in self.agent_ids},
            {mg: prices[mg][0] for mg in self.agent_ids},
        )
        unconstrained_total = float(unconstrained.total_p2p_kwh)
        return {
            "unconstrained_total_energy_kwh": unconstrained_total,
            "blocked_by_pair_capacity_kwh": max(0.0, unconstrained_total - actual_total),
            "pair_usage_kw": pair_usage,
            "pair_saturated": pair_saturated,
        }

    def _constraint_step(self, info: Mapping[str, Any]) -> dict[str, Any]:
        soc_by_mg: dict[str, float] = {}
        battery_by_mg: dict[str, float] = {}
        grid_by_mg: dict[str, float] = {}
        for mg in self.agent_ids:
            battery = self.env.cores[mg].battery
            soc = float(info["devices"][mg]["soc"])
            soc_violation = max(0.0, battery.soc_min - soc, soc - battery.soc_max)
            charge = float(info["devices"][mg]["charge_kw"])
            discharge = float(info["devices"][mg]["discharge_kw"])
            battery_violation = max(
                0.0,
                charge - battery.max_charge_kw,
                discharge - battery.max_discharge_kw,
                min(charge, discharge),
            )
            settlement = info["settlements"][mg]
            grid = self.env.grids[mg]
            grid_violation = max(
                0.0,
                float(settlement["grid_import_kw"]) - grid.import_limit_kw if grid.import_limit_kw is not None else 0.0,
                float(settlement["grid_export_kw"]) - grid.export_limit_kw if grid.export_limit_kw is not None else 0.0,
            )
            soc_by_mg[mg] = soc_violation
            battery_by_mg[mg] = battery_violation
            grid_by_mg[mg] = grid_violation
            self.violations["soc"].append(soc_violation)
            self.violations["battery_power"].append(battery_violation)
            self.violations["grid_power"].append(grid_violation)

        power_balance = sum(
            abs(float(info["settlements"][mg]["power_balance_residual_kw"]))
            for mg in self.agent_ids
        )
        p2p_sell = sum(float(info["p2p"]["by_microgrid"][mg]["p2p_sell_kw"])
                       for mg in self.agent_ids)
        p2p_buy = sum(float(info["p2p"]["by_microgrid"][mg]["p2p_buy_kw"])
                      for mg in self.agent_ids)
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

        for mg in self.agent_ids:
            economics[mg] = self._economic_step(mg, info)
            self._energy_step(mg, info)
            self._record_dr_step(mg, info)
            self.profit[mg] += economics[mg]["physical_profit"]
            self.rewards[mg] += float((rewards or {}).get(mg, info.get("team_reward", 0.0)))
            self.soc[mg].append(float(info["devices"][mg]["soc"]))

            action = np.asarray(raw_actions[mg], dtype=np.float64).reshape(-1)
            action_names = ["battery"]
            if action.size >= 2:
                action_names.append(
                    "dr_raw" if self.dr_enabled else "sharing_offer_ratio_raw"
                )
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
        capacity_diag = self._p2p_capacity_diagnostics(info)
        self._last_p2p_capacity_diag = capacity_diag
        self.p2p_unconstrained_total += float(
            capacity_diag["unconstrained_total_energy_kwh"]
        )
        self.p2p_blocked_by_capacity += float(
            capacity_diag["blocked_by_pair_capacity_kwh"]
        )
        for key, saturated in capacity_diag["pair_saturated"].items():
            self.pair_saturation_steps[key] += int(bool(saturated))
        for key, energy in p2p.get("pair_flows_kwh", {}).items():
            seller, buyer = str(key).split("->", 1)
            self.pair_energy[f"{seller}_to_{buyer}"] += float(energy)
        price = p2p.get("transfer_price_per_kwh")
        if price is not None and math.isfinite(float(price)):
            self.p2p_prices.append(float(price))
        for mg in self.agent_ids:
            peer = p2p["by_microgrid"][mg]
            payment = float(peer.get("p2p_transfer_payment", 0.0))
            self.p2p_payment[mg] += payment
            plan = info["plans"][mg]
            offer = float(peer.get("ess_p2p_offer_kw", 0.0))
            accepted = float(peer.get("ess_p2p_sell_kw", 0.0))
            self.sharing_action_masked_steps[mg] += int(
                bool(plan.get("sharing_action_masked", False))
            )
            if offer > self.tolerance:
                self.ess_offer_steps[mg] += 1
                self.ess_offer_any_fill_steps[mg] += int(accepted > self.tolerance)
                self.ess_offer_full_fill_steps[mg] += int(
                    accepted >= offer - self.tolerance
                )

        constraints = self._constraint_step(info)
        if self.capture_hourly:
            self.hourly_rows.append(
                self._hourly_row(raw_actions, info, economics, constraints)
            )

    def _hourly_row(self, raw_actions: Mapping[str, Sequence[float]], info: Mapping[str, Any],
                    economics: Mapping[str, Mapping[str, float]],
                    constraints: Mapping[str, Any]) -> dict[str, Any]:
        timestamp = str(info["timestamp"])
        try:
            hour = int(timestamp[11:13])
        except (TypeError, ValueError):
            hour = self.steps - 1
        row: dict[str, Any] = {
            "date": timestamp[:10], "hour": hour, "timestep": self.steps - 1,
            "p2p_unconstrained_total_energy_kwh": float(
                self._last_p2p_capacity_diag["unconstrained_total_energy_kwh"]
            ),
            "p2p_blocked_by_pair_capacity_kwh": float(
                self._last_p2p_capacity_diag["blocked_by_pair_capacity_kwh"]
            ),
        }
        for pair, usage in self._last_p2p_capacity_diag["pair_usage_kw"].items():
            row[f"p2p_{pair}_usage_kw"] = float(usage)
            row[f"p2p_{pair}_saturated"] = int(bool(
                self._last_p2p_capacity_diag["pair_saturated"][pair]
            ))
        for mg in self.agent_ids:
            plan = info["plans"][mg]
            settlement = info["settlements"][mg]
            device = info["devices"][mg]
            peer = info["p2p"]["by_microgrid"][mg]
            available = float(plan.get("renewable_kw", 0.0))
            curtailment = float(settlement.get("renewable_curtailment_kw", 0.0))
            actual_load = float(plan.get("actual_load_kw", plan.get("load_kw", 0.0)))
            row.update({
                f"{mg}_load_kw": actual_load,
                f"{mg}_base_load_kw": float(plan.get("base_load_kw", actual_load)),
                f"{mg}_flexible_load_kw": float(plan.get("flexible_load_kw", 0.0)),
                f"{mg}_actual_load_kw": actual_load,
                f"{mg}_renewable_available_kw": available,
                f"{mg}_renewable_used_kw": max(0.0, available - curtailment),
                f"{mg}_renewable_curtailment_kw": curtailment,
                f"{mg}_grid_import_kw": float(settlement["grid_import_kw"]),
                f"{mg}_grid_export_kw": float(settlement["grid_export_kw"]),
                f"{mg}_unmet_load_kw": float(settlement.get("unmet_load_kw", 0.0)),
                f"{mg}_power_balance_residual_kw": float(settlement["power_balance_residual_kw"]),
                f"{mg}_battery_charge_kw": float(device["charge_kw"]),
                f"{mg}_battery_discharge_kw": float(device["discharge_kw"]),
                f"{mg}_battery_charge_curtailed_kw": float(plan.get("battery_charge_curtailed_kw", 0.0)),
                f"{mg}_soc": float(device["soc"]),
                f"{mg}_p2p_buy_kw": float(peer["p2p_buy_kw"]),
                f"{mg}_p2p_sell_kw": float(peer["p2p_sell_kw"]),
                f"{mg}_renewable_p2p_sell_kw": float(peer.get("renewable_p2p_sell_kw", 0.0)),
                f"{mg}_ess_p2p_sell_kw": float(peer.get("ess_p2p_sell_kw", 0.0)),
                f"{mg}_p2p_to_load_kw": float(peer.get("p2p_to_load_kw", 0.0)),
                f"{mg}_p2p_to_battery_kw": float(peer.get("p2p_to_battery_kw", 0.0)),
                f"{mg}_ess_p2p_offer_kw": float(peer.get("ess_p2p_offer_kw", 0.0)),
                f"{mg}_ess_p2p_unmatched_kw": float(peer.get("ess_p2p_unmatched_kw", 0.0)),
                f"{mg}_ess_p2p_offer_uncapped_kw": float(
                    plan.get("ess_p2p_offer_uncapped_kw", peer.get("ess_p2p_offer_kw", 0.0))
                ),
                f"{mg}_ess_p2p_offer_cap_kw": float(plan.get("ess_p2p_offer_cap_kw", 0.0)),
                f"{mg}_ess_p2p_offer_projection_reduction_kw": float(
                    plan.get("ess_p2p_offer_projection_reduction_kw", 0.0)
                ),
                f"{mg}_ess_to_local_load_kw": float(plan.get("ess_to_local_kw", 0.0)),
                f"{mg}_requested_sharing_offer_ratio": float(
                    plan.get("requested_sharing_offer_ratio", 0.0)
                ),
                f"{mg}_sharing_offer_ratio": float(plan.get("sharing_offer_ratio", 0.0)),
                f"{mg}_sharing_action_masked": int(bool(plan.get("sharing_action_masked", False))),
                f"{mg}_ess_offer_projection_active": int(bool(plan.get("ess_offer_projection_active", False))),
                f"{mg}_grid_import_cost": economics[mg]["grid_import_cost"],
                f"{mg}_grid_export_revenue": economics[mg]["grid_export_revenue"],
                f"{mg}_p2p_purchase_cost": economics[mg]["p2p_purchase_cost"],
                f"{mg}_p2p_sale_revenue": economics[mg]["p2p_sale_revenue"],
                f"{mg}_battery_degradation_cost": economics[mg]["battery_degradation_cost"],
                f"{mg}_terminal_constraint_penalty": economics[mg]["terminal_soc_penalty"],
                f"{mg}_hourly_profit": economics[mg]["physical_profit"],
                f"{mg}_soc_violation": constraints["soc_by_microgrid"][mg],
                f"{mg}_battery_power_violation": constraints["battery_by_microgrid"][mg],
            })

            raw = np.asarray(raw_actions[mg], dtype=np.float64).reshape(-1)
            for index, value in enumerate(raw):
                row[f"{mg}_raw_action_{index}"] = float(value)
            row[f"{mg}_executed_action_0"] = float(device["discharge_kw"] - device["charge_kw"])

            for field in (
                "direct_local_discharge_kw", "direct_local_priority_reserve_kw",
                "direct_remaining_discharge_kw", "direct_trade_capacity_kw",
                "effective_local_action", "effective_trade_action",
                "battery_discharge_requested_kw", "dispatch_requested_power_kw",
            ):
                if field in plan:
                    row[f"{mg}_{field}"] = float(plan[field])

            if self.dr_enabled:
                row.update({
                    f"{mg}_raw_ess_action": float(plan.get("raw_ess_action", raw[0])),
                    f"{mg}_raw_dr_action": float(plan.get("raw_dr_action", raw[1])),
                    f"{mg}_effective_ess_dispatch": float(plan.get("effective_ess_dispatch", 0.0)),
                    f"{mg}_effective_dr_action": float(plan.get("effective_dr_action", 0.0)),
                    f"{mg}_ess_projection_delta": float(plan.get("ess_projection_delta", 0.0)),
                    f"{mg}_dr_projection_delta": float(plan.get("dr_projection_delta", 0.0)),
                    f"{mg}_ess_projected": int(bool(plan.get("ess_projected", False))),
                    f"{mg}_dr_projected": int(bool(plan.get("dr_projected", False))),
                    f"{mg}_dr_supply_infeasible": int(bool(plan.get("dr_supply_infeasible", False))),
                })
                row[f"{mg}_executed_action_1"] = float(plan.get("effective_dr_action", 0.0))
                dr_plan = info.get("dr_plans", {}).get(mg, {})
                dr_state = info.get("dr_state_after", {}).get(mg, {})
                kind = str(self.env.cfg_by_id[mg]["dr"]["type"])
                if kind == "industrial_mill":
                    row.update({
                        f"{mg}_mill_requested_duty": float(dr_plan.get("requested_duty", 0.0)),
                        f"{mg}_mill_effective_duty": float(dr_plan.get("effective_duty", 0.0)),
                        f"{mg}_mill_power_kw": float(dr_plan.get("flexible_load_kw", 0.0)),
                        f"{mg}_mill_production_tph": float(dr_plan.get("production_tph", 0.0)),
                        f"{mg}_silo_level_t": float(dr_state.get("silo_level_t", dr_plan.get("silo_after_t", 0.0))),
                        f"{mg}_silo_ratio": float(dr_state.get("silo_ratio", 0.0)),
                        f"{mg}_mill_feasible_min": float(dr_plan.get("feasible_min", 0.0)),
                        f"{mg}_mill_feasible_max": float(dr_plan.get("feasible_max", 0.0)),
                    })
                elif kind == "commercial_hvac":
                    row.update({
                        f"{mg}_hvac_effective_signed_power_kw": float(dr_plan.get("effective_signed_power_kw", 0.0)),
                        f"{mg}_hvac_heating_power_kw": float(dr_plan.get("heating_power_kw", 0.0)),
                        f"{mg}_hvac_cooling_power_kw": float(dr_plan.get("cooling_power_kw", 0.0)),
                        f"{mg}_t_air_c": float(dr_state.get("t_air_c", dr_plan.get("t_air_after_c", 0.0))),
                        f"{mg}_t_mass_c": float(dr_state.get("t_mass_c", dr_plan.get("t_mass_after_c", 0.0))),
                        f"{mg}_t_out_c": float(dr_state.get("t_out_c", 0.0)),
                        f"{mg}_occupied": float(dr_state.get("occupied", 0.0)),
                        f"{mg}_terminal_air_gap_c": float(dr_state.get("terminal_air_gap_c", 0.0)),
                        f"{mg}_terminal_mass_gap_c": float(dr_state.get("terminal_mass_gap_c", 0.0)),
                        f"{mg}_comfort_infeasible": int(bool(dr_plan.get("comfort_infeasible", False))),
                        f"{mg}_terminal_thermal_infeasible": int(bool(dr_plan.get("terminal_thermal_infeasible", False))),
                    })
                elif kind == "ev_charging":
                    row.update({
                        f"{mg}_ev_effective_power_kw": float(dr_plan.get("effective_power_kw", 0.0)),
                        f"{mg}_ev_soc": float(dr_state.get("ev_soc", dr_plan.get("soc_after", 0.0))),
                        f"{mg}_ev_soc_gap": float(dr_state.get("ev_soc_gap", 0.0)),
                        f"{mg}_ev_hours_to_departure": float(dr_state.get("ev_hours_to_departure", 0.0)),
                        f"{mg}_deadline_pressure": float(dr_state.get("deadline_pressure", 0.0)),
                        f"{mg}_ev_target_missed": int(bool(dr_plan.get("target_missed", False))),
                    })
            elif raw.size >= 2:
                row[f"{mg}_executed_action_1"] = float(plan.get("sharing_offer_ratio", 0.0))

        for seller in self.agent_ids:
            for buyer in self.agent_ids:
                if seller == buyer:
                    continue
                key = f"{seller}->{buyer}"
                energy = float(info["p2p"].get("pair_flows_kwh", {}).get(key, 0.0))
                row[f"p2p_{seller}_to_{buyer}_kw"] = (
                    energy / max(float(self.env.delta_t), EPSILON)
                )
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

    def _dr_summary(self, mg: str) -> dict[str, Any]:
        if not self.dr_enabled:
            return {}
        totals = self.dr_totals[mg]
        kind = str(self.env.cfg_by_id[mg]["dr"]["type"])
        summary: dict[str, Any] = {
            "type": kind,
            "flexible_load_energy_kwh": float(totals["flexible_load_energy_kwh"]),
            "projection_rate": safe_ratio(totals["projection_steps"], self.steps),
            "ess_projection_rate": safe_ratio(totals["ess_projection_steps"], self.steps),
            "supply_infeasible_rate": safe_ratio(totals["supply_infeasible_steps"], self.steps),
        }
        if kind == "industrial_mill":
            cfg = self.env.cfg_by_id[mg]["dr"]
            summary.update({
                "mill_energy_kwh": float(totals["mill_energy_kwh"]),
                "mean_effective_duty": safe_ratio(totals["mill_effective_duty_sum"], self.steps),
                "production_t": float(totals["mill_production_t"]),
                "terminal_silo_level_t": self.dr_last[mg].get("silo_level_t"),
                "terminal_silo_gap_t": (
                    self.dr_last[mg].get("silo_level_t", float(cfg["silo_initial_t"]))
                    - float(cfg["silo_initial_t"])
                ),
                "terminal_infeasible_rate": safe_ratio(totals["terminal_infeasible_steps"], self.steps),
            })
        elif kind == "commercial_hvac":
            summary.update({
                "hvac_energy_kwh": float(totals["hvac_energy_kwh"]),
                "heating_energy_kwh": float(totals["heating_energy_kwh"]),
                "cooling_energy_kwh": float(totals["cooling_energy_kwh"]),
                "comfort_infeasible_rate": safe_ratio(totals["comfort_infeasible_steps"], self.steps),
                "terminal_thermal_infeasible_rate": safe_ratio(totals["terminal_infeasible_steps"], self.steps),
                "terminal_t_air_c": self.dr_last[mg].get("t_air_c"),
                "terminal_t_mass_c": self.dr_last[mg].get("t_mass_c"),
            })
        elif kind == "ev_charging":
            cfg = self.env.cfg_by_id[mg]["dr"]
            departure_soc = self.dr_last[mg].get("ev_soc")
            summary.update({
                "ev_charge_energy_kwh": float(totals["ev_charge_energy_kwh"]),
                "departure_soc": departure_soc,
                "target_soc": float(cfg.get("target_soc", 0.8)),
                "target_met": bool(
                    departure_soc is not None and departure_soc >= float(cfg.get("target_soc", 0.8)) - 1e-7
                ),
                "target_missed_rate": safe_ratio(totals["target_missed_steps"], self.steps),
            })
        if kind == "ev_charging" and mg in self.ev_sessions:
            sessions = self.ev_sessions[mg]
            events = self.ev_departure_events.get(mg, [])
            demand = sum(float(item["demand_kwh"]) for item in sessions)
            shortfall = sum(float(item.get("shortfall_kwh", item.get("ev_departure_shortfall_kwh", 0.0))) for item in events)
            completed = sum(float(item.get("shortfall_kwh", item.get("ev_departure_shortfall_kwh", 0.0))) <= self.tolerance for item in events)
            summary.update({
                "sessions": sessions, "departure_events": events,
                "demand_kwh": demand, "departure_shortfall_kwh": shortfall,
                "departure_count": len(events), "completed_group_count": completed,
                "group_completion_rate": safe_ratio(completed, len(sessions)),
                "energy_satisfaction_rate": 1.0 - safe_ratio(shortfall, demand),
                "target_met": completed == len(sessions),
            })
            summary.pop("departure_soc", None)
        return summary

    def finalize_episode(self) -> dict[str, Any]:
        energy: dict[str, dict[str, float]] = {}
        economics: dict[str, dict[str, float]] = {}
        for mg in self.agent_ids:
            economics[mg] = {key: float(value) for key, value in self.economics[mg].items()}
            for key in (
                "grid_import_cost", "grid_export_revenue", "p2p_purchase_cost",
                "p2p_sale_revenue", "p2p_net_revenue", "battery_degradation_cost",
                "unmet_load_cost", "renewable_curtailment_cost",
                "terminal_soc_penalty", "physical_profit",
            ):
                economics[mg].setdefault(key, 0.0)
            energy[mg] = {key: float(value) for key, value in self.energy[mg].items()}
            for key in (
                "load_kwh", "base_load_kwh", "flexible_load_kwh", "actual_load_kwh",
                "renewable_available_kwh", "renewable_used_kwh",
                "renewable_curtailment_kwh", "grid_import_kwh", "grid_export_kwh",
                "p2p_buy_kwh", "p2p_sell_kwh", "renewable_p2p_sell_kwh",
                "ess_p2p_sell_kwh", "p2p_to_load_kwh", "p2p_to_battery_kwh",
                "renewable_p2p_offer_kwh", "ess_p2p_offer_kwh",
                "ess_p2p_unmatched_kwh", "ess_p2p_offer_uncapped_kwh",
                "ess_p2p_offer_projection_reduction_kwh", "ess_to_local_load_kwh",
                "battery_charge_kwh", "battery_discharge_kwh", "battery_throughput_kwh",
                "battery_charge_curtailed_kwh", "unmet_load_kwh",
            ):
                energy[mg].setdefault(key, 0.0)
            energy[mg]["renewable_utilization_rate"] = safe_ratio(
                energy[mg]["renewable_used_kwh"], energy[mg]["renewable_available_kwh"]
            )

            energy[mg]["curtailment_kwh"] = energy[mg]["renewable_curtailment_kwh"]
            energy[mg]["curtailment_rate"] = safe_ratio(energy[mg]["curtailment_kwh"], energy[mg]["renewable_available_kwh"])
            energy[mg]["renewable_non_curtailment_rate"] = energy[mg]["renewable_utilization_rate"]

        p2p_buy = {mg: energy[mg]["p2p_buy_kwh"] for mg in self.agent_ids}
        p2p_sell = {mg: energy[mg]["p2p_sell_kwh"] for mg in self.agent_ids}
        pair_capacity = getattr(self.env.energy_pool, "pair_capacity_kw", None)
        if pair_capacity is not None and not math.isfinite(float(pair_capacity)):
            pair_capacity = None
        p2p_stats = {
            "total_energy_kwh": float(self.p2p_total),
            "pair_capacity_kw": pair_capacity,
            "unconstrained_total_energy_kwh": float(self.p2p_unconstrained_total),
            "blocked_by_pair_capacity_kwh": float(self.p2p_blocked_by_capacity),
            "blocked_energy_rate": safe_ratio(
                self.p2p_blocked_by_capacity, self.p2p_unconstrained_total
            ),
            "pair_saturation_rate_by_pair": {
                pair: safe_ratio(count, self.steps)
                for pair, count in self.pair_saturation_steps.items()
            },
            "pair_saturation_step_count_by_pair": dict(self.pair_saturation_steps),
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
                mg: safe_ratio(energy[mg]["ess_p2p_sell_kwh"], energy[mg]["ess_p2p_offer_kwh"])
                for mg in self.agent_ids
            },
            "ess_offer_any_fill_rate_by_microgrid": {
                mg: safe_ratio(self.ess_offer_any_fill_steps[mg], self.ess_offer_steps[mg])
                for mg in self.agent_ids
            },
            "ess_offer_full_fill_rate_by_microgrid": {
                mg: safe_ratio(self.ess_offer_full_fill_steps[mg], self.ess_offer_steps[mg])
                for mg in self.agent_ids
            },
            "ess_offer_step_count_by_microgrid": dict(self.ess_offer_steps),
            "ess_offer_any_fill_step_count_by_microgrid": dict(self.ess_offer_any_fill_steps),
            "ess_offer_full_fill_step_count_by_microgrid": dict(self.ess_offer_full_fill_steps),
            "sharing_action_mask_rate_by_microgrid": {
                mg: safe_ratio(self.sharing_action_masked_steps[mg], self.steps)
                for mg in self.agent_ids
            },
            "ess_uncapped_offer_energy_by_microgrid_kwh": {
                mg: energy[mg]["ess_p2p_offer_uncapped_kwh"] for mg in self.agent_ids
            },
            "ess_offer_projection_reduction_by_microgrid_kwh": {
                mg: energy[mg]["ess_p2p_offer_projection_reduction_kwh"]
                for mg in self.agent_ids
            },
            "ess_offer_projection_retention_rate_by_microgrid": {
                mg: safe_ratio(
                    energy[mg]["ess_p2p_offer_kwh"], energy[mg]["ess_p2p_offer_uncapped_kwh"]
                ) for mg in self.agent_ids
            },
        }
        p2p_stats["ess_offer_acceptance_rate"] = safe_ratio(
            sum(energy[mg]["ess_p2p_sell_kwh"] for mg in self.agent_ids),
            sum(energy[mg]["ess_p2p_offer_kwh"] for mg in self.agent_ids),
        )

        soc_stats = {}
        for mg in self.agent_ids:
            battery = self.env.cores[mg].battery
            values = self.soc[mg]
            terminal = float(values[-1])
            target = float(battery.initial_soc)
            soc_stats[mg] = {
                "initial_soc": float(values[0]),
                "initial_energy_kwh": float(values[0]) * battery.capacity_kwh,
                "final_energy_kwh": terminal * battery.capacity_kwh,
                "energy_delta_kwh": (terminal - float(values[0])) * battery.capacity_kwh,
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
        grid_v = self._violation_summary(self.violations["grid_power"], "sum")
        battery_v = self._violation_summary(self.violations["battery_power"], "sum")
        constraints = {
            "feasible_episode": all(
                max(values, default=0.0) <= self.tolerance
                for values in self.violations.values()
            ),
            "soc_violation": soc_v,
            "power_balance_violation": power_v,
            "p2p_balance_violation": p2p_v,
            "grid_power_violation": {"count": grid_v["count"], "max": grid_v["max"]},
            "battery_power_violation": {"count": battery_v["count"], "max": battery_v["max"]},
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
            warnings.warn(
                f"Episode metric consistency check failed: {checks}",
                RuntimeWarning,
                stacklevel=2,
            )

        result = {
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
            "renewable_non_curtailment_rate": safe_ratio(total_used, total_available),
            "curtailment_kwh": sum(x["renewable_curtailment_kwh"] for x in energy.values()),
            "curtailment_rate": safe_ratio(sum(x["renewable_curtailment_kwh"] for x in energy.values()), total_available),
            "renewable_accounting_definition": "available minus curtailed; includes exports and storage charging, not final internal renewable consumption",
            "ev_sessions": self.ev_sessions,
            "ev_departure_events": self.ev_departure_events,
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
        if self.dr_enabled:
            result["dr_statistics_by_microgrid"] = {
                mg: self._dr_summary(mg) for mg in self.agent_ids
            }
        return result


def aggregate_episode_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    profits = [float(item["team_profit"]) for item in episodes]
    ids = list(episodes[0]["profit_by_microgrid"]) if episodes else []
    result = {
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
    if episodes and "dr_statistics_by_microgrid" in episodes[0]:
        result["dr_statistics_by_microgrid"] = {
            mg: {
                "flexible_load_energy_kwh_mean": float(np.mean([
                    day["dr_statistics_by_microgrid"][mg]["flexible_load_energy_kwh"]
                    for day in episodes
                ])),
                "projection_rate_mean": float(np.mean([
                    day["dr_statistics_by_microgrid"][mg]["projection_rate"]
                    for day in episodes
                ])),
            }
            for mg in ids
        }
    return result
