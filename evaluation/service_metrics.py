from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

import numpy as np


SERVICE_METRICS_SCHEMA_VERSION = "service_metrics_v2"
DEFAULT_TOLERANCE = 1e-7


class ServiceMetricDataError(ValueError):
    """Raised when a current-schema trajectory is incomplete or non-finite."""


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _float_field(
    row: Mapping[str, Any],
    field: str,
    *,
    strict: bool,
    missing: set[str],
    default: float = 0.0,
) -> float:
    if field not in row or not _finite(row.get(field)):
        missing.add(field)
        if strict:
            raise ServiceMetricDataError(f"Missing/non-finite service metric field: {field}")
        return float(default)
    return float(row[field])


def _bool_field(
    row: Mapping[str, Any],
    field: str,
    *,
    strict: bool,
    missing: set[str],
    default: bool = False,
) -> bool:
    if field not in row:
        missing.add(field)
        if strict:
            raise ServiceMetricDataError(f"Missing service metric field: {field}")
        return bool(default)
    value = row[field]
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    if isinstance(value, (bool, int, float, np.number)) and _finite(value):
        return bool(value)
    missing.add(field)
    if strict:
        raise ServiceMetricDataError(f"Invalid service metric boolean: {field}")
    return bool(default)


def _parse_timestamp(value: Any, field: str) -> datetime:
    if value is None:
        raise ServiceMetricDataError(f"Missing service metric timestamp: {field}")
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ServiceMetricDataError(f"Invalid timestamp in {field}: {value!r}") from exc


def _ratio(numerator: float, denominator: float, tolerance: float) -> float | None:
    return float(numerator / denominator) if denominator > tolerance else None


def _microgrid_config(config: Mapping[str, Any], kind: str) -> tuple[str, Mapping[str, Any]] | None:
    for item in config.get("microgrids", []):
        dr = item.get("dr", {})
        if str(dr.get("type", "")) == kind:
            return str(item["id"]), dr
    return None


def summarize_day_services(
    hourly_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    physical_feasible: bool | None,
    schema_version: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Summarize one natural day from executed physical-service observations.

    ``service_metrics_v2`` is strict: required columns must be present and finite,
    and result-state timestamps must equal action timestamps plus ``delta_t``.
    Older/unknown schemas are audit-only: missing columns are surfaced through
    ``data_complete``/``missing_fields`` and strict feasibility is left unknown.
    """
    current_schema = str(schema_version) == SERVICE_METRICS_SCHEMA_VERSION
    if not hourly_rows:
        if current_schema:
            raise ServiceMetricDataError("service_metrics_v2 requires at least one hourly row")
        return {
            "metrics_schema_version": str(schema_version),
            "data_complete": False,
            "missing_fields": ["hourly_rows"],
            "strict_feasible_day": None,
        }

    dt = float(config.get("time", {}).get("delta_t", 1.0))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time.delta_t must be positive and finite")
    ids = [str(item["id"]) for item in config.get("microgrids", [])]
    missing: set[str] = set()

    unmet_by_mg = {mg: 0.0 for mg in ids}
    unmet_hours_by_mg = {mg: 0 for mg in ids}
    dr_infeasible_hours_by_mg = {mg: 0 for mg in ids}
    unmet_system_hours = 0
    dr_infeasible_system_hours = 0

    for row in hourly_rows:
        unmet_flags: list[bool] = []
        dr_flags: list[bool] = []
        for mg in ids:
            unmet_kw = max(
                0.0,
                _float_field(
                    row,
                    f"{mg}_unmet_load_kw",
                    strict=current_schema,
                    missing=missing,
                ),
            )
            dr_bad = _bool_field(
                row,
                f"{mg}_dr_supply_infeasible",
                strict=current_schema,
                missing=missing,
            )
            unmet = unmet_kw > tolerance
            unmet_by_mg[mg] += unmet_kw * dt
            unmet_hours_by_mg[mg] += int(unmet)
            dr_infeasible_hours_by_mg[mg] += int(dr_bad)
            unmet_flags.append(unmet)
            dr_flags.append(dr_bad)
        unmet_system_hours += int(any(unmet_flags))
        dr_infeasible_system_hours += int(any(dr_flags))

    # HVAC: every action-t row contains the state at t+1.  Validate the timestamp
    # and compare that result state against the comfort band active at t+1.
    hvac = _microgrid_config(config, "commercial_hvac")
    hvac_result: dict[str, Any] = {
        "applicable": hvac is not None,
        "boundary_count": 0,
        "boundary_success_count": 0,
        "all_boundary_comfort_rate": None,
        "occupied_boundary_count": 0,
        "occupied_success_count": 0,
        "comfort_rate": None,
        "comfort_violation_degree_hours": 0.0,
        "max_comfort_violation_c": 0.0,
        "terminal_applicable": False,
        "terminal_success": None,
    }
    hvac_strict_ok = True
    if hvac is not None:
        mg, dr = hvac
        for row in hourly_rows:
            start = _parse_timestamp(row.get("timestamp"), "timestamp") if current_schema else None
            result_raw = row.get(f"{mg}_result_timestamp")
            if result_raw is None and not current_schema:
                result_raw = row.get("result_timestamp")
            if current_schema:
                result_ts = _parse_timestamp(result_raw, f"{mg}_result_timestamp")
                expected = start + timedelta(hours=dt)
                if abs((result_ts - expected).total_seconds()) > 1e-6:
                    raise ServiceMetricDataError(
                        f"{mg}: result timestamp {result_ts.isoformat()} does not equal "
                        f"action timestamp + delta_t ({expected.isoformat()})"
                    )
            t_air = _float_field(
                row, f"{mg}_t_air_c", strict=current_schema, missing=missing
            )
            low = _float_field(
                row, f"{mg}_result_comfort_low_c", strict=current_schema, missing=missing
            )
            high = _float_field(
                row, f"{mg}_result_comfort_high_c", strict=current_schema, missing=missing
            )
            occupied = _bool_field(
                row, f"{mg}_result_occupied", strict=current_schema, missing=missing
            )
            if low > high:
                if current_schema:
                    raise ServiceMetricDataError(f"{mg}: invalid result comfort band [{low}, {high}]")
                missing.add(f"{mg}_result_comfort_band")
            violation = max(0.0, low - t_air, t_air - high)
            success = violation <= tolerance
            hvac_result["boundary_count"] += 1
            hvac_result["boundary_success_count"] += int(success)
            hvac_result["comfort_violation_degree_hours"] += float(violation * dt)
            hvac_result["max_comfort_violation_c"] = max(
                float(hvac_result["max_comfort_violation_c"]), float(violation)
            )
            if occupied:
                hvac_result["occupied_boundary_count"] += 1
                hvac_result["occupied_success_count"] += int(success)
            hvac_strict_ok = hvac_strict_ok and success
        hvac_result["all_boundary_comfort_rate"] = _ratio(
            float(hvac_result["boundary_success_count"]),
            float(hvac_result["boundary_count"]),
            tolerance,
        )
        hvac_result["comfort_rate"] = _ratio(
            float(hvac_result["occupied_success_count"]),
            float(hvac_result["occupied_boundary_count"]),
            tolerance,
        )
        terminal_mode = str(dr.get("terminal_mode", "none"))
        hvac_result["terminal_applicable"] = terminal_mode != "none"
        if terminal_mode != "none":
            last = hourly_rows[-1]
            air_gap = abs(_float_field(
                last, f"{mg}_terminal_air_gap_c", strict=current_schema, missing=missing
            ))
            mass_gap = abs(_float_field(
                last, f"{mg}_terminal_mass_gap_c", strict=current_schema, missing=missing
            ))
            terminal_ok = bool(
                air_gap <= float(dr["terminal_air_tolerance_c"]) + tolerance
                and mass_gap <= float(dr["terminal_mass_tolerance_c"]) + tolerance
            )
            hvac_result["terminal_success"] = terminal_ok
            hvac_strict_ok = hvac_strict_ok and terminal_ok

    # Industrial mill: distinguish production coverage from actual downstream service.
    mill = _microgrid_config(config, "industrial_mill")
    mill_result: dict[str, Any] = {
        "applicable": mill is not None,
        "production_t": None,
        "consumption_required_t": None,
        "actual_consumption_t": None,
        "downstream_unserved_t": None,
        "production_coverage_ratio": None,
        "downstream_service_completion_rate": None,
        "inventory_start_t": None,
        "inventory_end_t": None,
        "inventory_drawdown_t": None,
        "balance_residual_t": None,
        "inventory_min_violation_t": 0.0,
        "inventory_max_violation_t": 0.0,
        "terminal_applicable": False,
        "terminal_success": None,
    }
    mill_strict_ok = True
    if mill is not None:
        mg, dr = mill
        production = 0.0
        required = 0.0
        consumed = 0.0
        unserved = 0.0
        min_violation = 0.0
        max_violation = 0.0
        first_before: float | None = None
        last_after: float | None = None
        for row in hourly_rows:
            actual_prod_t = _float_field(
                row, f"{mg}_mill_actual_production_t", strict=current_schema, missing=missing
            )
            required_t = _float_field(
                row, f"{mg}_mill_downstream_required_t", strict=current_schema, missing=missing
            )
            consumed_t = _float_field(
                row, f"{mg}_mill_actual_consumption_t", strict=current_schema, missing=missing
            )
            unserved_t = _float_field(
                row, f"{mg}_mill_downstream_unserved_t", strict=current_schema, missing=missing
            )
            before = _float_field(
                row, f"{mg}_mill_silo_before_t", strict=current_schema, missing=missing
            )
            after = _float_field(
                row, f"{mg}_silo_level_t", strict=current_schema, missing=missing
            )
            raw_after = _float_field(
                row, f"{mg}_mill_silo_after_unclipped_t", strict=current_schema, missing=missing
            )
            if first_before is None:
                first_before = before
            last_after = after
            production += actual_prod_t
            required += required_t
            consumed += consumed_t
            unserved += max(0.0, unserved_t)
            min_violation += max(0.0, float(dr["silo_min_t"]) - raw_after)
            max_violation += max(0.0, raw_after - float(dr["silo_max_t"]))
        start = float(first_before or 0.0)
        end = float(last_after or 0.0)
        mill_result.update({
            "production_t": float(production),
            "consumption_required_t": float(required),
            "actual_consumption_t": float(consumed),
            "downstream_unserved_t": float(unserved),
            "production_coverage_ratio": _ratio(production, required, tolerance),
            "downstream_service_completion_rate": _ratio(consumed, required, tolerance),
            "inventory_start_t": start,
            "inventory_end_t": end,
            "inventory_drawdown_t": float(start - end),
            "balance_residual_t": float(start + production - consumed - end),
            "inventory_min_violation_t": float(min_violation),
            "inventory_max_violation_t": float(max_violation),
        })
        mill_strict_ok = bool(
            unserved <= tolerance
            and min_violation <= tolerance
            and max_violation <= tolerance
        )
        terminal_mode = str(dr.get("terminal_mode", "none"))
        mill_result["terminal_applicable"] = terminal_mode != "none"
        if terminal_mode != "none":
            terminal_ok = abs(end - float(dr["silo_initial_t"])) <= (
                float(dr["terminal_tolerance_t"]) + tolerance
            )
            mill_result["terminal_success"] = bool(terminal_ok)
            mill_strict_ok = mill_strict_ok and bool(terminal_ok)

    # EV fields are cumulative event totals in the hourly logger; use the final/max
    # cumulative value instead of summing the same departure event repeatedly.
    ev = _microgrid_config(config, "ev_charging")
    ev_result: dict[str, Any] = {
        "applicable": ev is not None,
        "departure_count": 0,
        "completed_group_count": 0,
        "departure_shortfall_kwh": 0.0,
        "departure_success_rate": None,
    }
    ev_strict_ok = True
    if ev is not None:
        mg, _ = ev
        counts: list[int] = []
        completed: list[int] = []
        shortfalls: list[float] = []
        for row in hourly_rows:
            counts.append(int(round(_float_field(
                row, f"{mg}_ev_departure_count", strict=current_schema, missing=missing
            )))
            completed.append(int(round(_float_field(
                row, f"{mg}_ev_completed_group_count", strict=current_schema, missing=missing
            ))))
            shortfalls.append(max(0.0, _float_field(
                row, f"{mg}_ev_departure_shortfall_kwh", strict=current_schema, missing=missing
            )))
        departure_count = max(counts, default=0)
        completed_count = max(completed, default=0)
        shortfall = max(shortfalls, default=0.0)
        ev_result.update({
            "departure_count": int(departure_count),
            "completed_group_count": int(completed_count),
            "departure_shortfall_kwh": float(shortfall),
            "departure_success_rate": (
                float(completed_count / departure_count) if departure_count > 0 else None
            ),
        })
        ev_strict_ok = shortfall <= tolerance

    steps = len(hourly_rows)
    agent_hours = steps * max(len(ids), 1)
    unmet_total = float(sum(unmet_by_mg.values()))
    data_complete = not missing
    strict_feasible: bool | None
    if physical_feasible is None or not data_complete:
        strict_feasible = None
    else:
        strict_feasible = bool(
            physical_feasible
            and unmet_total <= tolerance
            and dr_infeasible_system_hours == 0
            and hvac_strict_ok
            and mill_strict_ok
            and ev_strict_ok
        )

    result: dict[str, Any] = {
        "metrics_schema_version": str(schema_version),
        "data_complete": bool(data_complete),
        "missing_fields": sorted(missing),
        "physical_feasible_day": None if physical_feasible is None else bool(physical_feasible),
        "strict_feasible_day": strict_feasible,
        "unmet_energy_kwh": unmet_total,
        "unmet_load_hour_count": int(unmet_system_hours),
        "unmet_load_hour_rate": float(unmet_system_hours / steps),
        "unmet_load_microgrid_hour_count": int(sum(unmet_hours_by_mg.values())),
        "unmet_load_microgrid_hour_rate": float(sum(unmet_hours_by_mg.values()) / agent_hours),
        "unmet_load_day": bool(unmet_system_hours > 0),
        "dr_supply_infeasible_hour_count": int(dr_infeasible_system_hours),
        "dr_supply_infeasible_rate": float(dr_infeasible_system_hours / steps),
        "dr_supply_infeasible_microgrid_hour_count": int(sum(dr_infeasible_hours_by_mg.values())),
        "dr_supply_infeasible_microgrid_hour_rate": float(sum(dr_infeasible_hours_by_mg.values()) / agent_hours),
        "dr_supply_infeasible_day": bool(dr_infeasible_system_hours > 0),
        "hvac": hvac_result,
        "mill": mill_result,
        "ev": ev_result,
        # here has aliases used by rollout dashboards and historical output readers.
        "hvac_comfort_rate": hvac_result["comfort_rate"],
        "hvac_terminal_joint_success": hvac_result["terminal_success"],
        "mill_production_coverage_ratio": mill_result["production_coverage_ratio"],
        "mill_production_completion_rate": mill_result["production_coverage_ratio"],
        "mill_downstream_service_completion_rate": mill_result["downstream_service_completion_rate"],
        "mill_terminal_success": mill_result["terminal_success"],
        "ev_departure_count": int(ev_result["departure_count"]),
        "ev_completed_group_count": int(ev_result["completed_group_count"]),
        "ev_departure_shortfall_kwh": float(ev_result["departure_shortfall_kwh"]),
        "ev_departure_success_rate": ev_result["departure_success_rate"],
    }
    for mg in ids:
        result[f"unmet_energy_{mg}_kwh"] = float(unmet_by_mgVmg])
        result[f"unmet_load_hour_rate_{mg}"] = float(unmet_hours_by_mg[mg] / steps)
        result[f"dr_supply_infeasible_rate_{mg}"] = float(
            dr_infeasible_hours_by_mgVmg] / steps
        )
    return result


def aggregate_service_days(
    daily_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Aggregate service counts/energies before computing rates.

    This intentionally avoids an unweighted mean of daily percentages whenever
    denominators can differ between days.
    """
    rows = list(daily_rows)
    complete_rows = [row for row in rows if bool(row.get("data_complete", False))]
    strict_values = [row.get("strict_feasible_day") for row in rows]
    strict_known = [bool(value) for value in strict_values if value is not None]

    unmet_values = [float(row.get("unmet_energy_kwh", 0.0)) for row in complete_rows]
    # Prefer explicit numerators/denominators carried by the nested service records.
    hvac_occ_success = sum(int(row.get("hvac", {}).get("occupied_success_count", 0)) for row in complete_rows)
    hvac_occ_count = sum(int(row.get("hvac", {}).get("occupied_boundary_count", 0)) for row in complete_rows)
    hvac_all_success = sum(int(row.get("hvac", {}).get("boundary_success_count", 0)) for row in complete_rows)
    hvac_all_count = sum(int(row.get("hvac", {}).get("boundary_count", 0)) for row in complete_rows)

    production = sum(float(row.get("mill", {}).get("production_t") or 0.0) for row in complete_rows)
    required = sum(float(row.get("mill", {}).get("consumption_required_t") or 0.0) for row in complete_rows)
    consumed = sum(float(row.get("mill", {}).get("actual_consumption_t") or 0.0) for row in complete_rows)
    unserved = sum(float(row.get("mill", {}).get("downstream_unserved_t") or 0.0) for row in complete_rows)

    ev_departures = sum(int(row.get("ev", {}).get("departure_count", 0)) for row in complete_rows)
    ev_completed = sum(int(row.get("ev", {}).get("completed_group_count", 0)) for row in complete_rows)
    ev_shortfall = sum(float(row.get("ev", {}).get("departure_shortfall_kwh", 0.0)) for row in complete_rows)

    terminal_hvac_values = [
        bool(row["hvac"]["terminal_success"])
        for row in complete_rows
        if row.get("hvac", {}).get("terminal_success") is not None
    ]
    terminal_mill_values = [
        bool(row["mill"]["terminal_success"])
        for row in complete_rows
        if row.get("mill", {}).get("terminal_success") is not None
    ]

    result = {
        "metrics_schema_version": SERVICE_METRICS_SCHEMA_VERSION,
        "days_count": len(rows),
        "data_complete_day_count": len(complete_rows),
        "data_complete_day_rate": float(len(complete_rows) / len(rows)) if rows else None,
        "missing_fields": sorted({
            field for row in rows for field in row.get("missing_fields", [])
        }),
        "strict_feasible_day_count": int(sum(strict_known)),
        "strict_feasible_day_denominator": len(strict_known),
        "strict_feasible_day_rate": (
            float(sum(strict_known) / len(strict_known)) if strict_known else None
        ),
        "unmet_energy_total_kwh": float(sum(unmet_values)),
        "unmet_energy_mean_kwh": float(np.mean(unmet_values)) if unmet_values else None,
        "unmet_load_day_count": int(sum(bool(row.get("unmet_load_day", False)) for row in complete_rows)),
        "unmet_load_day_rate": (
            float(np.mean([bool(row.get("unmet_load_day", False)) for row in complete_rows]))
            if complete_rows else None
        ),
        "dr_supply_infeasible_day_count": int(sum(
            bool(row.get("dr_supply_infeasible_day", False)) for row in complete_rows
        )),
        "dr_supply_infeasible_day_rate": (
            float(np.mean([bool(row.get("dr_supply_infeasible_day", False) for row in complete_rows]))
            if complete_rows else None
        ),
        "hvac_comfort_success_count": int(hvac_occ_success),
        "hvac_comfort_opportunity_count": int(hvac_occ_count),
        "hvac_comfort_rate": _ratio(hvac_occ_success, hvac_occ_count, tolerance),
        "hvac_all_boundary_success_count": int(hvac_all_success),
        "hvac_all_boundary_count": int(hvac_all_count),
        "hvac_all_boundary_comfort_rate": _ratio(hvac_all_success, hvac_all_count, tolerance),
        "hvac_comfort_violation_degree_hours": float(sum(
            float(row.get("hvac", {}).get("comfort_violation_degree_hours", 0.0))
            for row in complete_rows
         )),
        "hvac_terminal_joint_success_rate": (
            float(np.mean(terminal_hvac_values)) if terminal_hvac_values else None
        ),
        "mill_production_t": float(production),
        "mill_consumption_required_t": float(required),
        "mill_actual_consumption_t": float(consumed),
        "mill_downstream_unserved_t": float(unserved),
        "mill_production_coverage_ratio": _ratio(production, required, tolerance),
        "mill_production_completion_rate": _ratio(production, required, tolerance),
        "mill_downstream_service_completion_rate": _ratio(consumed, required, tolerance),
        "mill_terminal_success_rate": (
            float(np.mean(terminal_mill_values)) if terminal_mill_values else None
        ),
        "ev_departure_count": int(ev_departures),
        "ev_completed_group_count": int(ev_completed),
        "ev_departure_shortfall_total_kwh": float(ev_shortfall),
        "ev_departure_success_rate": _ratio(ev_completed, ev_departures, tolerance),
    }
    return result


def aggregate_cooperative_value(
    rows: Sequence[Mapping[str, Any]],
    *,
    monetary_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Aggregate cooperative value capture as ratio-of-sums on paired valid days."""
    included: list[str] = []
    excluded: list[dict[str, str]] = []
    daily_ratios: list[float] = []
    daily_ratio_defined: list[str] = []
    numerator_total = 0.0
    denominator_total = 0.0
    ordering_errors: list[str] = []

    for index, row in enumerate(rows):
        day = str(row.get("date", row.get("day", index)))
        values = {
            "algorithm": row.get("algorithm_joint_profit", row.get("team_profit")),
            "independent": row.get("independent_total_profit"),
            "coalition": row.get("cooperative_milp_profit", row.get("v_grand_coalition")),
        }
        if not all(_finite(value) for value in values.values()):
            excluded.append({"date": day, "reason": "missing_or_nonfinite_profit"})
            continue
        if int(row.get("coalition_solve_failure_count", 0) or 0) > 0:
            excluded.append({"date": day, "reason": "coalition_solver_failure"})
            continue
        algorithm = float(values["algorithm"])
        independent = float(values["independent"])
        coalition = float(values["coalition"])
        numerator = algorithm - independent
        denominator_raw = coalition - independent
        if denominator_raw < -monetary_tolerance:
            ordering_errors.append(day)
            excluded.append({"date": day, "reason": "coalition_value_ordering_error"})
            continue
        denominator = 0.0 if denominator_raw < 0.0 else denominator_raw
        included.append(day)
        numerator_total += numerator
        denominator_total += denominator
        if denominator > monetary_tolerance:
            daily_ratios.append(numerator / denominator)
            daily_ratio_defined.append(day)

    ratio_total = (
        numerator_total / denominator_total
        if denominator_total > monetary_tolerance else None
    )
    status = "ok" if ratio_total is not None else "undefined_no_positive_surplus"
    return {
        "aggregation_method": "ratio_of_sums",
        "monetary_tolerance": float(monetary_tolerance),
        "status": status,
        "cooperative_value_capture_ratio_total": ratio_total,
        "cooperative_value_capture_ratio": ratio_total,
        "cooperative_value_capture_ratio_daily_mean": (
            float(np.mean(daily_ratios)) if daily_ratios else None
        ),
        "cooperative_value_capture_ratio_daily_median": (
            float(np.median(daily_ratios)) if daily_ratios else None
        ),
        "numerator_total": float(numerator_total),
        "denominator_total": float(denominator_total),
        "included_days": included,
        "excluded_days": excluded,
        "daily_ratio_defined_days": daily_ratio_defined,
        "daily_ratio_defined_count": len(daily_ratio_defined),
        "coalition_value_ordering_error_days": ordering_errors,
        "coverage_rate": float(len(included) / len(rows)) if rows else None,
    }
