from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_TOLERANCE = 1e-7
_AGENT_UNMET_RE = re.compile(r"^(.+)_unmet_load_kw$")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], base_fields: Sequence[str]) -> None:
    fields = list(base_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.number)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([_as_float(row.get(field, 0.0)) for row in rows])) if rows else 0.0


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([_as_bool(row.get(field, False)) for row in rows])) if rows else 0.0


def _infer_agent_ids(fieldnames: Sequence[str]) -> list[str]:
    ids = []
    for field in fieldnames:
        match = _AGENT_UNMET_RE.match(str(field))
        if match and match.group(1) not in ids:
            ids.append(match.group(1))
    return ids


def _delta_t_from_summary(summary: Mapping[str, Any]) -> float:
    try:
        value = summary["experiment_contract"]["physical_settings"]["time"]["delta_t"]
        delta_t = float(value)
        if delta_t > 0.0:
            return delta_t
    except (KeyError, TypeError, ValueError):
        pass
    return 1.0


def _daily_service_metrics(
    hourly_rows: Sequence[Mapping[str, Any]],
    agent_ids: Sequence[str],
    *,
    delta_t: float,
    feasible_day: bool,
    tolerance: float,
) -> dict[str, Any]:
    steps = len(hourly_rows)
    agent_count = len(agent_ids)
    unmet_system_hours = 0
    unmet_microgrid_hours = 0
    dr_system_hours = 0
    dr_microgrid_hours = 0
    unmet_by_mg = {mg: 0.0 for mg in agent_ids}
    unmet_hours_by_mg = {mg: 0 for mg in agent_ids}
    dr_hours_by_mg = {mg: 0 for mg in agent_ids}

    for row in hourly_rows:
        unmet_flags: list[bool] = []
        dr_flags: list[bool] = []
        for mg in agent_ids:
            unmet_kw = max(0.0, _as_float(row.get(f"{mg}_unmet_load_kw", 0.0)))
            unmet_by_mg[mg] += unmet_kw * delta_t
            unmet = unmet_kw > tolerance
            dr_infeasible = _as_bool(row.get(f"{mg}_dr_supply_infeasible", False))
            unmet_flags.append(unmet)
            dr_flags.append(dr_infeasible)
            unmet_hours_by_mg[mg] += int(unmet)
            dr_hours_by_mg[mg] += int(dr_infeasible)
        unmet_system_hours += int(any(unmet_flags))
        unmet_microgrid_hours += sum(unmet_flags)
        dr_system_hours += int(any(dr_flags))
        dr_microgrid_hours += sum(dr_flags)

    unmet_energy = float(sum(unmet_by_mg.values()))
    unmet_day = unmet_system_hours > 0
    dr_day = dr_system_hours > 0
    strict_feasible = bool(feasible_day and not unmet_day and not dr_day)
    mg_hour_denominator = steps * agent_count

    result: dict[str, Any] = {
        "unmet_energy_kwh": unmet_energy,
        "unmet_load_hour_count": int(unmet_system_hours),
        "unmet_load_hour_rate": unmet_system_hours / steps if steps else 0.0,
        "unmet_load_microgrid_hour_count": int(unmet_microgrid_hours),
        "unmet_load_microgrid_hour_rate": (
            unmet_microgrid_hours / mg_hour_denominator if mg_hour_denominator else 0.0
        ),
        "unmet_load_day": bool(unmet_day),
        "dr_supply_infeasible_hour_count": int(dr_system_hours),
        "dr_supply_infeasible_rate": dr_system_hours / steps if steps else 0.0,
        "dr_supply_infeasible_microgrid_hour_count": int(dr_microgrid_hours),
        "dr_supply_infeasible_microgrid_hour_rate": (
            dr_microgrid_hours / mg_hour_denominator if mg_hour_denominator else 0.0
        ),
        "dr_supply_infeasible_day": bool(dr_day),
        "strict_feasible_day": strict_feasible,
    }
    for mg in agent_ids:
        result[f"unmet_energy_{mg}_kwh"] = float(unmet_by_mg[mg])
        result[f"unmet_load_hour_rate_{mg}"] = (
            unmet_hours_by_mg[mg] / steps if steps else 0.0
        )
        result[f"dr_supply_infeasible_rate_{mg}"] = (
            dr_hours_by_mg[mg] / steps if steps else 0.0
        )
    return result


def _aggregate_service_metrics(
    daily_rows: Sequence[Mapping[str, Any]], agent_ids: Sequence[str]
) -> dict[str, Any]:
    unmet_values = [_as_float(row.get("unmet_energy_kwh", 0.0)) for row in daily_rows]
    result: dict[str, Any] = {
        "days_count": len(daily_rows),
        "unmet_energy_total_kwh": float(np.sum(unmet_values)) if unmet_values else 0.0,
        "unmet_energy_mean_kwh": float(np.mean(unmet_values)) if unmet_values else 0.0,
        "unmet_energy_distribution_kwh": _distribution(unmet_values),
        "unmet_load_day_count": int(sum(_as_bool(row.get("unmet_load_day")) for row in daily_rows)),
        "unmet_load_day_rate": _rate(daily_rows, "unmet_load_day"),
        "unmet_load_hour_rate_mean": _mean(daily_rows, "unmet_load_hour_rate"),
        "unmet_load_microgrid_hour_rate_mean": _mean(
            daily_rows, "unmet_load_microgrid_hour_rate"
        ),
        "dr_supply_infeasible_day_count": int(
            sum(_as_bool(row.get("dr_supply_infeasible_day")) for row in daily_rows)
        ),
        "dr_supply_infeasible_day_rate": _rate(daily_rows, "dr_supply_infeasible_day"),
        "dr_supply_infeasible_rate_mean": _mean(daily_rows, "dr_supply_infeasible_rate"),
        "dr_supply_infeasible_microgrid_hour_rate_mean": _mean(
            daily_rows, "dr_supply_infeasible_microgrid_hour_rate"
        ),
        "strict_feasible_day_count": int(
            sum(_as_bool(row.get("strict_feasible_day")) for row in daily_rows)
        ),
        "strict_feasible_day_rate": _rate(daily_rows, "strict_feasible_day"),
        "by_microgrid": {},
    }
    for mg in agent_ids:
        unmet = [_as_float(row.get(f"unmet_energy_{mg}_kwh", 0.0)) for row in daily_rows]
        result["by_microgrid"][mg] = {
            "unmet_energy_total_kwh": float(np.sum(unmet)) if unmet else 0.0,
            "unmet_energy_mean_kwh": float(np.mean(unmet)) if unmet else 0.0,
            "unmet_load_hour_rate_mean": _mean(daily_rows, f"unmet_load_hour_rate_{mg}"),
            "dr_supply_infeasible_rate_mean": _mean(
                daily_rows, f"dr_supply_infeasible_rate_{mg}"
            ),
            "unmet_load_day_rate": float(np.mean([value > 0.0 for value in unmet])) if unmet else 0.0,
            "dr_supply_infeasible_day_rate": float(np.mean([
                _as_float(row.get(f"dr_supply_infeasible_rate_{mg}", 0.0)) > 0.0
                for row in daily_rows
            ])) if daily_rows else 0.0,
        }
    return result


def _summary_aliases(reliability: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strict_feasible_day_rate": reliability["strict_feasible_day_rate"],
        "unmet_energy_total_kwh": reliability["unmet_energy_total_kwh"],
        "unmet_energy_mean_kwh": reliability["unmet_energy_mean_kwh"],
        "unmet_load_day_rate": reliability["unmet_load_day_rate"],
        "unmet_load_hour_rate_mean": reliability["unmet_load_hour_rate_mean"],
        "unmet_load_microgrid_hour_rate_mean": reliability[
            "unmet_load_microgrid_hour_rate_mean"
        ],
        "dr_supply_infeasible_day_rate": reliability["dr_supply_infeasible_day_rate"],
        "dr_supply_infeasible_rate_mean": reliability["dr_supply_infeasible_rate_mean"],
        "dr_supply_infeasible_microgrid_hour_rate_mean": reliability[
            "dr_supply_infeasible_microgrid_hour_rate_mean"
        ],
    }


def _attach_summary_metrics(document: dict[str, Any], reliability: Mapping[str, Any]) -> None:
    document["service_reliability"] = dict(reliability)
    document.update(_summary_aliases(reliability))
    for key in ("algorithm", "matd3"):
        section = document.get(key)
        if isinstance(section, dict):
            section["service_reliability"] = dict(reliability)
            section["strict_feasible_rate"] = reliability["strict_feasible_day_rate"]
            section["unmet_energy_kwh_mean"] = reliability["unmet_energy_mean_kwh"]
            section["unmet_energy_kwh_total"] = reliability["unmet_energy_total_kwh"]
            section["unmet_load_hour_rate_mean"] = reliability["unmet_load_hour_rate_mean"]
            section["dr_supply_infeasible_rate_mean"] = reliability[
                "dr_supply_infeasible_rate_mean"
            ]


def _plot_constraint_reliability(run_dir: Path, daily_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [
        "Feasible Day",
        "Strict Feasible Day",
        "Unmet-Load Day",
        "DR Supply-Infeasible Day",
        "SOC Violation",
        "Power Balance Violation",
        "P2P Balance Violation",
    ]
    values = [
        100.0 * _rate(daily_rows, "feasible_day"),
        100.0 * _rate(daily_rows, "strict_feasible_day"),
        100.0 * _rate(daily_rows, "unmet_load_day"),
        100.0 * _rate(daily_rows, "dr_supply_infeasible_day"),
        100.0 * _rate(daily_rows, "soc_violation_day"),
        100.0 * _rate(daily_rows, "power_balance_violation_day"),
        100.0 * _rate(daily_rows, "p2p_balance_violation_day"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    positions = np.arange(len(labels))
    bars = ax.barh(positions, values)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 105.0)
    ax.set_xlabel("Test-Day Rate (%)")
    ax.set_title("Constraint, Supply Reliability, and Feasibility")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            min(value + 1.0, 102.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.1f}%",
            va="center",
            ha="left" if value < 100.0 else "right",
            fontsize=8,
        )
    ax.text(
        0.99,
        0.02,
        (
            f"Unmet energy: {_mean(daily_rows, 'unmet_energy_kwh'):.3f} kWh/day | "
            f"Unmet-hour rate: {100*_mean(daily_rows, 'unmet_load_hour_rate'):.2f}% | "
            f"DR supply-infeasible rate: {100*_mean(daily_rows, 'dr_supply_infeasible_rate'):.2f}%"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"12_constraint_feasibility.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def augment_run_outputs(
    run_dir: str | Path,
    *,
    generate_plot: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Add supply-reliability metrics without changing environment or policy behavior.

    Existing ``feasible_day`` semantics are preserved. ``strict_feasible_day`` is
    diagnostic-only and additionally requires zero unmet load and zero DR supply
    infeasibility on that test day.
    """
    root = Path(run_dir).resolve()
    daily_path = root / "test_daily_metrics.csv"
    hourly_dir = root / "test_hourly_trajectories"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing test daily metrics: {daily_path}")
    if not hourly_dir.is_dir():
        raise FileNotFoundError(f"Missing test hourly trajectories: {hourly_dir}")

    summary_document = _read_json(root / "test_summary.json", {}) or {}
    delta_t = _delta_t_from_summary(summary_document)
    base_fields, daily_rows = _read_csv(daily_path)
    if not daily_rows:
        raise ValueError("test_daily_metrics.csv contains no test days")

    agent_ids: list[str] = []
    augmented_rows: list[dict[str, Any]] = []
    for original in daily_rows:
        row: dict[str, Any] = dict(original)
        date = str(row.get("date", "")).strip()
        if not date:
            raise ValueError("Every test daily row must contain a date")
        hourly_path = hourly_dir / f"{date}.csv"
        if not hourly_path.exists():
            raise FileNotFoundError(f"Missing hourly trajectory for {date}: {hourly_path}")
        hourly_fields, hourly_rows = _read_csv(hourly_path)
        current_ids = _infer_agent_ids(hourly_fields)
        if not current_ids:
            raise ValueError(f"No *_unmet_load_kw columns found in {hourly_path}")
        if not agent_ids:
            agent_ids = current_ids
        elif current_ids != agent_ids:
            raise ValueError(
                f"Inconsistent microgrid columns in {hourly_path}: {current_ids} != {agent_ids}"
            )
        row.update(
            _daily_service_metrics(
                hourly_rows,
                agent_ids,
                delta_t=delta_t,
                feasible_day=_as_bool(row.get("feasible_day", False)),
                tolerance=float(tolerance),
            )
        )
        augmented_rows.append(row)

    reliability = _aggregate_service_metrics(augmented_rows, agent_ids)
    reliability["definition"] = {
        "feasible_day": "existing physical constraint feasibility; semantics unchanged",
        "strict_feasible_day": (
            "feasible_day AND no unmet-load hour AND no DR supply-infeasible hour"
        ),
        "unmet_load_hour_rate": "fraction of test-day hours where any microgrid has unmet load",
        "unmet_load_microgrid_hour_rate": (
            "fraction of microgrid-hours with unmet load across all microgrids"
        ),
        "dr_supply_infeasible_rate": (
            "fraction of test-day hours where any microgrid reports dr_supply_infeasible"
        ),
        "dr_supply_infeasible_microgrid_hour_rate": (
            "fraction of microgrid-hours reporting dr_supply_infeasible"
        ),
        "unmet_tolerance_kw": float(tolerance),
        "delta_t_hours": float(delta_t),
    }

    _write_csv(daily_path, augmented_rows, base_fields)
    for name in ("test_summary.json", "test_evaluation.json"):
        path = root / name
        if path.exists():
            document = _read_json(path, {}) or {}
            _attach_summary_metrics(document, reliability)
            _write_json(path, document)
    run_summary_path = root / "summary.json"
    if run_summary_path.exists():
        run_summary = _read_json(run_summary_path, {}) or {}
        _attach_summary_metrics(run_summary, reliability)
        _write_json(run_summary_path, run_summary)

    if generate_plot:
        _plot_constraint_reliability(root, augmented_rows)
    return reliability
