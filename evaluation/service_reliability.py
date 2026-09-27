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


def _config_from_summary(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        physical = summary["experiment_contract"]["physical_settings"]
    except (KeyError, TypeError):
        return None
    if not isinstance(physical, Mapping) or not physical.get("microgrids"):
        return None
    return dict(physical)


def _daily_service_metrics(
    hourly_rows: Sequence[Mapping[str, Any]],
    agent_ids: Sequence[str],
    *,
    delta_t: float,
    feasible_day: bool,
    tolerance: float,
    config: Mapping[str, Any] | None = None,
    schema_version: str = "legacy_unknown",
) -> dict[str, Any]:
    del agent_ids, delta_t
    from evaluation.service_metrics import summarize_day_services
    if config is None:
        # Historical runs without an embedded full physical contract are
        # audit-only.  Missing evidence remains null/incomplete rather than
        # silently reconstructed from today's YAML defaults.
        return summarize_day_services(
            hourly_rows,
            {"time": {"delta_t": 1.0}, "microgrids": []},
            bool(feasible_day),
            str(schema_version),
            tolerance=float(tolerance),
        )
    return summarize_day_services(
        hourly_rows,
        config,
        bool(feasible_day),
        str(schema_version),
        tolerance=float(tolerance),
    )


def _aggregate_service_metrics(
    daily_rows: Sequence[Mapping[str, Any]], agent_ids: Sequence[str]
) -> dict[str, Any]:
    del agent_ids
    from evaluation.service_metrics import aggregate_service_days
    result = aggregate_service_days(daily_rows)
    result["unmet_energy_distribution_kwh"] = _distribution([
        _as_float(row.get("unmet_energy_kwh"))
        for row in daily_rows
        if row.get("unmet_energy_kwh") is not None
    ])
    result["by_microgrid"] = {}
    ids = sorted({
        key[len("unmet_energy_"):-len("_kwh")]
        for row in daily_rows
        for key in row
        if str(key).startswith("unmet_energy_")
        and str(key).endswith("_kwh")
        and key != "unmet_energy_kwh"
    })
    for mg in ids:
        values = [
            _as_float(row.get(f"unmet_energy_{mg}_kwh"))
            for row in daily_rows
            if row.get(f"unmet_energy_{mg}_kwh") is not None
        ]
        result["by_microgrid"][mg] = {
            "unmet_energy_total_kwh": float(np.sum(values)) if values else 0.0,
            "unmet_energy_mean_kwh": float(np.mean(values)) if values else None,
        }
    # Compatibility aliases for the existing report/plot consumer.
    if "unmet_load_hour_rate_mean" not in result:
        result["unmet_load_hour_rate_mean"] = (
            float(np.mean([
                _as_float(row.get("unmet_load_hour_rate"))
                for row in daily_rows
                if row.get("unmet_load_hour_rate") is not None
            ]))
            if daily_rows else None
        )
    if "unmet_load_microgrid_hour_rate_mean" not in result:
        result["unmet_load_microgrid_hour_rate_mean"] = (
            float(np.mean([
                _as_float(row.get("unmet_load_microgrid_hour_rate"))
                for row in daily_rows
                if row.get("unmet_load_microgrid_hour_rate") is not None
            ]))
            if daily_rows else None
        )
    if "dr_supply_infeasible_rate_mean" not in result:
        result["dr_supply_infeasible_rate_mean"] = (
            float(np.mean([
                _as_float(row.get("dr_supply_infeasible_rate"))
                for row in daily_rows
                if row.get("dr_supply_infeasible_rate") is not None
            ]))
            if daily_rows else None
        )
    if "dr_supply_infeasible_microgrid_hour_rate_mean" not in result:
        result["dr_supply_infeasible_microgrid_hour_rate_mean"] = (
            float(np.mean([
                _as_float(row.get("dr_supply_infeasible_microgrid_hour_rate"))
                for row in daily_rows
                if row.get("dr_supply_infeasible_microgrid_hour_rate") is not None
            ]))
            if daily_rows else None
        )
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

    Existing ``feasible_day`` semantics are preserved.  Current runs are
    recomputed by the shared service_metrics_v2 implementation; historical runs
    without sufficient embedded evidence remain explicitly incomplete.
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
    physical_config = _config_from_summary(summary_document)
    schema_version = str(
        summary_document.get("experiment_contract", {}).get(
            "service_metrics_schema_version", "legacy_unknown"
        )
    )
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
                config=physical_config,
                schema_version=schema_version,
            )
        )
        augmented_rows.append(row)

    reliability = _aggregate_service_metrics(augmented_rows, agent_ids)
    reliability["definition"] = {
        "feasible_day": "existing physical constraint feasibility; semantics unchanged",
        "strict_feasible_day": (
            "physical feasibility plus zero unmet/DR failures and applicable "
            "HVAC, mill, and EV service constraints under service_metrics_v2"
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
