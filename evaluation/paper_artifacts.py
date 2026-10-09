from __future__ import annotations

import json
from pathlib import Path
from experiments.layout import artifact_directory, figure_directory, run_root
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


MICROGRIDS = ("mg1", "mg2", "mg3")
BLUE = "#2F6B9A"
GOLD = "#D5A021"
ORANGE = "#D97841"
OLIVE = "#7A8B45"
PINK = "#B75D75"
INK = "#263238"
GREY = "#AAB2B7"
LIGHT_GREY = "#E6E9EB"


def _environment_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("environment")
    return nested if isinstance(nested, Mapping) else config


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Paper artifacts require saved fields: {', '.join(missing)}")


def select_representative_day(hourly: pd.DataFrame) -> str:
    """Select the day nearest the test-set centroid of exogenous conditions.

    Only uncontrollable base load, available renewable power, and saved outdoor
    temperature are used, so algorithms evaluated on the same days select the
    same representative date regardless of their actions or rewards.
    """

    fields = [
        *(f"{mg}_base_load_kw" for mg in MICROGRIDS),
        *(f"{mg}_renewable_available_kw" for mg in MICROGRIDS),
    ]
    _require_columns(hourly, ["date", *fields])
    work = hourly.copy()
    work["base_load_energy"] = work[[f"{mg}_base_load_kw" for mg in MICROGRIDS]].sum(axis=1)
    work["renewable_available_energy"] = work[
        [f"{mg}_renewable_available_kw" for mg in MICROGRIDS]
    ].sum(axis=1)
    aggregations: dict[str, str] = {
        "base_load_energy": "sum",
        "renewable_available_energy": "sum",
    }
    if "mg2_t_out_c" in work:
        aggregations["mg2_t_out_c"] = "mean"
    daily = work.groupby("date", sort=True).agg(aggregations)
    if daily.empty:
        raise ValueError("Representative-day selection requires at least one saved test day")
    scale = daily.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    standardized = (daily - daily.mean()) / scale
    distance = np.sqrt(np.square(standardized).sum(axis=1))
    return str(distance.sort_values(kind="stable").index[0])


def _metric(
    metric_id: str,
    category: str,
    label: str,
    unit: str,
    value: float,
    direction: str,
) -> dict[str, Any]:
    stored_value = float(value) * 100.0 if unit == "%" else float(value)
    return {
        "metric_id": metric_id,
        "category": category,
        "metric": label,
        "unit": unit,
        "value": stored_value,
        "preferred_direction": direction,
    }


def calculate_paper_metrics(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Calculate paper-facing ESS and heterogeneous-DR metrics from saved test data."""

    env = _environment_config(config)
    microgrids = {str(item["id"]): item for item in env["microgrids"]}
    terminal_cfg = env["terminal_soc"]
    tolerance = float(terminal_cfg["tolerance"])
    penalty_rate = float(terminal_cfg["penalty_per_kwh"])
    required = ["date", "hour"]
    for mg in MICROGRIDS:
        required.extend([
            f"{mg}_soc", f"{mg}_battery_charge_kw", f"{mg}_battery_discharge_kw",
            f"{mg}_dr_projected", f"{mg}_dr_supply_infeasible",
        ])
    required.extend([
        "mg1_silo_level_t", "mg1_mill_production_tph",
        "mg2_t_air_c", "mg2_t_mass_c", "mg2_t_out_c", "mg2_occupied",
        "mg2_hvac_effective_signed_power_kw",
        "mg3_ev_soc", "mg3_ev_effective_power_kw",
    ])
    _require_columns(hourly, required)
    _require_columns(daily, ["algorithm_team_profit"])

    ordered = hourly.sort_values(["date", "hour"]).copy()
    grouped = ordered.groupby("date", sort=True)
    dates = list(grouped.groups)
    day_count = len(dates)
    if not day_count:
        raise ValueError("Paper metrics require at least one saved test day")

    rows: list[dict[str, Any]] = []
    current_service_v2 = bool(
        "metrics_schema_version" in daily.columns
        and daily["metrics_schema_version"].astype(str).eq("service_metrics_v2").all()
    )
    ess_equivalent = pd.Series(0.0, index=dates, dtype=float)
    success_columns: dict[str, pd.Series] = {}
    total_throughput = 0.0
    equivalent_cycles: list[float] = []
    absolute_deviations: list[pd.Series] = []

    for mg in MICROGRIDS:
        battery = microgrids[mg]["battery"]
        capacity = float(battery["capacity_kwh"])
        initial_soc = float(battery["initial_soc"])
        terminal = grouped[f"{mg}_soc"].last().reindex(dates)
        deviation = (terminal - initial_soc).abs()
        success_columns[mg] = deviation <= tolerance + 1e-12
        absolute_deviations.append(deviation)
        excess = (deviation - tolerance).clip(lower=0.0) * capacity
        ess_equivalent = ess_equivalent.add(excess, fill_value=0.0)
        throughput = grouped[
            [f"{mg}_battery_charge_kw", f"{mg}_battery_discharge_kw"]
        ].sum().sum(axis=1)
        total_throughput += float(throughput.mean())
        equivalent_cycles.append(float((throughput / (2.0 * capacity)).mean()))

    success_frame = pd.DataFrame(success_columns)
    abs_dev = pd.concat(absolute_deviations, ignore_index=True)
    if current_service_v2 and str(terminal_cfg.get("mode", "none")) == "none":
        rows.extend([
            _metric("ess_end_soc_abs_deviation", "ESS", "Mean absolute end-of-day SOC deviation (diagnostic)", "p.u.", abs_dev.mean(), "lower"),
            _metric("ess_throughput_kwh_day", "ESS", "Battery throughput", "kWh/day", total_throughput, "lower"),
            _metric("ess_equivalent_cycles_day", "ESS", "Mean equivalent full cycles", "cycles/day", np.mean(equivalent_cycles), "lower"),
        ])
    else:
        rows.extend([
            _metric("ess_terminal_success_microgrid_day", "ESS", "Terminal SOC success rate", "%", success_frame.to_numpy().mean(), "higher"),
            _metric("ess_all_microgrids_terminal_success_day", "ESS", "All-microgrid terminal SOC success rate", "%", success_frame.all(axis=1).mean(), "higher"),
            _metric("ess_terminal_abs_deviation", "ESS", "Mean absolute terminal SOC deviation", "p.u.", abs_dev.mean(), "lower"),
            _metric("ess_terminal_excess_kwh_day", "ESS", "Terminal SOC excess-equivalent energy", "kWh/day", ess_equivalent.mean(), "lower"),
            _metric("ess_throughput_kwh_day", "ESS", "Battery throughput", "kWh/day", total_throughput, "lower"),
            _metric("ess_equivalent_cycles_day", "ESS", "Mean equivalent full cycles", "cycles/day", np.mean(equivalent_cycles), "lower"),
        ])

    mill = microgrids["mg1"]["dr"]
    silo_terminal = grouped["mg1_silo_level_t"].last().reindex(dates)
    silo_gap = (silo_terminal - float(mill["silo_initial_t"])).abs()
    silo_tolerance = float(mill["terminal_tolerance_t"])
    mill_equivalent = (
        (silo_gap - silo_tolerance).clip(lower=0.0)
        / float(mill["rated_production_tph"])
        * float(mill["rated_power_kw"])
    )
    if current_service_v2:
        _require_columns(
            daily,
            [
                "mill_production_t",
                "mill_consumption_required_t",
                "mill_actual_consumption_t",
                "mill_downstream_unserved_t",
                "mill_inventory_drawdown_t",
            ],
        )
        production = pd.to_numeric(daily["mill_production_t"], errors="raise").sum()
        required_production = pd.to_numeric(
            daily["mill_consumption_required_t"], errors="raise"
        ).sum()
        actual_consumption = pd.to_numeric(
            daily["mill_actual_consumption_t"], errors="raise"
        ).sum()
        downstream_unserved = pd.to_numeric(
            daily["mill_downstream_unserved_t"], errors="raise"
        ).sum()
        coverage = production / required_production if required_production > 1e-12 else 0.0
        service = actual_consumption / required_production if required_production > 1e-12 else 0.0
        rows.extend([
            _metric("mill_production_coverage_ratio", "MG1 industrial mill", "Actual production coverage ratio", "%", coverage, "target_1"),
            _metric("mill_downstream_service_completion_rate", "MG1 industrial mill", "Downstream service completion rate", "%", service, "higher"),
            _metric("mill_downstream_unserved_t_day", "MG1 industrial mill", "Downstream unserved material", "t/day", downstream_unserved / day_count, "lower"),
            _metric("mill_inventory_drawdown_t_day", "MG1 industrial mill", "Mean end-of-day inventory drawdown", "t/day", pd.to_numeric(daily["mill_inventory_drawdown_t"], errors="raise").mean(), "diagnostic"),
        ])
    else:
        production = grouped["mg1_mill_production_tph"].sum()
        required_production = float(mill["downstream_consumption_tph"]) * 24.0
        rows.extend([
            _metric("mill_production_completion_rate", "MG1 industrial mill", "Production completion rate", "%", (production / required_production).mean(), "target_1"),
            _metric("mill_terminal_success_rate", "MG1 industrial mill", "Terminal silo success rate", "%", (silo_gap <= silo_tolerance + 1e-12).mean(), "higher"),
            _metric("mill_terminal_abs_gap_t", "MG1 industrial mill", "Mean terminal silo gap", "t", silo_gap.mean(), "lower"),
        ])

    hvac = microgrids["mg2"]["dr"]
    late_mask = ordered["hour"] >= (int(env.get("time", {}).get("episode_length", 24)) - 2)
    late_power = ordered.loc[late_mask, "mg2_hvac_effective_signed_power_kw"]
    if late_power.empty or not np.all(np.isfinite(late_power)):
        raise ValueError("Paper metrics require finite mg2_hvac_effective_signed_power_kw for late hours")
    late_power_p95 = float(np.quantile(late_power.abs(), 0.95))
    if current_service_v2:
        _require_columns(
            daily,
            [
                "hvac_occupied_success_count",
                "hvac_occupied_boundary_count",
                "hvac_comfort_violation_degree_hours",
                "hvac_max_comfort_violation_c",
            ],
        )
        occ_success = pd.to_numeric(
            daily["hvac_occupied_success_count"], errors="raise"
        ).sum()
        occ_count = pd.to_numeric(
            daily["hvac_occupied_boundary_count"], errors="raise"
        ).sum()
        comfort_rate = occ_success / occ_count if occ_count > 0 else 0.0
        rows.extend([
            _metric("hvac_occupied_comfort_rate", "MG2 HVAC", "Occupied comfort compliance rate", "%", comfort_rate, "higher"),
            _metric("hvac_occupied_degree_hours", "MG2 HVAC", "Occupied temperature violation", "°C·h/day", pd.to_numeric(daily["hvac_comfort_violation_degree_hours"], errors="raise").sum() / day_count, "lower"),
            _metric("hvac_max_comfort_violation_c", "MG2 HVAC", "Maximum comfort-bound violation", "°C", pd.to_numeric(daily["hvac_max_comfort_violation_c"], errors="raise").max(), "lower"),
            _metric("hvac_late_power_p95_kw", "MG2 HVAC", "Late HVAC absolute power p95", "kW", late_power_p95, "lower"),
        ])
    else:
        occupied = ordered["mg2_occupied"].astype(float) > 0.5
        comfort_low = np.where(occupied, float(hvac["occupied_comfort_c"][0]), float(hvac["unoccupied_comfort_c"][0]))
        comfort_high = np.where(occupied, float(hvac["occupied_comfort_c"][1]), float(hvac["unoccupied_comfort_c"][1]))
        comfort_violation = np.maximum(0.0, comfort_low - ordered["mg2_t_air_c"]) + np.maximum(
            0.0, ordered["mg2_t_air_c"] - comfort_high
        )
        occupied_violation = comfort_violation[occupied]
        occupied_days = max(1, ordered.loc[occupied, "date"].nunique())
        first_outdoor = grouped["mg2_t_out_c"].first().reindex(dates)
        initial_air = np.clip(23.0 + 0.20 * (first_outdoor - 23.0), 18.0, 28.0)
        initial_mass = np.clip(23.0 + 0.10 * (first_outdoor - 23.0), 18.0, 28.0)
        terminal_air = grouped["mg2_t_air_c"].last().reindex(dates)
        terminal_mass = grouped["mg2_t_mass_c"].last().reindex(dates)
        if {"mg2_terminal_air_gap_c", "mg2_terminal_mass_gap_c"}.issubset(ordered.columns):
            signed_air_gap = grouped["mg2_terminal_air_gap_c"].last().reindex(dates)
            signed_mass_gap = grouped["mg2_terminal_mass_gap_c"].last().reindex(dates)
            air_gap = signed_air_gap.abs()
            mass_gap = signed_mass_gap.abs()
        else:
            signed_air_gap = initial_air - terminal_air
            signed_mass_gap = initial_mass - terminal_mass
            air_gap = signed_air_gap.abs()
            mass_gap = signed_mass_gap.abs()
        air_tolerance = float(hvac["terminal_air_tolerance_c"])
        mass_tolerance = float(hvac["terminal_mass_tolerance_c"])
        air_cop = np.where(signed_air_gap < 0.0, float(hvac["cop_cool"]), float(hvac["cop_heat"]))
        mass_cop = np.where(signed_mass_gap < 0.0, float(hvac["cop_cool"]), float(hvac["cop_heat"]))
        hvac_equivalent = (
            (air_gap - air_tolerance).clip(lower=0.0) * float(hvac["c_air_kwh_per_c"]) / air_cop
            + (mass_gap - mass_tolerance).clip(lower=0.0) * float(hvac["c_mass_kwh_per_c"]) / mass_cop
        )
        comfort_rate = float((occupied_violation <= 1e-9).mean()) if len(occupied_violation) else 1.0
        rows.extend([
            _metric("hvac_occupied_comfort_rate", "MG2 HVAC", "Occupied comfort compliance rate", "%", comfort_rate, "higher"),
            _metric("hvac_occupied_degree_hours", "MG2 HVAC", "Occupied temperature violation", "°C·h/occupied day", occupied_violation.sum() / occupied_days, "lower"),
            _metric("hvac_terminal_joint_success_rate", "MG2 HVAC", "Joint terminal thermal success rate", "%", ((air_gap <= air_tolerance + 1e-12) & (mass_gap <= mass_tolerance + 1e-12)).mean(), "higher"),
            _metric("hvac_late_power_p95_kw", "MG2 HVAC", "Late HVAC absolute power p95", "kW", late_power_p95, "lower"),
        ])

    ev = microgrids["mg3"]["dr"]
    if current_service_v2:
        _require_columns(
            daily,
            [
                "ev_departure_count",
                "ev_completed_group_count",
                "ev_departure_shortfall_kwh",
            ],
        )
        departures = pd.to_numeric(daily["ev_departure_count"], errors="raise").sum()
        completed = pd.to_numeric(
            daily["ev_completed_group_count"], errors="raise"
        ).sum()
        shortfall = pd.to_numeric(
            daily["ev_departure_shortfall_kwh"], errors="raise"
        ).sum()
        success = completed / departures if departures > 0 else 0.0
        ev_energy = float(ordered["mg3_ev_effective_power_kw"].sum())
        rows.extend([
            _metric("ev_departure_target_success_rate", "MG3 EV", "EV cohort departure success rate", "%", success, "higher"),
            _metric("ev_departure_shortfall_kwh_day", "MG3 EV", "Departure energy shortfall", "kWh/day", shortfall / day_count, "lower"),
            _metric("ev_charge_energy_kwh_day", "MG3 EV", "Delivered EV charging energy", "kWh/day", ev_energy / day_count, "diagnostic"),
        ])
    else:
        departure_hour = int(ev["departure_hour"])
        departure = ordered.loc[ordered["hour"] == departure_hour - 1].set_index("date")["mg3_ev_soc"].reindex(dates)
        target_soc = float(ev["target_soc"])
        ev_shortfall = (target_soc - departure).clip(lower=0.0) * float(ev["fleet_capacity_kwh"])
        available_start = int(ev.get("available_start_hour", 0))
        valley_mask = ordered["hour"].between(available_start, max(available_start, departure_hour - 2))
        ev_energy = float(ordered["mg3_ev_effective_power_kw"].sum())
        valley_share = float(ordered.loc[valley_mask, "mg3_ev_effective_power_kw"].sum() / ev_energy) if ev_energy > 0.0 else 0.0
        rows.extend([
            _metric("ev_departure_target_success_rate", "MG3 EV", "Departure SOC target success rate", "%", (departure >= target_soc - 1e-7).mean(), "higher"),
            _metric("ev_departure_shortfall_kwh_day", "MG3 EV", "Departure energy shortfall", "kWh/day", ev_shortfall.mean(), "lower"),
            _metric("ev_valley_charge_share", "MG3 EV", "Pre-departure low-price charging share", "%", valley_share, "higher"),
        ])

    for mg in MICROGRIDS:
        scope = ordered
        if mg == "mg3" and not current_service_v2:
            ev_scope = microgrids["mg3"]["dr"]
            scope = ordered.loc[ordered["hour"].between(
                int(ev_scope.get("available_start_hour", 0)),
                int(ev_scope["departure_hour"]) - 1,
            )]
        rows.extend([
            _metric(f"{mg}_dr_projection_rate", "DR control", f"{mg.upper()} DR action projection rate", "%", scope[f"{mg}_dr_projected"].astype(float).mean(), "lower"),
            _metric(f"{mg}_dr_supply_infeasible_rate", "DR control", f"{mg.upper()} DR supply infeasible rate", "%", scope[f"{mg}_dr_supply_infeasible"].astype(float).mean(), "lower"),
        ])

    penalty_fields = [f"{mg}_terminal_constraint_penalty" for mg in MICROGRIDS]
    if current_service_v2 and str(terminal_cfg.get("mode", "none")) == "none":
        terminal_penalty = pd.Series(0.0, index=dates, dtype=float)
    elif set(penalty_fields).issubset(ordered.columns):
        terminal_penalty = ordered.assign(
            _terminal_penalty=ordered[penalty_fields].sum(axis=1)
        ).groupby("date")["_terminal_penalty"].sum().reindex(dates)
    else:
        economics_suffixes = (
            "grid_export_revenue", "grid_import_cost", "p2p_sale_revenue",
            "p2p_purchase_cost", "battery_degradation_cost", "unmet_load_kw",
            "renewable_curtailment_kw", "hourly_profit",
        )
        economics_fields = {
            f"{mg}_{suffix}" for mg in MICROGRIDS for suffix in economics_suffixes
        }
        if economics_fields.issubset(ordered.columns):
            cost = env.get("cost", {})
            unmet_rate = float(cost.get("unmet_load_cost_per_kwh", 0.0))
            curtailment_rate = float(
                cost.get("renewable_curtailment_cost_per_kwh", 0.0)
            )
            hourly_penalty = pd.Series(0.0, index=ordered.index, dtype=float)
            for mg in MICROGRIDS:
                profit_without_terminal = (
                    ordered[f"{mg}_grid_export_revenue"]
                    - ordered[f"{mg}_grid_import_cost"]
                    + ordered[f"{mg}_p2p_sale_revenue"]
                    - ordered[f"{mg}_p2p_purchase_cost"]
                    - ordered[f"{mg}_battery_degradation_cost"]
                    - ordered[f"{mg}_unmet_load_kw"] * unmet_rate
                    - ordered[f"{mg}_renewable_curtailment_kw"] * curtailment_rate
                )
                hourly_penalty += profit_without_terminal - ordered[f"{mg}_hourly_profit"]
            hourly_penalty = hourly_penalty.mask(hourly_penalty.abs() <= 1e-9, 0.0)
            terminal_penalty = ordered.assign(
                _terminal_penalty=hourly_penalty
            ).groupby("date")["_terminal_penalty"].sum().reindex(dates)
        else:
            terminal_equivalent = ess_equivalent + mill_equivalent + hvac_equivalent + ev_shortfall
            terminal_penalty = terminal_equivalent * penalty_rate
    total_cost = -pd.to_numeric(daily["algorithm_team_profit"], errors="coerce").mean()
    rows.extend([
        _metric("total_operating_cost_day", "Economics", "Total operating cost", "currency/day", total_cost, "lower"),
        _metric("terminal_task_penalty_day", "Economics", "Terminal task penalty", "currency/day", terminal_penalty.mean(), "lower"),
        _metric("nonterminal_operating_cost_day", "Economics", "Operating cost excluding terminal task penalty", "currency/day", total_cost - terminal_penalty.mean(), "lower"),
    ])
    return pd.DataFrame(rows)


def load_saved_run(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    root = artifact_directory(run_dir)
    daily_path = root / "test_daily_metrics.csv"
    config_path = root / "config_snapshot.yaml"
    trajectory_dir = root / "test_hourly_trajectories"
    if not daily_path.exists() or not config_path.exists() or not trajectory_dir.is_dir():
        raise FileNotFoundError(
            "Paper artifacts require test_daily_metrics.csv, config_snapshot.yaml, "
            "and test_hourly_trajectories/"
        )
    paths = sorted(trajectory_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError("Paper artifacts require saved hourly trajectory CSV files")
    hourly = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    daily = pd.read_csv(daily_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return hourly, daily, config


def load_saved_dr_reference(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the persisted deterministic DR reference without rerunning the environment."""

    root = artifact_directory(run_dir)
    daily_path = root / "dr_reference_daily_metrics.csv"
    trajectory_dir = root / "dr_reference_hourly_trajectories"
    paths = sorted(trajectory_dir.glob("*.csv")) if trajectory_dir.is_dir() else []
    if not daily_path.exists() or not paths:
        raise FileNotFoundError(
            "DR comparison figures require dr_reference_daily_metrics.csv and "
            "dr_reference_hourly_trajectories/"
        )
    return (
        pd.concat([pd.read_csv(path) for path in paths], ignore_index=True),
        pd.read_csv(daily_path),
    )


def calculate_dr_shift_metrics(
    algorithm_hourly: pd.DataFrame,
    reference_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate per-day load-shifting metrics relative to the saved rule baseline."""

    keys = ["date", "hour"]
    fields = [
        *(f"{mg}_flexible_load_kw" for mg in MICROGRIDS),
        *(f"{mg}_actual_load_kw" for mg in MICROGRIDS),
    ]
    _require_columns(algorithm_hourly, [*keys, *fields])
    _require_columns(reference_hourly, [*keys, *fields])
    algorithm = algorithm_hourly[[*keys, *fields]].copy()
    reference = reference_hourly[[*keys, *fields]].copy()
    merged = algorithm.merge(
        reference,
        on=keys,
        how="inner",
        suffixes=("_algorithm", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(algorithm) or len(merged) != len(reference):
        raise ValueError("Algorithm and DR-reference hourly results must contain identical dates/hours")
    rows: list[dict[str, float | str]] = []
    for date, frame in merged.groupby("date", sort=True):
        algorithm_flexible = sum(
            frame[f"{mg}_flexible_load_kw_algorithm"] for mg in MICROGRIDS
        )
        reference_flexible = sum(
            frame[f"{mg}_flexible_load_kw_reference"] for mg in MICROGRIDS
        )
        delta = algorithm_flexible - reference_flexible
        algorithm_actual = sum(
            frame[f"{mg}_actual_load_kw_algorithm"] for mg in MICROGRIDS
        )
        reference_actual = sum(
            frame[f"{mg}_actual_load_kw_reference"] for mg in MICROGRIDS
        )
        reference_peak = float(reference_actual.max())
        rows.append({
            "date": str(date),
            "shed_energy_kwh": float((-delta).clip(lower=0.0).sum()),
            "rebound_energy_kwh": float(delta.clip(lower=0.0).sum()),
            "net_flexible_energy_change_kwh": float(delta.sum()),
            "peak_load_reduction_kw": reference_peak - float(algorithm_actual.max()),
            "peak_load_reduction_pct": (
                100.0 * (reference_peak - float(algorithm_actual.max())) / reference_peak
                if reference_peak > 1e-12 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def write_run_paper_metrics(run_dir: str | Path) -> tuple[Path, Path]:
    root = artifact_directory(run_dir)
    hourly, daily, config = load_saved_run(root)
    metrics = calculate_paper_metrics(hourly, daily, config)
    csv_path = root / "paper_metrics.csv"
    json_path = root / "paper_metrics.json"
    metrics.to_csv(csv_path, index=False)
    payload = {
        "representative_day": select_representative_day(hourly),
        "test_days": int(hourly["date"].nunique()),
        "metrics": metrics.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def _algorithm_label(root: Path) -> str:
    for name in ("run_metadata.json", "test_summary.json"):
        path = root / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = payload.get("algorithm_display_name") or payload.get("algorithm")
        if label:
            return str(label)
    return root.name


def _save_figure(fig: Any, directory: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for extension in formats:
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, format=extension, dpi=180, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": ["Noto Sans SC", "Microsoft YaHei", "DejaVu Sans"],
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "grid.color": LIGHT_GREY,
        "grid.linewidth": 0.7, "legend.frameon": False,
    })


def _plot_dispatch(day: pd.DataFrame, mg: str, label: str, date: str) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    hour = day["hour"].to_numpy()
    positive = [
        ("Renewable used", day[f"{mg}_renewable_used_kw"].to_numpy(), OLIVE),
        ("ESS discharge", day[f"{mg}_battery_discharge_kw"].to_numpy(), BLUE),
        ("P2P import", day[f"{mg}_p2p_buy_kw"].to_numpy(), ORANGE),
        ("Grid import", day[f"{mg}_grid_import_kw"].to_numpy(), PINK),
    ]
    negative = [
        ("ESS charge", -day[f"{mg}_battery_charge_kw"].to_numpy(), "#76AAA6"),
        ("P2P export", -day[f"{mg}_p2p_sell_kw"].to_numpy(), "#D95A5A"),
        ("Grid export", -day[f"{mg}_grid_export_kw"].to_numpy(), "#E9C842"),
    ]
    bottom = np.zeros(len(day))
    for name, values, color in positive:
        ax.bar(hour, values, bottom=bottom, label=name, color=color, width=0.82)
        bottom += values
    bottom = np.zeros(len(day))
    for name, values, color in negative:
        ax.bar(hour, values, bottom=bottom, label=name, color=color, width=0.82)
        bottom += values
    ax.plot(hour, day[f"{mg}_actual_load_kw"], color=INK, marker="o", markersize=3,
            linewidth=1.8, linestyle="--", label="Actual load")
    ax.axhline(0.0, color=INK, linewidth=0.9)
    ax.set_title(f"Representative-day energy dispatch ({date}, {mg.upper()}, {label})")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (kW)")
    ax.set_xticks(range(24))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=4)
    fig.tight_layout()
    return fig


def _plot_soc(day: pd.DataFrame, env: Mapping[str, Any], label: str, date: str) -> Any:
    import matplotlib.pyplot as plt

    microgrids = {str(item["id"]): item for item in env["microgrids"]}
    tolerance = float(env["terminal_soc"]["tolerance"])
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.5), sharex=True)
    for ax, mg in zip(axes, MICROGRIDS, strict=True):
        battery = microgrids[mg]["battery"]
        initial = float(battery["initial_soc"])
        values = np.r_[initial, day[f"{mg}_soc"].to_numpy(dtype=float)]
        ax.plot(range(25), 100.0 * values, color=BLUE, marker="o", markersize=3.5,
                linewidth=2.0, label=mg.upper())
        if str(env["terminal_soc"].get("mode", "none")) != "none":
            ax.axhspan(
                100.0 * (initial - tolerance),
                100.0 * (initial + tolerance),
                color=GOLD, alpha=0.18, label="Terminal target band",
            )
        ax.axhline(100.0 * float(battery["soc_min"]), color=GREY, linestyle="--", linewidth=1.0)
        ax.axhline(100.0 * float(battery["soc_max"]), color=GREY, linestyle="--", linewidth=1.0)
        ax.set_ylabel("SOC (%)")
        ax.set_title(mg.upper(), loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="both")
        ax.legend(loc="best", ncols=2)
    axes[-1].set_xlabel("Hour boundary")
    axes[-1].set_xticks(range(25))
    fig.suptitle(f"Representative-day ESS state of charge ({date}, {label})", fontsize=14)
    fig.tight_layout()
    return fig


def _plot_p2p(day: pd.DataFrame, label: str, date: str) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.3), sharex=True)
    colors = [BLUE, ORANGE]
    for seller, ax in enumerate(axes, start=1):
        buyers = [buyer for buyer in range(1, 4) if buyer != seller]
        for color, buyer in zip(colors, buyers, strict=True):
            ax.plot(day["hour"], day[f"p2p_mg{seller}_to_mg{buyer}_kw"],
                    marker="o", markersize=3, linewidth=1.8, color=color,
                    label=f"MG{seller}→MG{buyer}")
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"MG{seller} outgoing P2P", loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="both")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Hour")
    axes[-1].set_xticks(range(24))
    fig.suptitle(f"Representative-day P2P power profiles ({date}, {label})", fontsize=14)
    fig.tight_layout()
    return fig


def _plot_dr_power(
    day: pd.DataFrame,
    reference: pd.DataFrame,
    label: str,
    date: str,
) -> Any:
    import matplotlib.pyplot as plt

    specifications = [
        ("MG1 mill", "mg1_mill_power_kw", "Mill power (kW)"),
        ("MG2 HVAC", "mg2_hvac_effective_signed_power_kw", "Signed HVAC power (kW)"),
        ("MG3 EV", "mg3_ev_effective_power_kw", "EV charging power (kW)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 9.0), sharex=True)
    hour = day["hour"].to_numpy()
    for row, (name, field, ylabel) in enumerate(specifications):
        axes[row, 0].plot(hour, reference[field], color=GREY, linestyle="--",
                          linewidth=1.8, label="Rule reference")
        axes[row, 0].plot(hour, day[field], color=BLUE, marker="o", markersize=3,
                          linewidth=1.8, label=label)
        axes[row, 0].set_title(name, loc="left", fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel(ylabel)
        axes[row, 0].legend(loc="best", ncols=2)

        delta = day[field].to_numpy(dtype=float) - reference[field].to_numpy(dtype=float)
        axes[row, 1].plot(hour, delta, color=INK, linewidth=1.5)
        axes[row, 1].fill_between(hour, 0.0, delta, where=delta < 0.0,
                                  color=BLUE, alpha=0.35, label="Shed / defer")
        axes[row, 1].fill_between(hour, 0.0, delta, where=delta > 0.0,
                                  color=GOLD, alpha=0.45, label="Rebound / advance")
        axes[row, 1].axhline(0.0, color=INK, linewidth=0.8)
        axes[row, 1].set_title(f"{label} − reference", loc="left", fontsize=10)
        axes[row, 1].set_ylabel("Power change (kW)")
        axes[row, 1].legend(loc="best", ncols=2)
        for ax in axes[row]:
            ax.grid(axis="both")
    for ax in axes[-1]:
        ax.set_xlabel("Hour")
        ax.set_xticks(range(24))
    fig.suptitle(f"Representative-day DR device power and signed change ({date})", fontsize=14)
    fig.tight_layout()
    return fig


def _plot_dr_load(
    day: pd.DataFrame,
    reference: pd.DataFrame,
    label: str,
    date: str,
) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.3), sharex=True)
    fields = [
        ("Aggregate actual load", "actual_load_kw", "Load (kW)"),
        ("Aggregate controllable load", "flexible_load_kw", "DR-device load (kW)"),
        ("External-grid net import", "grid", "Net grid import (kW)"),
    ]
    hour = day["hour"].to_numpy()
    for ax, (title, suffix, ylabel) in zip(axes, fields, strict=True):
        if suffix == "grid":
            algorithm_values = sum(
                day[f"{mg}_grid_import_kw"] - day[f"{mg}_grid_export_kw"] for mg in MICROGRIDS
            )
            reference_values = sum(
                reference[f"{mg}_grid_import_kw"] - reference[f"{mg}_grid_export_kw"]
                for mg in MICROGRIDS
            )
        else:
            algorithm_values = sum(day[f"{mg}_{suffix}"] for mg in MICROGRIDS)
            reference_values = sum(reference[f"{mg}_{suffix}"] for mg in MICROGRIDS)
        ax.plot(hour, reference_values, color=GREY, linestyle="--", linewidth=1.8,
                label="Rule reference")
        ax.plot(hour, algorithm_values, color=BLUE, marker="o", markersize=3,
                linewidth=1.8, label=label)
        ax.fill_between(hour, reference_values, algorithm_values,
                        where=np.asarray(algorithm_values) < np.asarray(reference_values),
                        color=BLUE, alpha=0.18, label="Reduction")
        ax.fill_between(hour, reference_values, algorithm_values,
                        where=np.asarray(algorithm_values) > np.asarray(reference_values),
                        color=GOLD, alpha=0.22, label="Increase")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="both")
        ax.legend(loc="best", ncols=4)
    axes[-1].set_xlabel("Hour")
    axes[-1].set_xticks(range(24))
    fig.suptitle(f"Representative-day system response versus rule reference ({date})", fontsize=14)
    fig.tight_layout()
    return fig


def _plot_dr_service(
    day: pd.DataFrame,
    reference: pd.DataFrame,
    env: Mapping[str, Any],
    label: str,
    date: str,
) -> Any:
    import matplotlib.pyplot as plt

    microgrids = {str(item["id"]): item for item in env["microgrids"]}
    mill = microgrids["mg1"]["dr"]
    hvac = microgrids["mg2"]["dr"]
    ev = microgrids["mg3"]["dr"]
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.8), sharex=True)

    axes[0].plot(day["hour"] + 1, day["mg1_silo_level_t"], color=BLUE, marker="o",
                 markersize=3, label=label)
    axes[0].axhspan(float(mill["silo_initial_t"]) - float(mill["terminal_tolerance_t"]),
                    float(mill["silo_initial_t"]) + float(mill["terminal_tolerance_t"]),
                    color=GOLD, alpha=0.18, label="Terminal target band")
    axes[0].plot(reference["hour"] + 1, reference["mg1_silo_level_t"], color=GREY,
                 linestyle="--", linewidth=2.0, zorder=4, label="Rule reference")
    axes[0].axhline(float(mill["silo_min_t"]), color=GREY, linestyle="--")
    axes[0].axhline(float(mill["silo_max_t"]), color=GREY, linestyle="--")
    axes[0].set_ylabel("Silo inventory (t)")
    axes[0].set_title("MG1 industrial mill service state", loc="left", fontsize=10, fontweight="bold")
    axes[0].legend(loc="best")

    occupied = day["mg2_occupied"].astype(float) > 0.5
    low = np.where(occupied, float(hvac["occupied_comfort_c"][0]), float(hvac["unoccupied_comfort_c"][0]))
    high = np.where(occupied, float(hvac["occupied_comfort_c"][1]), float(hvac["unoccupied_comfort_c"][1]))
    boundary_hour = day["hour"].to_numpy() + 1
    axes[1].fill_between(boundary_hour, low, high, color=GOLD, alpha=0.18, label="Comfort band")
    axes[1].plot(boundary_hour, reference["mg2_t_air_c"], color=GREY, linestyle="--",
                 label="Rule-reference indoor air")
    axes[1].plot(boundary_hour, day["mg2_t_air_c"], color=BLUE, marker="o", markersize=3,
                 label=f"{label} indoor air")
    axes[1].plot(boundary_hour, day["mg2_t_out_c"], color=GREY, linestyle="--", label="Outdoor")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_title("MG2 HVAC comfort state", loc="left", fontsize=10, fontweight="bold")
    axes[1].legend(loc="best", ncols=3)

    initial_ev = float(ev["initial_soc"])
    ev_soc = np.r_[initial_ev, day["mg3_ev_soc"].to_numpy(dtype=float)]
    reference_ev_soc = np.r_[initial_ev, reference["mg3_ev_soc"].to_numpy(dtype=float)]
    axes[2].plot(range(25), 100.0 * reference_ev_soc, color=GREY, linestyle="--",
                 label="Rule reference")
    axes[2].plot(range(25), 100.0 * ev_soc, color=BLUE, marker="o", markersize=3, label=label)
    axes[2].axhline(100.0 * float(ev["target_soc"]), color=ORANGE, linestyle="--", label="Departure target")
    axes[2].axvline(int(ev["departure_hour"]), color=INK, linestyle=":", label="Departure")
    axes[2].set_ylabel("EV SOC (%)")
    axes[2].set_title("MG3 EV charging service state", loc="left", fontsize=10, fontweight="bold")
    axes[2].legend(loc="best", ncols=3)
    axes[2].set_xlabel("Hour boundary")
    axes[2].set_xticks(range(25))
    for ax in axes:
        ax.grid(axis="both")
    fig.suptitle(f"Representative-day DR service states ({date}, {label})", fontsize=14)
    fig.tight_layout()
    return fig


def _plot_dr_test_set_summary(
    shift: pd.DataFrame,
    algorithm_hourly: pd.DataFrame,
    reference_daily: pd.DataFrame,
    env: Mapping[str, Any],
    label: str,
) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.7))
    axes[0].boxplot(
        [
            shift["shed_energy_kwh"],
            shift["rebound_energy_kwh"],
            shift["net_flexible_energy_change_kwh"],
        ],
        tick_labels=["Shed/defer", "Rebound/advance", "Net change"],
        patch_artist=True,
        boxprops={"facecolor": LIGHT_GREY, "edgecolor": INK},
        medianprops={"color": BLUE, "linewidth": 2},
    )
    axes[0].set_ylabel("Energy (kWh/day)")
    axes[0].set_title("Flexible-load shifting")

    axes[1].boxplot(
        [shift["peak_load_reduction_pct"].dropna()],
        tick_labels=[label],
        patch_artist=True,
        boxprops={"facecolor": "#DCEAF3", "edgecolor": INK},
        medianprops={"color": BLUE, "linewidth": 2},
    )
    axes[1].axhline(0.0, color=INK, linewidth=0.8)
    axes[1].set_ylabel("Peak reduction vs reference (%)")
    axes[1].set_title("Daily system peak")

    penalty_fields = [f"{mg}_terminal_constraint_penalty" for mg in MICROGRIDS]
    if set(penalty_fields).issubset(algorithm_hourly.columns):
        algorithm_penalty = algorithm_hourly[penalty_fields].sum(axis=1).groupby(
            algorithm_hourly["date"]
        ).sum()
    else:
        economics_suffixes = (
            "grid_export_revenue", "grid_import_cost", "p2p_sale_revenue",
            "p2p_purchase_cost", "battery_degradation_cost", "unmet_load_kw",
            "renewable_curtailment_kw", "hourly_profit",
        )
        economics_fields = {
            f"{mg}_{suffix}" for mg in MICROGRIDS for suffix in economics_suffixes
        }
        if economics_fields.issubset(algorithm_hourly.columns):
            cost = env.get("cost", {})
            unmet_rate = float(cost.get("unmet_load_cost_per_kwh", 0.0))
            curtailment_rate = float(
                cost.get("renewable_curtailment_cost_per_kwh", 0.0)
            )
            hourly_penalty = pd.Series(0.0, index=algorithm_hourly.index, dtype=float)
            for mg in MICROGRIDS:
                without_terminal = (
                    algorithm_hourly[f"{mg}_grid_export_revenue"]
                    - algorithm_hourly[f"{mg}_grid_import_cost"]
                    + algorithm_hourly[f"{mg}_p2p_sale_revenue"]
                    - algorithm_hourly[f"{mg}_p2p_purchase_cost"]
                    - algorithm_hourly[f"{mg}_battery_degradation_cost"]
                    - algorithm_hourly[f"{mg}_unmet_load_kw"] * unmet_rate
                    - algorithm_hourly[f"{mg}_renewable_curtailment_kw"] * curtailment_rate
                )
                hourly_penalty += without_terminal - algorithm_hourly[f"{mg}_hourly_profit"]
            algorithm_penalty = hourly_penalty.mask(
                hourly_penalty.abs() <= 1e-9, 0.0
            ).groupby(algorithm_hourly["date"]).sum()
        else:
            algorithm_penalty = pd.Series(np.nan, index=shift["date"], dtype=float)
    reference_penalty = pd.to_numeric(
        reference_daily.set_index("date").get("terminal_task_penalty", pd.Series(dtype=float)),
        errors="coerce",
    )
    penalty_data = [algorithm_penalty.dropna(), reference_penalty.dropna()]
    if all(len(values) for values in penalty_data):
        axes[2].boxplot(
            penalty_data,
            tick_labels=[label, "Rule reference"],
            patch_artist=True,
            boxprops={"facecolor": "#F6E8BE", "edgecolor": INK},
            medianprops={"color": ORANGE, "linewidth": 2},
        )
    else:
        axes[2].text(0.5, 0.5, "Terminal penalty not available\nin saved legacy fields",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_xticks([])
    axes[2].set_ylabel("Penalty (currency/day)")
    axes[2].set_title("Terminal service penalty")
    for ax in axes:
        ax.grid(axis="y")
    fig.suptitle("DR performance over all formal test days", fontsize=14)
    fig.tight_layout()
    return fig


def generate_paper_artifacts(
    run_dir: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Generate the paper figure group and metrics strictly from saved run files."""

    _configure_matplotlib()
    root = artifact_directory(run_dir)
    hourly, daily, config = load_saved_run(root)
    reference_hourly, reference_daily = load_saved_dr_reference(root)
    env = _environment_config(config)
    metrics = calculate_paper_metrics(hourly, daily, config)
    dr_shift_metrics = calculate_dr_shift_metrics(hourly, reference_hourly)
    representative = select_representative_day(hourly)
    day = hourly.loc[hourly["date"].astype(str) == representative].sort_values("hour")
    reference_day = reference_hourly.loc[
        reference_hourly["date"].astype(str) == representative
    ].sort_values("hour")
    if len(day) != 24:
        raise ValueError(f"Representative day {representative} must contain exactly 24 saved hourly rows")
    if len(reference_day) != 24:
        raise ValueError(
            f"DR reference for representative day {representative} must contain exactly 24 rows"
        )
    figure_dir = figure_directory(root)
    figure_dir.mkdir(parents=True, exist_ok=True)
    label = _algorithm_label(root)
    figures = [
        (f"{15 + index:02d}_representative_dispatch_{mg}", _plot_dispatch(day, mg, label, representative))
        for index, mg in enumerate(MICROGRIDS)
    ]
    figures.extend([
        ("18_representative_soc", _plot_soc(day, env, label, representative)),
        ("19_representative_p2p", _plot_p2p(day, label, representative)),
        ("20_representative_dr_power", _plot_dr_power(day, reference_day, label, representative)),
        ("21_representative_dr_load", _plot_dr_load(day, reference_day, label, representative)),
        ("22_representative_dr_service", _plot_dr_service(
            day, reference_day, env, label, representative,
        )),
        ("23_dr_test_set_summary", _plot_dr_test_set_summary(
            dr_shift_metrics, hourly, reference_daily, env, label,
        )),
    ])
    written: list[Path] = []
    for stem, figure in figures:
        written.extend(_save_figure(figure, figure_dir, stem, formats))

    metrics.to_csv(root / "paper_metrics.csv", index=False)
    dr_shift_metrics.to_csv(root / "dr_shift_metrics.csv", index=False)
    payload = {
        "representative_day": representative,
        "representative_day_selection": "nearest centroid of saved base load, available renewable energy, and outdoor temperature",
        "test_days": int(hourly["date"].nunique()),
        "algorithm_display_name": label,
        "metrics": metrics.to_dict(orient="records"),
    }
    (root / "paper_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return written


def _relative_improvement(baseline: float, candidate: float, direction: str) -> float:
    if direction == "target_1":
        base_gap = abs(baseline - 100.0)
        candidate_gap = abs(candidate - 100.0)
        return 0.0 if base_gap <= 1e-12 and candidate_gap <= 1e-12 else (
            np.nan if base_gap <= 1e-12 else 100.0 * (base_gap - candidate_gap) / base_gap
        )
    denominator = abs(baseline)
    if denominator <= 1e-12:
        return 0.0 if abs(candidate) <= 1e-12 else np.nan
    sign = 1.0 if direction == "higher" else -1.0
    return 100.0 * sign * (candidate - baseline) / denominator


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "N/A" if not np.isfinite(value) else f"{float(value):.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body]) + "\n"


def _comparison_table_figure(
    table: pd.DataFrame,
    baseline_label: str,
    candidate_label: str,
) -> Any:
    import matplotlib.pyplot as plt

    display = table[[
        "category", "metric", "unit", baseline_label, candidate_label,
        "absolute_change", "relative_improvement_pct",
    ]].copy()

    def number(value: Any) -> str:
        return "—" if not np.isfinite(float(value)) else f"{float(value):,.3f}"

    for column in (baseline_label, candidate_label, "absolute_change"):
        display[column] = display[column].map(number)
    display["relative_improvement_pct"] = display["relative_improvement_pct"].map(
        lambda value: "—" if not np.isfinite(float(value)) else f"{float(value):+.2f}%"
    )
    display.columns = [
        "Category", "Metric", "Unit", baseline_label, candidate_label,
        "Change", "Directional improvement",
    ]

    height = max(8.0, 0.31 * len(display) + 2.0)
    fig, ax = plt.subplots(figsize=(16.0, height))
    ax.axis("off")
    ax.set_title(
        f"Paper performance metrics: {baseline_label} vs {candidate_label}",
        fontsize=16,
        fontweight="bold",
        color=INK,
        pad=18,
    )
    rendered = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        colWidths=[0.13, 0.28, 0.11, 0.12, 0.13, 0.10, 0.13],
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.0, 0.04, 1.0, 0.90],
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8.3)
    category_colors = {
        category: color for category, color in zip(
            display["Category"].drop_duplicates(),
            ("#EEF4F7", "#F8F2E8", "#EEF3E8", "#F5ECEF", "#F0EEF6", "#EDF3F3"),
            strict=False,
        )
    }
    for (row, column), cell in rendered.get_celld().items():
        cell.set_edgecolor("#CDD4D8")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            category = str(display.iloc[row - 1, 0])
            cell.set_facecolor(category_colors.get(category, "white"))
            if column in (0, 1):
                cell.get_text().set_ha("left")
    source_notes = [
        str(note) for note in table.get("source_note", pd.Series(dtype=str)).dropna().unique()
        if str(note).strip()
    ]
    footer = (
        "All metrics use the full saved test set. Positive directional improvement favors the candidate; "
        "— means undefined from a zero baseline."
    )
    if source_notes:
        footer += "\n" + " ".join(source_notes)
    fig.text(
        0.5,
        0.012,
        footer,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#59636A",
    )
    fig.subplots_adjust(top=0.94, bottom=0.055, left=0.02, right=0.98)
    return fig


def write_comparison_tables(
    baseline_run: str | Path,
    candidate_run: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    baseline_root = artifact_directory(baseline_run)
    candidate_root = artifact_directory(candidate_run)
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_hourly, baseline_daily, baseline_config = load_saved_run(baseline_root)
    candidate_hourly, candidate_daily, candidate_config = load_saved_run(candidate_root)
    baseline_dates = set(baseline_hourly["date"].astype(str))
    candidate_dates = set(candidate_hourly["date"].astype(str))
    if baseline_dates != candidate_dates:
        raise ValueError("Algorithm comparison requires identical saved test dates")
    baseline = calculate_paper_metrics(baseline_hourly, baseline_daily, baseline_config)
    candidate = calculate_paper_metrics(candidate_hourly, candidate_daily, candidate_config)
    baseline_label = _algorithm_label(baseline_root)
    candidate_label = _algorithm_label(candidate_root)
    table = baseline[["metric_id", "category", "metric", "unit", "preferred_direction", "value"]].rename(
        columns={"value": baseline_label}
    ).merge(candidate[["metric_id", "value"]].rename(columns={"value": candidate_label}), on="metric_id", validate="one_to_one")
    table["absolute_change"] = table[candidate_label] - table[baseline_label]
    table["relative_improvement_pct"] = [
        _relative_improvement(float(base), float(candidate_value), str(direction))
        for base, candidate_value, direction in zip(
            table[baseline_label], table[candidate_label], table["preferred_direction"], strict=True
        )
    ]
    table["source_note"] = ""
    exact_hvac_fields = {"mg2_terminal_air_gap_c", "mg2_terminal_mass_gap_c"}
    if not (
        exact_hvac_fields.issubset(baseline_hourly.columns)
        and exact_hvac_fields.issubset(candidate_hourly.columns)
    ):
        table.loc[
            table["metric_id"] == "hvac_terminal_joint_success_rate", "source_note"
        ] = (
            "Legacy-run note: joint HVAC terminal success is reconstructed from saved "
            "temperatures because exact terminal gaps were not recorded."
        )
    columns = [
        "metric_id", "category", "metric", "unit", baseline_label, candidate_label,
        "absolute_change", "relative_improvement_pct", "preferred_direction", "source_note",
    ]
    table = table[columns]
    csv_path = output_root / "performance_comparison.csv"
    markdown_path = output_root / "performance_comparison.md"
    png_path = output_root / "performance_comparison.png"
    pdf_path = output_root / "performance_comparison.pdf"
    table.to_csv(csv_path, index=False)
    markdown_columns = [
        "category", "metric", "unit", baseline_label, candidate_label,
        "absolute_change", "relative_improvement_pct", "source_note",
    ]
    markdown_path.write_text(_markdown_table(table[markdown_columns]), encoding="utf-8")
    figure = _comparison_table_figure(table, baseline_label, candidate_label)
    figure.savefig(png_path, format="png", dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return [csv_path, markdown_path, png_path, pdf_path]
