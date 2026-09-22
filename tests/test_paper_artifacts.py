from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.paper_artifacts import (
    calculate_dr_shift_metrics,
    calculate_paper_metrics,
    generate_paper_artifacts,
    select_representative_day,
    write_comparison_tables,
)
from evaluation.plotting import generate_figures


def _representative_fixture(policy_offset: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for date, base, renewable, temperature in (
        ("2018-01-01", 100.0, 100.0, 10.0),
        ("2018-01-02", 200.0, 200.0, 20.0),
        ("2018-01-03", 310.0, 310.0, 31.0),
    ):
        for hour in range(24):
            row: dict[str, float | int | str] = {
                "date": date,
                "hour": hour,
                "mg2_t_out_c": temperature,
                "policy_dependent_value": policy_offset * (hour + 1),
            }
            for mg in ("mg1", "mg2", "mg3"):
                row[f"{mg}_base_load_kw"] = base / 72.0
                row[f"{mg}_renewable_available_kw"] = renewable / 72.0
            rows.append(row)
    return pd.DataFrame(rows)


def _environment_config() -> dict:
    return {
        "terminal_soc": {"tolerance": 0.03, "penalty_per_kwh": 1.5},
        "cost": {"unmet_load_cost_per_kwh": 20.0, "renewable_curtailment_cost_per_kwh": 0.02},
        "microgrids": [
            {
                "id": "mg1",
                "battery": {"capacity_kwh": 100.0, "initial_soc": 0.5,
                            "soc_min": 0.1, "soc_max": 0.9},
                "dr": {
                    "type": "industrial_mill", "rated_power_kw": 10.0,
                    "rated_production_tph": 1.0, "downstream_consumption_tph": 0.5,
                    "silo_initial_t": 5.0, "silo_min_t": 1.0, "silo_max_t": 9.0,
                    "terminal_tolerance_t": 0.2,
                },
            },
            {
                "id": "mg2",
                "battery": {"capacity_kwh": 100.0, "initial_soc": 0.5,
                            "soc_min": 0.1, "soc_max": 0.9},
                "dr": {
                    "type": "commercial_hvac", "c_air_kwh_per_c": 10.0,
                    "c_mass_kwh_per_c": 100.0, "cop_heat": 2.0, "cop_cool": 2.0,
                    "terminal_air_tolerance_c": 1.0,
                    "terminal_mass_tolerance_c": 0.5,
                    "occupied_comfort_c": [21.0, 25.0],
                    "unoccupied_comfort_c": [16.0, 30.0],
                },
            },
            {
                "id": "mg3",
                "battery": {"capacity_kwh": 100.0, "initial_soc": 0.5,
                            "soc_min": 0.1, "soc_max": 0.9},
                "dr": {
                    "type": "ev_charging", "fleet_capacity_kwh": 100.0,
                    "initial_soc": 0.5, "target_soc": 0.8,
                    "departure_hour": 7, "available_start_hour": 0,
                },
            },
        ],
    }


def _hourly_fixture() -> pd.DataFrame:
    rows = []
    for hour in range(24):
        row: dict[str, float | int | str] = {
            "date": "2018-01-01", "hour": hour,
            "mg1_soc": 0.55, "mg2_soc": 0.50, "mg3_soc": 0.47,
            "mg1_silo_level_t": 5.1, "mg1_mill_production_tph": 0.5,
            "mg2_t_air_c": 23.0, "mg2_t_mass_c": 23.0,
            "mg2_t_out_c": 23.0, "mg2_occupied": int(7 <= hour <= 18),
            "mg3_ev_soc": 0.8 if hour >= 6 else 0.5 + 0.05 * (hour + 1),
            "mg3_ev_effective_power_kw": 10.0 if hour <= 6 else 0.0,
        }
        for mg in ("mg1", "mg2", "mg3"):
            row.update({
                f"{mg}_battery_charge_kw": 1.0,
                f"{mg}_battery_discharge_kw": 1.0,
                f"{mg}_dr_projected": 0,
                f"{mg}_dr_supply_infeasible": 0,
                f"{mg}_flexible_load_kw": 2.0,
                f"{mg}_base_load_kw": 10.0,
                f"{mg}_renewable_available_kw": 5.0,
                f"{mg}_renewable_used_kw": 5.0,
                f"{mg}_actual_load_kw": 12.0,
                f"{mg}_grid_import_kw": 3.0,
                f"{mg}_grid_export_kw": 0.0,
                f"{mg}_p2p_buy_kw": 2.0,
                f"{mg}_p2p_sell_kw": 1.0,
            })
        row.update({
            "mg1_mill_power_kw": 2.0,
            "mg2_hvac_effective_signed_power_kw": 2.0,
            "p2p_mg1_to_mg2_kw": 1.0,
            "p2p_mg1_to_mg3_kw": 0.0,
            "p2p_mg2_to_mg1_kw": 0.0,
            "p2p_mg2_to_mg3_kw": 1.0,
            "p2p_mg3_to_mg1_kw": 1.0,
            "p2p_mg3_to_mg2_kw": 0.0,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _write_run_fixture(root: Path, *, display_name: str, cost: float) -> None:
    trajectory = root / "test_hourly_trajectories"
    trajectory.mkdir(parents=True)
    _hourly_fixture().to_csv(trajectory / "2018-01-01.csv", index=False)
    reference_trajectory = root / "dr_reference_hourly_trajectories"
    reference_trajectory.mkdir(parents=True)
    reference = _hourly_fixture()
    for mg in ("mg1", "mg2", "mg3"):
        reference[f"{mg}_flexible_load_kw"] = 3.0
        reference[f"{mg}_actual_load_kw"] = 13.0
        reference[f"{mg}_grid_import_kw"] = 4.0
    reference["mg1_mill_power_kw"] = 3.0
    reference["mg2_hvac_effective_signed_power_kw"] = 3.0
    reference["mg3_ev_effective_power_kw"] = 3.0
    reference.to_csv(reference_trajectory / "2018-01-01.csv", index=False)
    pd.DataFrame({"date": ["2018-01-01"], "algorithm_team_profit": [-cost]}).to_csv(
        root / "test_daily_metrics.csv", index=False
    )
    pd.DataFrame({
        "date": ["2018-01-01"],
        "reference_team_profit": [-cost - 10.0],
        "terminal_task_penalty": [0.0],
    }).to_csv(root / "dr_reference_daily_metrics.csv", index=False)
    (root / "dr_reference_summary.json").write_text(
        '{"policy_definition": {"name": "deterministic_rule_based_dr_reference"}}',
        encoding="utf-8",
    )
    (root / "config_snapshot.yaml").write_text(
        yaml.safe_dump({"environment": _environment_config()}, sort_keys=False), encoding="utf-8"
    )
    (root / "run_metadata.json").write_text(
        f'{{"algorithm_display_name": "{display_name}"}}', encoding="utf-8"
    )
    (root / "test_summary.json").write_text(
        f'{{"algorithm_display_name": "{display_name}", "comparisons": {{}}}}',
        encoding="utf-8",
    )


def test_representative_day_uses_only_policy_independent_conditions() -> None:
    first = _representative_fixture(policy_offset=1.0)
    second = _representative_fixture(policy_offset=10_000.0)

    assert select_representative_day(first) == "2018-01-02"
    assert select_representative_day(second) == "2018-01-02"


def test_paper_metrics_separate_soc_and_device_terminal_service() -> None:
    hourly = _hourly_fixture()
    hourly.loc[hourly["hour"].isin([0, 8]), "mg3_dr_projected"] = 1
    daily = pd.DataFrame({"date": ["2018-01-01"], "algorithm_team_profit": [-100.0]})

    metrics = calculate_paper_metrics(hourly, daily, deepcopy(_environment_config()))
    values = dict(zip(metrics["metric_id"], metrics["value"], strict=True))

    assert np.isclose(values["ess_terminal_success_microgrid_day"], 200.0 / 3.0)
    assert np.isclose(values["ess_terminal_excess_kwh_day"], 2.0)
    assert np.isclose(values["ess_throughput_kwh_day"], 144.0)
    assert np.isclose(values["ess_equivalent_cycles_day"], 0.24)
    assert np.isclose(values["mill_production_completion_rate"], 100.0)
    assert np.isclose(values["mill_terminal_success_rate"], 100.0)
    assert np.isclose(values["hvac_occupied_comfort_rate"], 100.0)
    assert np.isclose(values["hvac_terminal_joint_success_rate"], 100.0)
    assert np.isclose(values["ev_departure_target_success_rate"], 100.0)
    assert np.isclose(values["ev_departure_shortfall_kwh_day"], 0.0)
    assert np.isclose(values["mg3_dr_projection_rate"], 100.0 / 7.0)
    assert np.isclose(values["terminal_task_penalty_day"], 3.0)
    assert np.isclose(values["nonterminal_operating_cost_day"], 97.0)


def test_paper_metrics_prefer_saved_hvac_terminal_gaps() -> None:
    hourly = _hourly_fixture()
    hourly["mg2_terminal_air_gap_c"] = 0.0
    hourly["mg2_terminal_mass_gap_c"] = 0.0
    hourly.loc[hourly["hour"] == 23, "mg2_terminal_air_gap_c"] = 2.0
    daily = pd.DataFrame({"date": ["2018-01-01"], "algorithm_team_profit": [-100.0]})

    metrics = calculate_paper_metrics(hourly, daily, deepcopy(_environment_config()))
    values = dict(zip(metrics["metric_id"], metrics["value"], strict=True))

    assert values["hvac_terminal_joint_success_rate"] == 0.0
    assert np.isclose(values["terminal_task_penalty_day"], 10.5)


def test_paper_metrics_prefer_explicit_saved_terminal_penalty() -> None:
    hourly = _hourly_fixture()
    for mg in ("mg1", "mg2", "mg3"):
        hourly[f"{mg}_terminal_constraint_penalty"] = 0.0
    hourly.loc[hourly["hour"] == 23, "mg2_terminal_constraint_penalty"] = 12.3
    daily = pd.DataFrame({"date": ["2018-01-01"], "algorithm_team_profit": [-100.0]})

    metrics = calculate_paper_metrics(hourly, daily, deepcopy(_environment_config()))
    values = dict(zip(metrics["metric_id"], metrics["value"], strict=True))

    assert np.isclose(values["terminal_task_penalty_day"], 12.3)
    assert np.isclose(values["nonterminal_operating_cost_day"], 87.7)


def test_paper_metrics_reconstruct_terminal_penalty_from_saved_economics() -> None:
    hourly = _hourly_fixture()
    for mg in ("mg1", "mg2", "mg3"):
        for suffix in (
            "grid_export_revenue", "grid_import_cost", "p2p_sale_revenue",
            "p2p_purchase_cost", "battery_degradation_cost", "unmet_load_kw",
            "renewable_curtailment_kw", "hourly_profit",
        ):
            hourly[f"{mg}_{suffix}"] = 0.0
    hourly.loc[hourly["hour"] == 23, "mg2_hourly_profit"] = -12.3
    daily = pd.DataFrame({"date": ["2018-01-01"], "algorithm_team_profit": [-100.0]})

    metrics = calculate_paper_metrics(hourly, daily, deepcopy(_environment_config()))
    values = dict(zip(metrics["metric_id"], metrics["value"], strict=True))

    assert np.isclose(values["terminal_task_penalty_day"], 12.3)


def test_dr_shift_metrics_measure_signed_change_against_saved_reference() -> None:
    algorithm = _hourly_fixture()
    reference = algorithm.copy()
    reference.loc[:, [f"{mg}_flexible_load_kw" for mg in ("mg1", "mg2", "mg3")]] = 2.0
    algorithm.loc[:, [f"{mg}_flexible_load_kw" for mg in ("mg1", "mg2", "mg3")]] = 2.0
    algorithm.loc[algorithm["hour"] == 0, "mg1_flexible_load_kw"] = 1.0
    algorithm.loc[algorithm["hour"] == 1, "mg1_flexible_load_kw"] = 3.0
    algorithm["mg1_actual_load_kw"] = algorithm["mg1_base_load_kw"] + algorithm["mg1_flexible_load_kw"]

    metrics = calculate_dr_shift_metrics(algorithm, reference)

    assert metrics.loc[0, "shed_energy_kwh"] == 1.0
    assert metrics.loc[0, "rebound_energy_kwh"] == 1.0
    assert metrics.loc[0, "net_flexible_energy_change_kwh"] == 0.0


def test_generate_paper_artifacts_writes_nine_saved_result_figures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run_fixture(run_dir, display_name="MATD3", cost=100.0)

    paths = generate_paper_artifacts(run_dir, formats=("png",))

    assert {path.stem for path in paths} == {
        "15_representative_dispatch_mg1",
        "16_representative_dispatch_mg2",
        "17_representative_dispatch_mg3",
        "18_representative_soc",
        "19_representative_p2p",
        "20_representative_dr_power",
        "21_representative_dr_load",
        "22_representative_dr_service",
        "23_dr_test_set_summary",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert (run_dir / "paper_metrics.csv").exists()
    assert (run_dir / "paper_metrics.json").exists()


def test_comparison_tables_use_all_test_metrics_and_write_four_formats(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    _write_run_fixture(baseline, display_name="MATD3", cost=100.0)
    _write_run_fixture(candidate, display_name="MATD3-Transformer", cost=90.0)

    paths = write_comparison_tables(baseline, candidate, output)

    assert {path.suffix for path in paths} == {".csv", ".md", ".png", ".pdf"}
    assert (output / "performance_comparison.png").stat().st_size > 0
    table = pd.read_csv(output / "performance_comparison.csv")
    cost = table.loc[table["metric_id"] == "total_operating_cost_day"].iloc[0]
    assert cost["MATD3"] == 100.0
    assert cost["MATD3-Transformer"] == 90.0
    assert cost["relative_improvement_pct"] == 10.0
    hvac = table.loc[table["metric_id"] == "hvac_terminal_joint_success_rate"].iloc[0]
    assert "Legacy-run note" in hvac["source_note"]


def test_comparison_markdown_does_not_require_optional_tabulate(
    tmp_path: Path, monkeypatch,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    _write_run_fixture(baseline, display_name="MATD3", cost=100.0)
    _write_run_fixture(candidate, display_name="MATD3-Transformer", cost=90.0)

    def unavailable(*args, **kwargs):
        raise ImportError("tabulate is unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", unavailable)

    paths = write_comparison_tables(baseline, candidate, output)

    assert (output / "performance_comparison.md") in paths
    assert "Total operating cost" in (output / "performance_comparison.md").read_text(encoding="utf-8")


def test_standard_figure_generation_includes_paper_group(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run_fixture(run_dir, display_name="MATD3", cost=100.0)

    paths = generate_figures(run_dir, formats=("png",))

    assert len(paths) == 24
    assert (run_dir / "figures" / "22_representative_dr_service.png") in paths


def test_comparison_cli_builds_table_from_explicit_runs(tmp_path: Path) -> None:
    from scripts.build_paper_comparison import main

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    _write_run_fixture(baseline, display_name="MATD3", cost=100.0)
    _write_run_fixture(candidate, display_name="MATD3-Transformer", cost=90.0)

    result = main([
        "--baseline-run", str(baseline),
        "--candidate-run", str(candidate),
        "--output-dir", str(output),
    ])

    assert result == 0
    assert (output / "performance_comparison.png").exists()


def test_paper_metrics_include_hvac_late_power_p95() -> None:
    hourly = _hourly_fixture()
    hourly.loc[hourly["hour"].isin([22, 23]), "mg2_hvac_effective_signed_power_kw"] = [4.0, -6.0]
    daily = pd.DataFrame({"date": ["2018-01-01"], "algorithm_team_profit": [-100.0]})

    metrics = calculate_paper_metrics(hourly, daily, deepcopy(_environment_config()))
    values = dict(zip(metrics["metric_id"], metrics["value"], strict=True))

    assert "hvac_late_power_p95_kw" in values
    assert np.isclose(values["hvac_late_power_p95_kw"], 5.9, rtol=0.1)
