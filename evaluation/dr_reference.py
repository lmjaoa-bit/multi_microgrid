from __future__ import annotations

from pathlib import Path
from experiments.layout import artifact_directory
from typing import Any, Mapping, Sequence

import numpy as np

from envs.multi_microgrid_env import MultiMicrogridEnv
from utils.metrics_collector import write_csv, write_json


class DeterministicDRReferencePolicy:
    """Rule-based DR reference with neutral ESS dispatch.

    MG1 runs at the constant duty required to balance downstream material
    consumption, MG2 requests zero HVAC power and lets the physical comfort
    projection apply the minimum necessary intervention, and MG3 follows the
    business-as-usual rule of charging immediately until its target SOC.
    """

    def __init__(self, env: MultiMicrogridEnv) -> None:
        if env.action_dim != 2:
            raise ValueError("DR reference requires the two-channel ESS/DR action interface")
        self.env = env
        mill = env.cfg_by_id["mg1"]["dr"]
        duty = float(mill["downstream_consumption_tph"]) / float(
            mill["rated_production_tph"]
        )
        self._mill_raw = float(np.clip(2.0 * duty - 1.0, -1.0, 1.0))

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "deterministic_rule_based_dr_reference",
            "ess": "neutral dispatch (raw action 0)",
            "mg1": "constant mill duty balancing downstream consumption",
            "mg2": "zero requested HVAC power with minimum comfort projection",
            "mg3": "immediate charging until departure SOC target",
        }

    def __call__(self, observations: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        del observations
        # +1 requests the currently feasible EV maximum. When no cohort is
        # connected the physical EV projector maps the same request to 0 kW.
        return {
            "mg1": np.asarray([0.0, self._mill_raw], dtype=np.float32),
            "mg2": np.asarray([0.0, 0.0], dtype=np.float32),
            "mg3": np.asarray([0.0, 1.0], dtype=np.float32),
        }


def evaluate_dr_reference(
    env_config: Mapping[str, Any],
    *,
    split: str,
    days: list[str],
    seed: int,
    forecast_noise: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    from evaluation.evaluate_matd3 import evaluate_policy_on_days

    env = MultiMicrogridEnv(dict(env_config), split=split)
    policy = DeterministicDRReferencePolicy(env)
    summary, daily, hourly = evaluate_policy_on_days(
        env,
        days,
        policy,
        seed=int(seed),
        deterministic_environment=True,
        forecast_noise=bool(forecast_noise),
        capture_hourly=True,
    )
    summary = {
        **summary,
        "policy_definition": policy.definition,
        "deterministic_policy": True,
    }
    return summary, daily, hourly


def _daily_rows(
    daily: Sequence[Mapping[str, Any]],
    hourly_by_day: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in daily:
        date = str(day["date"])
        hourly = hourly_by_day[date]
        peak_total = max(
            sum(float(row[f"{mg}_actual_load_kw"]) for mg in ("mg1", "mg2", "mg3"))
            for row in hourly
        )
        energies = day["energy_statistics_by_microgrid"]
        economics = day["economic_breakdown_by_microgrid"]
        rows.append({
            "date": date,
            "test_seed": int(day["test_seed"]),
            "reference_team_profit": float(day["team_profit"]),
            "total_actual_load_kwh": float(sum(
                values["actual_load_kwh"] for values in energies.values()
            )),
            "total_flexible_load_kwh": float(sum(
                values["flexible_load_kwh"] for values in energies.values()
            )),
            "peak_total_load_kw": float(peak_total),
            "total_grid_import_kwh": float(sum(
                values["grid_import_kwh"] for values in energies.values()
            )),
            "total_grid_export_kwh": float(sum(
                values["grid_export_kwh"] for values in energies.values()
            )),
            "terminal_task_penalty": float(sum(
                values["terminal_soc_penalty"] for values in economics.values()
            )),
        })
    return rows


def save_dr_reference_results(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    daily: Sequence[Mapping[str, Any]],
    hourly_by_day: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[Path]:
    root = artifact_directory(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "dr_reference_summary.json"
    daily_path = root / "dr_reference_daily_metrics.csv"
    hourly_dir = root / "dr_reference_hourly_trajectories"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, dict(summary))
    write_csv(daily_path, _daily_rows(daily, hourly_by_day))
    for day, rows in hourly_by_day.items():
        write_csv(hourly_dir / f"{day}.csv", list(rows))
    return [summary_path, daily_path, hourly_dir]


def evaluate_and_save_dr_reference(
    env_config: Mapping[str, Any],
    *,
    split: str,
    days: list[str],
    seed: int,
    forecast_noise: bool,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Run the deterministic reference on the formal-test cases and persist it."""

    summary, daily, hourly_by_day = evaluate_dr_reference(
        env_config,
        split=split,
        days=days,
        seed=seed,
        forecast_noise=forecast_noise,
    )
    if output_dir is not None:
        save_dr_reference_results(output_dir, summary, daily, hourly_by_day)
    return summary, daily, hourly_by_day
