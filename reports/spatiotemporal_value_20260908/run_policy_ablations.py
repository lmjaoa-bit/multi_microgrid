"""Run paired policy ablations for temporal/spatial storage diagnostics.

This is analysis-only: it does not edit the algorithm, environment, or shared
evaluation pipeline.  Every scenario uses the same checkpoint, test dates,
per-day seeds, and deterministic forecast errors.
"""
from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.evaluate_matd3 import evaluate_policy_on_days
from utils.torch_runtime import configure_torch_runtime


RUN_DIR = ROOT / "outputs/matd3_dispatch/runs/MATD3_2026-09-08_15-46-18_seed42"
CHECKPOINT = RUN_DIR / "checkpoints/best_model.pt"
SNAPSHOT = RUN_DIR / "config_snapshot.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent / "policy_ablations"
TEST_SEED = 42
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_SAMPLES = 10_000


SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "no_p2p_local_storage",
        "label": "关闭全部P2P，仅本地储能",
        "p2p_enabled": False,
        "ess_p2p_enabled": True,
        "mask_peer_future": False,
    },
    {
        "id": "renewable_p2p_only",
        "label": "仅可再生能源P2P，禁止储能外送",
        "p2p_enabled": True,
        "ess_p2p_enabled": False,
        "mask_peer_future": False,
    },
    {
        "id": "full_without_peer_future",
        "label": "完整ESS-P2P，但不提供同行未来净负荷",
        "p2p_enabled": True,
        "ess_p2p_enabled": True,
        "mask_peer_future": True,
    },
    {
        "id": "full_spatiotemporal",
        "label": "完整时空协同",
        "p2p_enabled": True,
        "ess_p2p_enabled": True,
        "mask_peer_future": False,
    },
)


def _sum_section(day: Mapping[str, Any], section: str, field: str) -> float:
    return float(sum(values[field] for values in day[section].values()))


def _daily_row(scenario: Mapping[str, Any], day: Mapping[str, Any]) -> dict[str, Any]:
    economics = day["economic_breakdown_by_microgrid"]
    energy = day["energy_statistics_by_microgrid"]
    available = _sum_section(day, "energy_statistics_by_microgrid", "renewable_available_kwh")
    used = _sum_section(day, "energy_statistics_by_microgrid", "renewable_used_kwh")
    row = {
        "scenario_id": scenario["id"],
        "scenario_label": scenario["label"],
        "date": day["date"],
        "test_seed": day["test_seed"],
        "system_energy_cost": -float(day["team_profit"]),
        "team_profit": float(day["team_profit"]),
        "grid_import_cost": float(sum(x["grid_import_cost"] for x in economics.values())),
        "grid_export_revenue": float(sum(x["grid_export_revenue"] for x in economics.values())),
        "battery_degradation_cost": float(sum(x["battery_degradation_cost"] for x in economics.values())),
        "renewable_curtailment_cost": float(sum(x["renewable_curtailment_cost"] for x in economics.values())),
        "grid_import_kwh": float(sum(x["grid_import_kwh"] for x in energy.values())),
        "grid_export_kwh": float(sum(x["grid_export_kwh"] for x in energy.values())),
        "battery_charge_kwh": float(sum(x["battery_charge_kwh"] for x in energy.values())),
        "battery_discharge_kwh": float(sum(x["battery_discharge_kwh"] for x in energy.values())),
        "battery_throughput_kwh": float(sum(x["battery_throughput_kwh"] for x in energy.values())),
        "renewable_curtailment_kwh": float(sum(x["renewable_curtailment_kwh"] for x in energy.values())),
        "renewable_utilization_pct": 100.0 * used / available if available > 1e-12 else float("nan"),
        "p2p_total_energy_kwh": float(day["p2p_statistics"]["total_energy_kwh"]),
        "renewable_p2p_energy_kwh": float(sum(x["renewable_p2p_sell_kwh"] for x in energy.values())),
        "ess_p2p_energy_kwh": float(sum(x["ess_p2p_sell_kwh"] for x in energy.values())),
        "ess_p2p_offer_kwh": float(sum(
            day["p2p_statistics"]["ess_offer_energy_by_microgrid_kwh"].values()
        )),
        "ess_offer_acceptance_pct": 100.0 * float(
            day["p2p_statistics"]["ess_offer_acceptance_rate"]
        ),
        "feasible_day": bool(day["constraint_statistics"]["feasible_episode"]),
    }
    for mg in sorted(energy):
        mg_economics = economics[mg]
        row.update({
            f"{mg}_realized_cost": -float(day["profit_by_microgrid"][mg]),
            f"{mg}_external_resource_cost": float(
                mg_economics["grid_import_cost"]
                - mg_economics["grid_export_revenue"]
                + mg_economics["battery_degradation_cost"]
                + mg_economics["unmet_load_cost"]
                + mg_economics["renewable_curtailment_cost"]
            ),
            f"{mg}_p2p_net_revenue": float(mg_economics["p2p_net_revenue"]),
            f"{mg}_grid_import_kwh": float(energy[mg]["grid_import_kwh"]),
            f"{mg}_grid_export_kwh": float(energy[mg]["grid_export_kwh"]),
            f"{mg}_battery_throughput_kwh": float(energy[mg]["battery_throughput_kwh"]),
            f"{mg}_p2p_buy_kwh": float(energy[mg]["p2p_buy_kwh"]),
            f"{mg}_p2p_sell_kwh": float(energy[mg]["p2p_sell_kwh"]),
        })
    return row


def _bootstrap_mean_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = values.size
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 1000):
        stop = min(start + 1000, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _scenario_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    numeric = (
        "system_energy_cost", "team_profit", "grid_import_cost", "grid_export_revenue",
        "battery_degradation_cost", "renewable_curtailment_cost", "grid_import_kwh",
        "grid_export_kwh", "battery_charge_kwh", "battery_discharge_kwh",
        "battery_throughput_kwh", "renewable_curtailment_kwh",
        "renewable_utilization_pct", "p2p_total_energy_kwh", "renewable_p2p_energy_kwh",
        "ess_p2p_energy_kwh", "ess_p2p_offer_kwh", "ess_offer_acceptance_pct",
    )
    agents = sorted(key.removesuffix("_realized_cost") for key in rows[0] if key.endswith("_realized_cost"))
    offer_mean = _mean(rows, "ess_p2p_offer_kwh")
    accepted_mean = _mean(rows, "ess_p2p_energy_kwh")
    return {
        "scenario_id": rows[0]["scenario_id"],
        "scenario_label": rows[0]["scenario_label"],
        "test_days": len(rows),
        **{f"{key}_mean": _mean(rows, key) for key in numeric},
        "system_energy_cost_std": float(np.std(
            [float(row["system_energy_cost"]) for row in rows], ddof=1
        )),
        "feasible_day_rate": float(np.mean([bool(row["feasible_day"]) for row in rows])),
        "ess_offer_aggregate_acceptance_pct": (
            100.0 * accepted_mean / offer_mean if offer_mean > 1e-12 else 0.0
        ),
        "by_microgrid": {
            mg: {
                key: _mean(rows, f"{mg}_{key}")
                for key in (
                    "realized_cost", "external_resource_cost", "p2p_net_revenue",
                    "grid_import_kwh", "grid_export_kwh", "battery_throughput_kwh",
                    "p2p_buy_kwh", "p2p_sell_kwh",
                )
            }
            for mg in agents
        },
    }


def _paired_effect(
    candidate_rows: list[Mapping[str, Any]],
    reference_rows: list[Mapping[str, Any]],
    effect_name: str,
) -> dict[str, Any]:
    candidate = np.asarray([row["system_energy_cost"] for row in candidate_rows], dtype=np.float64)
    reference = np.asarray([row["system_energy_cost"] for row in reference_rows], dtype=np.float64)
    delta = candidate - reference
    agents = sorted(
        key.removesuffix("_realized_cost")
        for key in candidate_rows[0]
        if key.endswith("_realized_cost")
    )
    return {
        "effect": effect_name,
        "definition": "candidate system energy cost minus reference system energy cost; positive means the removed capability had value",
        "candidate": candidate_rows[0]["scenario_id"],
        "reference": reference_rows[0]["scenario_id"],
        "mean_extra_cost": float(delta.mean()),
        "median_extra_cost": float(np.median(delta)),
        "std_extra_cost": float(delta.std(ddof=1)),
        "paired_bootstrap_95pct_ci": _bootstrap_mean_ci(delta),
        "positive_days": int(np.sum(delta > 1e-8)),
        "tie_days": int(np.sum(np.abs(delta) <= 1e-8)),
        "negative_days": int(np.sum(delta < -1e-8)),
        "mean_extra_cost_pct_of_reference": float(100.0 * delta.mean() / reference.mean()),
        "mean_extra_realized_cost_by_microgrid": {
            mg: float(np.mean([
                float(candidate_rows[index][f"{mg}_realized_cost"])
                - float(reference_rows[index][f"{mg}_realized_cost"])
                for index in range(len(candidate_rows))
            ]))
            for mg in agents
        },
    }


def _masked_policy(algorithm: Any, schemas: Mapping[str, tuple[str, ...]]):
    masked_indices = {
        mg: [i for i, name in enumerate(schema) if "_net_load_t+" in name]
        for mg, schema in schemas.items()
    }

    def policy(observations: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        masked: dict[str, np.ndarray] = {}
        for mg, observation in observations.items():
            value = np.asarray(observation, dtype=np.float32).copy()
            value[masked_indices[mg]] = 0.0
            masked[mg] = value
        return algorithm.select_actions(masked, noise_std=0.0)

    return policy, masked_indices


def main() -> None:
    snapshot = yaml.safe_load(SNAPSHOT.read_text(encoding="utf-8"))
    base_env_config = copy.deepcopy(snapshot["environment"])
    base_env_config["_project_root"] = str(ROOT)
    algorithm_config = copy.deepcopy(snapshot["algorithm"])
    configure_torch_runtime(algorithm_config, "cpu")
    spec = get_algorithm_spec("matd3_dispatch")

    # Load the checkpoint against its original physical signature.  The actor
    # itself is then reused read-only in all rollout environments.  Creating it
    # directly in an ablated environment would correctly trigger the checkpoint
    # configuration guard even though deterministic action selection only uses
    # the actor and observation schema.
    reference_env = MultiMicrogridEnv(base_env_config, split="test")
    algorithm = spec.create(reference_env, algorithm_config, device="cpu", seed=TEST_SEED)
    algorithm.load(CHECKPOINT)

    all_rows: list[dict[str, Any]] = []
    rows_by_scenario: dict[str, list[dict[str, Any]]] = {}
    masks: dict[str, list[int]] = {}
    mask_names: dict[str, list[str]] = {}
    common_days: list[str] | None = None

    for scenario in SCENARIOS:
        env_config = copy.deepcopy(base_env_config)
        env_config["p2p"]["enabled"] = bool(scenario["p2p_enabled"])
        env_config.setdefault("ess_sharing", {})["ess_p2p_enabled"] = bool(
            scenario["ess_p2p_enabled"]
        )
        env = MultiMicrogridEnv(env_config, split="test")
        days = [str(day.date()) for day in env.data.valid_days]
        if common_days is None:
            common_days = days
        elif days != common_days:
            raise RuntimeError("Scenario test dates differ")

        if scenario["mask_peer_future"]:
            policy, scenario_masks = _masked_policy(algorithm, algorithm.observation_schemas)
            masks = scenario_masks
            mask_names = {
                mg: [algorithm.observation_schemas[mg][index] for index in indices]
                for mg, indices in masks.items()
            }
        else:
            policy = lambda observations, algorithm=algorithm: algorithm.select_actions(
                observations, noise_std=0.0
            )
        _, daily = evaluate_policy_on_days(
            env,
            days,
            policy,
            seed=TEST_SEED,
            deterministic_environment=True,
            capture_hourly=False,
        )
        rows = [_daily_row(scenario, day) for day in daily]
        rows_by_scenario[str(scenario["id"])] = rows
        all_rows.extend(rows)

    full = rows_by_scenario["full_spatiotemporal"]
    effects = [
        _paired_effect(
            rows_by_scenario["no_p2p_local_storage"], full,
            "全部P2P通道对当前策略的总价值",
        ),
        _paired_effect(
            rows_by_scenario["renewable_p2p_only"], full,
            "储能跨微网外送通道的边际价值",
        ),
        _paired_effect(
            rows_by_scenario["full_without_peer_future"], full,
            "同行未来4小时净负荷信息对当前策略的边际价值",
        ),
        _paired_effect(
            rows_by_scenario["no_p2p_local_storage"],
            rows_by_scenario["renewable_p2p_only"],
            "可再生能源P2P通道在无ESS外送时的价值",
        ),
    ]

    stored_summary = json.loads((RUN_DIR / "test_summary.json").read_text(encoding="utf-8"))
    stored_cost = -float(stored_summary["algorithm"]["team_profit_mean"])
    reproduced_cost = _mean(full, "system_energy_cost")
    result = {
        "analysis_type": "paired deterministic policy ablation without retraining",
        "checkpoint": str(CHECKPOINT),
        "source_run": str(RUN_DIR),
        "test_seed": TEST_SEED,
        "test_days": len(common_days or []),
        "common_random_numbers": "same dates, deterministic forecasts, initial-SOC rule, and per-day seeds",
        "peer_future_mask": {
            "replacement": 0.0,
            "interpretation": "normalized neutral net load",
            "indices_by_agent": masks,
            "feature_names_by_agent": mask_names,
        },
        "scenarios": [_scenario_summary(rows_by_scenario[item["id"]]) for item in SCENARIOS],
        "paired_effects": effects,
        "baseline_reproduction": {
            "stored_full_policy_cost_mean": stored_cost,
            "rerun_full_policy_cost_mean": reproduced_cost,
            "absolute_difference": abs(stored_cost - reproduced_cost),
            "passed_1e-6": abs(stored_cost - reproduced_cost) <= 1e-6,
        },
        "interpretation_limits": [
            "The four cases reuse one trained policy; they isolate reliance of the current policy, not the optimum obtainable after retraining each architecture.",
            "Zeroing peer forecasts is an inference-time information ablation and can introduce out-of-distribution observations.",
            "P2P payments are internal transfers and cancel at system level; system_energy_cost equals minus team profit.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "policy_ablation_daily.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (OUTPUT_DIR / "policy_ablation_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
