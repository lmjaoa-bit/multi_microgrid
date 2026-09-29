from __future__ import annotations

import csv
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.service_reliability import augment_run_outputs
from utils.metrics_collector import EpisodeMetricsCollector, distribution_statistics, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"


def test_episode_collector_has_dr_aware_schema_and_24_hour_rows() -> None:
    env = MultiMicrogridEnv(CONFIG, split="test")
    observations, _ = env.reset(day=str(env.data.valid_days[0].date()), deterministic=True)
    collector = EpisodeMetricsCollector(env, capture_hourly=True)
    done = False
    while not done:
        actions = {mg: np.zeros(env.action_dim, dtype=np.float32) for mg in env.agent_ids}
        observations, rewards, done, info = env.step(actions)
        collector.record_step(actions, info, rewards)
    metrics = collector.finalize_episode()

    assert len(collector.hourly_rows) == 24
    assert set(metrics["profit_by_microgrid"]) == set(env.agent_ids)
    assert set(metrics["economic_breakdown_by_microgrid"]) == set(env.agent_ids)
    assert set(metrics["energy_statistics_by_microgrid"]) == set(env.agent_ids)
    assert len(metrics["p2p_statistics"]["pair_energy_kwh"]) == 6
    assert abs(metrics["sanity_checks"]["profit_balance_error"]) <= 1e-7
    assert metrics["sanity_checks"]["passed"]

    first = collector.hourly_rows[0]
    for mg in env.agent_ids:
        assert f"{mg}_raw_ess_action" in first
        assert f"{mg}_raw_dr_action" in first
        assert f"{mg}_effective_dr_action" in first
        assert f"{mg}_base_load_kw" in first
        assert f"{mg}_flexible_load_kw" in first
        assert f"{mg}_actual_load_kw" in first
        assert f"{mg}_ess_projected" in first
        assert f"{mg}_dr_projected" in first
        assert f"{mg}_power_balance_residual_kw" in first
        assert f"{mg}_sharing_offer_ratio_raw" not in first

    assert "mg1_silo_level_t" in first
    assert "mg1_mill_effective_duty" in first
    assert "mg2_t_air_c" in first
    assert "mg2_t_mass_c" in first
    assert "mg2_hvac_effective_signed_power_kw" in first
    assert "mg3_ev_soc" in first
    assert "mg3_ev_effective_power_kw" in first

    assert "dr_statistics_by_microgrid" in metrics
    assert set(metrics["dr_statistics_by_microgrid"]) == set(env.agent_ids)
    assert "projection_rate" in metrics["dr_statistics_by_microgrid"]["mg1"]
    assert "flexible_load_energy_kwh" in metrics["dr_statistics_by_microgrid"]["mg1"]
    assert "ess_offer_acceptance_rate_by_microgrid" in metrics["p2p_statistics"]

    # Pair-capacity diagnostics are an exact same-offer counterfactual, not
    # unmatched ESS offer energy relabeled as network congestion.
    p2p_stats = metrics["p2p_statistics"]
    assert p2p_stats["pair_capacity_kw"] is None
    assert p2p_stats["unconstrained_total_energy_kwh"] + 1e-9 >= p2p_stats["total_energy_kwh"]
    assert p2p_stats["blocked_by_pair_capacity_kwh"] >= -1e-9
    assert np.isclose(
        p2p_stats["blocked_by_pair_capacity_kwh"],
        p2p_stats["unconstrained_total_energy_kwh"] - p2p_stats["total_energy_kwh"],
        atol=1e-8,
    )
    assert set(p2p_stats["pair_saturation_rate_by_pair"]) == {
        "mg1_mg2", "mg1_mg3", "mg2_mg3"
    }
    assert 0.0 <= p2p_stats["blocked_energy_rate"] <= 1.0 + 1e-9
    assert "p2p_blocked_by_pair_capacity_kwh" in first
    assert "p2p_unconstrained_total_energy_kwh" in first

    action_dims = metrics["action_statistics"]["by_microgrid"]["mg1"]["dimensions"]
    assert "battery" in action_dims
    assert "dr_raw" in action_dims
    assert "sharing_offer_ratio_raw" not in action_dims


def test_service_reliability_augments_daily_and_summary_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    hourly_dir = run_dir / "test_hourly_trajectories"
    hourly_dir.mkdir(parents=True)

    hourly_rows = [
        {
            "date": "2018-11-28", "hour": 3,
            "mg1_unmet_load_kw": 0.0, "mg2_unmet_load_kw": 0.0,
            "mg3_unmet_load_kw": 2.5,
            "mg1_dr_supply_infeasible": 0, "mg2_dr_supply_infeasible": 0,
            "mg3_dr_supply_infeasible": 1,
        },
        {
            "date": "2018-11-28", "hour": 4,
            "mg1_unmet_load_kw": 0.0, "mg2_unmet_load_kw": 0.0,
            "mg3_unmet_load_kw": 0.0,
            "mg1_dr_supply_infeasible": 0, "mg2_dr_supply_infeasible": 0,
            "mg3_dr_supply_infeasible": 0,
        },
    ]
    with (hourly_dir / "2018-11-28.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hourly_rows[0]))
        writer.writeheader()
        writer.writerows(hourly_rows)

    daily_rows = [{
        "date": "2018-11-28",
        "feasible_day": True,
        "soc_violation_day": False,
        "power_balance_violation_day": False,
        "p2p_balance_violation_day": False,
    }]
    with (run_dir / "test_daily_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(daily_rows[0]))
        writer.writeheader()
        writer.writerows(daily_rows)

    base_summary = {
        "test_days_count": 1,
        "algorithm": {"feasible_rate": 1.0},
        "matd3": {"feasible_rate": 1.0},
    }
    for name in ("test_summary.json", "test_evaluation.json"):
        (run_dir / name).write_text(json.dumps(base_summary), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"test_days_count": 1}), encoding="utf-8")

    reliability = augment_run_outputs(run_dir, generate_plot=False)

    assert reliability["unmet_energy_total_kwh"] == 2.5
    assert reliability["unmet_energy_mean_kwh"] == 2.5
    assert reliability["unmet_load_day_rate"] == 1.0
    assert reliability["unmet_load_hour_rate_mean"] == 0.5
    assert np.isclose(reliability["unmet_load_microgrid_hour_rate_mean"], 1.0 / 6.0)
    assert reliability["dr_supply_infeasible_day_rate"] == 1.0
    assert reliability["dr_supply_infeasible_rate_mean"] == 0.5
    assert np.isclose(reliability["dr_supply_infeasible_microgrid_hour_rate_mean"], 1.0 / 6.0)
    assert reliability["strict_feasible_day_rate"] is None
    assert reliability["by_microgrid"]["mg3"]["unmet_energy_total_kwh"] == 2.5

    with (run_dir / "test_daily_metrics.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["unmet_energy_kwh"]) == 2.5
    assert row["unmet_load_day"].lower() == "true"
    assert row["dr_supply_infeasible_day"].lower() == "true"
    assert row["strict_feasible_day"] in {"", "None"}
    assert float(row["unmet_load_hour_rate"]) == 0.5
    assert float(row["dr_supply_infeasible_rate"]) == 0.5
    assert float(row["unmet_energy_mg3_kwh"]) == 2.5
    assert float(row["dr_supply_infeasible_rate_mg3"]) == 0.5

    test_summary = json.loads((run_dir / "test_summary.json").read_text(encoding="utf-8"))
    assert test_summary["strict_feasible_day_rate"] is None
    assert test_summary["unmet_energy_total_kwh"] == 2.5
    assert test_summary["service_reliability"]["unmet_load_day_rate"] == 1.0
    assert test_summary["algorithm"]["service_reliability"]["strict_feasible_day_rate"] is None
    assert test_summary["matd3"]["service_reliability"]["strict_feasible_day_rate"] is None

    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert run_summary["strict_feasible_day_rate"] is None
    assert run_summary["service_reliability"]["unmet_energy_total_kwh"] == 2.5


def test_json_writer_converts_numpy_and_rejects_nonfinite_json(tmp_path: Path) -> None:
    target = tmp_path / "metrics.json"
    write_json(target, {"scalar": np.float32(1.5), "bad": float("nan")})
    value = json.loads(target.read_text(encoding="utf-8"))
    assert value["scalar"] == 1.5
    assert value["bad"] is None
    assert value["serialization_warnings"][0]["field"] == "bad"
    assert distribution_statistics([1.0, 2.0, 3.0])["median"] == 2.0


def test_append_actor_diagnostics_supports_heterogeneous_agents(tmp_path: Path) -> None:
    from experiments.training_pipeline import _append_actor_diagnostics

    csv_path = tmp_path / "actor_diagnostics.csv"
    record1 = {
        "actor_diagnostic_statistics": {
            "mg1": {
                "actor_dr_latent": {"mean": 0.1, "p95": 0.2, "max": 0.3, "count": 1, "nonfinite_count": 0},
            },
            "mg2": {
                "actor_dr_latent": {"mean": 0.2, "p95": 0.3, "max": 0.4, "count": 1, "nonfinite_count": 0},
                "actor_hvac_recovery_anchor": {"mean": 10.0, "p95": 11.0, "max": 12.0, "count": 1, "nonfinite_count": 0},
            },
            "mg3": {
                "actor_dr_latent": {"mean": 0.3, "p95": 0.4, "max": 0.5, "count": 1, "nonfinite_count": 0},
                "actor_ev_mandatory_floor": {"mean": 20.0, "p95": 21.0, "max": 22.0, "count": 1, "nonfinite_count": 0},
            },
        }
    }
    _append_actor_diagnostics(csv_path, episode=1, record=record1)
    assert csv_path.exists()

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 3
    assert reader[0]["microgrid"] == "mg1"
    assert reader[1]["microgrid"] == "mg2"
    assert reader[2]["microgrid"] == "mg3"
    assert "actor_hvac_recovery_anchor_mean" in reader[1]
    assert "actor_ev_mandatory_floor_mean" in reader[2]

    # Append second episode
    _append_actor_diagnostics(csv_path, episode=2, record=record1)
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader2 = list(csv.DictReader(f))
    assert len(reader2) == 6


def test_ess_offer_energy_accounting_applies_delta_t_once() -> None:
    battery = SimpleNamespace(soc=0.5, initial_soc=0.5, capacity_kwh=100.0)
    env = SimpleNamespace(
        agent_ids=["mg1"],
        action_design="single_dispatch",
        delta_t=0.5,
        cores={"mg1": SimpleNamespace(battery=battery)},
    )
    collector = EpisodeMetricsCollector(env)
    info = {
        "plans": {
            "mg1": {
                "renewable_kw": 0.0,
                "load_kw": 0.0,
                "ess_to_local_kw": 0.0,
                "ess_p2p_offer_uncapped_kw": 30.0,
                "ess_p2p_offer_projection_reduction_kw": 10.0,
            }
        },
        "settlements": {
            "mg1": {
                "grid_import_kw": 0.0,
                "grid_export_kw": 0.0,
                "renewable_curtailment_kw": 0.0,
                "unmet_load_kw": 0.0,
            }
        },
        "devices": {
            "mg1": {
                "charge_kw": 0.0,
                "discharge_kw": 12.0,
                "throughput_kwh": 6.0,
            }
        },
        "p2p": {
            "by_microgrid": {
                "mg1": {
                    "p2p_buy_kw": 0.0,
                    "p2p_sell_kw": 12.0,
                    "renewable_p2p_sell_kw": 0.0,
                    "ess_p2p_sell_kw": 12.0,
                    "p2p_to_load_kw": 0.0,
                    "p2p_to_battery_kw": 0.0,
                    "renewable_p2p_offer_kw": 0.0,
                    "ess_p2p_offer_kw": 20.0,
                    "ess_p2p_unmatched_kw": 8.0,
                }
            }
        },
    }
    step = collector._energy_step("mg1", info)
    assert step["ess_p2p_offer_uncapped_kwh"] == 15.0
    assert step["ess_p2p_offer_projection_reduction_kwh"] == 5.0
    assert step["ess_p2p_sell_kwh"] == 6.0
    assert step["ess_p2p_unmatched_kwh"] == 4.0
    assert step["ess_p2p_offer_uncapped_kwh"] == (
        step["ess_p2p_offer_projection_reduction_kwh"]
        + step["ess_p2p_sell_kwh"]
        + step["ess_p2p_unmatched_kwh"]
    )
    # throughput_kwh is already an energy value from BatteryModel.commit.
    assert step["battery_throughput_kwh"] == 6.0


def test_hvac_soft_comfort_summary_counts_real_violations_not_legacy_feasibility_flag() -> None:
    env = SimpleNamespace(
        agent_ids=["mg2"],
        action_design="single_dispatch_dr",
        delta_t=1.0,
        cfg_by_id={"mg2": {"dr": {"type": "commercial_hvac"}}},
        cores={"mg2": SimpleNamespace(
            battery=SimpleNamespace(soc=0.5, initial_soc=0.5, capacity_kwh=100.0)
        )},
    )
    collector = EpisodeMetricsCollector(env)
    collector.steps = 1
    collector._record_dr_step(
        "mg2",
        {
            "plans": {
                "mg2": {
                    "flexible_load_kw": 31.0,
                    "dr_projected": False,
                    "ess_projected": False,
                    "dr_supply_infeasible": False,
                }
            },
            "dr_plans": {
                "mg2": {
                    "flexible_load_kw": 31.0,
                    "heating_power_kw": 0.0,
                    "cooling_power_kw": 31.0,
                    "comfort_infeasible": False,
                    "terminal_thermal_infeasible": False,
                }
            },
            "dr_devices": {
                "mg2": {
                    "t_air_c": 19.0,
                    "t_mass_c": 20.0,
                    "comfort_violation_c": 2.0,
                    "comfort_degree_hours": 2.0,
                }
            },
            "dr_state_after": {
                "mg2": {
                    "t_air_c": 19.0,
                    "t_mass_c": 20.0,
                    "comfort_lower_margin_c": -2.0,
                    "comfort_upper_margin_c": 6.0,
                }
            },
        },
    )

    summary = collector._dr_summary("mg2")

    assert summary["comfort_violation_rate"] == 1.0
    assert summary["comfort_violation_degree_hours"] == 2.0
    assert summary["comfort_infeasible_rate"] == 1.0
    assert summary["legacy_comfort_infeasible_rate"] == 0.0
