from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.evaluate_matd3 import evaluate_policy_on_days
from utils.metrics_collector import EpisodeMetricsCollector, distribution_statistics, write_json
from scripts import export_runs_to_tensorboard as tensorboard_export


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative.yaml"


def test_episode_collector_has_unified_schema_and_24_hour_rows() -> None:
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
    assert "mg1_executed_action_0" in collector.hourly_rows[0]
    assert "p2p_mg1_to_mg2_kw" in collector.hourly_rows[0]
    assert "mg1_ess_p2p_offer_kw" in collector.hourly_rows[0]
    assert "mg1_ess_p2p_unmatched_kw" in collector.hourly_rows[0]
    assert "mg1_ess_p2p_offer_uncapped_kw" in collector.hourly_rows[0]
    assert "mg1_ess_p2p_offer_projection_reduction_kw" in collector.hourly_rows[0]
    assert "ess_offer_acceptance_rate_by_microgrid" in metrics["p2p_statistics"]
    assert "ess_offer_any_fill_rate_by_microgrid" in metrics["p2p_statistics"]
    assert "ess_offer_full_fill_rate_by_microgrid" in metrics["p2p_statistics"]
    assert "sharing_action_mask_rate_by_microgrid" in metrics["p2p_statistics"]
    assert "ess_offer_projection_retention_rate_by_microgrid" in metrics["p2p_statistics"]


def test_json_writer_converts_numpy_and_rejects_nonfinite_json(tmp_path: Path) -> None:
    target = tmp_path / "metrics.json"
    write_json(target, {"scalar": np.float32(1.5), "bad": float("nan")})
    value = json.loads(target.read_text(encoding="utf-8"))
    assert value["scalar"] == 1.5
    assert value["bad"] is None
    assert value["serialization_warnings"][0]["field"] == "bad"
    assert distribution_statistics([1.0, 2.0, 3.0])["median"] == 2.0


def test_tensorboard_export_writes_independent_and_shapley_tags(tmp_path, monkeypatch):
    run_dir = tmp_path / "outputs" / "matd3_shapley_q" / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    record = {
        "team_reward": 12.0,
        "independent_baseline_return": 8.0,
        "cooperation_surplus_return": 4.0,
        "shapley_q_mg1": 1.0,
        "shapley_q_mg2": 1.25,
        "shapley_q_mg3": 1.75,
        "shapley_q_sum": 4.0,
        "efficiency_residual": 0.1,
        "grand_decomposition_residual": -0.2,
    }
    (run_dir / "training_history.json").write_text(
        json.dumps([record]), encoding="utf-8"
    )

    calls = []

    class FakeWriter:
        def __init__(self, log_dir):
            self.log_dir = log_dir

        def add_scalar(self, tag, value, step):
            calls.append((tag, value, step))

        def flush(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(tensorboard_export, "SummaryWriter", FakeWriter)
    assert tensorboard_export.export_run(run_dir, tmp_path / "tb")
    tags = {tag for tag, _, _ in calls}
    assert {
        "reward/team", "reward/independent", "reward/cooperation_surplus",
        "q/shapley_mg1", "q/shapley_mg2", "q/shapley_mg3", "q/shapley_sum",
        "q/efficiency_residual", "q/grand_decomposition_residual",
    }.issubset(tags)


def test_evaluation_baseline_lifecycle_adds_matched_daily_fields():
    env = MultiMicrogridEnv(CONFIG, split="test")
    day = str(env.data.valid_days[0].date())
    starts = []

    def start(current_env, episode_seed):
        starts.append((current_env.last_reset_options["day"], episode_seed))

    def finish():
        return {"independent_baseline_return": -3.0,
                "independent_baseline_profit": -300.0}

    policy = lambda observations: {
        mg: np.zeros(env.action_dim, dtype=np.float32) for mg in env.agent_ids
    }
    _, daily = evaluate_policy_on_days(
        env, [day], policy, seed=73, deterministic_environment=True,
        episode_start_fn=start, episode_finalize_fn=finish,
    )
    row = daily[0]
    assert starts == [(row["date"] + " 00:00:00", 73)]
    assert row["grand_team_return"] == row["team_reward"]
    assert row["grand_team_profit"] == row["team_profit"]
    assert row["cooperation_surplus_return"] == row["grand_team_return"] + 3.0
    assert row["cooperation_surplus_profit"] == row["grand_team_profit"] + 300.0
