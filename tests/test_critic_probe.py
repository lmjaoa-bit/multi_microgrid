from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml


def test_expanded_dates_are_calendar_selected_one_per_month():
    module = probe()
    dates = [datetime(2018, month, day) for month in range(1, 13) for day in (22, 23, 24, 25, 26)]
    selected = module.seasonal_days(dates, limit=None, days_per_season=3)
    assert selected == [f"2018-{month:02d}-24" for month in range(1, 13)]
    assert module.seasonal_days(dates) == [f"2018-{month:02d}-24" for month in (1, 4, 7, 10)]
    expanded = module.seasonal_days(dates, limit=None, days_per_season=6)
    assert expanded == [f"2018-{month:02d}-{day}" for month in range(1, 13) for day in (23, 25)]


def test_expanded_sampling_rejects_missing_month_coverage():
    dates = [datetime(2018, month, 24) for month in (1, 4, 7, 10)]
    with pytest.raises(ValueError, match="month"):
        probe().seasonal_days(dates, limit=None, days_per_season=3)


def test_state_selection_replaces_cross_origin_physical_duplicates():
    module = probe()
    day = "2018-01-24"
    origins = ["baseline", "transformer"]
    pool = []
    for origin in origins:
        for hour in range(24):
            # Both origin trajectories have the same actual initial state.
            signature = "shared-initial" if hour == 0 else f"{origin}-{hour}"
            context = {key: -hour for key in ("ev_deadline", "hvac_boundary", "ess_boundary", "peak_pressure")}
            pool.append(module.StateCase(f"{origin}-{hour}", origin, day, 7, hour, "unselected", [],
                                         np.zeros(1), {}, signature, context))
    selected = module.select_cases(pool, origins, [day], 7, 6)
    assert len(selected) == 6
    assert len({case.signature for case in selected}) == 6
    assert sum(case.origin == "baseline" for case in selected) == 3
    assert sum(case.origin == "transformer" for case in selected) == 3
    assert sum(case.category == "ordinary" for case in selected) == 2


def test_stratified_metrics_expose_outliers_and_actual_harm():
    rows = []
    for regret, opportunity, correct, pairs in [(0., 2., 2, 2), (40., 0., 0, 1)]:
        row = dict(actor_improvement_cost=opportunity)
        for kind in ("q1", "q2", "qmin"):
            row.update({f"{kind}_selection_regret_cost": regret, f"{kind}_rank_correct_count": correct,
                        f"{kind}_rank_pair_count": pairs})
        rows.append(row)
    scores = probe().summarize_scores(rows)
    assert scores["q1_mean_selection_regret_cost"] == 20.
    assert scores["q1_rank_accuracy"] == pytest.approx(2 / 3)
    assert scores["q1_harmful_selection_count"] == 1
    assert scores["q1_helpful_selection_count"] == 1
    assert scores["q1_max_selection_regret_cost"] == 40.
    assert scores["q1_top_five_regret_share"] == 1.


def probe():
    assert importlib.util.find_spec("diagnostics.critic_probe") is not None, "Offline critic diagnosis is missing"
    return importlib.import_module("diagnostics.critic_probe")


def write_run(path, *, slug="matd3_dispatch_q_guided", finished="2026-10-01T12:00:00", episodes=5):
    path.mkdir(parents=True)
    (path / "checkpoints").mkdir()
    checkpoint = path / "checkpoints/best_model.pt"
    checkpoint.write_bytes(b"fixture-best-checkpoint")
    files = {
        "run_metadata.json": dict(algorithm_slug=slug, run_purpose="formal_training", train_episodes=episodes,
                                  formal_test_completed=True, end_time=finished, seed=42),
        "training_summary.json": dict(episodes=episodes, total_steps=episodes * 24, best_episode=4,
                                      checkpoint_selection=dict(best_checkpoint="checkpoints/best_model.pt",
                                                                constraint_eligible=True, best_episode=4)),
        "training_history.json": [dict(episode=i) for i in range(1, episodes + 1)],
        "test_summary.json": dict(algorithm_slug=slug, checkpoint_used=str(checkpoint), checkpoint_episode=4,
                                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                  test_days_count=2, status="ok"),
    }
    for name, value in files.items():
        (path / name).write_text(json.dumps(value), encoding="utf-8")
    snapshot = dict(environment=dict(time=dict(episode_length=24)),
                    algorithm=dict(algorithm=dict(name=slug), training=dict(episodes=episodes, test_episodes=2)))
    (path / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot), encoding="utf-8")
    return path


def edit_json(path, name, **updates):
    file = path / name
    value = json.loads(file.read_text(encoding="utf-8"))
    value.update(updates)
    file.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("name,updates", [
    ("run_metadata.json", dict(run_purpose="smoke_test")),
    ("run_metadata.json", dict(formal_test_completed=False)),
    ("run_metadata.json", dict(end_time=None)),
    ("training_summary.json", dict(episodes=4)),
    ("test_summary.json", dict(checkpoint_used="checkpoints/final_model.pt")),
    ("test_summary.json", dict(checkpoint_sha256="wrong")),
    ("test_summary.json", dict(checkpoint_episode=3)),
    ("test_summary.json", dict(test_days_count=1)),
])
def test_invalid_or_smoke_runs_cannot_be_selected(tmp_path, name, updates):
    run = write_run(tmp_path / "run")
    edit_json(run, name, **updates)
    with pytest.raises(ValueError):
        probe().validate_source_run(run, "matd3_dispatch_q_guided", min_episodes=5)


def test_truncated_history_is_rejected(tmp_path):
    run = write_run(tmp_path / "run")
    (run / "training_history.json").write_text('[{"episode":5}]', encoding="utf-8")
    with pytest.raises(ValueError):
        probe().validate_source_run(run, "matd3_dispatch_q_guided", min_episodes=5)


def test_missing_episode_evidence_is_rejected_even_with_expected_history_length(tmp_path):
    run = write_run(tmp_path / "run")
    (run / "training_history.json").write_text(json.dumps([{}] * 5), encoding="utf-8")
    with pytest.raises(ValueError):
        probe().validate_source_run(run, "matd3_dispatch_q_guided", min_episodes=5)


def test_nonobject_metadata_is_excluded_without_aborting_discovery(tmp_path):
    root = tmp_path / "working"
    full = write_run(root / "runs/full")
    broken = write_run(root / "runs/broken")
    (broken / "run_metadata.json").write_text('[]', encoding="utf-8")
    selected, excluded = probe().find_source_run(root, "matd3_dispatch_q_guided", min_episodes=5)
    assert selected.path == full.resolve()
    assert len(excluded) == 1


def test_bad_snapshot_is_excluded_without_hiding_the_good_full_run(tmp_path):
    root = tmp_path / "working"
    full = write_run(root / "runs/full")
    broken = write_run(root / "runs/broken", finished="2026-10-02T12:00:00")
    (broken / "config_snapshot.yaml").write_text('algorithm: [broken', encoding="utf-8")
    selected, excluded = probe().find_source_run(root, "matd3_dispatch_q_guided", min_episodes=5)
    assert selected.path == full.resolve()
    assert len(excluded) == 1


def test_latest_pointer_cannot_override_full_training_filter(tmp_path):
    root = tmp_path / "working"
    full = write_run(root / "runs/full")
    smoke = write_run(root / "runs/smoke", finished="2026-10-02T12:00:00")
    edit_json(smoke, "run_metadata.json", run_purpose="smoke_test")
    (root / "latest_run.json").write_text(json.dumps(dict(run_directory=str(smoke))), encoding="utf-8")
    selected, excluded = probe().find_source_run(root, "matd3_dispatch_q_guided", min_episodes=5)
    assert selected.path == full.resolve()
    assert len(excluded) == 1


def test_axis_candidates_do_not_change_other_controls_or_input():
    raw = np.array([[.25, -.2], [.3, -.4], [.1, .6]], dtype=np.float32)
    before = raw.copy()
    candidates = probe().axis_candidates(raw)
    assert len(candidates) == 13
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(candidates[0].raw, before)
    for item in candidates[1:]:
        assert np.count_nonzero(item.raw != before) == 1
        assert item.raw[item.agent_index, item.control_index] in (-.8, .8)


def test_pair_ranking_and_choice_loss_use_q1_and_ignore_near_ties():
    # Realized returns -10, -8, -8.005; the last two are tied within cost tolerance 1.
    scores = probe().score_choices(np.array([-10., -8., -8.005]), np.array([-1., -2., -3.]),
                                  reward_scale=100., tie_cost=1.)
    assert scores["selection_regret_cost"] == pytest.approx(200.)
    assert scores["actor_improvement_cost"] == pytest.approx(200.)
    assert scores["rank_pair_count"] == 2
    assert scores["rank_correct_count"] == 0
    assert scores["selected_index"] == 0


def test_nonfinite_scores_fail_instead_of_generating_a_report():
    with pytest.raises(ValueError):
        probe().score_choices(np.array([0., np.nan]), np.array([0., 1.]), reward_scale=100., tie_cost=1.)


def test_rollout_uses_one_team_reward_and_reacts_to_new_observations():
    class Env:
        agent_ids = ["mg1", "mg2", "mg3"]
        t = 1
        def step(self, actions):
            self.t += 1
            reward = float(self.t * 2)
            return {mg: np.array([self.t]) for mg in self.agent_ids}, {mg: reward for mg in self.agent_ids}, self.t == 3, {}
    class Policy:
        gamma = .5
        def actions(self, observations):
            assert observations["mg1"][0] == 2
            return {mg: np.zeros(2) for mg in observations}
    value, steps = probe().rollout_return(Env(), Policy(), {mg: np.zeros(2) for mg in Env.agent_ids})
    assert value == pytest.approx(7.)  # 4 + 0.5 * 6, not three times the team reward.
    assert steps == 2


def test_prefix_reconstruction_includes_departed_ev_history_and_peak():
    module = probe()
    from envs.multi_microgrid_env import MultiMicrogridEnv
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(config, split="val")
    day = str(env.data.valid_days[0].date())
    observations, _ = env.reset(day=day, seed=7, deterministic=True, forecast_noise=True)
    prefix = [{mg: np.array([-.3, .2], dtype=np.float32) for mg in env.agent_ids} for _ in range(10)]
    for actions in prefix:
        observations, _, _, _ = env.step(actions)
    signature = module.environment_signature(env, observations)
    case = module.StateCase("case", "source", day, 7, 10, "ev_deadline", prefix, env.global_state(), observations, signature, {})
    restored = module.restore_case(env, case, forecast_noise=True)
    assert module.environment_signature(env, restored) == signature
    assert env.t == 10
    assert env.daily_load_peak_kw > 0
    assert env.dr_devices["mg3"].departure_events
    case.signature = "corrupted"
    with pytest.raises(ValueError, match="reconstruction"):
        module.restore_case(env, case, forecast_noise=True)


def test_equivalent_ev_intents_reuse_return_only_after_matching_actual_next_state():
    module = probe()
    from envs.multi_microgrid_env import MultiMicrogridEnv
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(config, split="val")
    day = str(env.data.valid_days[0].date())
    observations, _ = env.reset(day=day, seed=7, deterministic=True, forecast_noise=True)
    fixed = {mg: np.array([-.3, .2], dtype=np.float32) for mg in env.agent_ids}
    prefix = []
    for _ in range(20):
        prefix.append({mg: value.copy() for mg, value in fixed.items()})
        observations, _, _, _ = env.step(fixed)
    case = module.StateCase("case", "source", day, 7, 20, "ordinary", prefix, env.global_state(), observations,
                            module.environment_signature(env, observations), {})
    class Policy:
        gamma = 1.
        def actions(self, observations):
            return {mg: value.copy() for mg, value in fixed.items()}
    raw = np.stack(list(fixed.values()))
    negative, positive = raw.copy(), raw.copy()
    negative[2, 1], positive[2, 1] = -.8, .8
    candidates = [module.Candidate(raw, None, None), module.Candidate(negative, 2, 1), module.Candidate(positive, 2, 1)]
    result = module.evaluate_candidates(env, Policy(), case, candidates, forecast_noise=True)
    assert result["rollouts"] == 1
    assert result["duplicate_of"] == [None, 0, 0]
    np.testing.assert_allclose(result["returns"], result["returns"][0], atol=0, rtol=0)
    for candidate in candidates:
        module.restore_case(env, case, forecast_noise=True)
        realized, _ = module.rollout_return(env, Policy(), {mg: candidate.raw[i] for i, mg in enumerate(env.agent_ids)})
        assert realized == pytest.approx(result["returns"][0], abs=1e-10)


def test_peak_state_sampling_uses_load_normalization_and_matches_actual_dispatch():
    module = probe()
    from envs.multi_microgrid_env import MultiMicrogridEnv
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(config, split="val")
    day = str(env.data.valid_days[0].date())
    env.reset(day=day, seed=7, deterministic=True, forecast_noise=False)
    raw = np.zeros((3, 2), dtype=np.float32)
    estimated = module.context_scores(env, env.global_state(), raw)["peak_pressure"]
    peak_before = env.daily_load_peak_kw
    _, _, _, info = env.step({mg: raw[i] for i, mg in enumerate(env.agent_ids)})
    assert estimated == pytest.approx(info["team_actual_load_kw"] - peak_before, abs=1e-4)


def test_real_formal_checkpoint_smoke_is_read_only(tmp_path):
    module = probe()
    root = Path(__file__).resolve().parents[1]
    slugs = ["matd3_dispatch_q_guided", "matd3_hierarchical_transformer_q_readout"]
    if any(not (root / "outputs/working_runs" / slug / "runs").exists() for slug in slugs):
        pytest.skip("Full training checkpoints are not available on this checkout")
    result = module.run_probe(root=root, output=tmp_path / "diagnosis", state_limit=4, day_limit=1,
                             device="cpu", forecast_seeds=[7, 1007])
    assert result["status"] == "complete"
    assert result["state_count"] == 4
    assert result["unique_state_count"] == 4
    assert result["model_count"] == 2
    assert result["source_integrity_verified"] is True
    assert result["reconstruction_failures"] == 0
    assert result["candidate_rollouts"] <= 104
    manifest = json.loads((tmp_path / "diagnosis/manifest.json").read_text(encoding="utf-8"))
    assert all(s["training_episodes"] >= 5000 for s in manifest["sources"])
    assert all(Path(s["checkpoint"]).name == "best_model.pt" for s in manifest["sources"])
    assert (tmp_path / "diagnosis/state_scores.csv").is_file()
    assert (tmp_path / "diagnosis/report.md").is_file()
    assert manifest["forecast_seeds"] == [7, 1007]
    pool = json.loads((tmp_path / "diagnosis/state_pool.json").read_text(encoding="utf-8"))
    assert {case["forecast_seed_base"] for case in pool} == {7, 1007}
    assert len({case["case_id"] for case in pool}) == 4
    assert (tmp_path / "diagnosis/stratified_scores.csv").is_file()
    assert set(result["models"][slugs[0]]["by_forecast_seed"]) == {"7", "1007"}
