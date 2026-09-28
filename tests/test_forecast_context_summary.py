from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import scripts.summarize_forecast_context as summary_module


CONFIG_PATHS = {
    "A": Path("configs/algorithms/matd3_forecast_context.yaml"),
    "B": Path("configs/algorithms/matd3_forecast_context/b_peer_h8.yaml"),
    "C": Path("configs/algorithms/matd3_forecast_context/c_local_h24.yaml"),
    "D": Path("configs/algorithms/matd3_forecast_context/d_peer_h24.yaml"),
}
GROUP_EFFECT = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 4.0}


def _json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_run(
    root: Path,
    *,
    group: str,
    seed: int = 42,
    config_mutator=None,
) -> Path:
    run_dir = root / f"run_{group}_seed{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    hourly_dir = run_dir / "test_hourly_trajectories"
    hourly_dir.mkdir()

    algorithm_bundle = yaml.safe_load(
        CONFIG_PATHS[group].read_text(encoding="utf-8")
    )
    if config_mutator is not None:
        config_mutator(algorithm_bundle)
    snapshot = {
        "environment": {
            "env": {"mode": "cooperative", "seed": 42},
            "observation": {
                "future_horizon": 24,
                "mask_forecast_beyond_episode": True,
            },
            "ess_sharing": {
                "control_mode": "single_dispatch",
                "observation_mode": "peer_energy_summary",
            },
        },
        "algorithm": algorithm_bundle,
        "experiment_contract": {
            "mode": "cooperative",
            "action_design": "single_dispatch_dr",
            "policy": "unregistered",
        },
    }
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    common_metadata = {
        "algorithm_slug": "matd3_forecast_context",
        "seed": seed,
        "train_episodes": 5000,
        "test_episodes": 55,
        "device": "cuda",
        "runtime": {
            "device": "cuda",
            "cpu_threads": 4,
            "interop_threads": 2,
        },
        "ess_sharing_control_mode": "single_dispatch",
        "ess_sharing_observation_mode": "peer_energy_summary",
        "evaluation_forecast_noise": False,
        "git_commit": "0123456789abcdef",
        "experiment_contract": copy.deepcopy(
            snapshot["experiment_contract"]
        ),
    }
    _json(run_dir / "run_metadata.json", common_metadata)
    _json(
        run_dir / "run_status.json",
        {
            "status": "COMPLETED",
            "completed_episode": 5000,
            "total_steps": 120000,
        },
    )
    _json(
        run_dir / "training_summary.json",
        {
            "episodes": 5000,
            "training_seconds": 123.0,
            "checkpoint_selection": {
                "best_episode": 4100,
                "best_checkpoint": "checkpoints/best_model.pt",
            },
        },
    )
    _json(
        run_dir / "test_summary.json",
        {
            "test_days_count": 55,
            "checkpoint_used": "checkpoints/best_model.pt",
            "checkpoint_episode": 4100,
        },
    )
    _json(
        run_dir / "summary.json",
        {
            "test_days_count": 55,
            "training_seconds": 123.0,
            "plotting_succeeded": True,
        },
    )

    network = algorithm_bundle["algorithm"]["network"]
    contract = {
        "horizon": network["forecast_horizon"],
        "include_peer_forecast": network["include_peer_forecast"],
        "peer_forecast_features": ["energy_balance"],
        "agent_order": ["mg1", "mg2", "mg3"],
        "per_agent": {},
        "power_normalizers_by_agent": {},
        "mask_semantics": "valid_lt_half_padding_v1",
        "local_balance_sign": "positive_deficit",
        "peer_balance_sign": "positive_surplus",
        "architecture": {
            "embed_dim": 32,
            "num_heads": 4,
            "num_layers": 2,
            "hidden_dim": 64,
        },
    }
    torch.save(
        {
            "algorithm_slug": "matd3_forecast_context",
            "forecast_context_version": "forecast_context_v1",
            "forecast_context_contract": contract,
        },
        run_dir / "checkpoints" / "best_model.pt",
    )

    rows = []
    start = pd.Timestamp("2026-01-01")
    for day_index in range(55):
        date = str((start + pd.Timedelta(days=day_index)).date())
        profit = 100.0 + day_index + GROUP_EFFECT[group]
        rows.append(
            {
                "date": date,
                "algorithm_team_profit": profit,
                "gain_vs_idle_abs": profit - 80.0,
                "algorithm_cooperation_surplus": 10.0 + GROUP_EFFECT[group],
                "cooperative_milp_cooperation_surplus": 20.0,
                "feasible_day": True,
                "renewable_curtailment_kwh": 5.0 - 0.1 * GROUP_EFFECT[group],
                "renewable_utilization_rate": 0.90 + 0.01 * GROUP_EFFECT[group],
                "p2p_total_energy_kwh": 20.0 + GROUP_EFFECT[group],
                "grid_import_kwh": 50.0 - GROUP_EFFECT[group],
                "grid_export_kwh": 2.0 + GROUP_EFFECT[group],
                "battery_charge_kwh": 10.0,
                "battery_discharge_kwh": 9.0,
            }
        )
        pd.DataFrame({"hour": np.arange(24)}).to_csv(
            hourly_dir / f"{date}.csv",
            index=False,
        )
    pd.DataFrame(rows).to_csv(
        run_dir / "test_daily_metrics.csv",
        index=False,
    )
    return run_dir


@pytest.fixture
def four_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        summary_module,
        "REPORT_ROOT",
        tmp_path / "reports" / "forecast_context",
    )
    source_root = tmp_path / "runs"
    source_root.mkdir()
    return [
        _build_run(source_root, group=group, seed=42)
        for group in ("A", "B", "C", "D")
    ]


def test_single_seed_summary_has_correct_paired_effects_and_unknown_seed_std(
    four_runs,
):
    source_hashes_before = {
        path: summary_module._source_hashes(path)
        for path in four_runs
    }
    output = summary_module.summarize_study(
        four_runs,
        study_id="seed42_exploratory",
        required_seeds=[42],
    )

    assert output.is_dir()
    assert {
        "study_manifest.json",
        "run_summary.csv",
        "paired_daily_deltas.csv",
        "factor_effects.csv",
        "README.md",
    } == {path.name for path in output.iterdir()}

    effects = pd.read_csv(output / "factor_effects.csv")
    expected = {
        "B-A": 1.0,
        "C-A": 2.0,
        "D-C": 2.0,
        "D-B": 3.0,
        "interaction_D-C-B+A": 1.0,
    }
    for effect, value in expected.items():
        seed_row = effects[
            (effects["effect"] == effect)
            & (effects["seed"].astype(str) == "42")
        ].iloc[0]
        assert seed_row["mean_daily_delta"] == pytest.approx(value)
        all_row = effects[
            (effects["effect"] == effect)
            & (effects["seed"].astype(str) == "all")
        ].iloc[0]
        assert all_row["mean_daily_delta"] == pytest.approx(value)
        assert pd.isna(all_row["std_across_seeds"])

    paired = pd.read_csv(output / "paired_daily_deltas.csv")
    assert len(paired) == 55 * 5
    assert set(paired["effect"]) == set(expected)

    manifest = json.loads(
        (output / "study_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["required_seeds"] == [42]
    assert len(manifest["runs"]) == 4
    assert {
        (item["group"], item["forecast_horizon"], item["include_peer_forecast"])
        for item in manifest["runs"]
    } == {
        ("A", 8, False),
        ("B", 8, True),
        ("C", 24, False),
        ("D", 24, True),
    }

    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "one training seed only" in readme
    assert "exploratory" in readme

    for path in four_runs:
        assert summary_module._source_hashes(path) == source_hashes_before[path]


def test_ratio_of_sums_not_mean_of_daily_ratios(four_runs):
    output = summary_module.summarize_study(
        four_runs,
        study_id="ratio_of_sums",
        required_seeds=[42],
    )
    rows = pd.read_csv(output / "run_summary.csv")
    row_a = rows[rows["group"] == "A"].iloc[0]
    assert row_a["cooperation_value_capture_ratio_of_sums"] == pytest.approx(0.5)


def test_duplicate_group_seed_is_rejected(four_runs):
    with pytest.raises(ValueError, match="Duplicate run"):
        summary_module.summarize_study(
            [*four_runs, four_runs[0]],
            study_id="duplicate",
            required_seeds=[42],
        )


def test_missing_group_is_rejected(four_runs):
    with pytest.raises(ValueError, match="Study run set mismatch"):
        summary_module.summarize_study(
            four_runs[:-1],
            study_id="missing_group",
            required_seeds=[42],
        )


def test_config_difference_outside_whitelist_is_rejected(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        summary_module,
        "REPORT_ROOT",
        tmp_path / "reports" / "forecast_context",
    )
    root = tmp_path / "runs"
    root.mkdir()
    runs = []
    for group in ("A", "B", "C", "D"):
        mutator = None
        if group == "B":
            def mutator(config):
                config["algorithm"]["actor_lr"] = 0.0009
        runs.append(
            _build_run(
                root,
                group=group,
                seed=42,
                config_mutator=mutator,
            )
        )

    with pytest.raises(ValueError, match="differs beyond"):
        summary_module.summarize_study(
            runs,
            study_id="contaminated",
            required_seeds=[42],
        )


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "/absolute", "UpperCase", "contains space", ""],
)
def test_study_id_path_traversal_and_invalid_names_rejected(
    four_runs,
    bad_id,
):
    with pytest.raises(ValueError, match="study_id"):
        summary_module.summarize_study(
            four_runs,
            study_id=bad_id,
            required_seeds=[42],
        )


def test_existing_report_directory_is_never_overwritten(four_runs):
    summary_module.summarize_study(
        four_runs,
        study_id="immutable",
        required_seeds=[42],
    )
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        summary_module.summarize_study(
            four_runs,
            study_id="immutable",
            required_seeds=[42],
        )


def test_summary_script_has_no_environment_algorithm_or_solver_imports():
    source = Path("scripts/summarize_forecast_context.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "from envs",
        "import envs",
        "from algorithms",
        "import algorithms",
        "from solvers",
        "import solvers",
        "evaluation.evaluate",
        "coalition_evaluator",
    )
    for fragment in forbidden:
        assert fragment not in source
