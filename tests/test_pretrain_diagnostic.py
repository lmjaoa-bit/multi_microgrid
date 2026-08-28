from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from envs.multi_microgrid_env import MultiMicrogridEnv
from scripts.pretrain_diagnostic import (
    build_static_timeseries,
    compute_curtailment_potential,
    compute_microgrid_statistics,
    compute_p2p_potential,
    run_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multi_microgrid.yaml"


def test_static_p2p_and_curtailment_diagnostics_obey_physical_bounds():
    env = MultiMicrogridEnv(CONFIG, split="train")
    static = build_static_timeseries(env)
    statistics = compute_microgrid_statistics(env, static)
    potential, p2p_summary, monthly, seasonal = compute_p2p_potential(env, static)
    comparison, curtailment = compute_curtailment_potential(env, static, potential)

    assert len(static) == len(env.data.df)
    assert set(statistics["microgrid"]) == set(env.agent_ids)
    assert np.all(
        potential["theoretical_p2p_match_kw"].to_numpy()
        <= potential["total_surplus_kw"].to_numpy() + 1e-12
    )
    assert np.all(
        potential["theoretical_p2p_match_kw"].to_numpy()
        <= potential["total_deficit_kw"].to_numpy() + 1e-12
    )
    assert p2p_summary["ideal_p2p_buy_kwh"] == pytest.approx(
        p2p_summary["ideal_p2p_sell_kwh"]
    )
    assert (comparison["no_p2p_curtailment_kw"] >= 0.0).all()
    assert (comparison["ideal_p2p_curtailment_kw"] >= 0.0).all()
    assert (
        comparison["ideal_p2p_curtailment_kw"]
        <= comparison["no_p2p_curtailment_kw"] + 1e-12
    ).all()
    assert curtailment["potential_curtailment_reduction_kwh"] >= 0.0
    assert np.isfinite(monthly.select_dtypes(include=[np.number]).to_numpy()).all()
    assert np.isfinite(seasonal.select_dtypes(include=[np.number]).to_numpy()).all()


def test_diagnostic_outputs_are_finite_train_only_and_seed_reproducible(tmp_path):
    base_env = MultiMicrogridEnv(CONFIG, split="train")
    config = copy.deepcopy(base_env.config)
    # A successful run proves the diagnostic never attempts to read val/test.
    config["data"]["val_path"] = "data/DO_NOT_READ_val.csv"
    config["data"]["test_path"] = "data/DO_NOT_READ_test.csv"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = run_diagnostic(
        config,
        first_dir,
        dynamic_days=1,
        random_days=1,
        seed=77,
        create_plots=False,
    )
    second = run_diagnostic(
        config,
        second_dir,
        dynamic_days=1,
        random_days=1,
        seed=77,
        create_plots=False,
    )
    assert first == second
    assert first["metadata"]["split"] == "train"

    required = {
        "summary.json",
        "summary.csv",
        "microgrid_statistics.csv",
        "p2p_potential.csv",
        "reward_statistics.csv",
        "rule_based_statistics.csv",
        "diagnostic_report.md",
    }
    assert required <= {path.name for path in first_dir.iterdir()}
    with (first_dir / "summary.json").open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded == first

    for path in first_dir.glob("*.csv"):
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy(dtype=np.float64)).all(), path.name

    rewards = pd.read_csv(first_dir / "reward_statistics.csv")
    assert set(rewards["policy"]) == {
        "legal_random_p2p",
        "rule_based_no_p2p",
        "rule_based_p2p",
    }
    assert set(rewards["microgrid"]) == set(base_env.agent_ids)
    assert np.isfinite(rewards.filter(regex=r"^(reward_|abs_reward_)").to_numpy()).all()

    rule_p2p = first["policy_cases"]["rule_based_p2p"]["by_microgrid"]
    total_buy = sum(values["p2p_buy_kwh"] for values in rule_p2p.values())
    total_sell = sum(values["p2p_sell_kwh"] for values in rule_p2p.values())
    assert total_buy == pytest.approx(total_sell, abs=1e-7)

    report = (first_dir / "diagnostic_report.md").read_text(encoding="utf-8")
    for heading in (
        "## Microgrid statistics",
        "## P2P potential",
        "## Renewable curtailment",
        "## Rule-based comparison",
        "## Reward statistics",
        "## Final diagnostic conclusion",
    ):
        assert heading in report
