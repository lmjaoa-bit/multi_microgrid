from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.evaluate_matd3 import _extract_rollout_constraint_metrics


def test_mill_terminal_or_production_accepts_completed_daily_production():
    env_config = yaml.safe_load(
        Path("configs/cooperative.yaml").read_text(encoding="utf-8")
    )
    env = MultiMicrogridEnv(env_config, split="train")
    mill_cfg = env.cfg_by_id["mg1"]["dr"]
    date = "2018-01-01"
    rows = []
    for hour in range(24):
        rows.append(
            {
                "date": date,
                "hour": hour,
                "mg1_unmet_load_kw": 0.0,
                "mg2_unmet_load_kw": 0.0,
                "mg3_unmet_load_kw": 0.0,
                "power_balance_violation_kw": 0.0,
                "p2p_balance_violation_kw": 0.0,
                "mg1_mill_production_tph": float(mill_cfg["downstream_consumption_tph"]),
                "mg1_silo_level_t": float(mill_cfg["silo_initial_t"]) + 10.0,
                "mg2_terminal_air_gap_c": 0.0,
                "mg2_terminal_mass_gap_c": 0.0,
                "mg2_occupied": 0.0,
                "mg2_t_air_c": 23.0,
                "mg2_dr_projected": 0.0,
                "mg3_dr_projected": 0.0,
                "mg3_ev_soc": 1.0,
            }
        )

    summary = _extract_rollout_constraint_metrics(
        env,
        [{"date": date, "constraint_statistics": {"feasible_episode": True}}],
        rows,
    )

    assert summary["mill_terminal_success_rate"] == 0.0
    assert summary["mill_production_completion_rate"] == pytest.approx(1.0, abs=1e-12)
    assert summary["mill_terminal_or_production_success_rate"] == pytest.approx(1.0, abs=1e-12)
