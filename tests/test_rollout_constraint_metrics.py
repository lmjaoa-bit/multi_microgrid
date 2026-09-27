from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from envs.multi_microgrid_env import MultiMicrogridEnv
from evaluation.evaluate_matd3 import _extract_rollout_constraint_metrics


def test_rollout_metrics_separate_mill_production_from_downstream_service():
    env_config = yaml.safe_load(
        Path("configs/cooperative.yaml").read_text(encoding="utf-8")
    )
    env = MultiMicrogridEnv(env_config, split="train")
    mill_cfg = env.cfg_by_id["mg1"]["dr"]
    date = "2018-01-01"
    start = datetime.fromisoformat(f"{date}T00:00:00")
    production_total = 16.51
    required_total = 23.40
    production_step = production_total / 24.0
    required_step = required_total / 24.0
    silo = float(mill_cfg["silo_initial_t"])
    rows = []

    for hour in range(24):
        before = silo
        raw_after = before + production_step - required_step
        silo = raw_after
        action_ts = start + timedelta(hours=hour)
        result_ts = action_ts + timedelta(hours=1)
        row = {
            "date": date,
            "hour": hour,
            "timestep": hour,
            "timestamp": action_ts.isoformat(),
            "mg1_unmet_load_kw": 0.0,
            "mg2_unmet_load_kw": 0.0,
            "mg3_unmet_load_kw": 0.0,
            "mg1_dr_supply_infeasible": 0,
            "mg2_dr_supply_infeasible": 0,
            "mg3_dr_supply_infeasible": 0,
            "mg1_mill_actual_production_t": production_step,
            "mg1_mill_downstream_required_t": required_step,
            "mg1_mill_actual_consumption_t": required_step,
            "mg1_mill_downstream_unserved_t": 0.0,
            "mg1_mill_silo_before_t": before,
            "mg1_mill_silo_after_unclipped_t": raw_after,
            "mg1_silo_level_t": silo,
            "mg2_result_timestamp": result_ts.isoformat(),
            "mg2_t_air_c": 23.0,
            "mg2_result_comfort_low_c": 16.0,
            "mg2_result_comfort_high_c": 30.0,
            "mg2_result_occupied": 0,
            "mg2_dr_projected": 0.0,
            "mg3_dr_projected": 0.0,
            "mg3_ev_departure_count": 3 if hour == 23 else 0,
            "mg3_ev_completed_group_count": 3 if hour == 23 else 0,
            "mg3_ev_departure_shortfall_kwh": 0.0,
        }
        rows.append(row)

    summary = _extract_rollout_constraint_metrics(
        env,
        [{"date": date, "constraint_statistics": {"feasible_episode": True}}],
        rows,
    )

    assert summary["strict_feasible_day_rate"] == pytest.approx(1.0)
    assert summary["mill_terminal_success_rate"] is None
    assert summary["hvac_terminal_joint_success_rate"] is None
    assert summary["mill_production_completion_rate"] == pytest.approx(
        production_total / required_total
    )
    assert summary["mill_production_coverage_ratio"] == pytest.approx(
        production_total / required_total
    )
    assert summary["mill_downstream_service_completion_rate"] == pytest.approx(1.0)
    assert summary["mill_terminal_or_production_success_rate"] == pytest.approx(
        production_total / required_total
    )
    assert summary["ev_departure_success_rate"] == pytest.approx(1.0)
