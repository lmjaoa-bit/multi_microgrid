from types import SimpleNamespace

import pytest

from evaluation.evaluate_matd3 import _extract_rollout_constraint_metrics
from evaluation.service_reliability import _daily_service_metrics
from utils.metrics_collector import EpisodeMetricsCollector, json_ready


def test_non_curtailment_includes_exports_and_reports_end_inventory():
    battery = SimpleNamespace(soc=0.1, initial_soc=0.1, capacity_kwh=100)
    env = SimpleNamespace(agent_ids=['mg1'], action_design='single_dispatch',
                          cores={'mg1': SimpleNamespace(battery=battery)},
                          energy_pool=SimpleNamespace(pair_capacity_kw=None))
    collector = EpisodeMetricsCollector(env)
    collector.energy['mg1'].update(renewable_available_kwh=100, renewable_used_kwh=80,
                                  renewable_curtailment_kwh=20, grid_export_kwh=70)
    collector.soc['mg1'].append(0.3)
    result = collector.finalize_episode()
    assert result['curtailment_kwh'] == 20
    assert result['curtailment_rate'] == pytest.approx(0.2)
    assert result['renewable_non_curtailment_rate'] == pytest.approx(0.8)
    assert result['renewable_utilization_rate'] == result['renewable_non_curtailment_rate']
    assert result['soc_statistics_by_microgrid']['mg1']['final_energy_kwh'] == 30
    assert result['soc_statistics_by_microgrid']['mg1']['energy_delta_kwh'] == pytest.approx(20)
    assert result['p2p_statistics']['pair_capacity_kw'] is None
    assert json_ready(result)[1] == []


def test_departure_service_uses_once_only_shortfall_and_completed_groups():
    rows = [dict(mg3_unmet_load_kw=0, mg3_ev_departure_count=1,
                 mg3_ev_completed_group_count=1, mg3_ev_departure_shortfall_kwh=0),
            dict(mg3_unmet_load_kw=0, mg3_ev_departure_count=3,
                 mg3_ev_completed_group_count=2, mg3_ev_departure_shortfall_kwh=4)]
    result = _daily_service_metrics(rows, ['mg3'], delta_t=1, feasible_day=True, tolerance=1e-7)
    assert result['ev_departure_shortfall_kwh'] == 4
    assert result['ev_departure_success_rate'] == pytest.approx(2 / 3)
    assert not result['strict_feasible_day']


def test_disabled_terminal_restoration_is_not_service_failure():
    env = SimpleNamespace(agent_ids=['mg1', 'mg2', 'mg3'], delta_t=1, episode_length=24,
        config={'terminal_soc': {'mode': 'none'}}, cfg_by_id={
            'mg1': {'dr': {'type': 'industrial_mill', 'silo_initial_t': 10.53}},
            'mg2': {'dr': {'type': 'commercial_hvac'}},
            'mg3': {'dr': {'type': 'ev_charging', 'cohorts': [{}, {}, {}]}}})
    rows = [dict(date='2026-01-01', hour=23, mg1_silo_level_t=0,
                 mg2_terminal_air_gap_c=8, mg2_terminal_mass_gap_c=8,
                 mg3_ev_departure_count=3, mg3_ev_completed_group_count=3)]
    result = _extract_rollout_constraint_metrics(env, [{'date': '2026-01-01'}], rows)
    assert result['hvac_terminal_joint_success_rate'] == 1
    assert result['mill_terminal_success_rate'] == 1
    assert result['ev_departure_success_rate'] == 1
