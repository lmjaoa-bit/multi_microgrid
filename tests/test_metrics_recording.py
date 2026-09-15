from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from envs.multi_microgrid_env import MultiMicrogridEnv
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
    assert p2p_stats["pair_capacity_kw"] == 75.0
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


def test_json_writer_converts_numpy_and_rejects_nonfinite_json(tmp_path: Path) -> None:
    target = tmp_path / "metrics.json"
    write_json(target, {"scalar": np.float32(1.5), "bad": float("nan")})
    value = json.loads(target.read_text(encoding="utf-8"))
    assert value["scalar"] == 1.5
    assert value["bad"] is None
    assert value["serialization_warnings"][0]["field"] == "bad"
    assert distribution_statistics([1.0, 2.0, 3.0])["median"] == 2.0
