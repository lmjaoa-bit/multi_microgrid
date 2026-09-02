from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from components.battery import BatteryModel
from core.action_codec import ActionCodec
from algorithms.registry import get_algorithm_spec
from envs.cooperative_energy_sharing import CooperativeEnergySharing
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cooperative_v1.yaml"


def test_p2p_energy_and_transfer_conservation() -> None:
    result = CooperativeEnergySharing().allocate({"mg1": 10.0, "mg2": -4.0, "mg3": -9.0}, 0.1, 0.3)
    assert result.total_p2p_kwh == 10.0
    assert sum(x["p2p_net_kw"] for x in result.by_microgrid.values()) == 0.0
    assert abs(sum(x["p2p_transfer_payment"] for x in result.by_microgrid.values())) < 1e-10
    assert set(result.pair_flows_kwh) == {"mg1->mg2", "mg1->mg3"}


def test_p2p_pair_capacity_limits_each_direction() -> None:
    result = CooperativeEnergySharing(pair_capacity_kw=12.0).allocate(
        {"mg1": 100.0, "mg2": -100.0}, 0.1, 0.3,
    )
    assert result.pair_flows_kwh == {"mg1->mg2": 12.0}
    assert result.by_microgrid["mg1"]["p2p_sell_kw"] == 12.0
    assert result.by_microgrid["mg2"]["p2p_buy_kw"] == 12.0


def test_structured_p2p_prioritizes_load_then_battery_and_tracks_source() -> None:
    result = CooperativeEnergySharing().allocate_structured(
        renewable_offers_kw={"mg1": 4.0, "mg2": 0.0, "mg3": 0.0},
        ess_offers_kw={"mg1": 10.0, "mg2": 0.0, "mg3": 0.0},
        load_demands_kw={"mg1": 0.0, "mg2": 5.0, "mg3": 0.0},
        charge_demands_kw={"mg1": 0.0, "mg2": 0.0, "mg3": 8.0},
        grid_sell_price=0.1,
        grid_buy_price=0.3,
    )
    assert np.isclose(result.by_microgrid["mg1"]["renewable_p2p_sell_kw"], 4.0)
    assert np.isclose(result.by_microgrid["mg1"]["ess_p2p_sell_kw"], 9.0)
    assert np.isclose(result.by_microgrid["mg2"]["p2p_to_load_kw"], 5.0)
    assert np.isclose(result.by_microgrid["mg3"]["p2p_to_battery_kw"], 8.0)
    assert np.isclose(result.by_microgrid["mg1"]["ess_p2p_unmatched_kw"], 1.0)


def test_sharing_action_is_offer_cap_not_executed_trade_ratio() -> None:
    codec = ActionCodec("explicit_offer")
    assert codec.decode([0.0, -1.0])["sharing_offer_ratio"] == 0.0
    assert codec.decode([0.0, 0.0])["sharing_offer_ratio"] == 0.5
    assert codec.decode([0.0, 1.0])["sharing_offer_ratio"] == 1.0

    result = CooperativeEnergySharing().allocate_structured(
        renewable_offers_kw={"mg1": 0.0, "mg2": 0.0, "mg3": 0.0},
        ess_offers_kw={"mg1": 10.0, "mg2": 0.0, "mg3": 0.0},
        load_demands_kw={"mg1": 0.0, "mg2": 2.0, "mg3": 0.0},
        charge_demands_kw={"mg1": 0.0, "mg2": 0.0, "mg3": 0.0},
        grid_sell_price=0.1,
        grid_buy_price=0.3,
    )
    assert result.by_microgrid["mg1"]["ess_p2p_offer_kw"] == 10.0
    assert result.by_microgrid["mg1"]["ess_p2p_sell_kw"] == 2.0
    assert result.by_microgrid["mg1"]["ess_p2p_unmatched_kw"] == 8.0


def test_independent_mode_disables_p2p() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["_project_root"] = str(ROOT)
    config["env"]["mode"] = "independent"
    env = MultiMicrogridEnv(config)
    env.reset(seed=1, deterministic=True)
    _, _, _, info = env.step({mg: np.zeros(1) for mg in env.agent_ids})
    assert info["p2p"]["total_p2p_kwh"] == 0.0


def test_battery_soc_and_power_feasibility() -> None:
    battery = BatteryModel(100, 20, 30, initial_soc=0.5, delta_t=1.0)
    battery.reset()
    mapped = battery.map_action(1.0)
    assert mapped["discharge_requested_kw"] <= 30.0
    state = battery.commit(0.0, mapped["discharge_requested_kw"])
    assert battery.soc_min <= state["soc"] <= battery.soc_max


def test_environment_has_one_battery_action_and_no_capacity_commitment_fields() -> None:
    env = MultiMicrogridEnv(CONFIG)
    env.reset(seed=2, deterministic=True)
    _, rewards, _, info = env.step({mg: np.asarray([0.0]) for mg in env.agent_ids})
    assert env.action_dim == 1
    assert len(set(rewards.values())) == 1
    assert "reserve" not in env.config
    assert "reserve" not in info
    assert all(abs(x["power_balance_residual_kw"]) < 1e-7 for x in info["settlements"].values())


def test_unmatched_full_offer_remains_in_battery() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["_project_root"] = str(ROOT)
    config["p2p"]["enabled"] = False
    config["ess_sharing"] = {"control_mode": "fixed_full_offer", "observation_mode": "local"}
    env = MultiMicrogridEnv(config, split="test")
    env.reset(day=str(env.data.valid_days[0].date()), seed=4, deterministic=True)
    actions = {mg: np.asarray([1.0 if mg == "mg1" else 0.0]) for mg in env.agent_ids}
    _, _, _, info = env.step(actions)
    plan = info["plans"]["mg1"]
    assert plan["ess_p2p_offer_kw"] > 0.0
    assert np.isclose(plan["ess_p2p_unmatched_kw"], plan["ess_p2p_offer_kw"])
    assert np.isclose(plan["battery_discharge_kw"], plan["ess_to_local_kw"])
    assert plan["battery_discharge_kw"] < plan["battery_discharge_requested_kw"]
    assert np.isclose(info["devices"]["mg1"]["discharge_kw"], plan["battery_discharge_kw"])


def test_ess_p2p_sale_never_coincides_with_external_grid_import() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["_project_root"] = str(ROOT)
    config = get_algorithm_spec("matd3_ess_sharing").build_environment_config(config)
    env = MultiMicrogridEnv(config, split="test")
    env.reset(day=str(env.data.valid_days[0].date()), seed=5, deterministic=True)
    actions = {
        "mg1": np.asarray([1.0, 1.0]),
        "mg2": np.asarray([-1.0, -1.0]),
        "mg3": np.asarray([-1.0, -1.0]),
    }
    _, _, _, info = env.step(actions)
    assert info["p2p"]["by_microgrid"]["mg1"]["ess_p2p_sell_kw"] > 0.0
    for mg in env.agent_ids:
        if info["p2p"]["by_microgrid"][mg]["p2p_sell_kw"] > 1e-8:
            assert info["settlements"][mg]["grid_import_kw"] <= 1e-8


def test_terminal_reachability_bounds() -> None:
    env = MultiMicrogridEnv(CONFIG)
    env.reset(seed=3, deterministic=True)
    done = False
    info = {}
    while not done:
        _, _, done, info = env.step({mg: np.asarray([1.0]) for mg in env.agent_ids})
    for core in env.cores.values():
        assert core.battery.soc_min <= float(core.battery.soc) <= core.battery.soc_max
        # Reachability projection prevents extreme drop
        assert float(core.battery.soc) >= core.battery.initial_soc - 0.03 - 1e-6
