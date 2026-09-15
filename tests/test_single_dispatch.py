from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from experiments.training_pipeline import _diagnostic_means

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG = ROOT / "configs" / "cooperative_legacy_pre_dr.yaml"
NEW_CONFIG = ROOT / "configs" / "cooperative.yaml"


def build(split="train", disabled=False, terminal_mode="none"):
    """Build the frozen 1-D ESS-only contract used by projector regression tests."""
    spec = get_algorithm_spec("crossgate")
    cfg = yaml.safe_load(LEGACY_CONFIG.read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    cfg["terminal_soc"]["mode"] = terminal_mode
    if terminal_mode != "none":
        for mg in cfg["microgrids"]:
            mg["battery"]["initial_soc"] = .55
    if disabled:
        cfg["p2p"]["enabled"] = False
    env = MultiMicrogridEnv(spec.build_environment_config(cfg), split=split)
    alg = spec.create(
        env,
        yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8")),
        seed=7,
    )
    return env, alg


def build_dr_matd3(split="train"):
    spec = get_algorithm_spec("matd3_dispatch")
    cfg = yaml.safe_load(NEW_CONFIG.read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(cfg), split=split)
    alg = spec.create(
        env,
        yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8")),
        seed=7,
    )
    return env, alg


def synthetic_state(env):
    _, info = env.reset(seed=7, deterministic=True)
    state = torch.tensor(info["global_state"], dtype=torch.float64).unsqueeze(0)
    index = {name: i for i, name in enumerate(env.global_state_builder.schema())}
    norm = env.global_state_builder.norm
    for mg in env.agent_ids:
        for name in ("load", "pv", "wind", "feasible_charge_min", "feasible_discharge_min"):
            state[:, index[f"{mg}_{name}"]] = 0
        for kind in ("charge", "discharge"):
            state[:, index[f"{mg}_feasible_{kind}_max"]] = 30 / getattr(
                env.cores[mg].battery, f"max_{kind}_kw"
            )
        state[:, index[f"{mg}_grid_buy_price"]] = .8 / norm[mg]["price"]
        state[:, index[f"{mg}_grid_sell_price"]] = .1 / norm[mg]["price"]
    state[:, index["mg1_load"]] = 10 / norm["mg1"]["load"]
    state[:, index["mg2_load"]] = 15 / norm["mg2"]["load"]
    return state, index


def test_zero_partial_discharge_and_offer_are_continuous():
    env, alg = build()
    state, _ = synthetic_state(env)
    for kw in (0, 2, 5, 9, 10, 20):
        raw = torch.tensor([[[kw / 30], [0.], [0.]]], dtype=torch.float64, requires_grad=True)
        effective, plans = alg.project_actions(state, raw)
        p = plans[0]
        assert float(p["local"].detach()) == pytest.approx(min(kw, 10))
        assert float(p["offer"].detach()) == pytest.approx(max(kw - 10, 0))
        assert float(p["local_priority_reserve"]) == 0
        if 0 < kw < 10:
            effective[:, 0, 0].sum().backward()
            assert raw.grad[0, 0, 0] > 0
    for edge in (0., 10 / 30):
        raw = torch.zeros((1, 3, 1), dtype=torch.float64)
        raw[0, 0, 0] = edge - 1e-7
        left = alg.project_actions(state, raw)[0]
        raw[0, 0, 0] = edge + 1e-7
        right = alg.project_actions(state, raw)[0]
        assert torch.max(torch.abs(right - left)) < 1e-6
    zero, plans = alg.project_actions(state, torch.zeros((1, 3, 1), dtype=torch.float64))
    assert not torch.any(zero)
    assert all(float(p["charge"]) == 0 for p in plans)


def test_partial_actions_do_not_force_all_potential_buyers_to_discharge():
    env, alg = build()
    state, index = synthetic_state(env)
    for mg in env.agent_ids:
        state[:, index[f"{mg}_load"]] = 10 / env.global_state_builder.norm[mg]["load"]
    raw = torch.full((1, 3, 1), .2, dtype=torch.float64, requires_grad=True)
    effective, plans = alg.project_actions(state, raw)
    assert all(float(p["local"].detach()) == pytest.approx(6) for p in plans)
    assert all(float(p["offer"].detach()) == 0 for p in plans)
    effective.sum().backward()
    assert torch.all(raw.grad > 0)


def test_capacity_price_masks_and_gradcheck():
    env, alg = build()
    state, index = synthetic_state(env)
    raw = torch.tensor([[[.7], [-.2], [.2]]], dtype=torch.float64, requires_grad=True)
    _, plans = alg.project_actions(state, raw)
    for p in plans:
        assert float((p["local"] + p["mandatory_export"] + p["offer"]).detach()) <= 30 + 1e-8
        assert not (float(p["charge"].detach()) > 0 and float(p["local"].detach()) > 0)
    assert torch.autograd.gradcheck(
        lambda a: alg.project_actions(state, a)[0], (raw,), eps=1e-6, atol=1e-4
    )
    state[:, index["mg1_grid_sell_price"]] = 2 / env.global_state_builder.norm["mg1"]["price"]
    assert float(alg.project_actions(state, raw)[1][0]["offer"].detach()) == 0
    with pytest.raises(ValueError, match="requires raw"):
        alg.project_actions(state, torch.zeros((1, 3, 2)))


@pytest.mark.parametrize("kind", ["charge", "discharge"])
def test_forced_intervals_use_the_whole_raw_range(kind):
    env, alg = build()
    state, index = synthetic_state(env)
    b = env.cores["mg1"].battery
    state[:, index[f"mg1_feasible_{kind}_min"]] = 6 / getattr(b, f"max_{kind}_kw")
    other = "charge" if kind == "discharge" else "discharge"
    state[:, index[f"mg1_feasible_{other}_max"]] = 0
    requested = []
    for value in (-1., -.5, 0., .5, 1.):
        raw = torch.tensor([[[value], [0.], [0.]]], dtype=torch.float64, requires_grad=True)
        _, plans = alg.project_actions(state, raw)
        p = plans[0]
        requested.append(float(p["requested_power"].detach()))
        assert bool(p["forced_dispatch"])
        p["requested_power"].sum().backward()
        # PyTorch versions differ on the derivative assigned exactly at a clamp
        # boundary. Interior raw actions must retain a strictly positive gradient;
        # endpoints are validated by the full-range mapping assertion below.
        if abs(value) < 1.0:
            assert raw.grad[0, 0, 0] > 0
        else:
            assert torch.isfinite(raw.grad[0, 0, 0])
        actual = p["local"] + p["mandatory_export"] + p["offer"]
        assert float(p["charge"].detach() if kind == "charge" else actual.detach()) >= 6 - 1e-7
    assert requested == pytest.approx(
        [-30, -24, -18, -12, -6] if kind == "charge" else [6, 12, 18, 24, 30]
    )


def test_crossgate_rollout_update_diagnostics_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    env, alg = build()
    observations, info = env.reset(seed=7, deterministic=True)
    state = info["global_state"]
    diagnostic_records = []
    assert env.action_dim == alg.action_dim == 1
    done = False
    for _ in range(24):
        actions = alg.select_actions(observations, .1)
        with torch.no_grad():
            effective, plans = alg.project_actions(
                torch.tensor(state).unsqueeze(0),
                torch.tensor(np.asarray([actions[mg] for mg in env.agent_ids])).unsqueeze(0),
            )
        next_obs, rewards, done, info = env.step(actions)
        for i, mg in enumerate(env.agent_ids):
            p = info["plans"][mg]
            assert p["effective_local_action"] == pytest.approx(float(effective[0, i, 0]), abs=2e-6)
            assert p["effective_trade_action"] == pytest.approx(float(effective[0, i, 1]), abs=2e-6)
            assert p["ess_p2p_offer_kw"] == pytest.approx(float(plans[i]["offer"]), abs=2e-4)
            assert p["direct_local_priority_reserve_kw"] == 0
            remaining = p["ess_flexible_discharge_kw"]
            assert p["sharing_offer_ratio"] == pytest.approx(
                p["ess_p2p_offer_kw"] / remaining if remaining > 1e-9 else 0.0
            )
            assert not (
                info["devices"][mg]["charge_kw"] > 1e-6
                and info["devices"][mg]["discharge_kw"] > 1e-6
            )
        alg.replay.add(
            observations, state, actions, rewards[env.agent_ids[0]],
            next_obs, info["global_state_after"], done,
        )
        metrics = alg.update(4)
        if "actor_diagnostics" in metrics:
            diagnostic_records.append(metrics)
            for diagnostic in metrics["actor_diagnostics"].values():
                assert diagnostic["post_clip_grad_norm"] <= alg.actor_max_grad_norm + 1e-5
                assert diagnostic["local_override_kw_max"] == 0
                assert "dispatch_action_grad_rms" in diagnostic
                assert np.isfinite(list(diagnostic.values())).all()
        observations, state = next_obs, info["global_state_after"]
    assert diagnostic_records and done
    aggregate = _diagnostic_means(diagnostic_records)
    assert "dispatch_action_grad_rms" in aggregate["actor_diagnostics"]["mg3"]

    path = tmp_path / "best_model.pt"
    alg.save(path)
    _, restored = build(split="test")
    restored.load(path)
    for mg, action in alg.select_actions(observations).items():
        assert np.array_equal(action, restored.select_actions(observations)[mg])

    # The migrated MATD3 uses a different algorithm/action space and must reject
    # a legacy CrossGate checkpoint rather than silently interpreting it.
    _, dr_matd3 = build_dr_matd3(split="test")
    with pytest.raises(ValueError, match="different algorithm"):
        dr_matd3.load(path)

    incompatible = build(disabled=True)[1]
    with pytest.raises(ValueError, match="physical projection"):
        incompatible.load(path)


def test_forced_terminal_export_without_market():
    env, _ = build(disabled=True, terminal_mode="penalty")
    env.reset(seed=1, deterministic=True)
    env.t = 23
    for mg in env.agent_ids:
        env.cores[mg].battery.soc = .7
        for field in ("load_kw", "pv_kw", "wind_kw", "renewable_kw"):
            env._episode_exogenous[mg][field][23] = 0.
    env._last_global_state = env._build_global_state()
    _, _, done, info = env.step({mg: [-1.] for mg in env.agent_ids})
    assert done
    for mg in env.agent_ids:
        b = env.cores[mg].battery
        assert info["plans"][mg]["mandatory_export_kw"] > 0
        assert info["plans"][mg]["dispatch_forced"]
        assert info["plans"][mg]["ess_p2p_offer_kw"] == 0
        assert b.soc <= b.initial_soc + .030001


def test_no_market_optional_energy_stays_in_battery():
    env, alg = build(disabled=True)
    state, _ = synthetic_state(env)
    _, plans = alg.project_actions(state, torch.ones((1, 3, 1), dtype=torch.float64))
    assert all(float(p["offer"]) == 0 for p in plans)
    assert float(plans[0]["local"]) == pytest.approx(10)
    assert float(plans[2]["local"]) == 0
