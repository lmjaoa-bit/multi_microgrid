import copy
import numpy as np
import pytest
import yaml
from optimization.common_constraints import DispatchData, solve_dispatch


def case(load, renewable):
    cfg = yaml.safe_load(open("configs/cooperative.yaml", encoding="utf-8"))
    ids = ["mg1", "mg2"]
    cfg["microgrids"] = [m for m in cfg["microgrids"] if m["id"] in ids]
    cfg["p2p"].update(pair_capacity_kw=None, allow_grid_resale=False, allow_ess_to_ess=False)
    cfg["terminal_soc"]["mode"] = "none"
    for m in cfg["microgrids"]:
        m["grid"].update(import_limit_kw=None, export_limit_kw=100)
        m["battery"].update(initial_soc=0.1, soc_min=0.1)
    data = DispatchData(["2018-04-27 00:00:00"], ids,
        dict(zip(ids, map(lambda x: np.array([x]), load))),
        dict(zip(ids, map(lambda x: np.array([x]), renewable))),
        np.array([0.6]), np.array([0.3]))
    return cfg, data


def test_unlimited_import_and_pairs_have_finite_physical_bounds():
    cfg, data = case([1000, 1000], [0, 0])
    result = solve_dispatch(cfg, data, data.agent_ids, True)
    assert result.trajectories.grid_import_kw.sum() == pytest.approx(2000)
    assert result.trajectories.unmet_load_kw.sum() == pytest.approx(0)


def test_battery_cannot_export_to_grid_even_at_high_sell_price():
    cfg, data = case([0, 0], [0, 0])
    cfg["microgrids"][0]["battery"]["initial_soc"] = 0.8
    result = solve_dispatch(cfg, data, data.agent_ids, True)
    assert result.trajectories.grid_export_kw.sum() == pytest.approx(0)
    assert result.trajectories.battery_discharge_kw.sum() == pytest.approx(0)


def test_battery_cannot_charge_peer_battery_to_meet_terminal_target():
    cfg, data = case([0, 0], [0, 0])
    cfg["microgrids"][0]["battery"]["initial_soc"] = 0.8
    cfg["terminal_soc"].update(mode="hard", tolerance=0)
    with pytest.raises(RuntimeError, match="failed"):
        solve_dispatch(cfg, data, data.agent_ids, True,
            terminal_target_soc_by_microgrid={"mg1":0.7,"mg2":0.2})
