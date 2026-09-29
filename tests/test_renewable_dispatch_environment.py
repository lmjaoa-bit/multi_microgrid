from pathlib import Path

import numpy as np
import pytest
import yaml

from core.data_manager import DataManager
from core.grid_settlement import ExternalGrid
from envs.cooperative_energy_sharing import CooperativeEnergySharing
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


def test_default_environment_uses_common_prices_and_noncyclic_inventory():
    cfg = yaml.safe_load((ROOT / 'configs/cooperative.yaml').read_text(encoding='utf-8'))
    cfg['_project_root'] = str(ROOT)
    data = DataManager(cfg, split='train').df
    for mg in cfg['microgrids']:
        name = mg['id']
        assert np.allclose(data[f'{name}_buy_price_per_kwh'], data['buy_price_per_kwh'])
        assert np.allclose(data[f'{name}_sell_price_per_kwh'], data['buy_price_per_kwh'] * .5)
        assert mg['battery']['initial_soc'] == mg['battery']['soc_min']
        assert mg['grid']['import_limit_kw'] is None
        assert mg['grid']['export_limit_kw'] == 0
    assert cfg['p2p']['pair_capacity_kw'] is None
    assert cfg['terminal_soc']['mode'] == 'none'
    assert cfg['reward']['semantics_version'] == 'generalized_system_cost_v2'
    assert cfg['reward']['calibration_status'] == 'locked'


def test_unbounded_import_does_not_clip_large_load_but_export_is_limited():
    grid = ExternalGrid(import_limit_kw=None, export_limit_kw=100.)
    imported = grid.settle(-5000.)
    assert imported['grid_import_kw'] == 5000.
    assert imported['unmet_load_kw'] == 0.
    exported = grid.settle(175.)
    assert exported['grid_export_kw'] == 100.
    assert exported['renewable_curtailment_kw'] == 75.
    assert exported['power_balance_residual_kw'] == 0.


def test_storage_supply_can_serve_peer_load_but_not_peer_storage():
    pool = CooperativeEnergySharing()
    result = pool.allocate_structured(
        renewable_offers_kw={'a': 0., 'b': 0., 'c': 0.},
        ess_offers_kw={'a': 60., 'b': 0., 'c': 0.},
        load_demands_kw={'a': 0., 'b': 20., 'c': 0.},
        charge_demands_kw={'a': 0., 'b': 0., 'c': 40.},
        grid_sell_price=.2, grid_buy_price=.5,
    )
    assert result.by_microgrid['b']['p2p_to_load_kw'] == 20.
    assert result.by_microgrid['c']['p2p_to_battery_kw'] == 0.
    assert result.by_microgrid['a']['ess_p2p_unmatched_kw'] == 40.


def test_default_full_day_has_finite_state_and_no_midnight_restoration():
    env = MultiMicrogridEnv(ROOT / 'configs/cooperative.yaml', split='test')
    obs, info = env.reset(deterministic=True)
    assert env.action_dim == 2
    assert 'ev_sessions' in info
    for _ in range(24):
        obs, _, done, info = env.step({mg: np.array([-1., -1.]) for mg in env.agent_ids})
        assert all(np.isfinite(value).all() for value in obs.values())
        assert np.isfinite(env.global_state()).all()
        assert info['terminal_constraint_penalty_total'] == 0.
    assert done
    for core in env.cores.values():
        assert core.battery.soc > core.battery.initial_soc


def test_renewable_first_environment_exposes_zero_source_balance_residuals():
    cfg = yaml.safe_load((ROOT / "configs" / "cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(cfg, split="test")
    observations, _ = env.reset(seed=314, deterministic=True, forecast_noise=False)
    actions = {
        mg: np.zeros(env.action_dim, dtype=np.float32)
        for mg in env.agent_ids
    }
    _, _, _, info = env.step(actions)
    assert set(info["source_balance_residuals"]) == set(env.agent_ids)
    for mg, residuals in info["source_balance_residuals"].items():
        assert set(residuals) == {
            "renewable_source_residual_kw",
            "load_sink_residual_kw",
            "charge_sink_residual_kw",
            "discharge_source_residual_kw",
            "external_surplus_residual_kw",
        }
        assert all(abs(float(value)) <= 1e-7 for value in residuals.values())
        for key, value in residuals.items():
            assert info["plans"][mg][key] == pytest.approx(value)
