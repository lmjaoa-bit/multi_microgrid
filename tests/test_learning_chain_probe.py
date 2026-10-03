from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


def module():
    assert importlib.util.find_spec('diagnostics.learning_chain_probe') is not None, 'Learning-chain diagnosis is missing'
    return importlib.import_module('diagnostics.learning_chain_probe')


def test_local_probes_keep_other_controls_and_measure_clipped_steps():
    raw = np.array([[.999, -.3], [.2, -.4], [.1, .6]], dtype=np.float32)
    before = raw.copy()
    candidates = module().local_candidates(raw, [.02, .1])
    assert len(candidates) == 25
    np.testing.assert_array_equal(raw, before)
    for candidate in candidates[1:]:
        assert np.count_nonzero(candidate.raw != before) <= 1
        assert np.abs(candidate.raw).max() <= 1
    first_positive = next(c for c in candidates if c.agent_index == 0 and c.control_index == 0 and c.sign == 1 and c.radius == .02)
    assert first_positive.raw[0, 0] - raw[0, 0] == pytest.approx(.001, abs=1e-7)


def test_direction_candidates_are_comparable_and_do_not_move_other_agents():
    raw = np.zeros((3, 2), dtype=np.float32)
    gradients = [dict(raw_gradient=np.array([2., -1.]), actor_response=np.array([0., 0.])) for _ in range(3)]
    candidates = module().direction_candidates(raw, gradients, .02)
    assert len(candidates) == 6
    np.testing.assert_allclose(candidates[0].raw[0], [.02, -.01])
    np.testing.assert_array_equal(candidates[0].raw[1:], raw[1:])
    np.testing.assert_array_equal(candidates[1].raw, raw)


def test_fixed_case_selection_balances_origins_and_covers_seasons_and_seeds():
    pool = []
    categories = ['ordinary', 'ev_deadline', 'hvac_boundary', 'ess_boundary', 'peak_pressure', 'ordinary']
    for seed in (7, 1007):
        for day_index in range(24):
            day = f'2018-{day_index // 2 + 1:02d}-{24 + day_index % 2}'
            for slot, category in enumerate(categories):
                pool.append(SimpleNamespace(day=day, forecast_seed_base=seed, category=category,
                    origin=['base', 'tq'][(slot + day_index) % 2], case_id=f'{seed}-{day}-{slot}'))
    selected = module().select_local_cases(pool)
    assert len(selected) == 48
    assert len({x.case_id for x in selected}) == 48
    for seed in (7, 1007):
        cases = [x for x in selected if x.forecast_seed_base == seed]
        assert sum(x.origin == 'base' for x in cases) == 12
        assert sum(x.origin == 'tq' for x in cases) == 12
    assert sorted(x.category for x in selected).count('ordinary') == 16


class LinearCritic(nn.Module):
    def q1(self, state, effective):
        return effective[:, :1] * 2 + effective[:, 3:4] * 3


class ToyModel:
    device = torch.device('cpu')
    agent_ids = ['mg1', 'mg2', 'mg3']
    critics = {'mg1': LinearCritic(), 'mg2': LinearCritic(), 'mg3': LinearCritic()}
    def project_actions(self, state, raw):
        # MG1 raw controls also affect MG2 effective dispatch.
        return torch.stack([raw[:, 0, 0], raw[:, 0, 1], raw[:, 0, 0] * 0,
                            raw[:, 0, 0] + raw[:, 1, 0], raw[:, 1, 1], raw[:, 1, 0] * 0,
                            raw[:, 2, 0], raw[:, 2, 1], raw[:, 2, 0] * 0], dim=-1).reshape(-1, 3, 3), []


def test_gradient_trace_includes_cross_agent_projection_effects():
    result = module().projection_trace(ToyModel(), np.zeros(2), np.zeros((3, 2)))
    assert result['jacobian'].shape == (9, 6)
    assert result['q1_raw_gradients'][0, 0] == pytest.approx(5.)
    assert result['q1_raw_gradients'][0, 2] == pytest.approx(3.)
    assert result['chain_residual_max'] < 1e-6


def test_actor_trace_restores_flags_and_keeps_weights_and_grads_unchanged():
    model = ToyModel()
    actor = nn.Module()
    actor.network = nn.Sequential(nn.Linear(2, 2), nn.Tanh())
    with torch.no_grad():
        actor.network[0].weight.zero_()
        actor.network[0].bias.copy_(torch.tensor([10., 0.]))
    actor.requires_grad_(False)
    model.actors = {'mg1': actor}
    def auxiliary(active, state, raw, observations):
        loss = .01 * actor.network[:-1](observations)[:, 0].square().sum()
        return loss, dict(dispatch_q_guidance_trigger_rate=torch.tensor(1.))
    model.actor_auxiliary_loss = auxiliary
    before = [p.detach().clone() for p in actor.parameters()]
    result = module().actor_trace(model, 'mg1', np.zeros(2), np.zeros(2), np.zeros((3, 2)), np.zeros(2))
    assert result['tanh_factors'][0] == 0.
    assert result['pre_jacobian_norms'][0] > 0
    assert result['post_jacobian_norms'][0] == 0
    assert result['main_gradient_norm'] == 0
    assert result['total_gradient_norm'] > 0
    assert result['auxiliary_trigger'] == 1
    assert all(not p.requires_grad and p.grad is None for p in actor.parameters())
    for original, parameter in zip(before, actor.parameters()):
        torch.testing.assert_close(original, parameter)


def test_local_classification_does_not_confuse_value_error_with_zero_projection():
    rows = [dict(realized_gain_cost=2., predicted_gain_cost=-1., linear_gain_cost=-.5,
                 physical_delta_norm=.1, physical_linear_residual=0.)]
    metric = dict(projection_jacobian_norm=1., tanh_factor=1., main_signal=1., actor_response=1.)
    result = module().classify_coordinate(rows, metric, broad_opportunity=3., auxiliary_trigger=0., tie_cost=.1)
    assert result['local_value_wrong_count'] == 1
    assert result['local_linear_direction_conflict_count'] == 1
    assert not result['projection_barrier_opportunity']


def test_projection_barrier_is_recorded_even_when_auxiliary_can_help():
    rows = [dict(realized_gain_cost=0., predicted_gain_cost=0., linear_gain_cost=0.,
                 physical_delta_norm=0., physical_linear_residual=0.)]
    metric = dict(projection_jacobian_norm=0., tanh_factor=.001, main_signal=0., actor_response=0.)
    result = module().classify_coordinate(rows, metric, broad_opportunity=5., auxiliary_trigger=1., tie_cost=.1)
    assert result['projection_barrier_opportunity']
    assert result['actor_saturation_opportunity']
    assert result['auxiliary_available']
    assert result['local_value_wrong_count'] == 0


def test_exact_curved_value_is_not_classified_as_a_wrong_critic_derivative():
    # An exact Q(u)=u-100*u**2 has positive derivative at u=0, but
    # a finite positive step can reduce Q. The projection is identity.
    rows = [dict(realized_gain_cost=-2., predicted_gain_cost=-2., linear_gain_cost=2.,
                 physical_delta_norm=.02, physical_linear_residual=0.)]
    metric = dict(projection_jacobian_norm=1., tanh_factor=1.)
    result = module().classify_coordinate(rows, metric, broad_opportunity=0., auxiliary_trigger=0., tie_cost=.1)
    assert result['local_value_wrong_count'] == 0
    assert result['local_linear_direction_conflict_count'] == 1
    assert result['local_q1_nonlinear_count'] == 1


def test_real_two_model_chain_smoke_uses_exact_phase_one_sources(tmp_path):
    root = Path(__file__).resolve().parents[1]
    probe = root / 'outputs/diagnostics/critic_probe/expanded_24d_2seeds_2026-10-02'
    if not (probe/'manifest.json').exists():
        pytest.skip('Expanded probe is not present on this checkout')
    result = module().run_chain_probe(root=root, probe=probe, output=tmp_path/'chain', state_limit=2)
    assert result['status'] == 'complete'
    assert result['state_count'] == 2
    assert result['source_integrity_verified']
    assert result['model_tensors_unchanged']
    assert (tmp_path/'chain/coordinate_results.csv').exists()
    assert (tmp_path/'chain/direction_results.csv').exists()
