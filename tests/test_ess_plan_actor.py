from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from algorithms.matd3_hierarchical_transformer.actor import HierarchicalTransformerActor
from algorithms.matd3_ess_plan_transformer.actor import EssPlanActor
from algorithms.matd3_ess_plan_transformer.plan_projection import decode_nominal_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def actor_case():
    torch.set_num_threads(1)
    cfg = yaml.safe_load((ROOT / 'configs/cooperative.yaml').read_text(encoding='utf-8'))
    env = MultiMicrogridEnv(get_algorithm_spec('matd3').build_environment_config(cfg), split='train')
    obs, _ = env.reset(seed=42, deterministic=True)
    mg = env.agent_ids[0]
    ref = HierarchicalTransformerActor(env.obs_builder.schema(mg), env.agent_ids, mg,
        mg_config=env.cfg_by_id[mg], power_normalizers=env.obs_builder.norm[mg],
        forecast_horizon=24, include_peer_forecast=True, embed_dim=8, num_heads=2,
        num_layers=1, hidden_dim=16)
    actor = EssPlanActor(ref, env.cfg_by_id[mg]['battery'], delta_t=1.)
    return env, actor, torch.tensor(obs[mg])[None]


def test_nominal_plan_obeys_efficiency_and_limits():
    b = dict(capacity_kwh=100., max_charge_kw=20., max_discharge_kw=20.,
             soc_min=.1, soc_max=.9, eta_charge=.9, eta_discharge=.8)
    intents = torch.tensor([[1., -1., 1., 0.], [-1., 1., -1., 0.]])
    initial = torch.tensor([[.1], [.9]])
    p, soc = decode_nominal_plan(intents, initial, torch.ones_like(intents, dtype=torch.bool), b, 1.)
    assert p[0, 0] == 0 and p[1, 0] == 0
    previous = torch.cat([initial, soc[:, :-1]], 1)
    expected = previous + torch.where(p < 0, -p*.9, -p/.8)/100.
    torch.testing.assert_close(soc, expected, atol=1e-6, rtol=0)
    assert torch.all((soc >= .1-1e-6) & (soc <= .9+1e-6))


def test_action_is_first_plan_step_and_invalid_slots_do_not_matter(actor_case):
    _, actor, obs = actor_case
    output = actor.forward_with_plan(obs)
    torch.testing.assert_close(output.actions[:, 0], output.intents[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(actor(obs), output.actions)
    last = obs.clone()
    last[:, actor.schema.index('hour_sin')] = np.sin(23*2*np.pi/24)
    last[:, actor.schema.index('hour_cos')] = np.cos(23*2*np.pi/24)
    for i, name in enumerate(actor.schema):
        if '_t+' in name:
            last[:, i] = 0 if 'forecast_valid' in name else float('nan')
    out = actor.forward_with_plan(last)
    assert out.mask.sum() == 1 and torch.isfinite(out.actions).all()
    assert torch.all(out.request_kw[:, 1:] == 0)
    assert torch.all(out.nominal_soc_after[:, 1:] == out.nominal_soc_after[:, :1])


def test_plan_has_forecast_gradient_without_training_dr(actor_case):
    _, actor, obs = actor_case
    out = actor.forward_with_plan(obs)
    out.intents.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.future_encoder.parameters())
    assert all(p.grad is None for p in actor.dr_actor.parameters())
    broken = obs.clone()
    broken[:, actor.schema.index('load_t+1')] = float('nan')
    with pytest.raises(ValueError, match='forecast'):
        actor(broken)


def test_replanning_uses_returned_soc_after_power_curtailment(actor_case):
    _, actor, obs = actor_case
    obs = obs.clone()
    obs[:, actor.soc_index] = .6
    with torch.no_grad():
        actor.intent_head.weight.zero_()
        actor.intent_head.bias.fill_(2.)
    original = actor.forward_with_plan(obs)
    # A curtailed discharge leaves more stored energy than the nominal plan.
    next_obs = obs.clone()
    next_obs[:, actor.soc_index] = obs[:, actor.soc_index]
    next_obs[:, actor.hour_indices[0]] = np.sin(2*np.pi/24)
    next_obs[:, actor.hour_indices[1]] = np.cos(2*np.pi/24)
    replanned = actor.forward_with_plan(next_obs)
    expected_power, expected_soc = decode_nominal_plan(replanned.intents,
        next_obs[:, actor.soc_index], replanned.mask, actor.battery, actor.delta_t)
    torch.testing.assert_close(replanned.request_kw, expected_power)
    torch.testing.assert_close(replanned.nominal_soc_after, expected_soc)
    assert original.nominal_soc_after[0, 0] < next_obs[0, actor.soc_index]
    wrong_power, wrong_soc = decode_nominal_plan(replanned.intents,
        original.nominal_soc_after[:, 0], replanned.mask, actor.battery, actor.delta_t)
    assert not torch.allclose(replanned.nominal_soc_after[:, 0], wrong_soc[:, 0])
