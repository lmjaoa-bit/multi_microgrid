from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case():
    from algorithms.matd3_plan_transformer.actor import PlanTransformerActor
    cfg = yaml.safe_load((ROOT / 'configs/cooperative.yaml').read_text(encoding='utf-8'))
    cfg['_project_root'] = str(ROOT)
    env = MultiMicrogridEnv(cfg, split='train')
    obs, _ = env.reset(seed=17, deterministic=True, forecast_noise=False)
    actor = PlanTransformerActor(env.obs_builder.schema('mg2'), env.agent_ids, 'mg2',
        env.cfg_by_id['mg2'], env.obs_builder.norm['mg2'], {})
    return env, actor, torch.tensor(obs['mg2'][None])


def test_h24_shape_and_day_end_masks(case):
    env, actor, initial = case
    for _ in range(23):
        obs, _, _, _ = env.step({mg: np.zeros(2) for mg in env.agent_ids})
    result = actor.forward_with_plan(torch.cat([initial, torch.tensor(obs['mg2'][None])]), True)
    assert result.actions.shape == (2, 2)
    assert result.soc_after.shape == (2, 24)
    assert result.attention.shape == (2, 2, 4, 26)
    assert result.plan_mask.sum(dim=1).tolist() == [24, 1]
    assert result.forecast_mask[1].sum() == 0
    assert torch.isfinite(result.actions).all()
    assert (result.actions.abs() <= 1).all()


def test_invalid_forecast_padding_cannot_affect_actions(case):
    _, actor, obs = case
    clean = obs.clone()
    bad = obs.clone()
    for k in range(1, 25):
        i = actor.schema.index(f'forecast_valid_t+{k}')
        clean[:, i] = bad[:, i] = 0
        for j, name in enumerate(actor.schema):
            if name.endswith(f'_t+{k}') and 'forecast_valid' not in name:
                bad[:, j] = float('nan')
    torch.testing.assert_close(actor(clean), actor(bad))
    invalid = clean.clone()
    invalid[:, actor.schema.index('soc')] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        actor(invalid)


def test_forecast_base_load_survives_net_load_merge(case):
    env, actor, obs = case
    variants = obs.repeat(2, 1)
    norm = env.obs_builder.norm['mg2']
    for row, (load, wind) in enumerate(((100, 50), (200, 150))):
        variants[row, actor.schema.index('load_t+1')] = load / norm['load']
        variants[row, actor.schema.index('wind_t+1')] = wind / norm['wind']
        variants[row, actor.schema.index('pv_t+1')] = 0
    f = actor._future_features(variants)
    torch.testing.assert_close(f[:, 0, 1], torch.full((2,), 50 / norm['power']))
    assert f[0, 0, 0] != f[1, 0, 0]


def test_auxiliary_losses_train_both_attention_layers_and_use_actual_power(case):
    from algorithms.matd3_plan_transformer.plan_losses import plan_loss_terms
    env, actor, obs = case
    output = actor.forward_with_plan(obs)
    batch = SimpleNamespace(mask=output.plan_mask.float(),
        soc_after={'mg2': torch.full((1, 24), .3)}, ess_net_kw={'mg2': torch.zeros(1, 24)})
    projected = {'actual_charge_kw': torch.tensor([3.]), 'actual_discharge_kw': torch.tensor([0.]),
                 'requested_power': torch.tensor([-80.])}
    terms = plan_loss_terms(output, batch, 'mg2', projected, env.cfg_by_id['mg2']['battery'], 1.)
    (terms.plan + terms.dynamics + terms.first).backward()
    assert all(sum(p.grad.abs().sum().item() for p in layer.parameters() if p.grad is not None) > 0
               for layer in actor.layers)
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
    # Requested power is deliberately different; changing it must not alter first loss.
    projected['requested_power'] = torch.tensor([65.])
    other = plan_loss_terms(output, batch, 'mg2', projected, env.cfg_by_id['mg2']['battery'], 1.)
    torch.testing.assert_close(terms.first, other.first)
