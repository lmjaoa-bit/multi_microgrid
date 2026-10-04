from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'matd3_plan_transformer'


@pytest.fixture
def case():
    torch.set_num_threads(1)
    spec = get_algorithm_spec(SLUG)
    cfg = yaml.safe_load((ROOT / 'configs/cooperative.yaml').read_text(encoding='utf-8'))
    cfg['_project_root'] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(cfg), split='train')
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding='utf-8'))
    settings['algorithm']['auxiliary_plan']['teacher'].update(population=2, elite=1, iterations=1, max_labels=1)
    model = spec.create(env, settings, device='cpu', seed=17)
    obs, info = env.reset(seed=17, deterministic=True, forecast_noise=False)
    return spec, env, settings, model, obs, info['global_state']


def test_registry_has_own_config_and_same_environment():
    spec = get_algorithm_spec(SLUG)
    cfg = yaml.safe_load((ROOT / 'configs/cooperative.yaml').read_text(encoding='utf-8'))
    assert spec.display_name == 'Plan-Transformer'
    assert spec.build_environment_config(cfg) == cfg
    assert yaml.safe_load(spec.default_config_path.read_text(encoding='utf-8'))['algorithm']['name'] == SLUG


def test_delayed_update_uses_auxiliary_pool_and_checkpoint_roundtrip(case, tmp_path):
    spec, env, settings, model, obs, state = case
    model.capture_training_context(env, obs, state, {'day': 'train', 'episode_index': 0, 'total_steps': 0})
    for _ in range(3):
        actions = model.select_actions(obs)
        assert all(a.shape == (2,) for a in actions.values())
        nxt, rewards, done, info = env.step(actions)
        model.replay.add(obs, state, actions, rewards['mg1'], nxt, info['global_state_after'], done)
        obs, state = nxt, info['global_state_after']
    first = model.update(2)
    assert 'actor_loss' not in first
    metrics = model.update(2)
    assert np.isfinite(metrics['actor_loss'])
    assert metrics['actor_auxiliary_update_count'] == 1
    assert metrics['auxiliary_label_count'] == 1
    for mg in env.agent_ids:
        assert metrics['actor_diagnostics'][mg]['auxiliary_attention_grad_norm'] > 0
        assert all(np.isfinite(x) for x in metrics['actor_diagnostics'][mg].values())
        assert all(x > 0 for x in metrics['actor_layer_grad_norms'][mg].values())
    path = tmp_path / 'checkpoint.pt'
    model.save(path)
    restored = spec.create(env, settings, device='cpu', seed=18)
    restored.load(path)
    for mg in env.agent_ids:
        np.testing.assert_array_equal(model.select_actions(obs)[mg], restored.select_actions(obs)[mg])
    assert restored.teacher.solve_calls == 0


@pytest.mark.parametrize('field', ['algorithm_slug', 'action_semantics', 'actor_contract', 'reward_v2_contract',
                                  'observation_schemas', 'state_schema', 'projection_signature'])
def test_checkpoint_mismatch_does_not_mutate_model(case, tmp_path, field):
    _, _, _, model, _, _ = case
    path = tmp_path / 'bad.pt'
    model.save(path)
    payload = torch.load(path, weights_only=True)
    payload[field] = 'invalid'
    torch.save(payload, path)
    before = deepcopy(model.actors.state_dict())
    with pytest.raises(ValueError):
        model.load(path)
    for name, tensor in model.actors.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_ablation_disables_teacher_without_changing_network(case):
    spec, env, settings, model, obs, state = case
    disabled = deepcopy(settings)
    disabled['algorithm']['auxiliary_plan'].update(lambda_plan=0, lambda_dyn=0, lambda_first=0)
    other = spec.create(env, disabled, device='cpu', seed=17)
    other.capture_training_context(env, obs, state, {})
    assert other.teacher.solve_calls == 0
    assert list(other.actors.state_dict()) == list(model.actors.state_dict())
    for mg in env.agent_ids:
        np.testing.assert_array_equal(model.select_actions(obs)[mg], other.select_actions(obs)[mg])
