from __future__ import annotations

import copy
import json
from collections import deque

import numpy as np
import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


SLUG = "matd3_dr_counterfactual"


def make_model(split="train", **guidance):
    spec = get_algorithm_spec(SLUG)
    config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    config["algorithm"].update(hidden_dims=[16, 16], learning_starts=0, batch_size=2)
    config["algorithm"]["dr_counterfactual"].update(guidance)
    env = MultiMicrogridEnv("configs/cooperative.yaml", split=split)
    model = spec.create(env, config, device="cpu", seed=42)
    obs, info = env.reset(seed=42, deterministic=True, forecast_noise=False)
    return env, model, obs, info


def terminal_context(env):
    env.t = 23
    hvac = env.dr_devices["mg2"]
    hvac.t_air = hvac.t_mass = 23.0
    hvac._frame.loc[23, hvac.temperature_column] = 23.0
    current, devices = env._exogenous_at(23), env._dr_states(23)
    env._last_observations = env._observations(current, devices)
    env._last_global_state = env._build_global_state(current, devices)
    return env._last_observations, env.global_state()


def fixed_policy(obs):
    return {mg: np.array([0.0, -1.0 if mg != "mg2" else 0.0], dtype=np.float32)
            for mg in obs}


def test_terminal_labels_use_real_reward_and_do_not_mutate_live_environment():
    env, model, _, _ = make_model()
    obs, state = terminal_context(env)
    actions = fixed_policy(obs)
    actions["mg2"][1] = 0.15
    original = copy.deepcopy(env)
    labels = model.teacher.compare(env, obs, state, actions, "mg2", fixed_policy, step=0)
    off = next(label for label in labels if label.candidate_raw == 0.0)
    trial = copy.deepcopy(original)
    changed = {mg: a.copy() for mg, a in actions.items()}
    changed["mg2"][1] = 0.0
    _, rewards, done, _ = trial.step(changed)
    assert done
    assert off.candidate_return == pytest.approx(rewards["mg2"])
    assert off.gain > 0.0
    assert off.actor_approved
    assert env.t == 23
    assert env.dr_devices["mg2"].t_air == 23.0
    np.testing.assert_array_equal(env.global_state(), state)
    np.testing.assert_array_equal(actions["mg2"], [0.0, np.float32(0.15)])


def test_teacher_clones_each_rollout_without_an_unused_extra_snapshot(monkeypatch):
    env, model, _, _ = make_model()
    obs, state = terminal_context(env)
    original_deepcopy = copy.deepcopy
    clones = []
    def observe(obj, *args, **kwargs):
        if isinstance(obj, MultiMicrogridEnv):
            clones.append(obj)
        return original_deepcopy(obj, *args, **kwargs)
    monkeypatch.setattr(copy, "deepcopy", observe)
    labels = model.teacher.compare(env, obs, state, fixed_policy(obs), "mg2", fixed_policy, step=0)
    assert labels
    assert len(clones) == 1 + len(labels)
    assert env.t == 23 and env.dr_devices["mg2"].t_air == 23.0
    np.testing.assert_array_equal(env.global_state(), state)


def test_ev_labels_keep_deadline_consequences_and_reject_missed_service():
    env, model, obs, _ = make_model()
    session = env.dr_devices["mg3"].sessions[0]
    hour = session["departure_hour"] - 1
    while env.t < hour:
        obs, _, _, _ = env.step(fixed_policy(obs))
    actions = fixed_policy(obs)
    actions["mg3"][1] = 1.0
    labels = model.teacher.compare(env, obs, env.global_state(), actions, "mg3",
                                   fixed_policy, step=0)
    off = next(label for label in labels if label.candidate_raw == -1.0)
    assert off.candidate_return < off.reference_return
    assert not off.service_safe
    assert not off.actor_approved
    assert off.rollout_steps == 24 - hour


@pytest.mark.parametrize("split", ["val", "test"])
def test_evaluation_contexts_never_generate_training_labels(split):
    env, model, obs, info = make_model(split, capture_probability=1.0)
    model.capture_training_context(env, obs, info["global_state"], {"total_steps": 0})
    actions = fixed_policy(obs)
    nxt, reward, done, after = env.step(actions)
    model.replay.add(obs, info["global_state"], actions, reward["mg1"], nxt,
                     after["global_state_after"], done)
    assert len(model.replay) == 1
    assert model.teacher.compare_calls == 0
    assert not model.replay.labels


def test_inference_does_not_use_counterfactual_teacher():
    env, model, obs, _ = make_model()
    before = model.teacher.compare_calls
    first = model.select_actions(obs, noise_std=0.0)
    second = model.select_actions(obs, noise_std=0.0)
    for mg in env.agent_ids:
        np.testing.assert_array_equal(first[mg], second[mg])
    assert model.teacher.compare_calls == before
    assert len(model.replay) == 0


def test_dr_actor_receives_gradient_even_when_tanh_is_saturated():
    from algorithms.matd3_dr_counterfactual.guidance import dr_logit_loss

    logits = torch.tensor([[-10.0, -10.0]], requires_grad=True)
    target = torch.tensor([0.0])
    loss = dr_logit_loss(logits, target, weight=1.0)
    gradient, = torch.autograd.grad(loss, logits)
    assert gradient[0, 1] < 0.0
    assert gradient[0, 0] == 0.0


def test_pair_loss_can_correct_wrong_value_ordering_in_both_critics():
    from algorithms.matd3_dr_counterfactual.guidance import critic_pair_loss

    reference = [torch.nn.Parameter(torch.tensor([[-0.5]])) for _ in range(2)]
    candidate = [torch.nn.Parameter(torch.tensor([[-0.7]])) for _ in range(2)]
    optimizer = torch.optim.SGD(reference + candidate, lr=0.1)
    for _ in range(30):
        loss = critic_pair_loss(tuple(reference), tuple(candidate), torch.tensor([[0.1]]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for base, alternative in zip(reference, candidate):
        assert float((alternative - base).detach()) == pytest.approx(0.1, abs=0.002)


def test_context_mismatch_is_rejected_without_contaminating_replay():
    env, model, obs, info = make_model(capture_probability=1.0)
    model.capture_training_context(env, obs, info["global_state"], {"total_steps": 0})
    actions = fixed_policy(obs)
    nxt, reward, done, after = env.step(actions)
    wrong = info["global_state"].copy()
    wrong[0] += 1.0
    with pytest.raises(ValueError, match="context"):
        model.replay.add(obs, wrong, actions, reward["mg1"], nxt,
                         after["global_state_after"], done)
    assert len(model.replay) == 0


def test_checkpoint_roundtrip_and_training_config_mismatch(tmp_path):
    env, model, obs, _ = make_model()
    expected = model.select_actions(obs, noise_std=0.0)
    path = tmp_path / "model.pt"
    model.save(path)
    _, loaded, _, _ = make_model()
    loaded.load(path)
    for mg in env.agent_ids:
        np.testing.assert_array_equal(loaded.select_actions(obs, noise_std=0.0)[mg], expected[mg])
    _, incompatible, _, _ = make_model(actor_weight=0.123)
    with pytest.raises(ValueError, match="counterfactual"):
        incompatible.load(path)


def test_real_critic_update_reverses_a_wrong_terminal_action_ranking():
    from algorithms.matd3.critic import TwinCritic

    env, model, _, _ = make_model(terminal_anchor_weight=0.0)
    obs, state = terminal_context(env)
    actions = fixed_policy(obs)
    actions["mg2"][1] = 0.15
    labels = model.teacher.compare(env, obs, state, actions, "mg2", fixed_policy, step=0)
    off = next(label for label in labels if label.candidate_raw == 0.0)
    model.replay.by_agent["mg2"] = deque([off], maxlen=8)
    critic = TwinCritic(len(state), 9, hidden_dims=())
    with torch.no_grad():
        for network in (critic.q1_network, critic.q2_network):
            network[0].weight.zero_()
            network[0].bias.zero_()
            network[0].weight[0, len(state) + 5] = 0.05
    model.critics["mg2"] = critic
    model.critic_optimizers["mg2"] = torch.optim.Adam(critic.parameters(), lr=0.02)
    states = torch.tensor(state[None])
    base, _ = model.project_actions(states, torch.tensor(off.reference_actions[None]))
    alternative, _ = model.project_actions(states, torch.tensor(off.candidate_actions[None]))

    def measured():
        with torch.no_grad():
            first = critic(states, base.flatten(1))
            second = critic(states, alternative.flatten(1))
        return [float(b - a) for a, b in zip(first, second)]

    assert all(value < 0 for value in measured())
    for _ in range(40):
        model._fit_counterfactual_critics()
    assert all(value > 0 for value in measured())


def test_real_actor_auxiliary_loss_recovers_saturated_hvac_without_ess_head_gradient():
    env, model, _, _ = make_model()
    obs, state = terminal_context(env)
    model.dispatch_q_guidance_weight = 0.0
    actor = model.actors["mg2"]
    with torch.no_grad():
        actor.network[-2].weight.zero_()
        actor.network[-2].bias[1] = 10.0
    actions = model.select_actions(obs, noise_std=0)
    labels = model.teacher.compare(env, obs, state, actions, "mg2", fixed_policy, step=0)
    best = next(label for label in labels if label.actor_approved)
    assert best.candidate_raw == 0.0
    model.replay.by_agent["mg2"] = deque(labels, maxlen=8)
    joint = torch.stack([model.actors[mg](torch.tensor(obs[mg][None]))
                         for mg in env.agent_ids], dim=1)
    loss, _ = model.actor_auxiliary_loss("mg2", torch.tensor(state[None]), joint,
                                       torch.tensor(obs["mg2"][None]))
    gradient, = torch.autograd.grad(loss, actor.network[-2].bias)
    assert gradient[1] > 0.0
    assert gradient[0] == 0.0


def test_disabled_guidance_preserves_matd3_q_actions_and_does_not_capture():
    env, model, obs, info = make_model(critic_weight=0.0, actor_weight=0.0,
                                     capture_probability=1.0)
    spec = get_algorithm_spec("matd3_dispatch_q_guided")
    config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    config["algorithm"].update(hidden_dims=[16, 16], learning_starts=0, batch_size=2)
    reference = spec.create(env, config, device="cpu", seed=42)
    for noise in (0.0, 0.15):
        expected = reference.select_actions(obs, noise_std=noise)
        actual = model.select_actions(obs, noise_std=noise)
        for mg in env.agent_ids:
            np.testing.assert_array_equal(actual[mg], expected[mg])
    model.capture_training_context(env, obs, info["global_state"], {"total_steps": 0})
    assert model.replay.pending_context is None


def test_replay_callback_stores_real_transition_separately_from_counterfactuals():
    env, model, _, _ = make_model(capture_probability=1.0)
    obs, state = terminal_context(env)
    model._capture_cursor = env.agent_ids.index("mg2")
    model.capture_training_context(env, obs, state, {"total_steps": 23})
    actions = fixed_policy(obs)
    actions["mg2"][1] = 0.15
    nxt, rewards, done, info = env.step(actions)
    model.replay.add(obs, state, actions, rewards["mg1"], nxt, info["global_state_after"], done)
    assert len(model.replay) == 1
    assert model.replay.labels
    np.testing.assert_array_equal(model.replay.storage[0].actions["mg2"], actions["mg2"])
    assert model.teacher.compare_calls == 1
    assert model.replay.pending_context is None


def test_stale_labels_and_service_unsafe_labels_do_not_guide_actor():
    env, model, _, _ = make_model()
    obs, state = terminal_context(env)
    actions = fixed_policy(obs)
    actions["mg2"][1] = 0.15
    labels = model.teacher.compare(env, obs, state, actions, "mg2", fixed_policy, step=0)
    model.replay.by_agent["mg2"] = deque(labels, maxlen=8)
    assert model._sample_labels("mg2", approved_only=True)
    model._training_step = model.guidance["max_age_steps"] + 1
    assert model._sample_labels("mg2") == []


@pytest.mark.parametrize("settings", [{"capacity": True}, {"capture_probability": 1.1},
                                      {"actor_weight": float("nan")}, {"typo": 1}])
def test_invalid_guidance_configuration_is_rejected(settings):
    with pytest.raises(ValueError, match="counterfactual"):
        make_model(**settings)


def test_formal_test_diagnostics_do_not_overwrite_training_evidence(tmp_path):
    _, training, _, _ = make_model()
    training.teacher.compare_calls = 7
    training.configure_diagnostics(tmp_path, {})
    training.set_diagnostic_phase(None, {"training_complete": True})
    _, evaluation, _, _ = make_model("test")
    evaluation.configure_diagnostics(tmp_path, {})
    evaluation.set_diagnostic_phase("formal_test", {})
    evaluation.set_diagnostic_phase(None, {"formal_test_complete": True})
    saved = json.loads((tmp_path / "dr_counterfactual_training_summary.json").read_text())
    assert saved["compare_calls"] == 7
    formal = json.loads((tmp_path / "dr_counterfactual_formal_test_summary.json").read_text())
    assert formal["compare_calls"] == 0


@pytest.mark.parametrize("field,value", [("dispatch_q_guidance_weight", 100.0),
                                          ("dispatch_q_guidance_margin", 5.0),
                                          ("policy_delay", 7), ("tau", 0.99)])
def test_checkpoint_rejects_changed_ess_guidance_and_td3_settings(tmp_path, field, value):
    env, model, _, _ = make_model()
    path = tmp_path / "model.pt"
    model.save(path)
    spec = get_algorithm_spec(SLUG)
    config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    config["algorithm"].update(hidden_dims=[16, 16], learning_starts=0, batch_size=2)
    config["algorithm"][field] = value
    changed = spec.create(env, config, device="cpu", seed=42)
    with pytest.raises(ValueError, match="algorithm configuration"):
        changed.load(path)


def test_hvac_probes_full_heat_when_half_power_cannot_meet_occupancy():
    env, model, _, _ = make_model()
    day = next(day for day in env.data.valid_days if day.weekday() < 5)
    env.reset(day=str(day.date()), seed=42, deterministic=True, forecast_noise=False)
    env.t = 6
    hvac = env.dr_devices["mg2"]
    hvac.t_air = hvac.t_mass = 17.0
    hvac._frame.loc[:, hvac.temperature_column] = 17.0
    hvac._frame.loc[:, hvac.ghi_column] = 0.0
    hvac._base_load_kw[:] = 0.0
    current, devices = env._exogenous_at(6), env._dr_states(6)
    obs = env._observations(current, devices)
    state = env._build_global_state(current, devices)
    actions = fixed_policy(obs)
    labels = model.teacher.compare(env, obs, state, actions, "mg2", fixed_policy, step=0)
    full = next(label for label in labels if label.candidate_raw == 1.0)
    assert full.service_safe
    half_trial, full_trial = copy.deepcopy(env), copy.deepcopy(env)
    for snapshot, raw in [(half_trial, 0.5), (full_trial, 1.0)]:
        changed = {mg: action.copy() for mg, action in actions.items()}
        changed["mg2"][1] = raw
        snapshot.step(changed)
    assert half_trial.dr_devices["mg2"].t_air < 21.0
    assert 21.0 <= full_trial.dr_devices["mg2"].t_air <= 25.0
