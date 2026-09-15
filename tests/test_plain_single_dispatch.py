from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from algorithms.matd3.actor import Actor
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from experiments.profiles import experiment_contract, profile, resolve_training_args
from experiments.training_pipeline import build_parser


ROOT = Path(__file__).resolve().parents[1]


def build(experiment_name):
    selected = profile(experiment_name)
    spec = get_algorithm_spec(selected["algorithm"])
    cfg = yaml.safe_load(selected["config"].read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(cfg))
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    return selected, spec, env, settings, spec.create(env, settings, seed=42)


def test_plain_matd3_uses_two_raw_actions_and_three_effective_critic_controls():
    torch.set_num_threads(1)
    selected, spec, env, settings, plain = build("matd3")
    observations, info = env.reset(seed=42, deterministic=True)

    assert selected["action_design"] == "single_dispatch_dr"
    assert env.action_dim == plain.action_dim == 2
    assert plain.raw_action_dim == 2
    assert plain.effective_action_dim == 3
    assert plain.projector.semantics == "single_dispatch_dr_v1"
    for mg in env.agent_ids:
        actor = plain.actors[mg]
        assert type(actor) is Actor
        assert [type(layer) for layer in actor.network] == [
            nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear, nn.Tanh
        ]
        assert [(layer.in_features, layer.out_features) for layer in actor.network
                if isinstance(layer, nn.Linear)] == [
            (env.observation_dims[mg], 128), (128, 128), (128, 2)
        ]
        assert observations[mg].shape == (env.observation_dims[mg],)

    raw = torch.zeros((1, 3, 2), dtype=torch.float32)
    state = torch.tensor(info["global_state"]).unsqueeze(0)
    projected, plans = plain.project_actions(state, raw)
    assert projected.shape == (1, 3, 3)
    assert len(plans) == 3

    contract = experiment_contract(
        spec, env.config, selected["config"], selected_experiment="matd3"
    )
    assert contract["policy"] == "mlp"
    assert contract["action_design"]["id"] == "single_dispatch_dr"
    assert contract["action_design"]["raw_action_dim"] == 2
    command = build_parser().parse_args(["--experiment", "matd3"])
    resolved = resolve_training_args(command)
    assert resolved.algorithm == "matd3_dispatch"
    assert resolved.config.name == "cooperative.yaml"


def test_crossgate_remains_on_frozen_pre_dr_contract():
    selected, spec, env, settings, cross = build("crossgate")
    assert selected["action_design"] == "single_dispatch"
    assert selected["config"].name == "cooperative_legacy_pre_dr.yaml"
    assert env.action_dim == cross.action_dim == 1
    assert "silo_ratio" not in env.obs_builder.schema("mg1")
    contract = experiment_contract(
        spec, env.config, selected["config"], selected_experiment="crossgate"
    )
    assert contract["action_design"]["id"] == "single_dispatch"
    assert contract["action_design"]["raw_action_dim"] == 1


def test_dr_and_legacy_profiles_do_not_claim_identical_physics():
    selected_dr, spec_dr, env_dr, _, _ = build("matd3")
    selected_old, spec_old, env_old, _, _ = build("crossgate")
    dr_contract = experiment_contract(
        spec_dr, env_dr.config, selected_dr["config"], selected_experiment="matd3"
    )
    old_contract = experiment_contract(
        spec_old, env_old.config, selected_old["config"], selected_experiment="crossgate"
    )
    assert dr_contract["physical_config_hash"] != old_contract["physical_config_hash"]
    assert dr_contract["resolved_environment_hash"] != old_contract["resolved_environment_hash"]


def test_plain_rejects_legacy_crossgate_checkpoint(tmp_path):
    _, _, _, _, baseline = build("matd3")
    _, _, _, _, grouped = build("crossgate")
    checkpoint = tmp_path / "grouped.pt"
    grouped.save(checkpoint)
    with pytest.raises(ValueError, match="different algorithm"):
        baseline.load(checkpoint)
