from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from algorithms.matd3.actor import Actor
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from experiments.profiles import experiment_contract, resolve_training_args
from experiments.training_pipeline import build_parser


ROOT = Path(__file__).resolve().parents[1]
SLUG = "matd3_dispatch"


def build(slug):
    spec = get_algorithm_spec(slug)
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(cfg))
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    return spec, env, settings, spec.create(env, settings, seed=42)


def test_plain_baseline_uses_canonical_actor_and_matched_dispatch_contract():
    torch.set_num_threads(1)
    spec, env, cfg, plain = build(SLUG)
    cross_spec, cross_env, cross_cfg, cross = build("crossgate")
    observations, info = env.reset(seed=42, deterministic=True)
    cross_obs, cross_info = cross_env.reset(seed=42, deterministic=True)
    assert env.obs_builder.schema("mg1") == cross_env.obs_builder.schema("mg1")
    assert plain.projection_signature == cross.projection_signature
    assert plain.state_schema == cross.state_schema
    assert plain.action_dim == cross.action_dim == 1
    for mg in env.agent_ids:
        actor = plain.actors[mg]
        assert type(actor) is Actor
        assert [type(layer) for layer in actor.network] == [
            nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear, nn.Tanh]
        assert [(layer.in_features, layer.out_features) for layer in actor.network
                if isinstance(layer, nn.Linear)] == [(env.observation_dims[mg], 128), (128, 128), (128, 1)]
        assert torch.equal(torch.tensor(observations[mg]), torch.tensor(cross_obs[mg]))
        for name, value in plain.critics[mg].state_dict().items():
            assert torch.equal(value, cross.critics[mg].state_dict()[name])
    assert torch.equal(torch.tensor(info["global_state"]), torch.tensor(cross_info["global_state"]))
    raw = torch.tensor([[[-.3], [.4], [.8]]])
    state = torch.tensor(info["global_state"]).unsqueeze(0)
    assert torch.equal(plain.project_actions(state, raw)[0], cross.project_actions(state, raw)[0])
    flat_settings, gated_settings = deepcopy(cfg), deepcopy(cross_cfg)
    flat_settings["algorithm"].pop("name")
    gated_settings["algorithm"].pop("name")
    gated_settings["algorithm"].pop("network")
    assert flat_settings == gated_settings
    contracts = [experiment_contract(s, e.config, ROOT / "configs/cooperative.yaml")
                 for s, e in [(spec, env), (cross_spec, cross_env)]]
    assert contracts[0]["resolved_environment_hash"] == contracts[1]["resolved_environment_hash"]
    assert contracts[0]["policy"] == "mlp"
    assert contracts[0]["action_design"]["id"] == "single_dispatch"
    command = build_parser().parse_args(["--experiment", "matd3"])
    assert resolve_training_args(command).algorithm == SLUG


@pytest.mark.parametrize("other", ["crossgate"])
def test_plain_rejects_grouped_actor_checkpoints(other, tmp_path):
    _, _, _, baseline = build(SLUG)
    _, _, _, grouped = build(other)
    checkpoint = tmp_path / "grouped.pt"
    grouped.save(checkpoint)
    with pytest.raises(ValueError, match="different algorithm"):
        baseline.load(checkpoint)
