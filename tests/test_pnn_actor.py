from __future__ import annotations

from pathlib import Path

import torch
import yaml

from algorithms.pnn.actor import PNNDispatchActor
from algorithms.pnn.algorithm import PNN
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]


def build_pnn():
    spec = get_algorithm_spec("pnn")
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(cfg))
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    return spec, env, settings, spec.create(env, settings, seed=42)


def test_pnn_initialization_and_forward():
    spec, env, settings, pnn = build_pnn()
    assert spec.slug == "pnn"
    assert type(pnn) is PNN
    assert pnn.action_dim == 2
    assert pnn.raw_action_dim == 2
    assert pnn.effective_action_dim == 3

    observations, _ = env.reset(seed=42, deterministic=True)
    actions = pnn.select_actions(observations, noise_std=0.0)

    for mg in env.agent_ids:
        actor = pnn.actors[mg]
        assert type(actor) is PNNDispatchActor
        assert actor.feature_dim == 32
        assert actions[mg].shape == (2,)
        assert all(-1.0 <= float(value) <= 1.0 for value in actions[mg])

    prods = pnn.product_diagnostics()
    for mg in env.agent_ids:
        assert "linear_signal_norm" in prods[mg]
        assert "product_signal_norm" in prods[mg]
        assert prods[mg]["product_signal_norm"] > 0


def test_pnn_gradient_flow():
    _, env, _, pnn = build_pnn()
    obs, _ = env.reset(seed=42, deterministic=True)
    for mg in env.agent_ids:
        actor = pnn.actors[mg]
        t_obs = torch.tensor(obs[mg], dtype=torch.float32).unsqueeze(0)
        action = actor(t_obs)
        assert action.shape == (1, 2)
        loss = (action ** 2).sum()
        pnn.actor_optimizers[mg].zero_grad()
        loss.backward()

        branches = actor.branch_modules()
        for branch_name, modules in branches.items():
            branch_norm = sum(
                parameter.grad.norm(2).item()
                for module in modules
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            assert branch_norm > 0, f"Branch {branch_name} for agent {mg} received zero gradient"


def test_pnn_save_and_load(tmp_path: Path):
    spec, env, settings, pnn = build_pnn()
    save_path = tmp_path / "pnn_checkpoint.pt"
    pnn.save(save_path)
    payload = torch.load(save_path, map_location="cpu", weights_only=True)
    assert payload["action_semantics"] == "single_dispatch_dr_v1"

    pnn_loaded = spec.create(env, settings, seed=99)
    pnn_loaded.load(save_path)

    for mg in env.agent_ids:
        for p1, p2 in zip(pnn.actors[mg].parameters(), pnn_loaded.actors[mg].parameters()):
            assert torch.equal(p1, p2)
