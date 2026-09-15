from __future__ import annotations

from pathlib import Path
import numpy as np
import yaml

from algorithms.milp_mpc.algorithm import MilpMpc
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG = ROOT / "configs" / "cooperative_legacy_pre_dr.yaml"


def build_mpc():
    spec = get_algorithm_spec("milp_mpc")
    cfg = yaml.safe_load(LEGACY_CONFIG.read_text(encoding="utf-8"))
    cfg["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(cfg), split="test")
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    return spec, env, settings, spec.create(env, settings, seed=42)


def test_milp_mpc_initialization_and_forward():
    spec, env, settings, mpc = build_mpc()
    assert spec.slug == "milp_mpc"
    assert type(mpc) is MilpMpc
    assert mpc.action_dim == 1
    assert mpc.horizon == 4

    observations, _ = env.reset(seed=42, deterministic=True)
    actions = mpc.select_actions(observations, noise_std=0.0)

    for mg in env.agent_ids:
        assert mg in actions
        assert actions[mg].shape == (1,)
        assert -1.0 <= float(actions[mg][0]) <= 1.0

    next_obs, _, done, _ = env.step(actions)
    assert not done
    for mg in env.agent_ids:
        assert mg in next_obs


def test_milp_mpc_save_and_load(tmp_path: Path):
    spec, env, settings, mpc = build_mpc()
    save_path = tmp_path / "mpc_checkpoint.pt"
    mpc.save(save_path)
    assert save_path.exists()

    mpc_loaded = spec.create(env, settings, seed=99)
    mpc_loaded.load(save_path)
