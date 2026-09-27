from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "configs" / "cooperative.yaml"
ALGORITHMS = ("matd3", "matd3_hierarchical_transformer")


@pytest.mark.parametrize("slug", ALGORITHMS)
def test_two_algorithm_replay_update_and_checkpoint_smoke(
    slug: str, tmp_path: Path
) -> None:
    spec = get_algorithm_spec(slug)
    base = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    base["_project_root"] = str(ROOT)
    env = MultiMicrogridEnv(spec.build_environment_config(base), split="train")
    settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    settings = deepcopy(settings)
    settings["algorithm"]["replay_capacity"] = 100
    settings.setdefault("logging", {})["diagnostic_interval"] = 2
    model = spec.create(env, settings, device="cpu", seed=17)

    observations, reset_info = env.reset(seed=17, deterministic=True)
    state = reset_info["global_state"]
    for _ in range(3):
        actions = model.select_actions(observations, noise_std=0.0)
        assert all(action.shape == (2,) for action in actions.values())
        next_observations, rewards, done, info = env.step(actions)
        model.replay.add(
            observations,
            state,
            actions,
            rewards[env.agent_ids[0]],
            next_observations,
            info["global_state_after"],
            done,
        )
        observations = next_observations
        state = info["global_state_after"]

    assert model.update(batch_size=2) == {}
    metrics = model.update(batch_size=2)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])

    checkpoint = tmp_path / f"{spec.slug}.pt"
    model.save(checkpoint)
    restored = spec.create(env, settings, device="cpu", seed=18)
    restored.load(checkpoint)
    assert restored.update_steps == model.update_steps

    legacy_checkpoint = tmp_path / f"{spec.slug}_legacy_v2.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["action_semantics"] = "single_dispatch_dr_v2"
    torch.save(payload, legacy_checkpoint)
    incompatible = spec.create(env, settings, device="cpu", seed=19)
    with pytest.raises(ValueError, match="action semantics"):
        incompatible.load(legacy_checkpoint)

    network = settings["algorithm"].get("network", {})
    if "transformer" in spec.slug:
        assert network["forecast_horizon"] == 8
