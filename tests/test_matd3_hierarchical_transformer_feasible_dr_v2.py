from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv

SLUG = "matd3_hierarchical_transformer_feasible_dr_v2"
CONFIG_ROOT = Path("configs/algorithms")


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_v2_plugin_and_ablation_configs_are_isolated():
    spec = get_algorithm_spec(SLUG)
    combined = _load(f"{SLUG}.yaml")
    hvac = _load(f"{SLUG}_hvac.yaml")
    ev = _load(f"{SLUG}_ev.yaml")

    assert spec.slug == SLUG
    assert spec.algorithm_class.__name__ == "Matd3HierarchicalTransformerFeasibleDRV2"
    assert Path(spec.default_config_path).name == f"{SLUG}.yaml"
    for config in (combined, hvac, ev):
        assert config["algorithm"]["name"] == SLUG

    assert combined["algorithm"]["adaptive_dr"]["hvac"]["recovery_anchor_enabled"] is True
    assert combined["algorithm"]["adaptive_dr"]["ev"]["economic_reference_enabled"] is True
    assert hvac["algorithm"]["adaptive_dr"]["hvac"]["recovery_anchor_enabled"] is True
    assert hvac["algorithm"]["adaptive_dr"]["ev"]["economic_reference_enabled"] is False
    assert ev["algorithm"]["adaptive_dr"]["hvac"]["recovery_anchor_enabled"] is False
    assert ev["algorithm"]["adaptive_dr"]["ev"]["economic_reference_enabled"] is True


@pytest.fixture
def v2_env_and_config():
    env_config = yaml.safe_load(Path("configs/cooperative.yaml").read_text(encoding="utf-8"))
    algorithm_config = _load(f"{SLUG}.yaml")
    env = MultiMicrogridEnv(env_config, split="train")
    return env, algorithm_config
