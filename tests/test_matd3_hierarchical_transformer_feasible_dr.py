from __future__ import annotations

from pathlib import Path

import yaml

from algorithms.registry import get_algorithm_spec


def test_feasible_dr_plugin_uses_its_own_config():
    spec = get_algorithm_spec("matd3_hierarchical_transformer_feasible_dr")
    config = yaml.safe_load(Path(spec.default_config_path).read_text(encoding="utf-8"))

    assert spec.slug == "matd3_hierarchical_transformer_feasible_dr"
    assert spec.algorithm_class.__name__ == "Matd3HierarchicalTransformerFeasibleDR"
    assert config["algorithm"]["name"] == spec.slug
    assert config["algorithm"]["feasible_dr"]["agents"] == {
        "mg1": False,
        "mg2": True,
        "mg3": True,
    }
