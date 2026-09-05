from __future__ import annotations

from pathlib import Path

import yaml

from algorithms.registry import available_algorithm_names, get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from experiments.layout import ExperimentLayout
from scripts.create_algorithm_variant import create_variant


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "configs" / "cooperative.yaml"


def test_matd3_plugin_is_discoverable_and_satisfies_pipeline_contract() -> None:
    assert "matd3_dispatch" in available_algorithm_names()
    spec = get_algorithm_spec("matd3")
    env = MultiMicrogridEnv(ENV_CONFIG, split="train")
    import yaml

    config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    algorithm = spec.create(env, config, seed=9)
    assert spec.slug == "matd3_dispatch"
    assert callable(algorithm.select_actions)
    assert callable(algorithm.replay.add)


def test_retained_plugins_have_isolated_configs_and_matching_interfaces():
    names = available_algorithm_names()
    assert set(names) >= {"crossgate", "matd3_dispatch"}
    base = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    dims = []
    for slug in names:
        spec = get_algorithm_spec(slug)
        settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
        assert settings["algorithm"]["name"] == slug
        env = MultiMicrogridEnv(spec.build_environment_config(base))
        assert env.action_dim == 1
        assert env.ess_sharing_control_mode == "single_dispatch"
        assert env.ess_sharing_observation_mode == "cross_microgrid"
        dims.append(env.observation_dims)
    for d in dims[1:]:
        assert d == dims[0]


def test_experiment_layout_isolates_algorithms_and_runs(tmp_path: Path) -> None:
    first = ExperimentLayout.create(tmp_path, "matd3", "MATD3", 42)
    second = ExperimentLayout.create(tmp_path, "candidate", "Candidate", 42)
    assert first.algorithm_root == (tmp_path / "matd3").resolve()
    assert second.algorithm_root == (tmp_path / "candidate").resolve()
    assert first.run_dir.parent.name == "runs"
    assert second.run_dir.parent.name == "runs"
    assert first.run_dir != second.run_dir
    assert first.checkpoints_dir.exists()


def test_variant_scaffold_owns_separate_code_and_config(tmp_path: Path) -> None:
    package, config = create_variant("matd3_trial", project_root=tmp_path)
    assert package == tmp_path / "algorithms" / "matd3_trial"
    assert config == tmp_path / "configs" / "algorithms" / "matd3_trial.yaml"
    assert "class Matd3Trial" in (package / "algorithm.py").read_text(encoding="utf-8")
    assert "name: matd3_trial" in config.read_text(encoding="utf-8")
    assert (ROOT / "algorithms" / "matd3" / "matd3.py").exists()
