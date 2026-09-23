from __future__ import annotations

from pathlib import Path

import yaml

from algorithms.registry import available_algorithm_names, get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from experiments.layout import ExperimentLayout
from scripts.create_algorithm_variant import create_variant


ROOT = Path(__file__).resolve().parents[1]
NEW_ENV_CONFIG = ROOT / "configs" / "cooperative.yaml"


def test_matd3_plugin_is_discoverable_and_satisfies_pipeline_contract() -> None:
    assert "matd3_dispatch" in available_algorithm_names()
    spec = get_algorithm_spec("matd3")
    env = MultiMicrogridEnv(NEW_ENV_CONFIG, split="train")
    config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    algorithm = spec.create(env, config, seed=9)
    assert spec.slug == "matd3_dispatch"
    assert env.action_dim == algorithm.action_dim == 2
    assert callable(algorithm.select_actions)
    assert callable(algorithm.replay.add)


def test_retained_plugins_keep_isolated_configs():
    names = available_algorithm_names()
    assert set(names) == {"matd3_dispatch", "matd3_hierarchical_transformer"}

    # Every retained plugin owns a matching algorithm config.
    for slug in names:
        spec = get_algorithm_spec(slug)
        settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
        assert settings["algorithm"]["name"] == slug

def test_matd3_and_ordinary_t_receive_same_peer_energy_summary():
    base = yaml.safe_load(NEW_ENV_CONFIG.read_text(encoding="utf-8"))
    specs = (get_algorithm_spec("matd3"), get_algorithm_spec("matd3_hierarchical_transformer"))
    assert specs[0].build_environment_config(base) == specs[1].build_environment_config(base)
    envs = [MultiMicrogridEnv(spec.build_environment_config(base)) for spec in specs]

    assert specs[0].display_name == "MATD3"
    assert specs[1].display_name == "普通 T"
    assert all(env.ess_sharing_observation_mode == "peer_energy_summary" for env in envs)
    assert envs[0].observation_dims == envs[1].observation_dims
    for mg in envs[0].agent_ids:
        assert envs[0].obs_builder.schema(mg) == envs[1].obs_builder.schema(mg)
        for peer in (other for other in envs[0].agent_ids if other != mg):
            schema = envs[0].obs_builder.schema(mg)
            assert f"{peer}_energy_balance" in schema
            assert f"{peer}_soc" not in schema


def test_only_matd3_and_ordinary_t_are_registered_for_new_runs():
    assert set(available_algorithm_names()) == {
        "matd3_dispatch",
        "matd3_hierarchical_transformer",
    }


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
