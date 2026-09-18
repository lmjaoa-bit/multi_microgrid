from __future__ import annotations

from pathlib import Path

import yaml

from algorithms.registry import available_algorithm_names, get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from experiments.layout import ExperimentLayout
from scripts.create_algorithm_variant import create_variant


ROOT = Path(__file__).resolve().parents[1]
NEW_ENV_CONFIG = ROOT / "configs" / "cooperative.yaml"
LEGACY_ENV_CONFIG = ROOT / "configs" / "cooperative_legacy_pre_dr.yaml"


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


def test_retained_plugins_keep_isolated_configs_and_explicit_environment_generations():
    names = available_algorithm_names()
    assert set(names) >= {"crossgate", "matd3_dispatch"}

    # Every retained plugin owns a matching algorithm config. This check does not
    # imply that every experimental plugin has already migrated to the new DR
    # action space.
    for slug in names:
        spec = get_algorithm_spec(slug)
        settings = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
        assert settings["algorithm"]["name"] == slug

    # The two public experiment generations are intentionally different while
    # sharing the same ESS single-dispatch physics underneath.
    matd3_spec = get_algorithm_spec("matd3_dispatch")
    matd3_base = yaml.safe_load(NEW_ENV_CONFIG.read_text(encoding="utf-8"))
    matd3_env = MultiMicrogridEnv(matd3_spec.build_environment_config(matd3_base))
    assert matd3_env.action_design == "single_dispatch_dr"
    assert matd3_env.action_dim == 2
    assert matd3_env.ess_sharing_control_mode == "single_dispatch"
    assert matd3_env.ess_sharing_observation_mode == "peer_energy_summary"

    cross_spec = get_algorithm_spec("crossgate")
    cross_base = yaml.safe_load(LEGACY_ENV_CONFIG.read_text(encoding="utf-8"))
    cross_env = MultiMicrogridEnv(cross_spec.build_environment_config(cross_base))
    assert cross_env.action_design == "single_dispatch"
    assert cross_env.action_dim == 1
    assert cross_env.ess_sharing_control_mode == "single_dispatch"
    assert cross_env.ess_sharing_observation_mode == "cross_microgrid"

    # The DR environment deliberately adds private device observations/state;
    # equal observation dimensions across generations are no longer required.
    assert matd3_env.observation_dims != cross_env.observation_dims
    assert matd3_env.global_state_dim > cross_env.global_state_dim


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
