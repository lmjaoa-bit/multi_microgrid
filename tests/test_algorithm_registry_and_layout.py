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
    assert "matd3" in available_algorithm_names()
    spec = get_algorithm_spec("matd3")
    env = MultiMicrogridEnv(ENV_CONFIG, split="train")
    import yaml

    config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    algorithm = spec.create(env, config, seed=9)
    assert spec.slug == "matd3"
    assert callable(algorithm.select_actions)
    assert callable(algorithm.replay.add)


def test_ess_study_plugins_have_isolated_configs_and_expected_interfaces() -> None:
    expected = {
        "matd3": (1, "residual_auto", "local"),
        "matd3_ess_sharing": (2, "explicit_offer", "cross_microgrid"),
        "matd3_renewable_p2p_only": (2, "explicit_offer", "cross_microgrid"),
        "milp_ess_sharing_mpc": (2, "explicit_offer", "cross_microgrid"),
        "milp_renewable_p2p_only": (2, "explicit_offer", "cross_microgrid"),
    }
    names = set(available_algorithm_names())
    base = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    base["_project_root"] = str(ROOT)
    observation_dims: dict[str, int] = {}
    for slug, (action_dim, control_mode, observation_mode) in expected.items():
        assert slug in names
        spec = get_algorithm_spec(slug)
        algorithm_config = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
        assert algorithm_config["algorithm"]["name"] == slug
        env = MultiMicrogridEnv(spec.build_environment_config(base), split="train")
        assert env.action_dim == action_dim
        assert env.ess_sharing_control_mode == control_mode
        assert env.ess_sharing_observation_mode == observation_mode
        assert env.ess_p2p_enabled is (
            slug not in {"matd3_renewable_p2p_only", "milp_renewable_p2p_only"}
        )
        observation_dims[slug] = env.observation_dims["mg1"]
    assert observation_dims["matd3_ess_sharing"] > observation_dims["matd3"]
    assert observation_dims["matd3_renewable_p2p_only"] == observation_dims["matd3_ess_sharing"]
    assert observation_dims["milp_ess_sharing_mpc"] == observation_dims["matd3_ess_sharing"]
    assert observation_dims["milp_renewable_p2p_only"] == observation_dims["matd3_ess_sharing"]


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
