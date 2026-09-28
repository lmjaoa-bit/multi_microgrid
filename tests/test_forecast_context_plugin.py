from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from algorithms.matd3_forecast_context.algorithm import Matd3ForecastContext
from algorithms.registry import available_algorithm_names, get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv


CONFIG_PATHS = {
    "A": Path("configs/algorithms/matd3_forecast_context.yaml"),
    "B": Path("configs/algorithms/matd3_forecast_context/b_peer_h8.yaml"),
    "C": Path("configs/algorithms/matd3_forecast_context/c_local_h24.yaml"),
    "D": Path("configs/algorithms/matd3_forecast_context/d_peer_h24.yaml"),
}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _env() -> MultiMicrogridEnv:
    config = _load(Path("configs/cooperative.yaml"))
    spec = get_algorithm_spec("matd3_forecast_context")
    return MultiMicrogridEnv(
        spec.build_environment_config(copy.deepcopy(config)),
        split="train",
    )


def test_plugin_registration_is_isolated_and_uses_group_a_default():
    spec = get_algorithm_spec("matd3_forecast_context")
    assert spec.slug == "matd3_forecast_context"
    assert spec.display_name == "普通 T（预测上下文实验）"
    assert spec.default_config_path == CONFIG_PATHS["A"].resolve()
    assert spec.environment_overrides == {
        "ess_sharing": {"observation_mode": "peer_energy_summary"}
    }
    assert "matd3_forecast_context" in available_algorithm_names()
    assert spec.algorithm_class is Matd3ForecastContext
    assert spec.algorithm_class._diagnostic_sampling_installed is True


def test_plugin_installs_grouped_sampled_diagnostics_not_plain_actor_path():
    spec = get_algorithm_spec("matd3_forecast_context")
    env = _env()
    algorithm = spec.create(
        env,
        _load(CONFIG_PATHS["A"]),
        device="cpu",
        seed=42,
    )
    assert algorithm._diagnostic_variant == "matd3_forecast_context"
    assert algorithm._diagnostic_plain_actor is False
    assert algorithm.update.__func__.__name__ == "fast_projected_update"


@pytest.mark.parametrize(
    ("group", "horizon", "peer"),
    [
        ("A", 8, False),
        ("B", 8, True),
        ("C", 24, False),
        ("D", 24, True),
    ],
)
def test_four_formal_configs_encode_exact_matrix(group, horizon, peer):
    config = _load(CONFIG_PATHS[group])
    assert config["experiment"] == {
        "study_id": "forecast_context_2x2",
        "group": group,
        "network_family": "transformer",
    }
    assert config["algorithm"]["name"] == "matd3_forecast_context"
    network = config["algorithm"]["network"]
    assert network["forecast_context_version"] == "forecast_context_v1"
    assert network["forecast_horizon"] == horizon
    assert network["include_peer_forecast"] is peer
    assert network["peer_forecast_features"] == ["energy_balance"]
    assert config["training"]["episodes"] == 5000
    assert config["training"]["test_episodes"] == 55


def test_configs_differ_only_by_group_horizon_and_peer_switch():
    sanitized = []
    for group in ("A", "B", "C", "D"):
        config = _load(CONFIG_PATHS[group])
        config = copy.deepcopy(config)
        config["experiment"].pop("group")
        config["algorithm"]["network"].pop("forecast_horizon")
        config["algorithm"]["network"].pop("include_peer_forecast")
        sanitized.append(config)
    assert sanitized[1:] == [sanitized[0]] * 3


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("wrong_name", "algorithm.name"),
        ("wrong_family", "network_family"),
        ("wrong_version", "forecast_context_version"),
        ("wrong_semantics", "feature semantics"),
        ("float_horizon", "forecast_horizon"),
        ("unsupported_horizon", "forecast_horizon"),
        ("non_bool_peer", "include_peer_forecast"),
        ("bad_peer_features", "peer_forecast_features"),
        ("string_peer_features", "peer_forecast_features"),
        ("group_mismatch", "group A requires"),
    ],
)
def test_invalid_study_configs_fail_fast(mutation, match):
    env = _env()
    config = _load(CONFIG_PATHS["A"])
    if mutation == "wrong_name":
        config["algorithm"]["name"] = "matd3_hierarchical_transformer"
    elif mutation == "wrong_family":
        config["experiment"]["network_family"] = "mlp"
    elif mutation == "wrong_version":
        config["algorithm"]["network"]["forecast_context_version"] = "v0"
    elif mutation == "wrong_semantics":
        config["algorithm"]["network"]["feature_semantics"] = "legacy"
    elif mutation == "float_horizon":
        config["algorithm"]["network"]["forecast_horizon"] = 8.0
    elif mutation == "unsupported_horizon":
        config["algorithm"]["network"]["forecast_horizon"] = 16
    elif mutation == "non_bool_peer":
        config["algorithm"]["network"]["include_peer_forecast"] = 0
    elif mutation == "bad_peer_features":
        config["algorithm"]["network"]["peer_forecast_features"] = [
            "energy_balance",
            "deficit",
        ]
    elif mutation == "string_peer_features":
        config["algorithm"]["network"]["peer_forecast_features"] = "energy_balance"
    elif mutation == "group_mismatch":
        config["algorithm"]["network"]["include_peer_forecast"] = True
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=match):
        Matd3ForecastContext.from_env(
            env,
            config,
            device="cpu",
            seed=42,
        )


def test_new_production_package_does_not_runtime_import_old_transformer():
    package = Path("algorithms/matd3_forecast_context")
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package.glob("*.py"))
    )
    assert "algorithms.matd3_hierarchical_transformer" not in combined


def test_public_experiment_catalog_is_not_extended_for_ablation_profiles():
    path = Path("configs/experiment_profiles.yaml")
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        assert "matd3_forecast_context" not in text
