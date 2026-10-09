from copy import deepcopy
import shutil

import pytest
import yaml

from algorithms.registry import available_algorithm_names, get_algorithm_spec
from core.action_codec import ActionCodec
from core.config_validation import validate_config
from experiments.profiles import (
    ROOT,
    catalog,
    default_profile,
    experiment_contract,
    profile,
    resolve_training_args,
    set_default,
)
from experiments.training_pipeline import build_parser


@pytest.fixture
def catalog_root(tmp_path):
    (tmp_path / "configs").mkdir()
    for name in (
        "action_designs",
        "environment_profiles",
        "experiment_profiles",
        "experiment_defaults",
        "cooperative",
    ):
        source = ROOT / "configs" / (name + ".yaml")
        if source.exists():
            shutil.copyfile(source, tmp_path / "configs" / (name + ".yaml"))
    return tmp_path


def command(*argv):
    return build_parser().parse_args(list(argv))


def test_default_and_public_selectors_keep_matd3_default(catalog_root):
    assert default_profile(catalog_root) == "matd3"
    assert resolve_training_args(command(), catalog_root).algorithm == "matd3_dispatch"
    assert {"matd3_dispatch", "matd3_hierarchical_transformer"} <= set(available_algorithm_names())
    assert get_algorithm_spec("matd3").slug == "matd3_dispatch"
    assert set(catalog("action_designs")) == {"single_dispatch", "single_dispatch_dr"}
    assert set(catalog("experiment_profiles")) == {"matd3", "ordinary_t"}

    assert resolve_training_args(command("--experiment", "ordinary_t"), catalog_root).algorithm == "matd3_hierarchical_transformer"
    assert resolve_training_args(command("--policy", "transformer"), catalog_root).algorithm == "matd3_hierarchical_transformer"

    set_default("ordinary_t", catalog_root)
    assert resolve_training_args(command(), catalog_root).algorithm == "matd3_hierarchical_transformer"
    assert resolve_training_args(command("--policy", "mlp"), catalog_root).algorithm == "matd3_dispatch"
    set_default(None, catalog_root)
    assert resolve_training_args(command(), catalog_root).algorithm == "matd3_dispatch"


def test_matd3_and_ordinary_t_share_the_environment_generation():
    matd3 = profile("matd3")
    ordinary_t = profile("ordinary_t")

    assert matd3["action_design"] == "single_dispatch_dr"
    assert matd3["environment"] == "cooperative_standard"
    assert ordinary_t["action_design"] == "single_dispatch_dr"
    assert ordinary_t["environment"] == "cooperative_standard"


def test_dr_action_design_declares_two_raw_controls():
    designs = catalog("action_designs")
    assert designs["single_dispatch"]["raw_actions"] == ["signed_dispatch_intent"]
    assert designs["single_dispatch_dr"]["raw_actions"] == [
        "signed_dispatch_intent",
        "dr_intent",
    ]


@pytest.mark.parametrize("mode", ["residual_auto", "explicit_offer", "direct_offer", "fixed_full_offer"])
def test_retired_ess_control_modes_are_rejected(mode):
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["ess_sharing"]["control_mode"] = mode
    with pytest.raises(ValueError):
        validate_config(cfg)
    with pytest.raises(ValueError):
        ActionCodec(mode)


def test_invalid_selections_and_conflicts(catalog_root):
    for argv in [
        ("--algorithm", "matd3", "--policy", "mlp"),
        ("--experiment", "matd3", "--action-design", "single_dispatch_dr"),
        ("--action-design", "direct_offer"),
        ("--experiment", "single_dispatch_matd3"),
    ]:
        with pytest.raises(ValueError):
            resolve_training_args(command(*argv), catalog_root)
    for slug in ("crossgate", "matd3_single_dispatch", "milp_ess_sharing_mpc"):
        with pytest.raises(KeyError):
            get_algorithm_spec(slug)
    with pytest.raises(ValueError):
        set_default("unknown", catalog_root)
    assert default_profile(catalog_root) == "matd3"


def test_contract_uses_explicit_action_design_instead_of_ess_control_mode():
    matd3_selected = profile("matd3")
    matd3_cfg_path = matd3_selected["config"]
    matd3_cfg = yaml.safe_load(matd3_cfg_path.read_text(encoding="utf-8"))
    matd3_spec = get_algorithm_spec(matd3_selected["algorithm"])
    matd3_contract = experiment_contract(
        matd3_spec,
        matd3_spec.build_environment_config(matd3_cfg),
        matd3_cfg_path,
        selected_experiment="matd3",
    )
    assert matd3_contract["action_design"]["id"] == "single_dispatch_dr"
    assert matd3_contract["action_design"]["raw_action_dim"] == 2
    assert matd3_contract["action_design"]["control_mode"] == "single_dispatch"

    ordinary_selected = profile("ordinary_t")
    ordinary_cfg_path = ordinary_selected["config"]
    ordinary_cfg = yaml.safe_load(ordinary_cfg_path.read_text(encoding="utf-8"))
    ordinary_spec = get_algorithm_spec(ordinary_selected["algorithm"])
    ordinary_contract = experiment_contract(
        ordinary_spec,
        ordinary_spec.build_environment_config(ordinary_cfg),
        ordinary_cfg_path,
        selected_experiment="ordinary_t",
    )
    assert ordinary_contract["policy"] == "transformer"
    assert ordinary_contract["action_design"]["id"] == "single_dispatch_dr"
    assert ordinary_contract["action_design"]["raw_action_dim"] == 2
    assert ordinary_contract["observation_mode"] == matd3_contract["observation_mode"] == "peer_energy_summary"

    assert matd3_contract["resolved_environment_hash"] == ordinary_contract["resolved_environment_hash"]
    assert matd3_contract["physical_config_hash"] == ordinary_contract["physical_config_hash"]

    modified = deepcopy(matd3_cfg)
    modified["p2p"]["pair_capacity_kw"] = 75.0
    changed = experiment_contract(
        matd3_spec,
        matd3_spec.build_environment_config(modified),
        matd3_cfg_path,
    )
    assert changed["physical_config_hash"] != matd3_contract["physical_config_hash"]
