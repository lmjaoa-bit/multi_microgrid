from copy import deepcopy
import shutil
import pytest
import yaml
from algorithms.registry import available_algorithm_names, get_algorithm_spec
from core.action_codec import ActionCodec
from core.config_validation import validate_config
from experiments.profiles import (ROOT, catalog, default_profile, experiment_contract,
                                  profile, resolve_training_args, set_default)
from experiments.training_pipeline import build_parser

@pytest.fixture
def catalog_root(tmp_path):
    (tmp_path / "configs").mkdir()
    for name in ("action_designs", "environment_profiles", "experiment_profiles", "experiment_defaults", "cooperative"):
        shutil.copyfile(ROOT / "configs" / (name + ".yaml"), tmp_path / "configs" / (name + ".yaml"))
    return tmp_path

def command(*argv):
    return build_parser().parse_args(list(argv))

def test_default_and_both_public_selectors(catalog_root):
    assert default_profile(catalog_root) == "matd3"
    assert resolve_training_args(command(), catalog_root).algorithm == "matd3_dispatch"
    assert {"matd3_dispatch", "crossgate"}.issubset(set(available_algorithm_names()))
    assert get_algorithm_spec("matd3").slug == "matd3_dispatch"
    assert set(catalog("action_designs")) == {"single_dispatch"}
    assert set(catalog("experiment_profiles")) == {"matd3", "crossgate"}
    for selector in ("--experiment", "--policy"):
        assert resolve_training_args(command(selector, "crossgate"), catalog_root).algorithm == "crossgate"
    set_default("crossgate", catalog_root)
    assert resolve_training_args(command(), catalog_root).algorithm == "crossgate"
    assert resolve_training_args(command("--policy", "mlp"), catalog_root).algorithm == "matd3_dispatch"
    set_default(None, catalog_root)
    assert resolve_training_args(command(), catalog_root).algorithm == "matd3_dispatch"

@pytest.mark.parametrize("mode", ["residual_auto", "explicit_offer", "direct_offer", "fixed_full_offer"])
def test_retired_actions_are_rejected(mode):
    cfg = yaml.safe_load((ROOT / "configs/cooperative.yaml").read_text(encoding="utf-8"))
    cfg["ess_sharing"]["control_mode"] = mode
    with pytest.raises(ValueError):
        validate_config(cfg)
    with pytest.raises(ValueError):
        ActionCodec(mode)

def test_invalid_selections_and_conflicts(catalog_root):
    for argv in [("--algorithm", "matd3", "--policy", "mlp"),
                 ("--experiment", "matd3", "--action-design", "single_dispatch"),
                 ("--action-design", "direct_offer"), ("--experiment", "single_dispatch_matd3")]:
        with pytest.raises(ValueError):
            resolve_training_args(command(*argv), catalog_root)
    for slug in ("crossgate_actor", "matd3_single_dispatch", "milp_ess_sharing_mpc"):
        with pytest.raises(KeyError):
            get_algorithm_spec(slug)
    with pytest.raises(ValueError):
        set_default("unknown", catalog_root)
    assert default_profile(catalog_root) == "matd3"

def test_both_profiles_have_identical_physics_observations_and_action_contracts():
    cfg_path = ROOT / "configs/cooperative.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    contracts = []
    for name in catalog("experiment_profiles"):
        selected = profile(name)
        spec = get_algorithm_spec(selected["algorithm"])
        contract = experiment_contract(spec, spec.build_environment_config(cfg), cfg_path,
                                       selected_experiment=name)
        assert contract["action_design"]["raw_action_dim"] == 1
        assert contract["action_design"]["id"] == "single_dispatch"
        contracts.append(contract)
    assert contracts[0]["physical_config_hash"] == contracts[1]["physical_config_hash"]
    assert contracts[0]["resolved_environment_hash"] == contracts[1]["resolved_environment_hash"]
    modified = deepcopy(cfg)
    modified["p2p"]["pair_capacity_kw"] *= .5
    spec = get_algorithm_spec("matd3")
    changed = experiment_contract(spec, spec.build_environment_config(modified), cfg_path)
    assert changed["physical_config_hash"] != contracts[0]["physical_config_hash"]
