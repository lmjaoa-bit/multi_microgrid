from __future__ import annotations

from experiments.profiles import experiment_contract
from algorithms.registry import get_algorithm_spec
from pathlib import Path
import yaml


def test_contract_embeds_service_metric_and_dispatch_semantics():
    root=Path(__file__).resolve().parents[1]
    cfg_path=root/"configs"/"cooperative.yaml"
    cfg=yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    spec=get_algorithm_spec("matd3_dispatch")
    env_cfg=spec.build_environment_config(cfg)
    contract=experiment_contract(
        spec,env_cfg,cfg_path,selected_experiment="matd3",root=root
    )
    assert contract["schema_version"]==2
    assert contract["service_metrics_schema_version"]=="service_metrics_v2"
    assert contract["dispatch_priority_mode"]=="renewable_load_first"
    assert contract["action_semantics"]=="single_dispatch_dr_v4_instantaneous_physical"
    assert contract["physical_settings"]["dispatch"]["priority_mode"]=="renewable_load_first"
    assert len(contract["physical_settings"]["microgrids"])==3


def test_contract_carries_explicit_v3_semantic_metadata():
    root=Path(__file__).resolve().parents[1]
    cfg_path=root/"configs"/"cooperative.yaml"
    cfg=yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    spec=get_algorithm_spec("matd3_dispatch")
    contract=experiment_contract(
        spec,
        spec.build_environment_config(cfg),
        cfg_path,
        selected_experiment="matd3",
        root=root,
    )
    assert contract["dispatch_semantics"] == "renewable_load_first_v2_no_grid_export"
    assert contract["action_semantics"] == "single_dispatch_dr_v4_instantaneous_physical"
    assert contract["metrics_schema_version"] == "service_metrics_v2"
    assert contract["hvac_safety_mode"] == "soft_comfort"
    assert contract["reward_semantics_version"] == "generalized_system_cost_v2"
    assert contract["reward_calibration_status"] == "locked"
    assert len(contract["reward_calibration_report_sha256"]) == 64
    assert contract["oracle_allocation_mode"] == "source_priority_free_allocation"
    assert contract["full_information_oracle"] is True
