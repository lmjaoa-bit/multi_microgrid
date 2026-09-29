"""Explicit experiment selection and immutable per-run contracts.

This module never eagerly imports experimental plugins. A deleted experimental
package cannot affect an explicit MATD3 run. Defaults only affect new invocations
without an explicit algorithm/profile; saved runs are never relabelled.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from core.reward_v2 import resolve_reward_v2_config

ROOT = Path(__file__).resolve().parents[1]


def catalog(name: str, root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / "configs" / (name + ".yaml")).read_text(encoding="utf-8")) or {}


def default_profile(root: Path = ROOT) -> str | None:
    return catalog("experiment_defaults", root).get("default_experiment")


def profile(name: str, root: Path = ROOT) -> dict[str, Any]:
    profiles = catalog("experiment_profiles", root)
    if name not in profiles:
        raise ValueError(f"Unknown experiment {name!r}; available: {', '.join(profiles)}")
    selected = deepcopy(profiles[name])
    if selected["action_design"] not in catalog("action_designs", root):
        raise ValueError(f"Unknown action design in {name}")
    environments = catalog("environment_profiles", root)
    if selected["environment"] not in environments:
        raise ValueError(f"Unknown environment profile in {name}")
    selected["config"] = (root / environments[selected["environment"]]["config"]).resolve()
    if not selected["config"].is_file():
        raise FileNotFoundError(selected["config"])
    selected["name"] = name
    return selected


def resolve_training_args(args, root: Path = ROOT):
    """Resolve CLI precedence without changing any physical configuration."""
    explicit_algorithm = getattr(args, "algorithm", None)
    experiment = getattr(args, "experiment", None)
    policy = getattr(args, "policy", None)
    action = getattr(args, "action_design", None)
    if explicit_algorithm and (experiment or policy or action):
        raise ValueError("--algorithm cannot be combined with --experiment/--policy/--action-design")
    if experiment and (policy or action):
        raise ValueError("--experiment is a complete bundle; do not combine it with --policy/--action-design")

    selected = None
    if not explicit_algorithm:
        if experiment:
            selected = profile(experiment, root)
        elif policy or action:
            # A public selector chooses among complete registered bundles. Do not
            # inherit the default MATD3 environment/action and accidentally force
            # another policy onto an incompatible physical generation.
            matches = []
            for name, item in catalog("experiment_profiles", root).items():
                if policy is not None and item["policy"] != policy:
                    continue
                if action is not None and item["action_design"] != action:
                    continue
                matches.append(name)
            if len(matches) != 1:
                requested = {"policy": policy, "action_design": action}
                raise ValueError(
                    f"No unique experiment for selectors {requested}; select --experiment"
                )
            selected = profile(matches[0], root)
        else:
            selected = profile(default_profile(root) or "matd3", root)
        args.algorithm = selected["algorithm"]

    args.config = Path(
        getattr(args, "config", None)
        or (selected["config"] if selected else root / "configs/cooperative.yaml")
    ).resolve()
    args.selected_experiment = selected["name"] if selected else None
    return args


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def experiment_contract(spec, env_config, config_path, *, selected_experiment=None, root: Path = ROOT):
    """Record resolved settings, never infer historical settings from current defaults."""
    cfg_path = Path(config_path).resolve()
    mode = env_config.get("ess_sharing", {}).get("control_mode", "single_dispatch")
    action_design_id = str(env_config.get("action_design", mode))
    designs = catalog("action_designs", root)
    design = deepcopy(designs.get(
        action_design_id,
        {"id": "custom", "version": action_design_id, "raw_actions": []},
    ))
    chosen = profile(selected_experiment, root) if selected_experiment else None
    if chosen and (
        chosen["algorithm"] != spec.slug
        or chosen["action_design"] != action_design_id
    ):
        raise ValueError("Experiment profile disagrees with the plugin's actual action contract")
    candidates = [
        item for item in catalog("experiment_profiles", root).values()
        if item["algorithm"] == spec.slug and item["action_design"] == action_design_id
    ]
    policy = chosen["policy"] if chosen else candidates[0]["policy"] if candidates else "unregistered"
    environment_name = "custom"
    for name, item in catalog("environment_profiles", root).items():
        if cfg_path == (root / item["config"]).resolve():
            environment_name = name
            break
    resolved = deepcopy(env_config)
    resolved.pop("_project_root", None)
    physics = deepcopy(resolved)
    # Controller-interface fields are excluded from the physical fingerprint.
    physics.pop("action_design", None)
    for key in ("control_mode", "observation_mode", "offer_projection"):
        physics.get("ess_sharing", {}).pop(key, None)
    if not physics.get("ess_sharing"):
        physics.pop("ess_sharing", None)
    design["control_mode"] = mode
    design["raw_action_dim"] = len(design["raw_actions"]) or None
    dispatch_priority_mode = str(
        env_config.get("dispatch", {}).get(
            "priority_mode", "legacy_local_first"
        )
    )
    reward_v2 = resolve_reward_v2_config(env_config)
    dispatch_semantics = (
        reward_v2["dispatch_semantics"]
        if reward_v2 is not None
        else "renewable_load_first_v1"
        if dispatch_priority_mode == "renewable_load_first"
        else "legacy_local_first_v1"
    )
    hvac_safety_mode = "legacy_one_step"
    for item in env_config.get("microgrids", []):
        dr = item.get("dr", {})
        if str(dr.get("type", "")) == "commercial_hvac":
            hvac_safety_mode = (
                "soft_comfort"
                if str(dr.get("control_semantics", "")) == "soft_comfort"
                else "forecast_viability"
                if bool(dr.get("forecast_viability", False))
                else "legacy_one_step"
            )
            break
    return {
        "schema_version": 2,
        "service_metrics_schema_version": "service_metrics_v2",
        "metrics_schema_version": "service_metrics_v2",
        "dispatch_priority_mode": dispatch_priority_mode,
        "dispatch_semantics": dispatch_semantics,
        "reward_semantics_version": (
            reward_v2["semantics_version"] if reward_v2 is not None else "legacy"
        ),
        "dr_projection_semantics": (
            reward_v2["dr_projection_semantics"] if reward_v2 is not None else None
        ),
        "hvac_control_semantics": (
            reward_v2["hvac_control_semantics"] if reward_v2 is not None else None
        ),
        "ev_control_semantics": (
            reward_v2["ev_control_semantics"] if reward_v2 is not None else None
        ),
        "observation_semantics_version": (
            reward_v2["observation_semantics_version"] if reward_v2 is not None else None
        ),
        "reward_discount_semantics": (
            reward_v2["reward_discount_semantics"] if reward_v2 is not None else None
        ),
        "comfort_metric_semantics": (
            reward_v2["comfort_metric_semantics"] if reward_v2 is not None else None
        ),
        "carbon_information_semantics": (
            reward_v2["carbon_information_semantics"] if reward_v2 is not None else None
        ),
        "settlement_value_semantics": (
            reward_v2["settlement_value_semantics"] if reward_v2 is not None else None
        ),
        "hvac_safety_mode": hvac_safety_mode,
        "oracle_allocation_mode": "source_priority_free_allocation",
        "full_information_oracle": True,
        "action_semantics": (
            "single_dispatch_dr_v4_instantaneous_physical"
            if reward_v2 is not None and action_design_id == "single_dispatch_dr"
            else "single_dispatch_dr_v3"
            if (
                action_design_id == "single_dispatch_dr"
                and str(env_config.get("dispatch", {}).get(
                    "priority_mode", "legacy_local_first"
                )) == "renewable_load_first"
            )
            else design.get("version")
        ),
        "experiment_profile": selected_experiment,
        "algorithm_slug": spec.slug,
        "policy": policy,
        "action_design": design,
        "observation_mode": env_config.get("ess_sharing", {}).get("observation_mode", "local"),
        "offer_projection": env_config.get("ess_sharing", {}).get("offer_projection", "none"),
        "environment_profile": environment_name,
        "environment_config": str(cfg_path),
        "environment_overrides": deepcopy(dict(spec.environment_overrides)),
        "resolved_environment_hash": _fingerprint(resolved),
        "physical_config_hash": _fingerprint(physics),
        "physical_settings": {
            key: deepcopy(env_config.get(key))
            for key in (
                "env", "dispatch", "p2p", "terminal_soc",
                "cost", "time", "data", "price", "microgrids",
            )
        },
        "ess_p2p_enabled": env_config.get("ess_sharing", {}).get("ess_p2p_enabled", True),
    }


def set_default(name: str | None, root: Path = ROOT):
    """Intentional future-only switch. Validate just the selected plugin."""
    if name is not None:
        selected = profile(name, root)
        from algorithms.registry import get_algorithm_spec
        spec = get_algorithm_spec(selected["algorithm"])
        cfg = yaml.safe_load(selected["config"].read_text(encoding="utf-8"))
        experiment_contract(
            spec,
            spec.build_environment_config(cfg),
            selected["config"],
            selected_experiment=name,
            root=root,
        )
        algorithm_cfg = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
        if algorithm_cfg.get("algorithm", {}).get("name") != spec.slug:
            raise ValueError("Selected plugin YAML name does not match")
    destination = root / "configs/experiment_defaults.yaml"
    temporary = destination.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump({"default_experiment": name}, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(destination)
