from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "forecast_context"
STUDY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
GROUPS = ("A", "B", "C", "D")
GROUP_MATRIX = {
    "A": (8, False),
    "B": (8, True),
    "C": (24, False),
    "D": (24, True),
}
PAIR_EFFECTS = (
    ("B-A", "B", "A"),
    ("C-A", "C", "A"),
    ("D-C", "D", "C"),
    ("D-B", "D", "B"),
)
REQUIRED_DAYS = 55


@dataclass
class RunRecord:
    path: Path
    group: str
    seed: int
    horizon: int
    include_peer_forecast: bool
    config_snapshot: dict[str, Any]
    algorithm_bundle: dict[str, Any]
    metadata: dict[str, Any]
    test_summary: dict[str, Any]
    summary: dict[str, Any]
    daily: pd.DataFrame
    checkpoint_contract: dict[str, Any]
    source_hashes: dict[str, str]
    referenced_file_hashes: dict[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _safe_resolve_under_report_root(study_id: str) -> Path:
    if not STUDY_ID_RE.fullmatch(study_id):
        raise ValueError(
            "study_id must match [a-z0-9][a-z0-9_-]*"
        )
    root = REPORT_ROOT.resolve()
    target = (root / study_id).resolve()
    if target.parent != root:
        raise ValueError("study_id escapes reports/forecast_context")
    if target.exists():
        raise FileExistsError(
            f"Study output already exists and will not be overwritten: {target}"
        )
    return target


def _algorithm_bundle(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    bundle = snapshot.get("algorithm")
    if not isinstance(bundle, Mapping):
        raise ValueError("config_snapshot.yaml is missing algorithm bundle")
    result = copy.deepcopy(dict(bundle))
    settings = result.get("algorithm")
    experiment = result.get("experiment")
    if not isinstance(settings, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError(
            "config_snapshot algorithm bundle must contain experiment and algorithm mappings"
        )
    return result


def _sanitized_algorithm_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(dict(bundle))
    experiment = clean.get("experiment")
    settings = clean.get("algorithm")
    if not isinstance(experiment, dict) or not isinstance(settings, dict):
        raise ValueError("Invalid algorithm bundle")
    experiment.pop("group", None)
    network = settings.get("network")
    if not isinstance(network, dict):
        raise ValueError("Algorithm network mapping is missing")
    network.pop("forecast_horizon", None)
    network.pop("include_peer_forecast", None)
    return clean


def _referenced_file_hashes(value: Any) -> dict[str, str]:
    candidates: set[Path] = set()

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for item in node.values():
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, str):
            raw = Path(node)
            possible = raw if raw.is_absolute() else PROJECT_ROOT / raw
            try:
                resolved = possible.resolve()
            except OSError:
                return
            if resolved.is_file():
                candidates.add(resolved)

    visit(value)
    return {
        str(path): _sha256(path)
        for path in sorted(candidates, key=lambda p: str(p))
    }


def _source_hashes(run_dir: Path) -> dict[str, str]:
    names = (
        "config_snapshot.yaml",
        "run_metadata.json",
        "run_status.json",
        "training_summary.json",
        "test_summary.json",
        "test_daily_metrics.csv",
        "summary.json",
    )
    result: dict[str, str] = {}
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{run_dir}: missing required result file {name}")
        result[name] = _sha256(path)
    checkpoint = run_dir / "checkpoints" / "best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"{run_dir}: missing checkpoints/best_model.pt"
        )
    result["checkpoints/best_model.pt"] = _sha256(checkpoint)
    return result


def _validate_daily(run_dir: Path, daily: pd.DataFrame) -> pd.DataFrame:
    if "date" not in daily.columns:
        raise ValueError(f"{run_dir}: test_daily_metrics.csv is missing date")
    if "algorithm_team_profit" not in daily.columns:
        raise ValueError(
            f"{run_dir}: test_daily_metrics.csv is missing algorithm_team_profit"
        )
    if len(daily) != REQUIRED_DAYS:
        raise ValueError(
            f"{run_dir}: formal study requires exactly {REQUIRED_DAYS} test days, "
            f"got {len(daily)}"
        )
    if daily["date"].isna().any() or daily["date"].duplicated().any():
        raise ValueError(f"{run_dir}: test dates must be present and unique")
    daily = daily.copy()
    daily["date"] = daily["date"].astype(str)
    profits = pd.to_numeric(
        daily["algorithm_team_profit"],
        errors="coerce",
    )
    if profits.isna().any() or not np.isfinite(profits.to_numpy()).all():
        raise ValueError(
            f"{run_dir}: algorithm_team_profit contains non-finite values"
        )

    trajectory_dir = run_dir / "test_hourly_trajectories"
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(
            f"{run_dir}: missing test_hourly_trajectories directory"
        )
    for date in daily["date"]:
        path = trajectory_dir / f"{date}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"{run_dir}: missing hourly trajectory for {date}"
            )
        hourly = pd.read_csv(path)
        if len(hourly) != 24:
            raise ValueError(
                f"{run_dir}: hourly trajectory {path.name} must contain 24 rows, "
                f"got {len(hourly)}"
            )
    return daily


def _load_run(run_dir: Path) -> RunRecord:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    source_hashes = _source_hashes(run_dir)
    snapshot_payload = yaml.safe_load(
        (run_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(snapshot_payload, Mapping):
        raise ValueError(f"{run_dir}: invalid config_snapshot.yaml")
    snapshot = copy.deepcopy(dict(snapshot_payload))
    bundle = _algorithm_bundle(snapshot)
    settings = bundle["algorithm"]
    experiment = bundle["experiment"]
    network = settings.get("network")
    if not isinstance(network, Mapping):
        raise ValueError(f"{run_dir}: missing algorithm.network")

    if settings.get("name") != "matd3_forecast_context":
        raise ValueError(
            f"{run_dir}: algorithm.name is not matd3_forecast_context"
        )
    group = experiment.get("group")
    if group not in GROUP_MATRIX:
        raise ValueError(f"{run_dir}: invalid experiment.group {group!r}")
    horizon = network.get("forecast_horizon")
    peer = network.get("include_peer_forecast")
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError(f"{run_dir}: invalid forecast_horizon")
    if not isinstance(peer, bool):
        raise ValueError(f"{run_dir}: invalid include_peer_forecast")
    expected = GROUP_MATRIX[str(group)]
    if (horizon, peer) != expected:
        raise ValueError(
            f"{run_dir}: group {group} does not match H/peer matrix"
        )

    metadata = _read_json(run_dir / "run_metadata.json")
    if metadata.get("algorithm_slug") != "matd3_forecast_context":
        raise ValueError(
            f"{run_dir}: run_metadata algorithm_slug mismatch"
        )
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{run_dir}: invalid run seed")

    status = _read_json(run_dir / "run_status.json")
    if status.get("status") != "COMPLETED":
        raise ValueError(
            f"{run_dir}: run status is not COMPLETED"
        )

    training_summary = _read_json(run_dir / "training_summary.json")
    checkpoint_selection = training_summary.get("checkpoint_selection")
    if not isinstance(checkpoint_selection, Mapping):
        raise ValueError(f"{run_dir}: missing checkpoint_selection metadata")
    if checkpoint_selection.get("best_checkpoint") != "checkpoints/best_model.pt":
        raise ValueError(
            f"{run_dir}: formal test must use checkpoints/best_model.pt"
        )

    test_summary = _read_json(run_dir / "test_summary.json")
    checkpoint_used = Path(str(test_summary.get("checkpoint_used", "")))
    if checkpoint_used.name != "best_model.pt":
        raise ValueError(
            f"{run_dir}: test_summary does not report best_model.pt as checkpoint_used"
        )
    if int(test_summary.get("test_days_count", -1)) != REQUIRED_DAYS:
        raise ValueError(
            f"{run_dir}: test_summary test_days_count must equal {REQUIRED_DAYS}"
        )

    summary = _read_json(run_dir / "summary.json")
    if int(summary.get("test_days_count", -1)) != REQUIRED_DAYS:
        raise ValueError(
            f"{run_dir}: summary test_days_count must equal {REQUIRED_DAYS}"
        )

    daily = _validate_daily(
        run_dir,
        pd.read_csv(run_dir / "test_daily_metrics.csv"),
    )

    checkpoint = torch.load(
        run_dir / "checkpoints" / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{run_dir}: invalid best checkpoint payload")
    if checkpoint.get("algorithm_slug") != "matd3_forecast_context":
        raise ValueError(f"{run_dir}: checkpoint algorithm_slug mismatch")
    contract = checkpoint.get("forecast_context_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(
            f"{run_dir}: checkpoint forecast_context_contract missing"
        )
    contract = copy.deepcopy(dict(contract))
    if int(contract.get("horizon", -1)) != horizon:
        raise ValueError(
            f"{run_dir}: checkpoint horizon disagrees with algorithm config"
        )
    if contract.get("include_peer_forecast") is not peer:
        raise ValueError(
            f"{run_dir}: checkpoint peer switch disagrees with algorithm config"
        )
    if checkpoint.get("forecast_context_version") != "forecast_context_v1":
        raise ValueError(
            f"{run_dir}: checkpoint forecast context version mismatch"
        )

    environment = snapshot.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError(f"{run_dir}: snapshot environment mapping is missing")

    return RunRecord(
        path=run_dir,
        group=str(group),
        seed=int(seed),
        horizon=int(horizon),
        include_peer_forecast=bool(peer),
        config_snapshot=snapshot,
        algorithm_bundle=bundle,
        metadata=metadata,
        test_summary=test_summary,
        summary=summary,
        daily=daily,
        checkpoint_contract=contract,
        source_hashes=source_hashes,
        referenced_file_hashes=_referenced_file_hashes(environment),
    )


def _assert_study_consistency(
    records: Sequence[RunRecord],
    required_seeds: Sequence[int],
) -> dict[tuple[str, int], RunRecord]:
    if not required_seeds:
        raise ValueError("required_seeds must not be empty")
    normalized_seeds: list[int] = []
    for seed in required_seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("required_seeds must contain integers")
        if seed in normalized_seeds:
            raise ValueError(f"Duplicate required seed: {seed}")
        normalized_seeds.append(int(seed))

    by_key: dict[tuple[str, int], RunRecord] = {}
    for record in records:
        key = (record.group, record.seed)
        if key in by_key:
            raise ValueError(
                f"Duplicate run supplied for group={record.group}, seed={record.seed}"
            )
        by_key[key] = record

    expected_keys = {
        (group, seed)
        for seed in normalized_seeds
        for group in GROUPS
    }
    actual_keys = set(by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Study run set mismatch; missing={missing}, extra={extra}"
        )

    environment_reference = _canonical(records[0].config_snapshot["environment"])
    algorithm_reference = _canonical(
        _sanitized_algorithm_bundle(records[0].algorithm_bundle)
    )
    dates_reference = set(records[0].daily["date"].astype(str))
    file_hash_reference = records[0].referenced_file_hashes

    for record in records[1:]:
        if _canonical(record.config_snapshot["environment"]) != environment_reference:
            raise ValueError(
                f"{record.path}: environment config differs across study runs"
            )
        if _canonical(
            _sanitized_algorithm_bundle(record.algorithm_bundle)
        ) != algorithm_reference:
            raise ValueError(
                f"{record.path}: algorithm config differs beyond group/H/peer whitelist"
            )
        dates = set(record.daily["date"].astype(str))
        if dates != dates_reference:
            raise ValueError(
                f"{record.path}: test date set differs across study runs"
            )
        if record.referenced_file_hashes != file_hash_reference:
            raise ValueError(
                f"{record.path}: referenced environment/data file hashes differ"
            )

    # Runtime and budget must be paired within each seed. Unknown public policy
    # values such as "unregistered" are accepted because the plugin is identified
    # by algorithm_slug and its explicit config.
    runtime_keys = (
        "train_episodes",
        "test_episodes",
        "device",
        "runtime",
        "ess_sharing_control_mode",
        "ess_sharing_observation_mode",
        "evaluation_forecast_noise",
    )
    for seed in normalized_seeds:
        seed_records = [by_key[(group, seed)] for group in GROUPS]
        baseline = seed_records[0].metadata
        for record in seed_records[1:]:
            for key in runtime_keys:
                if baseline.get(key) != record.metadata.get(key):
                    raise ValueError(
                        f"seed {seed}: run_metadata {key!r} differs across groups"
                    )
            baseline_contract = baseline.get("experiment_contract")
            current_contract = record.metadata.get("experiment_contract")
            if isinstance(baseline_contract, Mapping) and isinstance(
                current_contract, Mapping
            ):
                baseline_contract = dict(baseline_contract)
                current_contract = dict(current_contract)
                baseline_contract.pop("policy", None)
                current_contract.pop("policy", None)
                if baseline_contract != current_contract:
                    raise ValueError(
                        f"seed {seed}: experiment_contract differs across groups"
                    )

    return by_key


def _numeric_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _ratio_of_sums(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
) -> float | None:
    if numerator not in frame.columns or denominator not in frame.columns:
        return None
    num = pd.to_numeric(frame[numerator], errors="coerce")
    den = pd.to_numeric(frame[denominator], errors="coerce")
    mask = np.isfinite(num) & np.isfinite(den)
    if not mask.any():
        return None
    denominator_sum = float(den[mask].sum())
    if abs(denominator_sum) <= 1e-12:
        return None
    return float(num[mask].sum()) / denominator_sum


def _run_summary_row(record: RunRecord) -> dict[str, Any]:
    daily = record.daily
    result: dict[str, Any] = {
        "group": record.group,
        "seed": record.seed,
        "run_directory": str(record.path),
        "forecast_horizon": record.horizon,
        "include_peer_forecast": record.include_peer_forecast,
        "actor_parameter_count": record.summary.get("actor_parameter_count"),
        "checkpoint_episode": record.test_summary.get("checkpoint_episode"),
        "team_profit_mean": _numeric_mean(daily, "algorithm_team_profit"),
        "team_profit_median": float(
            pd.to_numeric(daily["algorithm_team_profit"]).median()
        ),
        "team_profit_worst_day": float(
            pd.to_numeric(daily["algorithm_team_profit"]).min()
        ),
        "gain_vs_idle_mean_abs": _numeric_mean(daily, "gain_vs_idle_abs"),
        "negative_cooperation_surplus_days": (
            int(
                (
                    pd.to_numeric(
                        daily["algorithm_cooperation_surplus"],
                        errors="coerce",
                    )
                    < 0.0
                ).sum()
            )
            if "algorithm_cooperation_surplus" in daily.columns
            else None
        ),
        "strict_feasible_day_rate": (
            float(
                pd.to_numeric(
                    daily["strict_feasible_day"],
                    errors="coerce",
                ).mean()
            )
            if "strict_feasible_day" in daily.columns
            else (
                float(pd.to_numeric(daily["feasible_day"], errors="coerce").mean())
                if "feasible_day" in daily.columns
                else None
            )
        ),
        "unmet_energy_mean_kwh": _numeric_mean(daily, "unmet_energy_kwh"),
        "renewable_curtailment_mean_kwh": _numeric_mean(
            daily,
            "renewable_curtailment_kwh",
        ),
        "renewable_utilization_rate_mean": _numeric_mean(
            daily,
            "renewable_utilization_rate",
        ),
        "p2p_total_energy_mean_kwh": _numeric_mean(
            daily,
            "p2p_total_energy_kwh",
        ),
        "grid_import_mean_kwh": _numeric_mean(daily, "grid_import_kwh"),
        "grid_export_mean_kwh": _numeric_mean(daily, "grid_export_kwh"),
        "battery_charge_mean_kwh": _numeric_mean(
            daily,
            "battery_charge_kwh",
        ),
        "battery_discharge_mean_kwh": _numeric_mean(
            daily,
            "battery_discharge_kwh",
        ),
        "cooperation_value_capture_ratio_of_sums": _ratio_of_sums(
            daily,
            "algorithm_cooperation_surplus",
            "cooperative_milp_cooperation_surplus",
        ),
        "training_seconds": record.summary.get("training_seconds"),
    }

    optional_metrics = (
        "hvac_comfort_rate",
        "hvac_violation_degree_hours",
        "ev_departure_success_rate",
        "mill_downstream_service_completion_rate",
        "curtailment_reduction_kwh",
        "ess_offer_acceptance_rate",
    )
    for metric in optional_metrics:
        result[f"{metric}_mean"] = _numeric_mean(daily, metric)
    return result


def _paired_daily_rows(
    by_key: Mapping[tuple[str, int], RunRecord],
    required_seeds: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in required_seeds:
        indexed = {
            group: by_key[(group, seed)].daily.set_index("date")
            for group in GROUPS
        }
        common_dates = set(indexed["A"].index)
        for group in GROUPS[1:]:
            common_dates &= set(indexed[group].index)
        if len(common_dates) != REQUIRED_DAYS:
            raise ValueError(
                f"seed {seed}: expected {REQUIRED_DAYS} paired dates, got {len(common_dates)}"
            )

        for effect, high_group, low_group in PAIR_EFFECTS:
            high = indexed[high_group]
            low = indexed[low_group]
            for date in sorted(common_dates):
                high_profit = float(high.loc[date, "algorithm_team_profit"])
                low_profit = float(low.loc[date, "algorithm_team_profit"])
                row: dict[str, Any] = {
                    "seed": seed,
                    "date": date,
                    "effect": effect,
                    "high_group": high_group,
                    "low_group": low_group,
                    "team_profit_delta": high_profit - low_profit,
                }
                for metric in (
                    "unmet_energy_kwh",
                    "renewable_curtailment_kwh",
                    "renewable_utilization_rate",
                    "p2p_total_energy_kwh",
                    "grid_import_kwh",
                    "grid_export_kwh",
                ):
                    if metric in high.columns and metric in low.columns:
                        high_value = pd.to_numeric(
                            pd.Series([high.loc[date, metric]]),
                            errors="coerce",
                        ).iloc[0]
                        low_value = pd.to_numeric(
                            pd.Series([low.loc[date, metric]]),
                            errors="coerce",
                        ).iloc[0]
                        row[f"{metric}_delta"] = (
                            float(high_value - low_value)
                            if math.isfinite(high_value) and math.isfinite(low_value)
                            else np.nan
                        )
                rows.append(row)

        # Interaction is paired by date, not by row position.
        for date in sorted(common_dates):
            profit = {
                group: float(
                    indexed[group].loc[date, "algorithm_team_profit"]
                )
                for group in GROUPS
            }
            rows.append(
                {
                    "seed": seed,
                    "date": date,
                    "effect": "interaction_D-C-B+A",
                    "high_group": "D-C",
                    "low_group": "B-A",
                    "team_profit_delta": (
                        profit["D"]
                        - profit["C"]
                        - profit["B"]
                        + profit["A"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _factor_effects(
    paired: pd.DataFrame,
    required_seeds: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    effects = [item[0] for item in PAIR_EFFECTS] + ["interaction_D-C-B+A"]
    seed_means: dict[str, list[float]] = {effect: [] for effect in effects}

    for effect in effects:
        for seed in required_seeds:
            subset = paired[
                (paired["effect"] == effect)
                & (paired["seed"] == seed)
            ]
            values = pd.to_numeric(
                subset["team_profit_delta"],
                errors="coerce",
            ).dropna()
            if len(values) != REQUIRED_DAYS:
                raise ValueError(
                    f"{effect}, seed={seed}: incomplete paired daily effects"
                )
            mean_value = float(values.mean())
            seed_means[effect].append(mean_value)
            rows.append(
                {
                    "effect": effect,
                    "seed": seed,
                    "mean_daily_delta": mean_value,
                    "median_daily_delta": float(values.median()),
                    "worst_daily_delta": float(values.min()),
                    "win_rate": float((values > 0.0).mean()),
                    "seed_mean": mean_value,
                    "std_across_seeds": np.nan,
                }
            )

        values = np.asarray(seed_means[effect], dtype=np.float64)
        rows.append(
            {
                "effect": effect,
                "seed": "all",
                "mean_daily_delta": float(values.mean()),
                "median_daily_delta": np.nan,
                "worst_daily_delta": np.nan,
                "win_rate": np.nan,
                "seed_mean": float(values.mean()),
                "std_across_seeds": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_readme(
    target: Path,
    study_id: str,
    records: Sequence[RunRecord],
    required_seeds: Sequence[int],
) -> None:
    single_seed_note = (
        "This report contains one training seed only. Cross-seed variance is "
        "unknown and std_across_seeds is intentionally left blank; treat the "
        "results as exploratory rather than a multi-seed confirmation."
        if len(required_seeds) == 1
        else (
            "Cross-seed summaries use sample standard deviation (ddof=1); "
            "daily observations are paired within seed and are not treated as "
            "independent training runs."
        )
    )
    lines = [
        f"# Forecast Context Study: {study_id}",
        "",
        f"Required seeds: {', '.join(map(str, required_seeds))}",
        "",
        single_seed_note,
        "",
        "Factor definitions:",
        "",
        "- B-A: add peer future balance at H=8.",
        "- C-A: extend local forecast horizon from 8 to 24 without peer future.",
        "- D-C: add peer future balance at H=24.",
        "- D-B: extend horizon from 8 to 24 with peer future enabled.",
        "- interaction: D-C-B+A.",
        "",
        "The primary effect is the date-paired team-profit difference. "
        "Cooperation value capture is computed as a ratio of sums, not an "
        "average of daily ratios. The shared 55-day test set is not a new blind "
        "holdout, and MILP remains a full-information reference rather than pure "
        "learning error.",
        "",
        "Included runs:",
        "",
    ]
    for record in sorted(records, key=lambda item: (item.seed, item.group)):
        lines.append(
            f"- group {record.group}, seed {record.seed}: {record.path}"
        )
    lines.extend(
        [
            "",
            "Source runs and checkpoints are read-only. This report does not "
            "rerun policies, environments, settlement, or MILP.",
            "",
        ]
    )
    (target / "README.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def summarize_study(
    run_dirs: Sequence[Path],
    *,
    study_id: str,
    required_seeds: Sequence[int],
) -> Path:
    """Read saved results and create a new immutable-by-default study report."""
    target = _safe_resolve_under_report_root(study_id)
    records = [_load_run(Path(path)) for path in run_dirs]
    if not records:
        raise ValueError("At least one --run directory is required")

    required_seed_list = [int(seed) for seed in required_seeds]
    by_key = _assert_study_consistency(records, required_seed_list)

    before_hashes = {
        str(record.path): copy.deepcopy(record.source_hashes)
        for record in records
    }

    run_summary = pd.DataFrame(
        [_run_summary_row(record) for record in records]
    ).sort_values(["seed", "group"])
    paired = _paired_daily_rows(by_key, required_seed_list)
    effects = _factor_effects(paired, required_seed_list)

    manifest = {
        "study_id": study_id,
        "algorithm_slug": "matd3_forecast_context",
        "required_seeds": required_seed_list,
        "required_groups": list(GROUPS),
        "test_days_per_run": REQUIRED_DAYS,
        "aggregation": {
            "daily_pairing_key": "date",
            "cross_seed_std_ddof": 1,
            "single_seed_std": None,
            "cooperation_value_capture": (
                "sum(algorithm_cooperation_surplus) / "
                "sum(cooperative_milp_cooperation_surplus)"
            ),
        },
        "runs": [
            {
                "run_directory": str(record.path),
                "group": record.group,
                "seed": record.seed,
                "forecast_horizon": record.horizon,
                "include_peer_forecast": record.include_peer_forecast,
                "checkpoint_episode": record.test_summary.get(
                    "checkpoint_episode"
                ),
                "git_commit": record.metadata.get("git_commit"),
                "device": record.metadata.get("device"),
                "runtime": record.metadata.get("runtime"),
                "source_file_sha256": record.source_hashes,
                "referenced_environment_file_sha256": (
                    record.referenced_file_hashes
                ),
                "forecast_context_contract": record.checkpoint_contract,
            }
            for record in sorted(
                records,
                key=lambda item: (item.seed, item.group),
            )
        ],
    }

    target.mkdir(parents=True, exist_ok=False)
    (target / "study_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    run_summary.to_csv(target / "run_summary.csv", index=False)
    paired.to_csv(target / "paired_daily_deltas.csv", index=False)
    effects.to_csv(target / "factor_effects.csv", index=False)
    _write_readme(
        target,
        study_id,
        records,
        required_seed_list,
    )

    for record in records:
        after = _source_hashes(record.path)
        if after != before_hashes[str(record.path)]:
            raise RuntimeError(
                f"Source run changed while summarizing: {record.path}"
            )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize saved forecast-context ablation runs"
    )
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=Path,
        help="Explicit run directory; repeat once per included group/seed",
    )
    parser.add_argument(
        "--require-seeds",
        nargs="+",
        required=True,
        type=int,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = summarize_study(
        args.run,
        study_id=args.study_id,
        required_seeds=args.require_seeds,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
