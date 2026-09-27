from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from evaluation.service_metrics import (
    SERVICE_METRICS_SCHEMA_VERSION,
    aggregate_service_days,
    summarize_day_services,
)


SOURCE_FILES = (
    "test_summary.json",
    "test_daily_metrics.csv",
    "config_snapshot.yaml",
)


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r",encoding="utf-8",newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value,bool):
        return value
    return str(value).strip().lower() in {"1","true","yes","y","on"}


def _flatten(prefix: str, value: Any, target: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key,item in value.items():
            _flatten(f"{prefix}_{key}" if prefix else str(key),item,target)
    elif isinstance(value,(list,tuple,set)):
        target[prefix]=json.dumps(list(value),ensure_ascii=False)
    else:
        target[prefix]=value


def _validate_output(run: Path, output: Path) -> None:
    if output == run or run in output.parents:
        raise ValueError("Audit output must not equal or live inside the source run")
    parts=[part.lower() for part in output.parts]
    for index in range(len(parts)-1):
        if parts[index]=="outputs" and parts[index+1]=="releases":
            raise ValueError("Audit output must not live inside outputs/releases")
    if output.exists():
        raise FileExistsError(
            "Audit output already exists; refusing to overwrite prior audit evidence"
        )


def _source_paths(run: Path) -> list[Path]:
    paths=[]
    for name in SOURCE_FILES:
        path=run/name
        if path.is_file():
            paths.append(path)
    hourly=run/"test_hourly_trajectories"
    if hourly.is_dir():
        paths.extend(sorted(hourly.glob("*.csv")))
    return paths


def _configuration_source(run: Path, summary: Mapping[str, Any]) -> str | None:
    physical = summary.get("experiment_contract", {}).get("physical_settings")
    if isinstance(physical, Mapping) and physical.get("microgrids"):
        return "test_summary.json:experiment_contract.physical_settings"
    snapshot = run / "config_snapshot.yaml"
    if snapshot.is_file():
        document = yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}
        env = document.get("environment")
        if isinstance(env, Mapping) and env.get("microgrids"):
            return "config_snapshot.yaml:environment"
    return None


def _source_metadata(source: Path, paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        stat = path.stat()
        result[str(path.relative_to(source))] = {
            "sha256": _sha256(path),
            "size_bytes": int(stat.st_size),
            "modified_time_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        }
    return result


def _load_config(run: Path, summary: Mapping[str,Any]) -> dict[str,Any] | None:
    physical=summary.get("experiment_contract",{}).get("physical_settings")
    if isinstance(physical,Mapping) and physical.get("microgrids"):
        return dict(physical)
    snapshot=run/"config_snapshot.yaml"
    if snapshot.is_file():
        document=yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}
        env=document.get("environment")
        if isinstance(env,Mapping) and env.get("microgrids"):
            return dict(env)
    return None


def audit_saved_run(run: str|Path, output: str|Path) -> dict[str,Any]:
    source=Path(run).resolve()
    destination=Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {source}")
    _validate_output(source,destination)

    source_paths=_source_paths(source)
    required={source/"test_summary.json",source/"test_daily_metrics.csv"}
    if not required.issubset(set(source_paths)):
        missing=sorted(str(p.name) for p in required if p not in source_paths)
        raise FileNotFoundError(f"Missing saved run evidence: {missing}")
    source_files_before = _source_metadata(source, source_paths)
    before = {
        name: item["sha256"] for name, item in source_files_before.items()
    }

    summary=_read_json(source/"test_summary.json")
    config=_load_config(source,summary)
    config_source = _configuration_source(source, summary)
    daily=_read_csv(source/"test_daily_metrics.csv")
    schema=str(
        summary.get("experiment_contract",{}).get(
            "service_metrics_schema_version","legacy_unknown"
        )
    )

    audited=[]
    audited_nested=[]
    for row in daily:
        date=str(row.get("date","")).strip()
        hourly_path=source/"test_hourly_trajectories"/f"{date}.csv"
        if not hourly_path.is_file():
            metrics={
                "metrics_schema_version":schema,
                "data_complete":False,
                "missing_fields":["hourly_trajectory"],
                "strict_feasible_day":None,
            }
        elif config is None:
            metrics={
                "metrics_schema_version":schema,
                "data_complete":False,
                "missing_fields":["physical_configuration"],
                "strict_feasible_day":None,
            }
        else:
            hourly=_read_csv(hourly_path)
            metrics=summarize_day_services(
                hourly,
                config,
                _as_bool(row.get("feasible_day",False)),
                schema,
            )
        nested={"date":date, **metrics}
        audited_nested.append(nested)
        result={"date":date}
        _flatten("",metrics,result)
        audited.append(result)

    aggregate=aggregate_service_days(audited_nested)
    source_files_after = _source_metadata(source, source_paths)
    after = {
        name: item["sha256"] for name, item in source_files_after.items()
    }
    if before != after:
        raise RuntimeError("Source run changed while auditing; refusing output")

    destination.mkdir(parents=True,exist_ok=False)
    daily_path=destination/"audit_daily_metrics.csv"
    fields=[]
    for row in audited:
        for key in row:
            if key not in fields:
                fields.append(key)
    with daily_path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(audited)

    audit_summary={
        "source_run":str(source),
        "source_metrics_schema_version":schema,
        "audit_metrics_schema_version":SERVICE_METRICS_SCHEMA_VERSION,
        "source_day_count":len(daily),
        "aggregate":aggregate,
        "historical_reinterpretation":False,
        "note":(
            "Only evidence present in the saved run is recomputed. Missing "
            "service fields remain incomplete/null; current defaults are never "
            "used to fabricate historical evidence."
        ),
    }
    (destination/"audit_summary.json").write_text(
        json.dumps(audit_summary,ensure_ascii=False,indent=2,allow_nan=False),
        encoding="utf-8",
    )
    manifest={
        "source_run":str(source),
        "output":str(destination),
        "source_hashes_before":before,
        "source_hashes_after":after,
        "source_hashes_unchanged":before==after,
        "source_metrics_schema_version":schema,
        "audit_metrics_schema_version":SERVICE_METRICS_SCHEMA_VERSION,
        "configuration_source": config_source,
        "source_dates": [str(row.get("date", "")).strip() for row in daily],
        "source_files": source_files_before,
        "missing_evidence_days":[
            row["date"] for row in audited_nested
            if not bool(row.get("data_complete",False))
        ],
        "missing_fields_by_day": {
            row["date"]: list(row.get("missing_fields", []))
            for row in audited_nested
            if row.get("missing_fields")
        },
    }
    (destination/"audit_manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),
        encoding="utf-8",
    )
    return audit_summary


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(
        description="Read-only service-metric audit for a saved run"
    )
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    return parser


def main() -> int:
    args=build_parser().parse_args()
    audit_saved_run(args.run,args.output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
