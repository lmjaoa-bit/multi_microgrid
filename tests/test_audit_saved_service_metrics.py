from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.audit_saved_service_metrics import audit_saved_run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> Path:
    run=root/"saved_run"
    hourly=run/"test_hourly_trajectories"
    hourly.mkdir(parents=True)
    config={
        "time":{"delta_t":1.0},
        "microgrids":[],
    }
    (run/"config_snapshot.yaml").write_text(
        yaml.safe_dump({"environment":config}),encoding="utf-8"
    )
    (run/"test_summary.json").write_text(
        json.dumps({"experiment_contract":{
            "service_metrics_schema_version":"legacy_unknown"
        }}),encoding="utf-8"
    )
    with (run/"test_daily_metrics.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=["date","feasible_day"])
        w.writeheader(); w.writerow({"date":"2026-01-01","feasible_day":1})
    with (hourly/"2026-01-01.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=["timestamp"])
        w.writeheader(); w.writerow({"timestamp":"2026-01-01T00:00:00"})
    return run


def test_audit_is_read_only_and_marks_missing_historical_evidence(tmp_path):
    run=_fixture(tmp_path)
    before={p:_sha(p) for p in run.rglob("*") if p.is_file()}
    output=tmp_path/"audit"
    result=audit_saved_run(run,output)
    after={p:_sha(p) for p in run.rglob("*") if p.is_file()}
    assert before==after
    manifest=json.loads((output/"audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_hashes_unchanged"] is True
    assert manifest["missing_evidence_days"]==["2026-01-01"]
    assert manifest["source_dates"] == ["2026-01-01"]
    assert manifest["configuration_source"] is None
    assert manifest["missing_fields_by_day"] == {
        "2026-01-01": ["physical_configuration"]
    }
    assert set(manifest["source_files"]) >= {
        "test_summary.json",
        "test_daily_metrics.csv",
        "config_snapshot.yaml",
        "test_hourly_trajectories/2026-01-01.csv",
    }
    for metadata in manifest["source_files"].values():
        assert len(metadata["sha256"]) == 64
        assert metadata["size_bytes"] >= 0
        assert metadata["modified_time_utc"].endswith("+00:00")
    assert result["historical_reinterpretation"] is False


def test_audit_rejects_output_inside_run(tmp_path):
    run=_fixture(tmp_path)
    with pytest.raises(ValueError):
        audit_saved_run(run,run/"audit")


def test_audit_refuses_existing_output(tmp_path):
    run=_fixture(tmp_path)
    output=tmp_path/"audit"
    output.mkdir()
    with pytest.raises(FileExistsError):
        audit_saved_run(run,output)


def test_audit_rejects_releases_tree(tmp_path):
    run=_fixture(tmp_path)
    with pytest.raises(ValueError):
        audit_saved_run(run,tmp_path/"outputs"/"releases"/"audit")
