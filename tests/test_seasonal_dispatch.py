from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from evaluation import plotting


def _profiles() -> pd.DataFrame:
    rows = []
    for month in (1, 4, 7, 10):
        for day, offset in ((1, -1), (2, 0), (3, 2)):
            for hour in range(24):
                row = {"date": f"2018-{month:02d}-{day:02d}", "hour": hour,
                       "mg2_t_out_c": 10 + month + offset, "p2p_price": 0.2 + hour / 100,
                       "reward": offset * 1000, "action": offset}
                for mg in ("mg1", "mg2", "mg3"):
                    row[f"{mg}_base_load_kw"] = 30 + hour + offset
                    row[f"{mg}_renewable_available_kw"] = 10 + offset
                rows.append(row)
    return pd.DataFrame(rows)


def _selector():
    selector = getattr(plotting, "select_seasonal_days", None)
    assert callable(selector), "The shared plotter needs seasonal representative-day selection"
    return selector


def test_seasonal_dates_ignore_policy_and_input_order():
    hourly = _profiles()
    wanted = {"spring": "2018-04-02", "summer": "2018-07-02",
              "autumn": "2018-10-02", "winter": "2018-01-02"}
    assert _selector()(hourly) == wanted
    hourly["reward"] *= -999
    hourly["action"] += 99999
    assert _selector()(hourly.sample(frac=1, random_state=42)) == wanted


def test_selection_uses_hourly_shape_not_only_daily_energy():
    hourly = _profiles().query("date.str.startswith('2018-04')").copy()
    # Every day has identical daily energy; day 2 alone has the central shape.
    for day, amplitude in ((1, -1), (2, 0), (3, 2)):
        mask = hourly.date == f"2018-04-{day:02d}"
        for mg in ("mg1", "mg2", "mg3"):
            hourly.loc[mask, f"{mg}_base_load_kw"] = 30 + amplitude * np.r_[np.ones(12), -np.ones(12)]
            hourly.loc[mask, f"{mg}_renewable_available_kw"] = 10
        hourly.loc[mask, "mg2_t_out_c"] = 20
    assert _selector()(hourly) == {"spring": "2018-04-02"}


@pytest.mark.parametrize("defect", ["missing_hour", "duplicate_hour", "nan"])
def test_invalid_hourly_conditions_cannot_select_a_day(defect):
    hourly = _profiles()
    if defect == "missing_hour":
        hourly = hourly.iloc[1:]
    elif defect == "duplicate_hour":
        hourly = pd.concat([hourly, hourly.iloc[:1]])
    else:
        hourly.loc[0, "mg1_base_load_kw"] = np.nan
    with pytest.raises(ValueError):
        _selector()(hourly)


def test_regeneration_removes_old_figures_and_preserves_saved_evidence(tmp_path: Path):
    run = tmp_path / "run"
    trajectory = run / "test_hourly_trajectories"
    trajectory.mkdir(parents=True)
    for date, day in _profiles().groupby("date"):
        day.to_csv(trajectory / f"{date}.csv", index=False)
    pd.DataFrame({"date": sorted(_profiles().date.unique())}).to_csv(run / "test_daily_metrics.csv", index=False)
    (run / "config_snapshot.yaml").write_text(yaml.safe_dump({"environment": {}}), encoding="utf-8")
    (run / "test_summary.json").write_text("{}", encoding="utf-8")
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "best_model.pt").write_bytes(b"checkpoint evidence")
    figures = run / "figures"
    figures.mkdir()
    (figures / "14_transfer_payments.png").write_bytes(b"old generated image")
    (figures / "notes.txt").write_text("keep me", encoding="utf-8")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob("*") if p.is_file() and figures not in p.parents}
    paths = plotting.generate_figures(run, formats=("png",))
    assert len(paths) == 6
    assert not (figures / "14_transfer_payments.png").exists()
    assert (figures / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    manifest = json.loads((run / "plot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seasonal_days"] == {"spring": "2018-04-02", "summer": "2018-07-02", "autumn": "2018-10-02", "winter": "2018-01-02"}
    assert set(manifest["figures"]) == {p.name for p in paths}
    assert set(manifest["source_sha256"]) >= {"test_hourly_trajectories/2018-04-02.csv", "config_snapshot.yaml"}


def test_completed_run_migration_preserves_checkpoint_and_repairs_paths(tmp_path: Path):
    from experiments import layout
    run = tmp_path / "run"
    (run / "figures").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    checkpoint = run / "checkpoints" / "best_model.pt"
    checkpoint.write_bytes(b"unchanged model")
    (run / "training_history.json").write_text('[{"episode": 1, "reward": -2.5}]', encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"run_directory": str(run), "checkpoint": str(checkpoint), "best_checkpoint": "checkpoints/best_model.pt"}), encoding="utf-8")
    migrate = getattr(layout, "organize_run_artifacts", None)
    assert callable(migrate), "Completed runs must support migration to the two-folder layout"
    migrate(run)
    assert {path.name for path in run.iterdir()} == {"figures", "artifacts"}
    assert (run / "artifacts/checkpoints/best_model.pt").read_bytes() == b"unchanged model"
    assert (run / "artifacts/training_history.json").read_text(encoding="utf-8") == '[{"episode": 1, "reward": -2.5}]'
    summary = json.loads((run / "artifacts/summary.json").read_text(encoding="utf-8"))
    assert summary["run_directory"] == str(run)
    assert summary["checkpoint"] == str(run / "artifacts/checkpoints/best_model.pt")
    assert summary["best_checkpoint"] == "artifacts/checkpoints/best_model.pt"
    migrate(run)  # Repeating a completed migration is harmless.


def test_migration_rejects_immutable_release(tmp_path: Path):
    from experiments import layout
    (tmp_path / "release_manifest.json").write_text("{}", encoding="utf-8")
    migrate = getattr(layout, "organize_run_artifacts", None)
    assert callable(migrate)
    with pytest.raises(ValueError, match="immutable"):
        migrate(tmp_path)
    assert (tmp_path / "release_manifest.json").exists()
