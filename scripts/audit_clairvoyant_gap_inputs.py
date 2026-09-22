"""Audit 55-day oracle handoff against repository ORIGINAL exogenous data and exact P2P clearing.

This is an input/clearing audit, NOT an environment-equivalent MILP or causal estimate.
Usage:
python scripts/audit_clairvoyant_gap_inputs.py --handoff-zip "C:/path/全知MILP差距分解_交接数据包.zip"
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data_manager import DataManager
from core.profiles import HeterogeneousProfileModel
from core.tariffs import prices_from_row
from envs.cooperative_energy_sharing import CooperativeEnergySharing

MG = ("mg1", "mg2", "mg3")
ALG = ("matd3", "transformer")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(name)))


def _max_error(got: object, expected: object) -> float:
    return float(np.max(np.abs(np.asarray(got, dtype=float) - np.asarray(expected, dtype=float))))


def audit(config_path: Path, archive: Path, output: Path, tolerance: float) -> dict:
    config_path = config_path.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["_project_root"] = str(config_path.parent.parent)
    test = DataManager(cfg, split="test", seed=42)
    train = DataManager(cfg, split="train", seed=42)
    profiles = HeterogeneousProfileModel(cfg, train.df)
    profiles.cache_frame(test.df)
    pool = CooperativeEnergySharing(float(cfg["time"].get("delta_t", 1.0)),
                                   bool(cfg["p2p"].get("enabled", True)),
                                   float(cfg["p2p"]["pair_capacity_kw"]))
    with zipfile.ZipFile(archive) as z:
        daily = _read_csv(z, "data/revised_55_counterfactual_daily.csv")
        if daily["day"].duplicated().any() or len(daily) != 55:
            raise ValueError("Handoff daily table must have exactly 55 unique dates.")
        dates = daily["day"].astype(str).tolist()
        out_rows = []
        events = []
        for day in dates:
            original = test.get_day(day)
            if len(original) != 24:
                raise ValueError(f"{day}: original data not a 24-hour day")
            for algorithm in ALG:
                recorded = _read_csv(z, f"data/rl_hourly/{algorithm}/{day}.csv")
                if len(recorded) != 24 or recorded["hour"].tolist() != list(range(24)):
                    raise ValueError(f"{algorithm} {day}: archived hours are not 0..23")
                if set(recorded["date"].astype(str)) != {day}:
                    raise ValueError(f"{algorithm} {day}: date mismatch")
                checks = {}
                for mg in MG:
                    exo = profiles.exogenous_series(mg, original)
                    for col, expect in (
                        (f"{mg}_base_load_kw", exo["load_kw"]),
                        (f"{mg}_renewable_available_kw", exo["renewable_kw"]),
                    ):
                        checks[col] = _max_error(recorded[col], expect)
                buy = {mg: np.asarray([prices_from_row(row, mg)[0] for _, row in original.iterrows()])
                       for mg in MG}
                sell = {mg: np.asarray([prices_from_row(row, mg)[1] for _, row in original.iterrows()])
                        for mg in MG}
                fields = {f"p2p_{a}_to_{b}_kw": 0.0 for a in MG for b in MG if a != b}
                fields.update({f"{mg}_{suffix}": 0.0
                              for mg in MG for suffix in ("p2p_buy_kw", "p2p_sell_kw",
                              "renewable_p2p_sell_kw", "ess_p2p_sell_kw",
                              "p2p_to_load_kw", "p2p_to_battery_kw")})
                fields["p2p_total_energy_kwh"] = 0.0
                for t in range(24):
                    row = recorded.iloc[t]
                    renewable_offers, ess_offers, load_demands, charge_demands = {}, {}, {}, {}
                    for mg in MG:
                        load = float(row[f"{mg}_actual_load_kw"])
                        renewable = float(row[f"{mg}_renewable_available_kw"])
                        # The planned ESS charge may be curtailed by a later grid
                        # limit; the offer/demand stage uses PRE-curtailment charge.
                        requested_charge = (float(row[f"{mg}_battery_charge_kw"])
                                            + float(row[f"{mg}_battery_charge_curtailed_kw"]))
                        surplus = max(0.0, renewable - load)
                        renewable_to_battery = min(requested_charge, surplus)
                        renewable_offers[mg] = max(0.0, surplus - renewable_to_battery)
                        charge_demands[mg] = max(0.0, requested_charge - renewable_to_battery)
                        load_demands[mg] = max(0.0, load - renewable -
                                              float(row[f"{mg}_direct_local_discharge_kw"]))
                        ess_offers[mg] = float(row[f"{mg}_ess_p2p_offer_kw"])
                    cleared = pool.allocate_structured(
                        renewable_offers, ess_offers, load_demands, charge_demands,
                        {mg: float(sell[mg][t]) for mg in MG},
                        {mg: float(buy[mg][t]) for mg in MG},
                    )
                    for mg in MG:
                        for field in ("p2p_buy_kw", "p2p_sell_kw", "renewable_p2p_sell_kw",
                                      "ess_p2p_sell_kw", "p2p_to_load_kw", "p2p_to_battery_kw"):
                            name = f"{mg}_{field}"
                            fields[name] = max(fields[name], abs(
                                float(cleared.by_microgrid[mg][field]) - float(row[name])))
                    for first in MG:
                        for second in MG:
                            if first == second:
                                continue
                            name = f"p2p_{first}_to_{second}_kw"
                            expected = float(cleared.pair_flows_kwh.get(f"{first}->{second}", 0.0))
                            expected /= float(cfg["time"].get("delta_t", 1.0))
                            fields[name] = max(fields[name], abs(expected - float(row[name])))
                    fields["p2p_total_energy_kwh"] = max(
                        fields["p2p_total_energy_kwh"],
                        abs(float(cleared.total_p2p_kwh) -
                            float(row["p2p_total_energy_kwh"])))
                checks.update(fields)
                exo_err = max(checks[k] for k in checks if k.endswith(("_base_load_kw",
                    "_renewable_available_kw")))
                p2p_err = max(fields.values())
                missed = float(recorded[[f"{mg}_unmet_load_kw" for mg in MG]].sum().sum())
                hvac_violations = int(pd.to_numeric(recorded["mg2_comfort_infeasible"]).fillna(0).gt(0).sum())
                result = dict(day=day, algorithm=algorithm, original_exogenous_max_abs_kw=exo_err,
                              original_p2p_replay_max_abs_kw=p2p_err,
                              unmet_kwh=missed * float(cfg["time"].get("delta_t", 1.0)),
                              hvac_comfort_infeasible_hours=hvac_violations,
                              source_exogenous="original_repository_spatial_market_2018",
                              clearing="CooperativeEnergySharing.allocate_structured",
                              exogenous_pass=exo_err <= tolerance,
                              p2p_replay_pass=p2p_err <= tolerance,
                              replay_pass=exo_err <= tolerance and p2p_err <= tolerance)
                out_rows.append(result)
                for metric, err in checks.items():
                    if err > tolerance:
                        events.append(dict(day=day, algorithm=algorithm, metric=metric,
                                           max_abs_error=err))
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(output / "original_input_p2p_audit_110.csv", index=False)
        pd.DataFrame(events, columns=["day", "algorithm", "metric", "max_abs_error"]).to_csv(
            output / "original_input_p2p_mismatches.csv", index=False)
        status = {"scope": "55_days_x_2_policies_original_inputs_and_recorded_action_P2P",
                  "not_certified": ["MILP environment equivalence", "fixed action cost parity",
                                    "realizable DR/ESS causal value"],
                  "config_sha256": _sha(config_path),
                  "test_csv_sha256": _sha(Path(cfg["_project_root"]) / cfg["data"]["test_path"]),
                  "train_csv_sha256": _sha(Path(cfg["_project_root"]) / cfg["data"]["train_path"]),
                  "handoff_zip_sha256": _sha(archive),
                  "days": len(dates), "records": len(out_rows),
                  "exogenous_pass": sum(r["exogenous_pass"] for r in out_rows),
                  "p2p_replay_pass": sum(r["p2p_replay_pass"] for r in out_rows),
                  "both_pass": sum(r["replay_pass"] for r in out_rows),
                  "mismatch_count": len(events),
                  "outstanding_failed_fixed_action_days": ["2018-01-29", "2018-02-26"]}
        (output / "audit_summary.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return status


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--handoff-zip", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/cooperative.yaml")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/clairvoyant_gap_input_audit")
    p.add_argument("--tolerance", type=float, default=1e-5)
    a = p.parse_args()
    summary = audit(a.config, a.handoff_zip, a.output, a.tolerance)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["both_pass"] == summary["records"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
