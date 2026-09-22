"""Audit saved RL actions against the CURRENT environment, not a surrogate MILP.

Run from the repository root:
 python scripts/audit_clairvoyant_environment_replay.py --package "path/to/全知MILP差距分解_交接数据包.zip"

The uploaded handoff contains historical RL hourly traces; the environment's
DataManager reads the actual checked-out data/spatial_market_2018_{test,train}.csv.
A failing day is retained with its reason. This does not solve any MILP.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.tariffs import prices_from_row
from envs.multi_microgrid_env import MultiMicrogridEnv


IDS = ("mg1", "mg2", "mg3")
FIELDS = (
    ("actual_load_kw", "plans", "actual_load_kw"),
    ("battery_charge_kw", "plans", "battery_charge_kw"),
    ("battery_discharge_kw", "plans", "battery_discharge_kw"),
    ("grid_import_kw", "settlements", "grid_import_kw"),
    ("grid_export_kw", "settlements", "grid_export_kw"),
    ("unmet_load_kw", "settlements", "unmet_load_kw"),
    ("renewable_curtailment_kw", "settlements", "renewable_curtailment_kw"),
)


def csv_in_zip(z: zipfile.ZipFile, path: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(path)))


def compare(errors: list[dict], maxima: dict[str, float], label: str,
            observed: float, expected: float, day: str, hour: int,
            algorithm: str, tolerance: float) -> None:
    delta = abs(float(observed) - float(expected))
    maxima[label] = max(maxima.get(label, 0.0), delta)
    if not np.isfinite(delta) or delta > tolerance:
        if len(errors) < 35:
            errors.append(dict(algorithm=algorithm, day=day, hour=hour,
                               field=label, recorded=float(expected),
                               replayed=float(observed), abs_error=float(delta)))


def audit(z: zipfile.ZipFile, env: MultiMicrogridEnv, alg: str,
          daily: pd.DataFrame, tolerance: float, first_only: bool) -> tuple[list[dict], list[dict]]:
    summaries, failures = [], []
    for item in daily.itertuples(index=False):
        day = str(item.date)[:10]
        seed = int(item.test_seed)
        rec = dict(algorithm=alg, day=day, test_seed=seed,
                   model_version="recorded_actions_current_environment_replay",
                   source_exogenous="repository_DataManager_original_spatial_CSV",
                   replay_status="not_started")
        mismatches: list[dict] = []
        maxima: dict[str, float] = {}
        steps = 0
        try:
            history = csv_in_zip(z, f"data/rl_hourly/{alg}/{day}.csv")
            if len(history) != 24 or sorted(history.hour.astype(int).tolist()) != list(range(24)):
                raise ValueError(f"Expected 24 hours (0..23), got {len(history)} rows")
            history = history.sort_values("hour").reset_index(drop=True)
            env.reset(day=day, seed=seed, deterministic=True, forecast_noise=True)
            if len(env.episode_df) != 24:
                raise ValueError("Repository DataManager did not return a 24h natural day")
            total_replay_cost = 0.0
            for hour, saved in history.iterrows():
                actual = env._exogenous_at(hour)
                for mg in IDS:
                    compare(mismatches, maxima, f"{mg}.base_load_kw",
                            actual[mg]["load_kw"], saved[f"{mg}_base_load_kw"],
                            day, hour, alg, tolerance)
                    compare(mismatches, maxima, f"{mg}.renewable_kw",
                            actual[mg]["renewable_kw"], saved[f"{mg}_renewable_available_kw"],
                            day, hour, alg, tolerance)
                if any(k.endswith((".base_load_kw", ".renewable_kw")) and v > tolerance
                       for k, v in maxima.items()):
                    rec["replay_status"] = "original_exogenous_mismatch"
                    break

                # Reconstruct the four exact pool input categories from the
                # SAVED projector outputs. Actual ESS charge may be curtailed
                # by the grid after the market, so use requested charge here.
                renewable_offer, ess_offer, load_demand, charge_demand = {}, {}, {}, {}
                for mg in IDS:
                    load = float(saved[f"{mg}_actual_load_kw"])
                    renewable = float(saved[f"{mg}_renewable_available_kw"])
                    requested_charge = (float(saved[f"{mg}_battery_charge_kw"])
                                        + float(saved[f"{mg}_battery_charge_curtailed_kw"]))
                    surplus = max(0.0, renewable - load)
                    renewable_to_battery = min(requested_charge, surplus)
                    renewable_offer[mg] = max(0.0, surplus - renewable_to_battery)
                    ess_offer[mg] = float(saved[f"{mg}_ess_p2p_offer_kw"])
                    load_demand[mg] = max(0.0, max(0.0, load-renewable)
                                          - float(saved[f"{mg}_direct_local_discharge_kw"]))
                    charge_demand[mg] = max(0.0, requested_charge - renewable_to_battery)
                pricing = {mg: prices_from_row(env.episode_df.iloc[hour], mg) for mg in IDS}
                matched = env.energy_pool.allocate_structured(
                    renewable_offer, ess_offer, load_demand, charge_demand,
                    {mg: pricing[mg][1] for mg in IDS},
                    {mg: pricing[mg][0] for mg in IDS},
                )
                for mg in IDS:
                    peer = matched.by_microgrid[mg]
                    for column, key in (("p2p_buy_kw", "p2p_buy_kw"),
                                        ("p2p_sell_kw", "p2p_sell_kw"),
                                        ("p2p_to_load_kw", "p2p_to_load_kw"),
                                        ("p2p_to_battery_kw", "p2p_to_battery_kw"),
                                        ("renewable_p2p_sell_kw", "renewable_p2p_sell_kw"),
                                        ("ess_p2p_sell_kw", "ess_p2p_sell_kw")):
                        compare(mismatches, maxima, f"matcher.{mg}.{key}",
                                peer[key], saved[f"{mg}_{column}"],
                                day, hour, alg, tolerance)
                for source in IDS:
                    for dest in IDS:
                        if source == dest:
                            continue
                        key = f"{source}->{dest}"
                        compare(mismatches, maxima, f"matcher.flow.{key}",
                                matched.pair_flows_kwh.get(key, 0.0)/env.delta_t,
                                saved[f"p2p_{source}_to_{dest}_kw"],
                                day, hour, alg, tolerance)
                if any(k.startswith("matcher.") and v > tolerance for k, v in maxima.items()):
                    rec["replay_status"] = "pool_clearing_mismatch"
                    break

                actions = {mg: np.array([float(saved[f"{mg}_raw_action_0"]),
                                         float(saved[f"{mg}_raw_action_1"])], dtype=np.float64)
                           for mg in IDS}
                _, _, done, info = env.step(actions)
                steps += 1
                total_replay_cost -= float(info["team_profit"])
                for mg in IDS:
                    for suffix, source, field in FIELDS:
                        compare(mismatches, maxima, f"rollout.{mg}.{suffix}",
                                info[source][mg][field],
                                saved[f"{mg}_{suffix}"],
                                day, hour, alg, tolerance)
                    compare(mismatches, maxima, f"rollout.{mg}.soc",
                            info["devices"][mg]["soc"], saved[f"{mg}_soc"],
                            day, hour, alg, tolerance)
                    compare(mismatches, maxima, f"rollout.{mg}.p2p_buy_kw",
                            info["p2p"]["by_microgrid"][mg]["p2p_buy_kw"],
                            saved[f"{mg}_p2p_buy_kw"], day, hour, alg, tolerance)
                    compare(mismatches, maxima, f"rollout.{mg}.p2p_sell_kw",
                            info["p2p"]["by_microgrid"][mg]["p2p_sell_kw"],
                            saved[f"{mg}_p2p_sell_kw"], day, hour, alg, tolerance)
                    compare(mismatches, maxima, f"rollout.{mg}.hourly_profit",
                            info["costs"][mg]["physical_profit"],
                            saved[f"{mg}_hourly_profit"], day, hour, alg, tolerance)
                for key, column in (("t_air_c", "mg2_t_air_c"),
                                     ("t_mass_c", "mg2_t_mass_c")):
                    compare(mismatches, maxima, f"rollout.mg2.{key}",
                            info["dr_devices"]["mg2"][key], saved[column],
                            day, hour, alg, tolerance)
                if any(k.startswith("rollout.") and v > tolerance for k, v in maxima.items()):
                    rec["replay_status"] = "recorded_actions_rollout_mismatch"
                    break
                if bool(done) != (hour == 23):
                    raise RuntimeError(f"done flag mismatch at hour {hour}")
            if steps == 24 and rec["replay_status"] == "not_started":
                compare(mismatches, maxima, "daily.rl_total_cost", total_replay_cost,
                        -float(item.algorithm_team_profit), day, 24, alg, max(tolerance, 1e-3))
                rec["replay_status"] = ("passed" if not mismatches else
                                        "recorded_actions_rollout_mismatch")
            rec["replayed_steps"] = steps
            rec["max_abs_error_by_field"] = json.dumps(maxima, ensure_ascii=False)
            if mismatches:
                rec["first_difference"] = json.dumps(mismatches[0], ensure_ascii=False)
                failures.extend(mismatches)
        except Exception as exc:
            rec.update(replay_status="exception",
                       first_difference=f"{type(exc).__name__}: {exc}",
                       replayed_steps=steps)
        summaries.append(rec)
        print(json.dumps({k:v for k,v in rec.items() if k != "max_abs_error_by_field"},
                         ensure_ascii=False), flush=True)
        if first_only and rec["replay_status"] != "passed":
            break
    return summaries, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True,
                        help="Original 全知MILP差距分解_交接数据包.zip")
    parser.add_argument("--config", type=Path, default=Path("configs/cooperative.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/clairvoyant_environment_replay_audit"))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--first-failure-only", action="store_true")
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    result, violations = [], []
    with zipfile.ZipFile(a.package) as z:
        for alg in ("matd3", "transformer"):
            daily = csv_in_zip(z, f"data/{alg}_daily_metrics.csv")
            if len(daily) != 55 or daily.date.nunique() != 55:
                raise ValueError(f"{alg}: expected exactly 55 unique test days, found {len(daily)}")
            env = MultiMicrogridEnv(a.config, split="test")
            rows, issues = audit(z, env, alg, daily, a.tolerance, a.first_failure_only)
            result.extend(rows); violations.extend(issues)
    pd.DataFrame(result).to_csv(a.output / "paired_replay_daily.csv", index=False)
    pd.DataFrame(violations).to_csv(a.output / "replay_mismatches.csv", index=False)
    counts = pd.DataFrame(result).groupby(["algorithm", "replay_status"]).size()
    report = {"model_version": "recorded_actions_current_environment_replay",
              "source_exogenous": "repository_DataManager_original_spatial_CSV",
              "config_path": str(a.config), "package": str(a.package),
              "tolerance": a.tolerance, "status_counts": {
                  f"{alg}:{status}": int(count) for (alg,status),count in counts.items()
              }, "all_110_passed": len(result) == 110 and all(
                  row["replay_status"] == "passed" for row in result)}
    (a.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_110_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
