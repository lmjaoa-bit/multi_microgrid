"""Twenty-four-hour clairvoyant cooperative MILP; NOT a runnable RL policy.

This separate entry point leaves the historical ESS-only coalition MILP and the
normal MATD3/Transformer training pipeline unchanged. The cooperative pool is
a source-aware flow RELAXATION, not an exact reimplementation of its sequential
local-first clearing; its objective is a diagnostic planning reference.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_manager import DataManager
from optimization.common_constraints import build_dispatch_data, load_config, solve_dispatch


def _read_rl_daily(path: Path, dates: list[str], label: str) -> dict[str, float]:
    frame = pd.read_csv(path)
    required = {"date", "algorithm_team_profit"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{label}: expected CSV columns {sorted(required)}")
    dates_in_csv = frame["date"].astype(str).str[:10]
    if dates_in_csv.duplicated().any():
        raise ValueError(f"{label}: duplicate test dates in {path}")
    result = dict(zip(dates_in_csv, frame["algorithm_team_profit"].astype(float)))
    missing = sorted(set(dates) - result.keys())
    if missing:
        raise ValueError(f"{label}: missing {len(missing)} test dates, e.g. {missing[:3]}")
    return result


def run(
    config: Path,
    *,
    split: str,
    dates: list[str],
    output: Path,
    matd3_csv: Path | None = None,
    transformer_csv: Path | None = None,
    solver_output: bool = False,
) -> dict:
    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate dates in oracle evaluation")
    output.mkdir(parents=True, exist_ok=False)
    (output / "hourly").mkdir()
    day_results = []
    errors = []
    for date in dates:
        try:
            cfg, dispatch = build_dispatch_data(config, split=split, day=date)
            if dispatch.horizon != 24:
                raise ValueError(f"Expected a complete 24h day, got {dispatch.horizon}")
            result = solve_dispatch(
                cfg, dispatch, dispatch.agent_ids, cooperative=True,
                solver_output=solver_output, include_dr=True,
                p2p_export_source="unrestricted",
            )
            if result.status != "kOptimal":
                raise RuntimeError(f"Day {date}: solver status {result.status}")
            result.trajectories.to_csv(output / "hourly" / f"{date}.csv", index=False)
            day_results.append({
                "date": date,
                "oracle_profit": result.total_profit,
                "oracle_cost": result.total_cost,
                "solver_status": result.status,
                "p2p_energy_kwh": sum(result.pair_flows_kwh.values()),
                "gross_load_peak_kw": (
                    result.trajectories.groupby("timestamp")["load_kw"].sum().max()
                ),
                "pair_flows_kwh": result.pair_flows_kwh,
            })
        except (ValueError, RuntimeError) as exc:
            errors.append({"date": date, "error": str(exc)})
            # Do not mix a partial day-set with the 55-day RL averages.
            continue

    completed = [row["date"] for row in day_results]
    audit = {
        "model": "24h_full_information_cooperative_dr_milp",
        "status": "complete" if not errors else "incomplete",
        "split": split,
        "days_requested": dates,
        "days_completed": completed,
        "errors": errors,
        "solver_requirement": "kOptimal",
        "reference_type": "joint_optimization_with_relaxed_p2p_clearing",
        "not_a_verified_environment_equivalent_upper_bound": True,
        "assumptions": [
            "All 24 hourly exogenous load/renewable/price/temperature observations are known exactly.",
            "Industrial silo and EV charging models are linear and use current physical parameters.",
            "HVAC uses the environment's 2R2C state equations with strictly enforced temperature bands.",
            "End-of-day ESS, mill, HVAC and EV deviations use environment penalty rates.",
            "P2P uses capacity, energy balance and source-aware export bounds rather than the exact sequential pool-clearing mechanism.",
            "Unmet base/flexible load is a power-balance slack, not a full replica of delivered-fraction device-state updates.",
            "An infeasible day is an audit failure, not evidence that the RL policy beats the MILP.",
        ],
    }
    if completed and not errors:
        daily = pd.DataFrame(day_results)
        audit["oracle_profit_mean"] = float(daily["oracle_profit"].mean())
        audit["oracle_cost_mean"] = float(daily["oracle_cost"].mean())
        audit["oracle_gross_load_peak_mean_kw"] = float(daily["gross_load_peak_kw"].mean())

        if matd3_csv is not None or transformer_csv is not None:
            if split != "test":
                raise ValueError("RL comparison requires --split test")
            for label, csv in (("matd3", matd3_csv), ("transformer", transformer_csv)):
                if csv is None:
                    continue
                rl = _read_rl_daily(csv, completed, label)
                daily[f"{label}_profit"] = daily["date"].map(rl)
                daily[f"{label}_cost"] = -daily[f"{label}_profit"]
                daily[f"{label}_cost_minus_oracle"] = (
                    daily[f"{label}_cost"] - daily["oracle_cost"]
                )
                audit[f"{label}_cost_mean"] = float(daily[f"{label}_cost"].mean())
                audit[f"{label}_cost_minus_oracle_mean"] = float(
                    daily[f"{label}_cost_minus_oracle"].mean()
                )
                audit[f"{label}_days_below_oracle_cost"] = int(
                    (daily[f"{label}_cost_minus_oracle"] < -1e-5).sum()
                )
            if len(completed) != 55:
                audit["comparison_warning"] = (
                    f"Only {len(completed)} days selected: do not compare against 55-day published means"
                )
        daily.drop(columns=["pair_flows_kwh"]).to_csv(output / "daily_comparison.csv", index=False)
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Complete-DR 24h clairvoyant MILP audit")
    parser.add_argument("--config", type=Path, default=Path("configs/cooperative.yaml"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--day", help="A YYYY-MM-DD date from the requested split")
    group.add_argument("--all-test-days", action="store_true",
                       help="Evaluate all complete natural days in the chosen split")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-matd3", type=Path,
                        help="test_daily_metrics.csv from the aligned MATD3 best-checkpoint test")
    parser.add_argument("--compare-transformer", type=Path,
                        help="test_daily_metrics.csv from ordinary hierarchical Transformer test")
    parser.add_argument("--solver-output", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    days = ([args.day] if args.day is not None else [
        str(day.date()) for day in DataManager(config, split=args.split).valid_days
    ])
    root = args.output or (
        Path("outputs/working_runs/oracle_full_information/runs") /
        ("ORACLE_FULL_INFORMATION_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    report = run(args.config, split=args.split, dates=days, output=root,
                 matd3_csv=args.compare_matd3,
                 transformer_csv=args.compare_transformer,
                 solver_output=args.solver_output)
    print(json.dumps({"output": str(root), **report}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
