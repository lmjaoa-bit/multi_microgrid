"""Six single-device controlled counterfactuals in the source-aware revised MILP.

Requires the user's handoff ZIP and the original repository spatial train/test
CSVs. Every cost reduction is measured against the SAME fixed-all-device
scenario. The P2P matcher remains relaxed: these are diagnostic, not achievable
real-world or environment-equivalent causal savings.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data_manager import DataManager
from core.profiles import HeterogeneousProfileModel
from core.tariffs import buy_price_column, sell_price_column

IDS = frozenset(("mg1", "mg2", "mg3"))
CASES = {
    "fixed_all": (IDS, IDS),
    "release_mg1_ess": (IDS, IDS - {"mg1"}),
    "release_mg2_ess": (IDS, IDS - {"mg2"}),
    "release_mg3_ess": (IDS, IDS - {"mg3"}),
    "release_mg1_mill": (IDS - {"mg1"}, IDS),
    "release_mg2_hvac": (IDS - {"mg2"}, IDS),
    "release_mg3_ev": (IDS - {"mg3"}, IDS),
}


def load_model(source):
    """Patch only the supplied, source-aware revised solver's fixing selectors."""
    if "Source-aware sale when selling" not in source:
        raise ValueError("The handoff does not contain the revised source-aware MILP")

    def replace(old, new, count=1):
        nonlocal source
        found = source.count(old)
        if found != count:
            raise ValueError(
                f"Unsupported handoff MILP version: {old[:65]!r}, "
                f"expected {count} occurrences, found {found}"
            )
        source = source.replace(old, new)

    replace(
        "              fixed_dr: pd.DataFrame | None = None,\n"
        "              fixed_ess: pd.DataFrame | None = None) -> OracleResult:",
        "              fixed_dr: pd.DataFrame | None = None,\n"
        "              fixed_ess: pd.DataFrame | None = None,\n"
        "              fixed_dr_ids: frozenset[str] | None = None,\n"
        "              fixed_ess_ids: frozenset[str] | None = None) -> OracleResult:"
    )
    replace(
        '    specs = {mg["id"]: mg for mg in cfg["microgrids"]}',
        '    fixed_dr_ids = frozenset(ids if fixed_dr_ids is None else fixed_dr_ids)\n'
        '    fixed_ess_ids = frozenset(ids if fixed_ess_ids is None else fixed_ess_ids)\n'
        '    if not fixed_dr_ids.issubset(ids) or not fixed_ess_ids.issubset(ids):\n'
        '        raise ValueError("Unknown microgrid ID in fixed device set")\n'
        '    if fixed_dr is None: fixed_dr_ids = frozenset()\n'
        '    if fixed_ess is None: fixed_ess_ids = frozenset()\n'
        '    specs = {mg["id"]: mg for mg in cfg["microgrids"]}'
    )
    replace(
        "                          if fixed_dr is not None else None)",
        "                          if fixed_dr is not None and mg in fixed_dr_ids else None)", 2
    )
    replace(
        "                if fixed_dr is None:\n"
        "                    fixed_h = fixed_c = None",
        "                if fixed_dr is None or mg not in fixed_dr_ids:\n"
        "                    fixed_h = fixed_c = None"
    )
    replace(
        "                        if fixed_ess is not None else None)",
        "                        if fixed_ess is not None and mg in fixed_ess_ids else None)", 2
    )
    model = types.ModuleType("handoff_single_device_revised_milp")
    model.__file__ = "<handoff/code/perfect_foresight_dr.py>"
    sys.modules[model.__name__] = model
    exec(compile(source, model.__file__, "exec"), model.__dict__)
    return model


def execute(args):
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["_project_root"] = str(config_path.parent.parent)
    test = DataManager(cfg, split="test", seed=42)
    train = DataManager(cfg, split="train", seed=42)
    profiles = HeterogeneousProfileModel(cfg, train.df)
    profiles.cache_frame(test.df)
    cache = {}

    def dispatch_data(config, split="test", day=None):
        if split != "test" or day is None:
            raise ValueError("Only explicit original test days are supported")
        if day not in cache:
            frame = test.get_day(day)
            mg_ids = [str(m["id"]) for m in cfg["microgrids"]]
            exo = {mg: profiles.exogenous_series(mg, frame) for mg in mg_ids}
            cache[day] = types.SimpleNamespace(
                timestamps=[str(t) for t in frame["timestamp"]],
                horizon=len(frame),
                agent_ids=mg_ids,
                load_kw={mg: exo[mg]["load_kw"] for mg in mg_ids},
                renewable_kw={mg: exo[mg]["renewable_kw"] for mg in mg_ids},
                buy_price={
                    mg: frame[buy_price_column(mg)].to_numpy(float) for mg in mg_ids
                },
                sell_price={
                    mg: frame[sell_price_column(mg)].to_numpy(float) for mg in mg_ids
                },
            )
        return cfg, cache[day]

    zpath = args.handoff_zip.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    gates = []
    with zipfile.ZipFile(zpath) as z:
        original = z.read("code/perfect_foresight_dr.py")
        source_sha = sha256(original).hexdigest()
        model = load_model(original.decode("utf-8"))
        model.build_dispatch_data = dispatch_data
        # The physical DR model must reset from the real preprocessed weather.
        model.DataManager = lambda _cfg, split="test", seed=42: test
        history = pd.read_csv(io.BytesIO(
            z.read("data/revised_55_counterfactual_daily.csv")
        ))
        dates = history["day"].astype(str).tolist()
        if len(dates) != 55 or len(set(dates)) != 55:
            raise ValueError("Expected 55 unique dates in the revised handoff CSV")
        actual_dates = {str(d.date()) for d in test.valid_days}
        if not set(dates).issubset(actual_dates):
            raise ValueError("The original spatial test CSV is missing handoff days")
        if args.day:
            if args.day not in dates:
                raise ValueError(f"Unknown test day: {args.day}")
            dates = [args.day]
        history = history.set_index("day")
        for index, day in enumerate(dates, 1):
            _, inp = dispatch_data(cfg, day=day)
            recorded = {}
            exo_error = {}
            for alg in ("matd3", "transformer"):
                f = pd.read_csv(io.BytesIO(z.read(
                    f"data/rl_hourly/{alg}/{day}.csv"
                )))
                if (len(f) != 24 or f["hour"].tolist() != list(range(24))
                        or set(f["date"].astype(str)) != {day}):
                    raise ValueError(f"Invalid hourly RL trajectory: {alg} {day}")
                deviations = [
                    float(np.max(np.abs(
                        f[f"{mg}_{kind}"].to_numpy(float) - expected[mg]
                    )))
                    for mg in IDS
                    for kind, expected in (
                        ("base_load_kw", inp.load_kw),
                        ("renewable_available_kw", inp.renewable_kw),
                    )
                ]
                exo_error[alg] = max(deviations)
                recorded[alg] = f

            historic_free = float(history.loc[day, "new_oracle_cost"])
            free = {"status": "failed", "cost": None, "error": ""}
            try:
                f = model.solve_day(
                    cfg, day, split="test", time_limit=args.time_limit,
                    mip_gap=args.mip_gap,
                )
                free = {
                    "status": f.status, "cost": float(f.total_cost),
                    "gap": f.mip_gap, "residual": f.max_violation, "error": "",
                }
            except (ValueError, RuntimeError) as error:
                free["error"] = str(error)
            baseline_ok = (
                free["status"] == "optimal"
                and free.get("residual", float("inf")) <= 1e-5
                and abs(free["cost"] - historic_free) <= args.baseline_tolerance
            )
            gates.append({
                "date": day, "archived_revised_free_cost": historic_free,
                "original_data_free_cost": free["cost"],
                "archived_baseline_reproduced": baseline_ok,
                "matd3_exogenous_error_kw": exo_error["matd3"],
                "transformer_exogenous_error_kw": exo_error["transformer"],
                "free_status": free["status"], "free_error": free["error"],
            })
            pd.DataFrame(gates).to_csv(out / "baseline_reproduction.csv", index=False)
            for alg in ("matd3", "transformer"):
                fixed_cost = None
                fixed_status = None
                for scenario, (fix_dr, fix_ess) in CASES.items():
                    item = {
                        "date": day, "algorithm": alg, "scenario": scenario,
                        "model_version": "source_aware_revised_single_device_v1",
                        "source_exogenous": "original_repository_spatial_train_test",
                        "revised_model_sha256": source_sha,
                        "original_data_free_cost": free["cost"],
                        "historical_free_baseline_pass": baseline_ok,
                        "rl_exogenous_max_abs_kw": exo_error[alg],
                        "status": "not_run", "error": "",
                    }
                    if exo_error[alg] > args.exogenous_tolerance:
                        item.update(status="input_mismatch",
                                    error="Original data differs from RL actual load or RES")
                    elif free["status"] != "optimal":
                        item.update(status="free_failed", error=free["error"])
                    elif not baseline_ok:
                        item.update(status="historical_baseline_mismatch",
                                    error="Original-data free objective differs from handoff")
                    else:
                        try:
                            solved = model.solve_day(
                                cfg, day, split="test",
                                time_limit=args.time_limit,
                                mip_gap=args.mip_gap,
                                fixed_dr=recorded[alg],
                                fixed_ess=recorded[alg],
                                fixed_dr_ids=fix_dr,
                                fixed_ess_ids=fix_ess,
                            )
                            item.update(
                                status=solved.status, cost=float(solved.total_cost),
                                mip_gap=solved.mip_gap,
                                primal_residual=solved.max_violation,
                                **{f"cost_{k}": v
                                   for k, v in solved.component_costs.items()},
                            )
                            if solved.status != "optimal" or solved.max_violation > 1e-5:
                                item.update(status="nonoptimal_or_infeasible",
                                            error="Solver status or residual is unacceptable")
                            if scenario == "fixed_all" and item["status"] == "optimal":
                                archived = history.loc[day, f"{alg}_fixed_both_cost"]
                                if pd.notna(archived) and abs(item["cost"] - archived) > args.baseline_tolerance:
                                    item.update(status="fixed_all_baseline_mismatch",
                                                error="Fixed-all objective differs from handoff")
                        except (ValueError, RuntimeError) as error:
                            item.update(status="failed", error=str(error))
                    if scenario == "fixed_all":
                        fixed_cost, fixed_status = item.get("cost"), item["status"]
                    elif fixed_status == "optimal" and item["status"] == "optimal":
                        reduction = fixed_cost - item["cost"]
                        if reduction < -args.baseline_tolerance:
                            item.update(status="monotonicity_failed",
                                        error="Releasing one device raised optimal cost")
                        else:
                            item["controlled_cost_reduction"] = reduction
                    rows.append(item)
                    pd.DataFrame(rows).to_csv(
                        out / "single_device_scenarios.csv", index=False
                    )
                print(f"[{index}/{len(dates)}] {day} {alg}: "
                      f"fixed_all={fixed_status}", flush=True)

    df = pd.DataFrame(rows)
    status = df.pivot_table(
        index=["date", "algorithm"], columns="scenario",
        values="status", aggfunc="first"
    ).eq("optimal").reindex(columns=list(CASES), fill_value=False).all(axis=1)
    pair = status.unstack("algorithm").reindex(
        columns=["matd3", "transformer"], fill_value=False
    )
    common_dates = pair.loc[pair.all(axis=1)].index.tolist()
    gains = {}
    for alg in ("matd3", "transformer"):
        gains[alg] = {}
        for case in CASES:
            if case == "fixed_all":
                continue
            observed = df.loc[
                df["date"].isin(common_dates)
                & df["algorithm"].eq(alg)
                & df["scenario"].eq(case), "controlled_cost_reduction"
            ]
            gains[alg][case] = float(observed.mean()) if len(observed) else None
    fail = df.loc[df.status.ne("optimal"),
                  ["date", "algorithm", "scenario", "status", "error"]]
    summary = {
        "scope": "revised_model_internal_single_device_diagnostic_only",
        "environment_equivalence_certified": False,
        "source_exogenous": "original_repository_spatial_train_test",
        "model_source_sha256": source_sha,
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "test_data_sha256": sha256(
            (Path(cfg["_project_root"]) / cfg["data"]["test_path"]).read_bytes()
        ).hexdigest(),
        "train_data_sha256": sha256(
            (Path(cfg["_project_root"]) / cfg["data"]["train_path"]).read_bytes()
        ).hexdigest(),
        "days_requested": len(dates), "common_complete_count": len(common_dates),
        "common_complete_days": common_dates,
        "mean_controlled_cost_reduction": gains,
        "failed_scenarios": fail.fillna("").to_dict("records"),
    }
    (out / "single_device_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(out), "paired_complete_days": len(common_dates),
        "mean_controlled_cost_reduction": gains,
        "failed_scenarios": len(fail),
    }, indent=2, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-zip", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cooperative.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/single_device_cf")
    parser.add_argument("--day", help="Optional single YYYY-MM-DD for smoke test")
    parser.add_argument("--time-limit", type=float, default=120.)
    parser.add_argument("--mip-gap", type=float, default=1e-5)
    parser.add_argument("--baseline-tolerance", type=float, default=0.02)
    parser.add_argument("--exogenous-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    result = execute(args)
    return 0 if result["common_complete_count"] == result["days_requested"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
