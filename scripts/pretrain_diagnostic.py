from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.multi_microgrid_env import MultiMicrogridEnv  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "multi_microgrid.yaml"
DEFAULT_OUTPUT = ROOT / "outputs" / "pretrain_diagnostic"
REWARD_PERCENTILES = (0.01, 0.05, 0.50, 0.95, 0.99)
SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def _python_scalar(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _python_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_scalar(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_python_scalar(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Diagnostic output contains NaN or infinite values.")
    return value


def _assert_finite_frame(frame: pd.DataFrame, label: str) -> None:
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"{label} contains NaN or infinite numeric values.")


def build_static_timeseries(env: MultiMicrogridEnv) -> pd.DataFrame:
    """Build train-only physical load/PV/wind series for every microgrid."""

    if env.split != "train":
        raise ValueError("Pre-training static diagnostics must use the train split only.")
    frame = env.data.df.copy()
    result = pd.DataFrame({"timestamp": frame[env.data.ts_col].copy()})
    ghi = frame["ghi_w_per_m2"].to_numpy(dtype=np.float64)
    wind_speed = frame["wind_speed_mps"].to_numpy(dtype=np.float64)
    pv_cf = np.clip(ghi / env.profiles.ghi_reference, 0.0, 1.0)
    wind_cf = np.fromiter(
        (env.profiles.wind_capacity_factor(value) for value in wind_speed),
        dtype=np.float64,
        count=len(frame),
    )

    for mg in env.agent_ids:
        cfg = env.profiles.mg_cfg[mg]
        load = env.profiles.load_series(mg, frame)
        pv = float(cfg["renewables"]["pv_capacity_kw"]) * pv_cf
        wind = float(cfg["renewables"]["wind_capacity_kw"]) * wind_cf
        renewable = pv + wind
        result[f"{mg}_load_kw"] = load
        result[f"{mg}_pv_kw"] = pv
        result[f"{mg}_wind_kw"] = wind
        result[f"{mg}_res_kw"] = renewable
        result[f"{mg}_net_res_kw"] = renewable - load

    result["month"] = result["timestamp"].dt.month.astype(int)
    result["season"] = result["month"].map(SEASON_BY_MONTH)
    result["hour"] = result["timestamp"].dt.hour.astype(int)
    _assert_finite_frame(result, "Static train time series")
    return result


def compute_microgrid_statistics(
    env: MultiMicrogridEnv,
    static: pd.DataFrame,
) -> pd.DataFrame:
    threshold_kw = float(env.config["p2p"].get("min_trade_power_kw", 1.0))
    rows = []
    for mg in env.agent_ids:
        load = static[f"{mg}_load_kw"].to_numpy(dtype=np.float64)
        pv = static[f"{mg}_pv_kw"].to_numpy(dtype=np.float64)
        wind = static[f"{mg}_wind_kw"].to_numpy(dtype=np.float64)
        renewable = static[f"{mg}_res_kw"].to_numpy(dtype=np.float64)
        net = static[f"{mg}_net_res_kw"].to_numpy(dtype=np.float64)
        load_energy = float(load.sum() * env.delta_t)
        res_energy = float(renewable.sum() * env.delta_t)
        rows.append(
            {
                "microgrid": mg,
                "load_energy_kwh": load_energy,
                "load_mean_kw": float(load.mean()),
                "load_peak_kw": float(load.max()),
                "pv_energy_kwh": float(pv.sum() * env.delta_t),
                "wind_energy_kwh": float(wind.sum() * env.delta_t),
                "res_energy_kwh": res_energy,
                "res_to_load_ratio": res_energy / load_energy if load_energy > 0.0 else 0.0,
                "natural_surplus_hour_ratio": float(np.mean(net > 0.0)),
                "natural_deficit_hour_ratio": float(np.mean(net < 0.0)),
                "below_p2p_threshold_hour_ratio": float(np.mean(np.abs(net) < threshold_kw)),
                "p2p_threshold_kw": threshold_kw,
            }
        )
    result = pd.DataFrame(rows)
    _assert_finite_frame(result, "Microgrid statistics")
    return result


def compute_p2p_potential(
    env: MultiMicrogridEnv,
    static: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    surplus_columns = [np.maximum(static[f"{mg}_net_res_kw"].to_numpy(), 0.0) for mg in env.agent_ids]
    deficit_columns = [np.maximum(-static[f"{mg}_net_res_kw"].to_numpy(), 0.0) for mg in env.agent_ids]
    total_surplus = np.sum(np.stack(surplus_columns), axis=0)
    total_deficit = np.sum(np.stack(deficit_columns), axis=0)
    theoretical_match = np.minimum(total_surplus, total_deficit)

    potential = pd.DataFrame(
        {
            "timestamp": static["timestamp"],
            "total_surplus_kw": total_surplus,
            "total_deficit_kw": total_deficit,
            "theoretical_p2p_match_kw": theoretical_match,
            "month": static["month"],
            "season": static["season"],
            "hour": static["hour"],
        }
    )
    total_load_energy = sum(
        float(static[f"{mg}_load_kw"].sum() * env.delta_t)
        for mg in env.agent_ids
    )
    match_energy = float(theoretical_match.sum() * env.delta_t)
    opportunity = theoretical_match > 1e-12
    summary = {
        "p2p_opportunity_hours": int(opportunity.sum()),
        "p2p_opportunity_ratio": float(opportunity.mean()),
        "theoretical_p2p_match_kwh": match_energy,
        "theoretical_p2p_match_to_total_load_ratio": (
            match_energy / total_load_energy if total_load_energy > 0.0 else 0.0
        ),
        "total_load_energy_kwh": total_load_energy,
        "ideal_p2p_buy_kwh": match_energy,
        "ideal_p2p_sell_kwh": match_energy,
    }

    def aggregate(group_field: str) -> pd.DataFrame:
        grouped = potential.groupby(group_field, sort=True, observed=True)
        rows = []
        for key, group in grouped:
            match = group["theoretical_p2p_match_kw"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    group_field: key,
                    "hours": int(len(group)),
                    "p2p_opportunity_hours": int(np.sum(match > 1e-12)),
                    "p2p_opportunity_ratio": float(np.mean(match > 1e-12)),
                    "theoretical_p2p_match_kwh": float(match.sum() * env.delta_t),
                    "mean_theoretical_p2p_match_kw": float(match.mean()),
                }
            )
        return pd.DataFrame(rows)

    monthly = aggregate("month")
    seasonal = aggregate("season")
    _assert_finite_frame(potential, "P2P potential")
    _assert_finite_frame(monthly, "Monthly P2P potential")
    _assert_finite_frame(seasonal, "Seasonal P2P potential")
    return potential, summary, monthly, seasonal


def compute_curtailment_potential(
    env: MultiMicrogridEnv,
    static: pd.DataFrame,
    potential: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    no_p2p_kw = np.zeros(len(static), dtype=np.float64)
    for mg in env.agent_ids:
        surplus = np.maximum(static[f"{mg}_net_res_kw"].to_numpy(dtype=np.float64), 0.0)
        no_p2p_kw += np.maximum(surplus - float(env.grid[mg].export_limit_kw), 0.0)

    theoretical_match = potential["theoretical_p2p_match_kw"].to_numpy(dtype=np.float64)
    # Ideal-P2P prioritizes energy that would otherwise be curtailed. This gives
    # the maximum physically available curtailment reduction without batteries,
    # generators, bid failures, losses, or fees.
    ideal_p2p_kw = np.maximum(no_p2p_kw - theoretical_match, 0.0)
    comparison = potential.copy()
    comparison["no_p2p_curtailment_kw"] = no_p2p_kw
    comparison["ideal_p2p_curtailment_kw"] = ideal_p2p_kw

    no_p2p_kwh = float(no_p2p_kw.sum() * env.delta_t)
    ideal_p2p_kwh = float(ideal_p2p_kw.sum() * env.delta_t)
    reduction = max(0.0, no_p2p_kwh - ideal_p2p_kwh)
    summary = {
        "no_p2p_res_curtailment_kwh": no_p2p_kwh,
        "ideal_p2p_res_curtailment_kwh": ideal_p2p_kwh,
        "potential_curtailment_reduction_kwh": reduction,
        "potential_curtailment_reduction_ratio": (
            reduction / no_p2p_kwh if no_p2p_kwh > 0.0 else 0.0
        ),
    }
    _assert_finite_frame(comparison, "Curtailment potential")
    return comparison, summary


def select_evenly_spaced_days(valid_days: Sequence[pd.Timestamp], count: int) -> list[pd.Timestamp]:
    if count <= 0:
        raise ValueError("Diagnostic day count must be positive.")
    if not valid_days:
        raise ValueError("No valid training days are available.")
    count = min(int(count), len(valid_days))
    indices = np.linspace(0, len(valid_days) - 1, num=count, dtype=int)
    return [pd.Timestamp(valid_days[index]) for index in np.unique(indices)]


def rule_based_actions(env: MultiMicrogridEnv, _: np.random.Generator) -> Dict[str, np.ndarray]:
    # Battery idle and MT OFF isolate the effect of midpoint P2P clearing.
    return {
        mg: np.asarray([0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        for mg in env.agent_ids
    }


def random_legal_actions(env: MultiMicrogridEnv, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    actions: Dict[str, np.ndarray] = {}
    for mg in env.agent_ids:
        commitment = float(rng.choice([-1.0, 1.0]))
        generator_setpoint = float(rng.uniform(-1.0, 1.0)) if commitment > 0.0 else -1.0
        actions[mg] = np.asarray(
            [
                rng.uniform(-1.0, 1.0),
                commitment,
                generator_setpoint,
                rng.uniform(-1.0, 1.0),
            ],
            dtype=np.float32,
        )
    return actions


def _empty_microgrid_totals() -> Dict[str, float]:
    return {
        "total_cost": 0.0,
        "grid_buy_kwh": 0.0,
        "grid_sell_kwh": 0.0,
        "p2p_buy_kwh": 0.0,
        "p2p_sell_kwh": 0.0,
        "mt_generation_kwh": 0.0,
        "battery_charge_kwh": 0.0,
        "battery_discharge_kwh": 0.0,
        "res_curtailment_kwh": 0.0,
        "unmet_load_kwh": 0.0,
        "p2p_buy_cost": 0.0,
        "p2p_sell_revenue": 0.0,
        "p2p_net_cost": 0.0,
    }


def run_policy_case(
    config: str | Path | Mapping[str, Any],
    days: Sequence[pd.Timestamp],
    *,
    policy_name: str,
    p2p_enabled: bool,
    action_fn: Callable[[MultiMicrogridEnv, np.random.Generator], Dict[str, np.ndarray]],
    seed: int,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    env = MultiMicrogridEnv(config, split="train")
    env.p2p_enabled = bool(p2p_enabled)
    rng = np.random.default_rng(int(seed))
    by_microgrid = {mg: _empty_microgrid_totals() for mg in env.agent_ids}
    reward_rows: list[Dict[str, Any]] = []
    system = {
        "system_total_cost": 0.0,
        "total_p2p_traded_kwh": 0.0,
        "total_grid_import_kwh": 0.0,
        "total_grid_export_kwh": 0.0,
        "total_res_curtailment_kwh": 0.0,
        "total_unmet_load_kwh": 0.0,
    }

    for episode_index, day in enumerate(days):
        env.reset(
            day=str(pd.Timestamp(day).date()),
            seed=int(seed) + episode_index,
            deterministic=True,
        )
        done = False
        while not done:
            actions = action_fn(env, rng)
            _, rewards, done, info = env.step(actions)
            system["total_p2p_traded_kwh"] += float(info["market"]["total_trade_kwh"])
            for mg in env.agent_ids:
                settlement = info["settlements"][mg]
                device = info["devices"][mg]
                costs = info["costs"][mg]
                market = info["market"]["by_microgrid"][mg]
                totals = by_microgrid[mg]
                totals["total_cost"] += float(costs["total_cost"])
                totals["grid_buy_kwh"] += float(settlement["grid_buy_kw"]) * env.delta_t
                totals["grid_sell_kwh"] += float(settlement["grid_sell_kw"]) * env.delta_t
                totals["p2p_buy_kwh"] += float(market["p2p_buy_kwh"])
                totals["p2p_sell_kwh"] += float(market["p2p_sell_kwh"])
                totals["mt_generation_kwh"] += float(device["generator"]["energy_kwh"])
                totals["battery_charge_kwh"] += float(device["battery"]["charge_kw"]) * env.delta_t
                totals["battery_discharge_kwh"] += float(device["battery"]["discharge_kw"]) * env.delta_t
                totals["res_curtailment_kwh"] += float(settlement["renewable_curtailment_kw"]) * env.delta_t
                totals["unmet_load_kwh"] += float(settlement["unmet_load_kw"]) * env.delta_t
                totals["p2p_buy_cost"] += float(market["p2p_buy_cost"])
                totals["p2p_sell_revenue"] += float(market["p2p_sell_revenue"])
                totals["p2p_net_cost"] += float(market["p2p_net_cost"])

                reward_rows.append(
                    {
                        "policy": policy_name,
                        "microgrid": mg,
                        "reward": float(rewards[mg]),
                        "total_cost": float(costs["total_cost"]),
                        "grid_cost": float(costs["grid_cost"]),
                        "generator_cost": float(costs["generator_cost"]),
                        "generator_fuel_cost": float(device["generator"]["fuel_cost"]),
                        "generator_om_cost": float(device["generator"]["om_cost"]),
                        "generator_startup_cost": float(device["generator"]["startup_cost"]),
                        "battery_degradation_cost": float(costs["battery_degradation_cost"]),
                        "carbon_cost": float(costs["carbon_cost"]),
                        "p2p_net_cost": float(costs["p2p_net_cost"]),
                        "p2p_buy_cost": float(market["p2p_buy_cost"]),
                        "p2p_sell_revenue": float(market["p2p_sell_revenue"]),
                        "unmet_load_cost": float(costs["unmet_load_cost"]),
                        "controllable_dump_cost": float(costs["controllable_dump_cost"]),
                    }
                )

    for mg, totals in by_microgrid.items():
        system["system_total_cost"] += totals["total_cost"]
        system["total_grid_import_kwh"] += totals["grid_buy_kwh"]
        system["total_grid_export_kwh"] += totals["grid_sell_kwh"]
        system["total_res_curtailment_kwh"] += totals["res_curtailment_kwh"]
        system["total_unmet_load_kwh"] += totals["unmet_load_kwh"]

    reward_frame = pd.DataFrame(reward_rows)
    _assert_finite_frame(reward_frame, f"{policy_name} reward records")
    result = {
        "policy": policy_name,
        "p2p_enabled": bool(p2p_enabled),
        "episode_count": int(len(days)),
        "step_count": int(len(days) * env.episode_length),
        "by_microgrid": by_microgrid,
        "system": system,
    }
    return result, reward_frame


def compute_reward_statistics(reward_records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cost_columns = [
        "total_cost", "grid_cost", "generator_cost", "generator_fuel_cost",
        "generator_om_cost", "generator_startup_cost", "battery_degradation_cost",
        "carbon_cost", "p2p_net_cost", "p2p_buy_cost", "p2p_sell_revenue",
        "unmet_load_cost", "controllable_dump_cost",
    ]
    for (policy, mg), group in reward_records.groupby(["policy", "microgrid"], sort=True):
        rewards = group["reward"].to_numpy(dtype=np.float64)
        quantiles = np.quantile(rewards, REWARD_PERCENTILES)
        row: Dict[str, Any] = {
            "policy": policy,
            "microgrid": mg,
            "steps": int(len(group)),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(ddof=0)),
            "reward_min": float(rewards.min()),
            "reward_max": float(rewards.max()),
            "reward_p01": float(quantiles[0]),
            "reward_p05": float(quantiles[1]),
            "reward_p50": float(quantiles[2]),
            "reward_p95": float(quantiles[3]),
            "reward_p99": float(quantiles[4]),
            "abs_reward_p50": float(np.median(np.abs(rewards))),
            "abs_reward_p95": float(np.quantile(np.abs(rewards), 0.95)),
            "abs_reward_gt_10_ratio": float(np.mean(np.abs(rewards) > 10.0)),
            "abs_reward_lt_1e_3_ratio": float(np.mean(np.abs(rewards) < 1e-3)),
        }
        for column in cost_columns:
            values = group[column].to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_total"] = float(values.sum())
        rows.append(row)
    result = pd.DataFrame(rows)
    _assert_finite_frame(result, "Reward statistics")
    return result


def _rule_based_statistics_frame(cases: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for case_name, case in cases.items():
        for mg, metrics in case["by_microgrid"].items():
            rows.append({"case": case_name, "scope": mg, **metrics})
        rows.append({"case": case_name, "scope": "system", **case["system"]})
    result = pd.DataFrame(rows).fillna(0.0)
    _assert_finite_frame(result, "Rule-based statistics")
    return result


def _summary_frame(summary: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for record in summary["microgrid_statistics"]:
        mg = record["microgrid"]
        for metric, value in record.items():
            if metric != "microgrid":
                rows.append({"section": "microgrid", "scope": mg, "metric": metric, "value": value})
    for section in ("p2p_summary", "curtailment_summary"):
        for metric, value in summary[section].items():
            rows.append({"section": section, "scope": "system", "metric": metric, "value": value})
    for case_name, case in summary["policy_cases"].items():
        for metric, value in case["system"].items():
            rows.append({"section": "policy_case", "scope": case_name, "metric": metric, "value": value})
    result = pd.DataFrame(rows)
    _assert_finite_frame(result, "Summary CSV")
    return result


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):,.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 3) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for _, record in frame.loc[:, columns].iterrows():
        values = []
        for column in columns:
            value = record[column]
            values.append(_fmt(value, digits) if isinstance(value, (int, float, np.number)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _diagnostic_conclusions(
    microgrid_stats: pd.DataFrame,
    p2p_summary: Mapping[str, float],
    curtailment_summary: Mapping[str, float],
    cases: Mapping[str, Mapping[str, Any]],
    reward_stats: pd.DataFrame,
) -> Dict[str, str]:
    opportunity_ratio = float(p2p_summary["p2p_opportunity_ratio"])
    match_ratio = float(p2p_summary["theoretical_p2p_match_to_total_load_ratio"])
    surplus_ratios = microgrid_stats.set_index("microgrid")["natural_surplus_hour_ratio"]
    curtailment_reduction = float(curtailment_summary["potential_curtailment_reduction_ratio"])
    rule_no = cases["rule_based_no_p2p"]["system"]
    rule_yes = cases["rule_based_p2p"]["system"]
    cost_delta = float(rule_yes["system_total_cost"] - rule_no["system_total_cost"])
    high_reward_share = float(reward_stats["abs_reward_gt_10_ratio"].max())
    tiny_reward_share = float(reward_stats["abs_reward_lt_1e_3_ratio"].max())
    median_abs_min = float(reward_stats["abs_reward_p50"].min())
    median_abs_max = float(reward_stats["abs_reward_p50"].max())
    p95_abs_max = float(reward_stats["abs_reward_p95"].max())
    worst_abs = float(
        np.maximum(
            np.abs(reward_stats["reward_min"].to_numpy(dtype=np.float64)),
            np.abs(reward_stats["reward_max"].to_numpy(dtype=np.float64)),
        ).max()
    )

    return {
        "p2p_space": (
            f"P2P complementarity appears visible: opportunities occur in {opportunity_ratio:.1%} "
            f"of train hours and theoretical matching equals {match_ratio:.1%} of total load."
        ),
        "heterogeneity": (
            f"Natural seller-hour ratios are MG1 {surplus_ratios['mg1']:.1%}, "
            f"MG2 {surplus_ratios['mg2']:.1%}, and MG3 {surplus_ratios['mg3']:.1%}. "
            "Roles are structurally distinct, while MG3 has only sparse seller hours and should be monitored for role concentration."
        ),
        "curtailment": (
            f"Ideal matching could reduce static no-P2P curtailment by {curtailment_reduction:.1%}; "
            "this is an engineering upper bound rather than a trained result."
        ),
        "reward_scale": (
            f"Median |reward| spans {median_abs_min:.3f}–{median_abs_max:.3f}, the largest |reward| "
            f"p95 is {p95_abs_max:.3f}, and the worst observed magnitude is {worst_abs:.3f}. "
            f"The |reward|>10 share is {high_reward_share:.2%} and the largest |reward|<1e-3 share "
            f"is {tiny_reward_share:.2%}; reward_scale=100 appears numerically reasonable for a pilot."
        ),
        "rule_based_cost": (
            f"Rule-based P2P changes sampled system cost by {cost_delta:+,.2f} monetary units "
            "versus the identical no-P2P policy."
        ),
    }


def generate_plots(
    output_dir: Path,
    env: MultiMicrogridEnv,
    static: pd.DataFrame,
    potential: pd.DataFrame,
    curtailment_summary: Mapping[str, float],
    cases: Mapping[str, Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_days = [pd.Timestamp(day) for day in env.data.valid_days]
    daily_load = []
    for day in valid_days:
        mask = static["timestamp"].dt.normalize() == day.normalize()
        daily_load.append(sum(float(static.loc[mask, f"{mg}_load_kw"].sum()) for mg in env.agent_ids))
    median_load = float(np.median(daily_load))
    representative_day = valid_days[int(np.argmin(np.abs(np.asarray(daily_load) - median_load)))]
    day_mask = static["timestamp"].dt.normalize() == representative_day.normalize()
    day_frame = static.loc[day_mask]

    colors = {"load": "#1f2937", "pv": "#d97706", "wind": "#2563eb", "res": "#059669"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for axis, mg in zip(axes, env.agent_ids):
        hours = day_frame["hour"]
        axis.plot(hours, day_frame[f"{mg}_load_kw"], label="Load", color=colors["load"], linewidth=2.0)
        axis.plot(hours, day_frame[f"{mg}_pv_kw"], label="PV", color=colors["pv"], linestyle="--")
        axis.plot(hours, day_frame[f"{mg}_wind_kw"], label="Wind", color=colors["wind"], linestyle=":")
        axis.plot(hours, day_frame[f"{mg}_res_kw"], label="RES", color=colors["res"], linewidth=2.0)
        axis.set_title(mg.upper())
        axis.set_ylabel("Power (kW)")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, frameon=False)
    axes[-1].set_xlabel("Hour of day")
    axes[-1].set_xticks(range(0, 24, 2))
    fig.suptitle(f"Representative train day profiles — {representative_day.date()}")
    fig.tight_layout()
    fig.savefig(output_dir / "daily_profiles.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    hourly = potential.groupby("hour", sort=True)[
        ["total_surplus_kw", "total_deficit_kw", "theoretical_p2p_match_kw"]
    ].mean()
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(hourly.index, hourly["total_surplus_kw"], label="Total surplus", color="#dc2626", linestyle="--")
    axis.plot(hourly.index, hourly["total_deficit_kw"], label="Total deficit", color="#2563eb", linestyle=":")
    axis.plot(hourly.index, hourly["theoretical_p2p_match_kw"], label="Theoretical match", color="#059669", linewidth=2.5)
    axis.set(title="Average hourly natural P2P potential — train split", xlabel="Hour of day", ylabel="Mean power (kW)")
    axis.set_xticks(range(0, 24, 2))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "p2p_potential.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    static_days = max(1, int(static["timestamp"].dt.normalize().nunique()))
    labels = ["Static\nNo-P2P", "Static\nIdeal-P2P", "Rule\nNo-P2P", "Rule\nP2P"]
    values = [
        float(curtailment_summary["no_p2p_res_curtailment_kwh"]) / static_days,
        float(curtailment_summary["ideal_p2p_res_curtailment_kwh"]) / static_days,
        float(cases["rule_based_no_p2p"]["system"]["total_res_curtailment_kwh"]) / cases["rule_based_no_p2p"]["episode_count"],
        float(cases["rule_based_p2p"]["system"]["total_res_curtailment_kwh"]) / cases["rule_based_p2p"]["episode_count"],
    ]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(labels, values, color=["#6b7280", "#059669", "#9ca3af", "#10b981"])
    axis.bar_label(bars, fmt="%.1f", padding=3)
    axis.set(title="Renewable curtailment comparison", ylabel="Average curtailment (kWh/day)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "curtailment_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    microgrid_stats: pd.DataFrame,
    reward_stats: pd.DataFrame,
) -> None:
    p2p = summary["p2p_summary"]
    curtailment = summary["curtailment_summary"]
    cases = summary["policy_cases"]
    rule_no = cases["rule_based_no_p2p"]["system"]
    rule_yes = cases["rule_based_p2p"]["system"]
    conclusions = summary["diagnostic_conclusions"]

    mg_table = microgrid_stats.copy()
    mg_table["seller_hour_ratio"] = mg_table["natural_surplus_hour_ratio"]
    mg_table["buyer_hour_ratio"] = mg_table["natural_deficit_hour_ratio"]
    reward_view = reward_stats[
        [
            "policy", "microgrid", "reward_mean", "reward_std", "reward_min",
            "reward_max", "reward_p01", "reward_p05", "reward_p50", "reward_p95", "reward_p99",
        ]
    ]
    seasonal_view = pd.DataFrame(summary["seasonal_p2p_potential"])

    report = f"""# Pre-training Diagnostic Report

## Executive summary

- {conclusions['p2p_space']}
- {conclusions['heterogeneity']}
- {conclusions['curtailment']}
- {conclusions['reward_scale']}
- {conclusions['rule_based_cost']}

All conclusions are engineering diagnostics based on the current Environment V1. They are not trained results or hard research pass/fail criteria.

## Microgrid statistics

The complete train split is transformed with train-derived profile statistics. Natural seller/buyer roles use `PV + wind - load` before Battery, MT, P2P, or grid settlement.

{_markdown_table(mg_table, ['microgrid', 'load_mean_kw', 'load_peak_kw', 'res_to_load_ratio', 'seller_hour_ratio', 'buyer_hour_ratio', 'below_p2p_threshold_hour_ratio'])}

![Representative daily load and renewable profiles](daily_profiles.png)

The representative date is the valid training day closest to median system daily load. The chart is descriptive and does not select a favorable P2P day.

## P2P potential

- P2P opportunity hours: **{int(p2p['p2p_opportunity_hours']):,}**
- P2P opportunity ratio: **{float(p2p['p2p_opportunity_ratio']):.2%}**
- Theoretical P2P match energy: **{_fmt(p2p['theoretical_p2p_match_kwh'], 2)} kWh**
- P2P match / total load: **{float(p2p['theoretical_p2p_match_to_total_load_ratio']):.2%}**

![Average hourly P2P potential](p2p_potential.png)

The theoretical match is `min(total natural surplus, total natural deficit)` at each one-hour step. It ignores Battery, MT, bids, losses, and fees.

{_markdown_table(seasonal_view, ['season', 'hours', 'p2p_opportunity_ratio', 'theoretical_p2p_match_kwh', 'mean_theoretical_p2p_match_kw'])}

Seasonal totals are shown above; monthly values are retained in `summary.json`.

## Renewable curtailment

- No-P2P RES curtailment: **{_fmt(curtailment['no_p2p_res_curtailment_kwh'], 2)} kWh**
- Ideal-P2P RES curtailment: **{_fmt(curtailment['ideal_p2p_res_curtailment_kwh'], 2)} kWh**
- Potential reduction: **{_fmt(curtailment['potential_curtailment_reduction_kwh'], 2)} kWh**
- Potential reduction ratio: **{float(curtailment['potential_curtailment_reduction_ratio']):.2%}**

![Curtailment comparison](curtailment_comparison.png)

Ideal-P2P is an upper bound that prioritizes otherwise-curtailed renewable surplus before grid-exportable surplus. Dynamic rule-based bars are normalized per sampled day so they are comparable with full-train static averages.

## Rule-based comparison

Both cases use Battery idle, MT OFF, and midpoint bid action. The only difference is whether P2P clearing is enabled.

| Metric | No-P2P | P2P | Difference |
| --- | ---: | ---: | ---: |
| System total cost | {_fmt(rule_no['system_total_cost'], 2)} | {_fmt(rule_yes['system_total_cost'], 2)} | {_fmt(rule_yes['system_total_cost'] - rule_no['system_total_cost'], 2)} |
| P2P traded energy (kWh) | {_fmt(rule_no['total_p2p_traded_kwh'], 2)} | {_fmt(rule_yes['total_p2p_traded_kwh'], 2)} | {_fmt(rule_yes['total_p2p_traded_kwh'] - rule_no['total_p2p_traded_kwh'], 2)} |
| Grid import (kWh) | {_fmt(rule_no['total_grid_import_kwh'], 2)} | {_fmt(rule_yes['total_grid_import_kwh'], 2)} | {_fmt(rule_yes['total_grid_import_kwh'] - rule_no['total_grid_import_kwh'], 2)} |
| Grid export (kWh) | {_fmt(rule_no['total_grid_export_kwh'], 2)} | {_fmt(rule_yes['total_grid_export_kwh'], 2)} | {_fmt(rule_yes['total_grid_export_kwh'] - rule_no['total_grid_export_kwh'], 2)} |
| RES curtailment (kWh) | {_fmt(rule_no['total_res_curtailment_kwh'], 2)} | {_fmt(rule_yes['total_res_curtailment_kwh'], 2)} | {_fmt(rule_yes['total_res_curtailment_kwh'] - rule_no['total_res_curtailment_kwh'], 2)} |

Detailed per-MG accounting is available in `rule_based_statistics.csv`.

## Reward statistics

Reward remains exactly `-total_cost / 100`, without clipping or additional bonuses. Statistics below cover rule-based No-P2P, rule-based P2P, and legal random P2P policies.

{_markdown_table(reward_view, list(reward_view.columns))}

Raw cost-component means and totals, including fuel, O&M, startup, battery degradation, carbon, P2P, dump, and unmet-load terms, are retained in `reward_statistics.csv`.

## Final diagnostic conclusion

1. **P2P space:** {conclusions['p2p_space']}
2. **Role heterogeneity:** {conclusions['heterogeneity']}
3. **Renewable utilization:** {conclusions['curtailment']}
4. **Reward scale:** {conclusions['reward_scale']}
5. **Rule-based economics:** {conclusions['rule_based_cost']}

Taken together, the current engineering evidence supports proceeding to a bounded 100–300 episode Conditional MATD3 pilot. The pilot should monitor MG3 role concentration, replay filling, critic-loss stability, Q-scale EMA behavior, and per-MG returns. This is not a convergence claim. The script never changes environment parameters or reward scaling.

## Metric definitions

- **Energy totals:** hourly power summed over the selected rows and multiplied by the fixed one-hour time step.
- **Natural RES net power:** `PV + wind - load`, before Battery, MT, P2P, or grid settlement.
- **Seller/buyer hour ratios:** fractions of train hours with positive/negative natural RES net power.
- **Below-threshold ratio:** fraction of hours where absolute natural RES net power is below the configured 1 kW P2P order threshold.
- **Theoretical P2P match:** hourly minimum of system natural surplus and natural deficit; opportunity hours have a strictly positive match.
- **Static No-P2P curtailment:** per-MG natural surplus above its fixed 15 kW grid-export limit.
- **Ideal-P2P curtailment:** static upper bound after matched demand first consumes energy that would otherwise be curtailed.
- **Dynamic accounting:** sums the environment's executed grid, market, device, curtailment, unmet-load, and cost fields over sampled episodes.
- **Reward statistics:** per-step `-total_cost / 100` distribution using population standard deviation and empirical percentiles.

## Scope and reproducibility

- Static source: complete configured train split only.
- Dynamic sample: {summary['metadata']['dynamic_days']} evenly spaced valid train days per rule-based case.
- Random reward sample: {summary['metadata']['random_days']} evenly spaced valid train days.
- Seed: {summary['metadata']['seed']}.
- Time step: 1 hour; episode length: 24 hours.
- Generated files are deterministic for the same configuration, day counts, seed, and library versions.
"""
    (output_dir / "diagnostic_report.md").write_text(report, encoding="utf-8")


def run_diagnostic(
    config: str | Path | Mapping[str, Any] = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    dynamic_days: int = 30,
    random_days: int = 30,
    seed: int = 2026,
    create_plots: bool = True,
) -> Dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = MultiMicrogridEnv(config, split="train")
    static = build_static_timeseries(env)
    microgrid_stats = compute_microgrid_statistics(env, static)
    potential, p2p_summary, monthly, seasonal = compute_p2p_potential(env, static)
    potential, curtailment_summary = compute_curtailment_potential(env, static, potential)

    rule_days = select_evenly_spaced_days(env.data.valid_days, dynamic_days)
    random_sample_days = select_evenly_spaced_days(env.data.valid_days, random_days)
    rule_no, rewards_no = run_policy_case(
        config, rule_days, policy_name="rule_based_no_p2p", p2p_enabled=False,
        action_fn=rule_based_actions, seed=seed,
    )
    rule_yes, rewards_yes = run_policy_case(
        config, rule_days, policy_name="rule_based_p2p", p2p_enabled=True,
        action_fn=rule_based_actions, seed=seed,
    )
    random_case, rewards_random = run_policy_case(
        config, random_sample_days, policy_name="legal_random_p2p", p2p_enabled=True,
        action_fn=random_legal_actions, seed=seed,
    )
    cases = {
        "rule_based_no_p2p": rule_no,
        "rule_based_p2p": rule_yes,
        "legal_random_p2p": random_case,
    }
    reward_records = pd.concat([rewards_no, rewards_yes, rewards_random], ignore_index=True)
    reward_stats = compute_reward_statistics(reward_records)
    conclusions = _diagnostic_conclusions(
        microgrid_stats, p2p_summary, curtailment_summary, cases, reward_stats
    )

    summary: Dict[str, Any] = {
        "metadata": {
            "split": "train",
            "train_path": str(env.config["data"]["train_path"]),
            "train_start": str(static["timestamp"].min()),
            "train_end": str(static["timestamp"].max()),
            "train_hours": int(len(static)),
            "dynamic_days": int(len(rule_days)),
            "random_days": int(len(random_sample_days)),
            "seed": int(seed),
            "delta_t_hours": float(env.delta_t),
            "episode_length_hours": int(env.episode_length),
        },
        "microgrid_statistics": microgrid_stats.to_dict(orient="records"),
        "p2p_summary": p2p_summary,
        "monthly_p2p_potential": monthly.to_dict(orient="records"),
        "seasonal_p2p_potential": seasonal.to_dict(orient="records"),
        "curtailment_summary": curtailment_summary,
        "policy_cases": cases,
        "reward_statistics": reward_stats.to_dict(orient="records"),
        "chart_map": [
            {
                "segment": "Microgrid statistics",
                "question": "Do structural load and renewable profiles create distinct natural roles?",
                "family": "multi-series line",
                "fields": ["load_kw", "pv_kw", "wind_kw", "res_kw"],
                "file": "daily_profiles.png",
            },
            {
                "segment": "P2P potential",
                "question": "At which hours is natural supply-demand complementarity largest?",
                "family": "multi-series line",
                "fields": ["total_surplus_kw", "total_deficit_kw", "theoretical_p2p_match_kw"],
                "file": "p2p_potential.png",
            },
            {
                "segment": "Renewable curtailment",
                "question": "How much curtailment reduction is available statically and dynamically?",
                "family": "comparison bar",
                "fields": ["average_curtailment_kwh_per_day"],
                "file": "curtailment_comparison.png",
            },
        ],
        "diagnostic_conclusions": conclusions,
    }
    summary = _python_scalar(summary)

    microgrid_stats.to_csv(output / "microgrid_statistics.csv", index=False)
    potential.to_csv(output / "p2p_potential.csv", index=False)
    reward_stats.to_csv(output / "reward_statistics.csv", index=False)
    _rule_based_statistics_frame(cases).to_csv(output / "rule_based_statistics.csv", index=False)
    _summary_frame(summary).to_csv(output / "summary.csv", index=False)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

    if create_plots:
        generate_plots(output, env, static, potential, curtailment_summary, cases)
    generate_report(output, summary, microgrid_stats, reward_stats)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train-only pre-training diagnostics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dynamic-days", type=int, default=30)
    parser.add_argument("--random-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_diagnostic(
        args.config,
        args.output_dir,
        dynamic_days=args.dynamic_days,
        random_days=args.random_days,
        seed=args.seed,
        create_plots=True,
    )
    p2p = summary["p2p_summary"]
    curtailment = summary["curtailment_summary"]
    rule = summary["policy_cases"]["rule_based_p2p"]["system"]
    print(f"Output: {Path(args.output_dir).resolve()}")
    print(f"P2P opportunity ratio: {p2p['p2p_opportunity_ratio']:.2%}")
    print(f"Theoretical P2P match: {p2p['theoretical_p2p_match_kwh']:.2f} kWh")
    print(f"Ideal curtailment reduction: {curtailment['potential_curtailment_reduction_ratio']:.2%}")
    print(f"Rule-based P2P system cost: {rule['system_total_cost']:.2f}")


if __name__ == "__main__":
    main()
