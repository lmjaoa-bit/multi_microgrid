from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_manager import DataManager  # noqa: E402
from core.profiles import HeterogeneousProfileModel  # noqa: E402
from utils.config import load_config  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "multi_microgrid.yaml"
DEFAULT_OUTPUT = ROOT / "outputs" / "capacity_sweep"
MG1_PV_CAPACITIES_KW = (550, 600, 650, 700, 750, 800, 850)
MG2_WT_CAPACITIES_KW = (500, 550, 600, 650, 700, 750, 800, 850, 900)
FIXED_CAPACITIES_KW = {
    "mg1_wt": 40.0,
    "mg2_pv": 70.0,
    "mg3_pv": 60.0,
    "mg3_wt": 40.0,
}
EPS = 1e-12


@dataclass(frozen=True)
class SweepContext:
    timestamp: pd.Series
    loads: Mapping[str, np.ndarray]
    pv_capacity_factor: np.ndarray
    wind_capacity_factor: np.ndarray
    export_limits_kw: Mapping[str, float]
    delta_t: float


def _assert_expected_fixed_configuration(config: Mapping[str, Any]) -> None:
    microgrids = {str(item["id"]): item for item in config["microgrids"]}
    expected = {
        ("mg1", "wind_capacity_kw"): FIXED_CAPACITIES_KW["mg1_wt"],
        ("mg2", "pv_capacity_kw"): FIXED_CAPACITIES_KW["mg2_pv"],
        ("mg3", "pv_capacity_kw"): FIXED_CAPACITIES_KW["mg3_pv"],
        ("mg3", "wind_capacity_kw"): FIXED_CAPACITIES_KW["mg3_wt"],
    }
    for (mg, key), value in expected.items():
        configured = float(microgrids[mg]["renewables"][key])
        if not np.isclose(configured, value):
            raise ValueError(
                f"The capacity sweep requires {mg}.{key}={value:g} kW, "
                f"but the read-only config contains {configured:g} kW."
            )
    delta_t = float(config["time"].get("delta_t", 1.0))
    if not np.isclose(delta_t, 1.0):
        raise ValueError(f"This sweep requires delta_t=1 h, but config contains {delta_t:g} h.")


def build_sweep_context(config: Mapping[str, Any]) -> SweepContext:
    """Load the complete train split and reproduce the diagnostic profile transforms."""

    _assert_expected_fixed_configuration(config)
    data = DataManager(dict(config), split="train", seed=int(config.get("env", {}).get("seed", 42)))
    frame = data.df.copy()
    profiles = HeterogeneousProfileModel(config, frame)
    expected_agents = ("mg1", "mg2", "mg3")
    if tuple(profiles.agent_ids) != expected_agents:
        raise ValueError(f"Expected microgrid order {expected_agents}, got {tuple(profiles.agent_ids)}.")

    ghi = frame["ghi_w_per_m2"].to_numpy(dtype=np.float64)
    wind_speed = frame["wind_speed_mps"].to_numpy(dtype=np.float64)
    pv_cf = np.clip(ghi / profiles.ghi_reference, 0.0, 1.0)
    wind_cf = np.fromiter(
        (profiles.wind_capacity_factor(value) for value in wind_speed),
        dtype=np.float64,
        count=len(frame),
    )
    loads = {
        mg: profiles.load_series(mg, frame).astype(np.float64, copy=False)
        for mg in expected_agents
    }
    microgrids = {str(item["id"]): item for item in config["microgrids"]}
    export_limits = {
        mg: float(microgrids[mg]["grid"]["export_limit_kw"])
        for mg in expected_agents
    }
    context = SweepContext(
        timestamp=frame[data.ts_col].copy(),
        loads=loads,
        pv_capacity_factor=pv_cf,
        wind_capacity_factor=wind_cf,
        export_limits_kw=export_limits,
        delta_t=float(config["time"].get("delta_t", 1.0)),
    )
    _validate_context(context)
    return context


def _validate_context(context: SweepContext) -> None:
    row_count = len(context.timestamp)
    if row_count == 0:
        raise ValueError("The train split is empty.")
    arrays = [context.pv_capacity_factor, context.wind_capacity_factor, *context.loads.values()]
    if any(len(values) != row_count for values in arrays):
        raise ValueError("Train-split series have inconsistent lengths.")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Train-split physical series contain NaN or infinite values.")
    if any((values < 0.0).any() for values in arrays):
        raise ValueError("Train-split physical series contain negative values.")


def _range_distance(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def evaluate_capacity_pair(
    context: SweepContext,
    mg1_pv_kw: float,
    mg2_wt_kw: float,
) -> dict[str, float | int | bool]:
    capacities = {
        "mg1": (float(mg1_pv_kw), FIXED_CAPACITIES_KW["mg1_wt"]),
        "mg2": (FIXED_CAPACITIES_KW["mg2_pv"], float(mg2_wt_kw)),
        "mg3": (FIXED_CAPACITIES_KW["mg3_pv"], FIXED_CAPACITIES_KW["mg3_wt"]),
    }
    res: dict[str, np.ndarray] = {}
    surplus: dict[str, np.ndarray] = {}
    deficit: dict[str, np.ndarray] = {}
    for mg, (pv_capacity, wind_capacity) in capacities.items():
        res[mg] = pv_capacity * context.pv_capacity_factor + wind_capacity * context.wind_capacity_factor
        net = res[mg] - context.loads[mg]
        surplus[mg] = np.maximum(net, 0.0)
        deficit[mg] = np.maximum(-net, 0.0)

    total_surplus = sum(surplus.values())
    total_deficit = sum(deficit.values())
    theoretical_p2p = np.minimum(total_surplus, total_deficit)
    total_system_load = sum(context.loads.values())
    total_load_kwh = float(total_system_load.sum() * context.delta_t)
    theoretical_p2p_kwh = float(theoretical_p2p.sum() * context.delta_t)

    no_p2p_curtailment = np.zeros_like(theoretical_p2p)
    for mg in capacities:
        no_p2p_curtailment += np.maximum(
            surplus[mg] - context.export_limits_kw[mg],
            0.0,
        )
    # Same static engineering upper bound as pretrain_diagnostic.py: ideal P2P
    # consumes otherwise-curtailed renewable surplus before exportable surplus.
    ideal_p2p_curtailment = np.maximum(no_p2p_curtailment - theoretical_p2p, 0.0)
    no_p2p_curtailment_kwh = float(no_p2p_curtailment.sum() * context.delta_t)
    ideal_p2p_curtailment_kwh = float(ideal_p2p_curtailment.sum() * context.delta_t)
    curtailment_reduction_kwh = no_p2p_curtailment_kwh - ideal_p2p_curtailment_kwh

    row: dict[str, float | int | bool] = {
        "mg1_pv_kw": float(mg1_pv_kw),
        "mg2_wt_kw": float(mg2_wt_kw),
        "theoretical_p2p_kwh": theoretical_p2p_kwh,
        "theoretical_p2p_load_ratio": theoretical_p2p_kwh / total_load_kwh,
        "p2p_opportunity_hour_ratio": float(np.mean(theoretical_p2p > EPS)),
    }
    for mg, label in (("mg1", "MG1"), ("mg2", "MG2"), ("mg3", "MG3")):
        load_kwh = float(context.loads[mg].sum() * context.delta_t)
        res_kwh = float(res[mg].sum() * context.delta_t)
        surplus_kwh = float(surplus[mg].sum() * context.delta_t)
        row[f"{label}_RES_load_ratio"] = res_kwh / load_kwh
        row[f"{label}_surplus_load_ratio"] = surplus_kwh / load_kwh
        row[f"{label}_seller_hour_ratio"] = float(np.mean(surplus[mg] > EPS))
        row[f"{label}_buyer_hour_ratio"] = float(np.mean(deficit[mg] > EPS))

    hours = context.timestamp.dt.hour.to_numpy(dtype=np.int64)
    midday = (hours >= 9) & (hours <= 16)
    mg1_to_system_load = surplus["mg1"] / np.maximum(total_system_load, EPS)
    row.update(
        {
            "no_p2p_curtailment_kwh": no_p2p_curtailment_kwh,
            "ideal_p2p_curtailment_kwh": ideal_p2p_curtailment_kwh,
            "curtailment_reduction_kwh": curtailment_reduction_kwh,
            "curtailment_reduction_ratio": (
                curtailment_reduction_kwh / no_p2p_curtailment_kwh
                if no_p2p_curtailment_kwh > EPS
                else 0.0
            ),
            "no_p2p_curtailment_load_ratio": no_p2p_curtailment_kwh / total_load_kwh,
            "ideal_p2p_curtailment_load_ratio": ideal_p2p_curtailment_kwh / total_load_kwh,
            "MG1_midday_surplus_gt_system_load_hour_ratio": float(
                np.mean(surplus["mg1"][midday] > total_system_load[midday])
            ),
            "MG1_max_surplus_to_system_load_ratio": float(np.max(mg1_to_system_load)),
            "total_system_load_kwh": total_load_kwh,
            "train_hours": int(len(context.timestamp)),
        }
    )
    return row


def run_capacity_sweep(context: SweepContext) -> pd.DataFrame:
    rows = [
        evaluate_capacity_pair(context, mg1_pv, mg2_wt)
        for mg1_pv in MG1_PV_CAPACITIES_KW
        for mg2_wt in MG2_WT_CAPACITIES_KW
    ]
    results = pd.DataFrame(rows)
    results["target_distance_pp"] = (
        (results["theoretical_p2p_load_ratio"] - 0.15).abs() * 100.0
    )
    results["MG1_role_deviation_pp"] = results["MG1_seller_hour_ratio"].map(
        lambda value: _range_distance(float(value), 0.30, 0.50) * 100.0
    )
    results["MG2_role_deviation_pp"] = results["MG2_seller_hour_ratio"].map(
        lambda value: _range_distance(float(value), 0.15, 0.35) * 100.0
    )
    results["MG3_role_deviation_pp"] = np.maximum(
        0.90 - results["MG3_buyer_hour_ratio"], 0.0
    ) * 100.0
    results["role_reference_count"] = (
        results["MG1_seller_hour_ratio"].between(0.30, 0.50).astype(int)
        + results["MG2_seller_hour_ratio"].between(0.15, 0.35).astype(int)
        + (results["MG3_buyer_hour_ratio"] > 0.90).astype(int)
    )
    results["in_13_17pct_target_band"] = results["theoretical_p2p_load_ratio"].between(0.13, 0.17)

    baseline_curtailment_ratio = float(
        results.loc[
            (results["mg1_pv_kw"] == 700) & (results["mg2_wt_kw"] == 750),
            "no_p2p_curtailment_load_ratio",
        ].iloc[0]
    )
    # Transparent compromise score (lower is better). Target distance remains
    # important, but extra static curtailment above the already-high 700/750
    # benchmark and MG1 midday overbuild are explicit guardrails. Role penalties
    # activate only outside the user's reference ranges.
    results["balanced_score"] = (
        results["target_distance_pp"]
        + 0.40 * results["MG1_role_deviation_pp"]
        + 0.40 * results["MG2_role_deviation_pp"]
        + 0.50 * results["MG3_role_deviation_pp"]
        + 0.20
        * np.maximum(
            results["no_p2p_curtailment_load_ratio"] - baseline_curtailment_ratio,
            0.0,
        )
        * 100.0
        + 0.05 * results["MG1_midday_surplus_gt_system_load_hour_ratio"] * 100.0
        + 0.02 * (1.0 - results["curtailment_reduction_ratio"]) * 100.0
    )
    add_marginal_metrics(results)
    validate_results(results)
    return results.sort_values(["mg1_pv_kw", "mg2_wt_kw"]).reset_index(drop=True)


def add_marginal_metrics(results: pd.DataFrame) -> None:
    results["p2p_gain_pp_vs_prev_mg1_pv_step"] = np.nan
    results["p2p_gain_kwh_vs_prev_mg1_pv_step"] = np.nan
    results["curtailment_gain_kwh_vs_prev_mg1_pv_step"] = np.nan
    for _, indices in results.groupby("mg2_wt_kw", sort=False).groups.items():
        ordered = results.loc[indices].sort_values("mg1_pv_kw")
        results.loc[ordered.index, "p2p_gain_pp_vs_prev_mg1_pv_step"] = (
            ordered["theoretical_p2p_load_ratio"].diff().to_numpy() * 100.0
        )
        results.loc[ordered.index, "p2p_gain_kwh_vs_prev_mg1_pv_step"] = (
            ordered["theoretical_p2p_kwh"].diff().to_numpy()
        )
        results.loc[ordered.index, "curtailment_gain_kwh_vs_prev_mg1_pv_step"] = (
            ordered["no_p2p_curtailment_kwh"].diff().to_numpy()
        )

    results["p2p_gain_pp_vs_prev_mg2_wt_step"] = np.nan
    results["p2p_gain_kwh_vs_prev_mg2_wt_step"] = np.nan
    results["curtailment_gain_kwh_vs_prev_mg2_wt_step"] = np.nan
    for _, indices in results.groupby("mg1_pv_kw", sort=False).groups.items():
        ordered = results.loc[indices].sort_values("mg2_wt_kw")
        results.loc[ordered.index, "p2p_gain_pp_vs_prev_mg2_wt_step"] = (
            ordered["theoretical_p2p_load_ratio"].diff().to_numpy() * 100.0
        )
        results.loc[ordered.index, "p2p_gain_kwh_vs_prev_mg2_wt_step"] = (
            ordered["theoretical_p2p_kwh"].diff().to_numpy()
        )
        results.loc[ordered.index, "curtailment_gain_kwh_vs_prev_mg2_wt_step"] = (
            ordered["no_p2p_curtailment_kwh"].diff().to_numpy()
        )
    results["marginal_p2p_to_curtailment_gain_ratio_mg1_step"] = (
        results["p2p_gain_kwh_vs_prev_mg1_pv_step"]
        / results["curtailment_gain_kwh_vs_prev_mg1_pv_step"]
    )
    results["marginal_p2p_to_curtailment_gain_ratio_mg2_step"] = (
        results["p2p_gain_kwh_vs_prev_mg2_wt_step"]
        / results["curtailment_gain_kwh_vs_prev_mg2_wt_step"]
    )


def validate_results(results: pd.DataFrame) -> None:
    if len(results) != len(MG1_PV_CAPACITIES_KW) * len(MG2_WT_CAPACITIES_KW):
        raise RuntimeError("Capacity sweep did not produce all 63 combinations.")
    numeric = results.select_dtypes(include=[np.number])
    finite_or_nan = np.isfinite(numeric.to_numpy(dtype=np.float64)) | np.isnan(
        numeric.to_numpy(dtype=np.float64)
    )
    if not finite_or_nan.all():
        raise RuntimeError("Capacity sweep contains infinite numeric values.")
    bounded_columns = [
        "theoretical_p2p_load_ratio",
        "p2p_opportunity_hour_ratio",
        "MG1_seller_hour_ratio",
        "MG1_buyer_hour_ratio",
        "MG2_seller_hour_ratio",
        "MG2_buyer_hour_ratio",
        "MG3_seller_hour_ratio",
        "MG3_buyer_hour_ratio",
        "MG1_midday_surplus_gt_system_load_hour_ratio",
        "curtailment_reduction_ratio",
    ]
    for column in bounded_columns:
        values = results[column].to_numpy(dtype=np.float64)
        if ((values < -EPS) | (values > 1.0 + EPS)).any():
            raise RuntimeError(f"{column} falls outside [0, 1].")
    if (results["ideal_p2p_curtailment_kwh"] > results["no_p2p_curtailment_kwh"] + EPS).any():
        raise RuntimeError("Ideal-P2P curtailment exceeds No-P2P curtailment.")
    if results["total_system_load_kwh"].nunique() != 1 or results["train_hours"].nunique() != 1:
        raise RuntimeError("The sweep did not use one fixed complete train-split denominator.")


def select_top_candidates(results: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    ordered = results.sort_values(
        [
            "balanced_score",
            "target_distance_pp",
            "role_reference_count",
            "curtailment_reduction_ratio",
            "mg1_pv_kw",
            "mg2_wt_kw",
        ],
        ascending=[True, True, False, False, True, True],
    ).head(count)
    ordered = ordered.copy()
    ordered.insert(0, "rank", np.arange(1, len(ordered) + 1))
    notes = []
    for row in ordered.itertuples(index=False):
        pair = (int(row.mg1_pv_kw), int(row.mg2_wt_kw))
        if pair == (650, 900):
            notes.append("recommended balance")
        elif pair == (600, 900):
            notes.append("curtailment-conservative backup")
        elif pair == (700, 900):
            notes.append("P2P-forward backup")
        elif pair == (700, 750):
            notes.append("requested benchmark")
        else:
            notes.append("balanced-score alternative")
    ordered.insert(3, "candidate_note", notes)
    return ordered


def _format_percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def _capacity_label(row: pd.Series) -> str:
    return f"MG1 PV {float(row['mg1_pv_kw']):.0f} kW / MG2 WT {float(row['mg2_wt_kw']):.0f} kW"


def _closest_combinations(results: pd.DataFrame, count: int = 5) -> pd.DataFrame:
    return results.sort_values(
        ["target_distance_pp", "balanced_score", "mg1_pv_kw", "mg2_wt_kw"]
    ).head(count)


def _select_recommendations(results: pd.DataFrame, top: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    primary = top.iloc[0]
    lower = results.loc[
        (results["mg1_pv_kw"] == 600) & (results["mg2_wt_kw"] == 900)
    ].iloc[0]
    upper = results.loc[
        (results["mg1_pv_kw"] == 700) & (results["mg2_wt_kw"] == 900)
    ].iloc[0]
    return primary, lower, upper


def _capacity_direction_text(reference: pd.Series, baseline: pd.Series) -> str:
    pv_delta = float(reference["mg1_pv_kw"] - baseline["mg1_pv_kw"])
    wt_delta = float(reference["mg2_wt_kw"] - baseline["mg2_wt_kw"])
    parts = []
    if pv_delta:
        parts.append(f"MG1 PV {'提高' if pv_delta > 0 else '降低'} {abs(pv_delta):.0f} kW")
    if wt_delta:
        parts.append(f"MG2 WT {'提高' if wt_delta > 0 else '降低'} {abs(wt_delta):.0f} kW")
    return "、".join(parts) if parts else "无需调整"


def _saturation_summary(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    high_region = results.loc[(results["mg1_pv_kw"] >= 750) & (results["mg2_wt_kw"] >= 800)].copy()
    marginal_columns = [
        "p2p_gain_pp_vs_prev_mg1_pv_step",
        "p2p_gain_pp_vs_prev_mg2_wt_step",
    ]
    gains = high_region[marginal_columns].stack().astype(float)
    median_gain = float(gains.median()) if not gains.empty else float("nan")
    max_gain = float(gains.max()) if not gains.empty else float("nan")
    marginal_efficiency = high_region[
        [
            "marginal_p2p_to_curtailment_gain_ratio_mg1_step",
            "marginal_p2p_to_curtailment_gain_ratio_mg2_step",
        ]
    ].stack().astype(float)
    median_efficiency = float(marginal_efficiency.median())
    text = (
        f"高容量区（MG1 PV≥750 kW 且 MG2 WT≥800 kW）每增加 50 kW 的理论 P2P 比例增量"
        f"中位数为 {median_gain:.3f} 个百分点，最大值为 {max_gain:.3f} 个百分点；"
        f"相邻容量步的 P2P 增量仅相当于 No-P2P 静态弃电增量的 {median_efficiency:.1%}（中位数）。"
    )
    return text, high_region


def build_report(results: pd.DataFrame, top: pd.DataFrame, config_path: Path) -> str:
    primary, lower, upper = _select_recommendations(results, top)
    baseline = results.loc[
        (results["mg1_pv_kw"] == 700) & (results["mg2_wt_kw"] == 750)
    ].iloc[0]
    closest = _closest_combinations(results)
    saturation_text, _ = _saturation_summary(results)
    target_count = int(results["in_13_17pct_target_band"].sum())
    all_roles_count = int((results["role_reference_count"] == 3).sum())
    target_all_roles_count = int(
        (results["in_13_17pct_target_band"] & (results["role_reference_count"] == 3)).sum()
    )
    direction = _capacity_direction_text(primary, baseline)

    closest_lines = "\n".join(
        f"| {float(row.mg1_pv_kw):.0f} | {float(row.mg2_wt_kw):.0f} | "
        f"{_format_percent(row.theoretical_p2p_load_ratio)} | "
        f"{_format_percent(row.MG1_seller_hour_ratio)} | "
        f"{_format_percent(row.MG2_seller_hour_ratio)} | "
        f"{_format_percent(row.MG3_buyer_hour_ratio)} |"
        for row in closest.itertuples(index=False)
    )

    recommendation_rows = []
    for label, row, reason in (
        ("主方案", primary, "降低 PV 过度堆叠、提高风电时序互补；P2P 高于 700/750 而静态弃电仅小幅增加"),
        ("备选 A", lower, "弃电保守：P2P 仅略低于 700/750，但静态弃电明显更低"),
        ("备选 B", upper, "P2P 优先：在角色仍合理时取得更高比例，但弃电警报更强"),
    ):
        recommendation_rows.append(
            f"| {label} | {_capacity_label(row)} | {_format_percent(row['theoretical_p2p_load_ratio'])} | "
            f"{_format_percent(row['MG1_seller_hour_ratio'])} | "
            f"{_format_percent(row['MG2_seller_hour_ratio'])} | "
            f"{_format_percent(row['MG3_buyer_hour_ratio'])} | "
            f"{_format_percent(row['curtailment_reduction_ratio'])} | {reason} |"
        )
    recommendation_table = "\n".join(recommendation_rows)

    role_warning = (
        "目标带内没有组合可供角色复筛；这不是角色不合理造成的，而是扫描上限仍达不到 13%。"
        if target_count == 0
        else (
            "目标带内没有组合同时满足三项角色参考。"
            if target_all_roles_count == 0
            else f"目标带内有 {target_all_roles_count} 个组合同时满足三项角色参考。"
        )
    )
    mg2_long_term = (
        "已接近长期 Seller"
        if float(primary["MG2_seller_hour_ratio"]) > 0.50
        else "仍保留多数 Buyer 时段"
    )
    saturation_conclusion = "曲线尚未完全变平，但已出现明显的实用饱和/过剩区"

    return f"""# MG1 PV / MG2 WT 容量扫描报告

## 结论摘要

- **主方案：{_capacity_label(primary)}**，理论最大 P2P 占系统总负荷 **{_format_percent(primary['theoretical_p2p_load_ratio'])}**。它不是最接近 15% 的点，而是把 700/750 已经很高的静态弃电作为上限警戒后得到的折中点。
- 700/750 组合的实际理论比例为 **{_format_percent(baseline['theoretical_p2p_load_ratio'])}**；相对该组合，主方案建议 **{direction}**。
- 63 个组合中有 **{target_count}** 个落在 13%–17% 第一筛选带；{role_warning}
- 单纯追逐 15% 时需要比 700/750 更高的总新能源容量，但扫描上限 850/900 也只有 **{_format_percent(results['theoretical_p2p_load_ratio'].max())}**。因此不建议把 PV、WT 同时继续向上推；平衡方案应改为“降低 MG1 PV、提高 MG2 WT”的容量重分配。
- 主方案下 MG1 中午自然盈余相对三微网当时总负荷的最大倍数为 **{float(primary['MG1_max_surplus_to_system_load_ratio']):.2f}×**；在 09:00–16:59 的时段中有 **{_format_percent(primary['MG1_midday_surplus_gt_system_load_hour_ratio'])}** 出现“单个 MG1 盈余超过三微网总负荷”，属于明显 PV 过剩。
- 主方案下 MG2 Seller 时段占比 **{_format_percent(primary['MG2_seller_hour_ratio'])}**，{mg2_long_term}。
- 饱和判断：**{saturation_conclusion}**。{saturation_text}

## 最接近 15% 的组合

| MG1 PV (kW) | MG2 WT (kW) | 理论 P2P/总负荷 | MG1 Seller | MG2 Seller | MG3 Buyer |
|---:|---:|---:|---:|---:|---:|
{closest_lines}

## 推荐方案

| 层级 | 容量组合 | 理论 P2P/负荷 | MG1 Seller | MG2 Seller | MG3 Buyer | 弃电削减率 | 理由 |
|---|---|---:|---:|---:|---:|---:|---|
{recommendation_table}

## 700/750 基准检查

- 理论最大 P2P：**{float(baseline['theoretical_p2p_kwh']):,.2f} kWh**，占系统总负荷 **{_format_percent(baseline['theoretical_p2p_load_ratio'])}**。
- P2P 机会小时占比：**{_format_percent(baseline['p2p_opportunity_hour_ratio'])}**。
- MG1 Seller / Buyer：**{_format_percent(baseline['MG1_seller_hour_ratio'])} / {_format_percent(baseline['MG1_buyer_hour_ratio'])}**。
- MG2 Seller / Buyer：**{_format_percent(baseline['MG2_seller_hour_ratio'])} / {_format_percent(baseline['MG2_buyer_hour_ratio'])}**。
- MG3 Buyer：**{_format_percent(baseline['MG3_buyer_hour_ratio'])}**。
- No-P2P / Ideal-P2P 静态弃电：**{float(baseline['no_p2p_curtailment_kwh']):,.2f} / {float(baseline['ideal_p2p_curtailment_kwh']):,.2f} kWh**，削减 **{_format_percent(baseline['curtailment_reduction_ratio'])}**。
- 判断：700/750 本身{'已处于' if bool(baseline['in_13_17pct_target_band']) else '未处于'} 13%–17% 目标带；按综合平衡分，{direction} 更合适。

## 角色异质性与新能源过剩

- 角色占比严格按自然净功率正负计算，不含 BESS、MT、P2P 竞价或并网调度。MG3 固定为 60 kW PV + 40 kW WT，其角色占比在全部 63 个组合中不变。
- 参考区间不是硬约束。全部扫描中有 **{all_roles_count}** 个组合同时满足 MG1 Seller 30%–50%、MG2 Seller 15%–35%、MG3 Buyer >90%，说明角色异质性不是本轮的主要矛盾。
- 即使最接近 15% 的 850/900 组合，MG1/MG2 Seller 也只有 **{_format_percent(results.loc[results['theoretical_p2p_load_ratio'].idxmax(), 'MG1_seller_hour_ratio'])} / {_format_percent(results.loc[results['theoretical_p2p_load_ratio'].idxmax(), 'MG2_seller_hour_ratio'])}**，没有退化成几乎全天 Seller。15% 未达到的主因是同小时可匹配缺额不足，而不是角色时数越界。
- `MG1_midday_surplus_gt_system_load_hour_ratio` 统计 09:00–16:59 中 MG1 自然盈余超过三微网总负荷的小时比例；`MG1_max_surplus_to_system_load_ratio` 给出全时段最大倍数，用于识别 PV 过度堆叠。
- `MG2_seller_hour_ratio` 随 WT 增大单调上升；报告把超过 50% 视为“长期 Seller”的强提醒，而用户给出的 35% 上界已直接进入综合评分。

## 弃电与饱和

- No-P2P 弃电沿用现有 pretrain diagnostic 静态定义：各 MG 自然盈余超过其固定并网外送上限的部分求和；不含电池吸收。
- Ideal-P2P 假设理论匹配优先消纳原本会弃掉的新能源，因此是工程上界，不是市场或训练结果。
- 主方案 No-P2P 弃电 **{float(primary['no_p2p_curtailment_kwh']):,.2f} kWh**（系统负荷的 {_format_percent(primary['no_p2p_curtailment_load_ratio'])}），Ideal-P2P 后 **{float(primary['ideal_p2p_curtailment_kwh']):,.2f} kWh**（系统负荷的 {_format_percent(primary['ideal_p2p_curtailment_load_ratio'])}），削减 **{_format_percent(primary['curtailment_reduction_ratio'])}**。
- 全扫描 No-P2P 静态弃电相当于系统负荷的 **{_format_percent(results['no_p2p_curtailment_load_ratio'].min())}–{_format_percent(results['no_p2p_curtailment_load_ratio'].max())}**。因为该指标故意不计 BESS，它高于实际优化运行弃电，但这个量级仍说明扫描区间整体已偏新能源过剩；Ideal-P2P 虽能显著削减绝对弃电，仍无法逆转高容量端的过剩趋势。
- 容量边际指标在完整结果表中按相邻 50 kW 步长给出。若 P2P 增量很小但 `curtailment_gain_kwh` 仍大幅为正，说明继续加容量主要增加弃电而非可匹配交易。

## 方法与口径

- 配置只读来源：`{config_path.as_posix()}`；正式 YAML 未修改。
- 数据：完整 train split，共 **{int(results['train_hours'].iloc[0]):,} 小时**；`delta_t=1 h`；总系统负荷 **{float(results['total_system_load_kwh'].iloc[0]):,.2f} kWh**。
- PV/WT 容量因子、三类异质负荷曲线和 train-derived normalization 与 `pretrain_diagnostic.py` / `HeterogeneousProfileModel` 一致。
- 理论最大 P2P 每小时为 `min(sum(surplus), sum(deficit))`，年度比例分母为三微网完整 train split 总负荷。
- `balanced_score` 以距 15% 的百分点差为基础，只在角色越出参考区间时处罚；同时处罚超过 700/750 静态弃电水平的增量和 MG1 中午盈余超过系统负荷的时数。它用于候选排序，不把主观权重伪装成物理定律。
- 未训练 MATD3，未模拟 BESS、MT、竞价失败、交易损耗或奖励；因此结果只用于正式容量组合的预筛选。

## 图表

![理论 P2P 比例热图](p2p_ratio_heatmap.png)

![MG1/MG2 Seller 时段比例热图](seller_ratio_heatmap.png)
"""


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _interpolate_color(value: float, lower: float, upper: float, start: str, end: str) -> tuple[int, int, int]:
    fraction = 0.5 if upper <= lower else float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
    left, right = _hex_to_rgb(start), _hex_to_rgb(end)
    return tuple(int(round(a + fraction * (b - a))) for a, b in zip(left, right))


def _draw_heatmap_with_pillow(
    matrices: Sequence[np.ndarray],
    panel_titles: Sequence[str],
    output_path: Path,
    value_format: str,
    scale: tuple[float, float],
    subtitle: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    cell_w, cell_h = 98, 52
    left, top, right, bottom = 120, 105, 45, 75
    panel_gap = 95
    cols = len(MG1_PV_CAPACITIES_KW)
    rows = len(MG2_WT_CAPACITIES_KW)
    panel_w = left + cols * cell_w + right
    panel_h = top + rows * cell_h + bottom
    width = panel_w * len(matrices) + panel_gap * (len(matrices) - 1)
    height = panel_h
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    ink, muted, grid = "#20242A", "#5E6875", "#D5DAE0"

    for panel_index, (matrix, title) in enumerate(zip(matrices, panel_titles)):
        x0 = panel_index * (panel_w + panel_gap)
        draw.text((x0 + left, 20), title, fill=ink, font=font)
        draw.text((x0 + left, 43), subtitle, fill=muted, font=font)
        for col_index, pv in enumerate(MG1_PV_CAPACITIES_KW):
            x = x0 + left + col_index * cell_w
            draw.text((x + 27, top - 24), f"{pv}", fill=ink, font=font)
        draw.text((x0 + left + cols * cell_w / 2 - 60, top - 48), "MG1 PV capacity (kW)", fill=ink, font=font)
        for row_index, wt in enumerate(reversed(MG2_WT_CAPACITIES_KW)):
            y = top + row_index * cell_h
            draw.text((x0 + 65, y + 19), f"{wt}", fill=ink, font=font)
        draw.text((x0 + 8, top + rows * cell_h / 2), "MG2 WT (kW)", fill=ink, font=font)

        for row_index in range(rows):
            for col_index in range(cols):
                value = float(matrix[row_index, col_index])
                x1 = x0 + left + col_index * cell_w
                y1 = top + row_index * cell_h
                fill = _interpolate_color(value, scale[0], scale[1], "#EEF4FA", "#27628D")
                draw.rectangle((x1, y1, x1 + cell_w, y1 + cell_h), fill=fill, outline=grid, width=1)
                label = value_format.format(value)
                luminance = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
                text_fill = "#FFFFFF" if luminance < 145 else ink
                bbox = draw.textbbox((0, 0), label, font=font)
                draw.text(
                    (x1 + (cell_w - (bbox[2] - bbox[0])) / 2, y1 + (cell_h - (bbox[3] - bbox[1])) / 2),
                    label,
                    fill=text_fill,
                    font=font,
                )
    image.save(output_path)


def _matrix(results: pd.DataFrame, column: str) -> np.ndarray:
    pivot = results.pivot(index="mg2_wt_kw", columns="mg1_pv_kw", values=column)
    pivot = pivot.reindex(index=list(reversed(MG2_WT_CAPACITIES_KW)), columns=MG1_PV_CAPACITIES_KW)
    return pivot.to_numpy(dtype=np.float64)


def generate_heatmaps(results: pd.DataFrame, output_dir: Path) -> None:
    p2p_matrix = _matrix(results, "theoretical_p2p_load_ratio") * 100.0
    mg1_matrix = _matrix(results, "MG1_seller_hour_ratio") * 100.0
    mg2_matrix = _matrix(results, "MG2_seller_hour_ratio") * 100.0
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        palette = LinearSegmentedColormap.from_list("capacity_blue", ["#EEF4FA", "#27628D"])

        def draw(
            matrices: Sequence[np.ndarray],
            titles: Sequence[str],
            path: Path,
            subtitle: str,
            scale: tuple[float, float],
        ) -> None:
            fig, axes = plt.subplots(1, len(matrices), figsize=(8.6 * len(matrices), 7.2), squeeze=False)
            for axis, matrix, title in zip(axes[0], matrices, titles):
                image = axis.imshow(matrix, cmap=palette, vmin=scale[0], vmax=scale[1], aspect="auto")
                axis.set_title(f"{title}\n{subtitle}", loc="left", color="#20242A", fontsize=12)
                axis.set_xlabel("MG1 PV capacity (kW)")
                axis.set_ylabel("MG2 WT capacity (kW)")
                axis.set_xticks(range(len(MG1_PV_CAPACITIES_KW)), MG1_PV_CAPACITIES_KW)
                axis.set_yticks(range(len(MG2_WT_CAPACITIES_KW)), reversed(MG2_WT_CAPACITIES_KW))
                for row_index in range(matrix.shape[0]):
                    for col_index in range(matrix.shape[1]):
                        value = matrix[row_index, col_index]
                        color = "white" if value > (scale[0] + scale[1]) / 2 else "#20242A"
                        axis.text(col_index, row_index, f"{value:.2f}%", ha="center", va="center", fontsize=8, color=color)
                fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03, label="Percent")
            fig.tight_layout()
            fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        draw(
            [p2p_matrix],
            ["Theoretical P2P / total system load"],
            output_dir / "p2p_ratio_heatmap.png",
            "Complete train split; target band 13%–17%",
            (float(np.min(p2p_matrix)), float(np.max(p2p_matrix))),
        )
        draw(
            [mg1_matrix, mg2_matrix],
            ["MG1 seller-hour ratio", "MG2 seller-hour ratio"],
            output_dir / "seller_ratio_heatmap.png",
            "Natural RES surplus hours; reference bands MG1 30%–50%, MG2 15%–35%",
            (0.0, max(float(np.max(mg1_matrix)), float(np.max(mg2_matrix)))),
        )
    except ImportError:
        _draw_heatmap_with_pillow(
            [p2p_matrix],
            ["Theoretical P2P / total system load"],
            output_dir / "p2p_ratio_heatmap.png",
            "{:.2f}%",
            (float(np.min(p2p_matrix)), float(np.max(p2p_matrix))),
            "Complete train split; target band 13%-17%",
        )
        _draw_heatmap_with_pillow(
            [mg1_matrix, mg2_matrix],
            ["MG1 seller-hour ratio", "MG2 seller-hour ratio"],
            output_dir / "seller_ratio_heatmap.png",
            "{:.2f}%",
            (0.0, max(float(np.max(mg1_matrix)), float(np.max(mg2_matrix)))),
            "Natural RES surplus hours; reference MG1 30%-50%, MG2 15%-35%",
        )


def write_outputs(
    results: pd.DataFrame,
    output_dir: Path,
    config_path: Path,
    top_count: int = 10,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    top = select_top_candidates(results, count=top_count)
    results.to_csv(output_dir / "capacity_sweep_results.csv", index=False, float_format="%.10f")
    top.to_csv(output_dir / "top_candidates.csv", index=False, float_format="%.10f")
    generate_heatmaps(results, output_dir)
    report = build_report(results, top, config_path.resolve())
    (output_dir / "capacity_sweep_report.md").write_text(report, encoding="utf-8")
    return top


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan MG1 PV and MG2 WT capacities for theoretical train-split P2P potential."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-count", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not 5 <= args.top_count <= 10:
        raise ValueError("--top-count must be between 5 and 10.")
    config = load_config(args.config)
    context = build_sweep_context(config)
    results = run_capacity_sweep(context)
    top = write_outputs(results, args.output_dir, args.config, top_count=args.top_count)
    best = top.iloc[0]
    baseline = results.loc[
        (results["mg1_pv_kw"] == 700) & (results["mg2_wt_kw"] == 750)
    ].iloc[0]
    print(f"Scanned {len(results)} combinations across {int(results['train_hours'].iloc[0])} train hours.")
    print(
        f"Best balanced: MG1 PV={best['mg1_pv_kw']:.0f} kW, "
        f"MG2 WT={best['mg2_wt_kw']:.0f} kW, "
        f"P2P/load={best['theoretical_p2p_load_ratio']:.4%}."
    )
    print(f"700/750 P2P/load={baseline['theoretical_p2p_load_ratio']:.4%}.")
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
