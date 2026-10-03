"""Saved-result-only evidence and Markdown reporting for the H24 dueling study."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

DUELING = "matd3_mlp_dueling_critic_h24_q_guided"
H24 = "matd3_mlp_transformer_critic_h24_q_guided"
MATD3Q = "matd3_dispatch_q_guided"
LABELS = {DUELING: "Dueling", H24: "H24", MATD3Q: "MATD3Q"}
WINDOWS = ((0, 500), (2000, 2500), (4500, 5000))
ACTOR_METRICS = ("dispatch_action_grad_rms", "dispatch_saturation_rate", "dr_saturation_rate",
                 "critic_adv_share", "critic_dqda_dispatch_rms", "critic_dqda_dr_rms")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _mean(values):
    numbers = [number for value in values if (number := _finite(value)) is not None]
    return float(np.mean(numbers)) if numbers else None


def _fmt(value):
    number = _finite(value)
    return "—" if number is None else f"{number:.6g}"


def paired_deltas(dueling, comparator):
    """Align each seed and date; reject missing, duplicate or mismatched scenarios."""
    if set(dueling) != {42, 43, 44} or set(comparator) != set(dueling):
        raise ValueError("Paired evidence must contain seeds 42, 43 and 44")
    differences, shared_dates = [], None
    for seed in sorted(dueling):
        left = {row["date"]: row for row in dueling[seed]}
        right = {row["date"]: row for row in comparator[seed]}
        if len(left) != len(dueling[seed]) or len(right) != len(comparator[seed]):
            raise ValueError("Duplicate paired dates")
        if not left or set(left) != set(right) or (shared_dates is not None and set(left) != shared_dates):
            raise ValueError("Mismatched paired dates")
        shared_dates = set(left)
        values = []
        for day in sorted(left):
            for key in ("generalized_reference_scenario_fingerprint", "generalized_reference_initial_state_fingerprint"):
                if left[day].get(key) != right[day].get(key):
                    raise ValueError(f"Paired {key} mismatch at seed {seed}, {day}")
            a, b = _finite(left[day].get("algorithm_team_profit")), _finite(right[day].get("algorithm_team_profit"))
            if a is None or b is None:
                raise ValueError("Nonfinite paired profit")
            values.append(a - b)
        differences.append(values)
    return np.asarray(differences, dtype=float)


def paired_bootstrap(differences, *, repetitions=5000, seed=20261003):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires a finite nonempty seed-by-day matrix")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("Bootstrap repetitions must be a positive integer")
    rng = np.random.default_rng(seed)
    seeds, days = values.shape
    seed_indices = rng.integers(seeds, size=(repetitions, seeds))
    day_indices = rng.integers(days, size=(repetitions, seeds, days))
    means = values[seed_indices[:, :, None], day_indices].mean(axis=(1, 2))
    low, high = np.quantile(means, [.025, .975])
    return dict(mean=float(values.mean()), ci_low=float(low), ci_high=float(high),
                win_rate=float((values > 0).mean()), pairs=int(values.size),
                repetitions=repetitions, bootstrap_seed=seed,
                resampling="paired hierarchical: seeds, then days within each selected seed")


def judge_study(dueling, matd3q, profit_gap, lower_ci):
    numbers = [_finite(dueling.get(name)) for name in
               ("dispatch_action_grad_rms", "dispatch_saturation_rate", "critic_adv_share")]
    baseline = _finite(matd3q.get("dispatch_action_grad_rms"))
    if any(v is None for v in (*numbers, baseline, _finite(profit_gap), _finite(lower_ci))):
        return dict(mechanism=None, performance=None, conclusion="待完成")
    gradient, saturation, share = numbers
    mechanism = gradient >= baseline and saturation <= .35 and .02 <= share <= .8
    performance = profit_gap > 0 and lower_ci > 0
    conclusion = "继续投入" if mechanism and performance else "部分成立" if mechanism else "不成立"
    return dict(mechanism=mechanism, performance=performance, conclusion=conclusion)


def read_run(path):
    root = Path(path)
    daily, test = _csv(root / "test_daily_metrics.csv"), _json(root / "test_summary.json")
    history = _json(root / "training_history.json")
    diagnostics = _csv(root / "actor_diagnostics.csv")
    statistics = {
        "profit": _mean(row["algorithm_team_profit"] for row in daily),
        "idle_gain": _mean(row["gain_vs_idle_abs"] for row in daily),
        "strict_feasible_rate": test["strict_feasible_day_rate"],
        "generalized_cost": _mean(row["algorithm_generalized_system_cost"] for row in daily),
        "curtailment_kwh": _mean(row["renewable_curtailment_kwh"] for row in daily),
        "seconds_per_episode": _json(root / "training_summary.json")["seconds_per_episode_mean"],
    }
    for cost in ("grid", "peak", "curtail", "comfort", "ev"):
        statistics[f"cost_{cost}"] = test["algorithm"][f"cost_{cost}_mean"]
    windows = {}
    for low, high in WINDOWS:
        training = [r for r in history if low <= int(r["episode"]) <= high]
        actors = [r for r in diagnostics if low <= int(r["episode"]) <= high]
        values = {key: _mean(r.get(key) for r in training) for key in ("critic_loss", "td_error_std")}
        values.update({key: _mean(r.get(key + "_mean") for r in actors) for key in ACTOR_METRICS})
        windows[(low, high)] = values
    # Read weights for parameter counts only; never instantiate an environment
    # or policy, perform inference, or rerun a MILP while generating this report.
    import torch
    saved = torch.load(root / "checkpoints/best_model.pt", map_location="cpu", weights_only=True)
    critic_counts = {}
    for key, tensor in saved["critics"].items():
        if tensor.is_floating_point():  # The three critics' fixed indices are integer buffers.
            agent = key.split(".")[0]
            critic_counts[agent] = critic_counts.get(agent, 0) + tensor.numel()
    counts = dict(actor=history[-1].get("actor_parameter_counts"), critic_twin=critic_counts)
    prefix = "mg1.q1_network."
    if any(key.startswith(prefix + "advantage_") for key in saved["critics"]):
        counts["advantage_per_q"] = sum(t.numel() for key, t in saved["critics"].items()
                                          if key.startswith(prefix + "advantage_"))
        counts["value_per_q"] = critic_counts["mg1"] // 2 - counts["advantage_per_q"]
    return dict(path=str(root), daily=daily, statistics=statistics, windows=windows, parameters=counts)


def write_report(manifest, output):
    completed = [job for job in manifest["jobs"] if job.get("status") == "COMPLETED"]
    runs = {(job["algorithm"], job["seed"]): read_run(job["run_directory"]) for job in completed}
    lines = ["# Dueling H24 正式对照实验报告", "", f"已核验完成：{len(runs)}/9。",
             f"代码提交：`{manifest['git_commit']}`。环境哈希：`{manifest.get('environment_hash', '待核验')}`。",
             "三个算法全部从本次共享修复后的相同提交重新训练；历史 seed-42 结果未复用，保留原始产物。",
             "", "## 每 seed 结果", "", "| 算法 | seed | 测试利润 | 相对 idle 增益 | 严格可行日比例 | 广义成本 | 秒/episode |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for job in manifest["jobs"]:
        data = runs.get((job["algorithm"], job["seed"]))
        values = data["statistics"] if data else {}
        cells = [_fmt(values.get(k)) for k in ("profit", "idle_gain", "strict_feasible_rate", "generalized_cost", "seconds_per_episode")]
        lines.append(f"| {LABELS[job['algorithm']]} | {job['seed']} | " + " | ".join(cells) + " |")
    lines += ["", "## 3-seed 均值 ± 标准差", "", "| 算法 | 指标 | n seeds | 均值 | 样本标准差 |", "|---|---|---:|---:|---:|"]
    aggregated, diagnostics = {}, {}
    for slug in LABELS:
        cohort = [data for (algorithm, _), data in runs.items() if algorithm == slug]
        aggregated[slug] = {key: _mean(data["statistics"][key] for data in cohort)
                            for key in ("profit", "idle_gain", "strict_feasible_rate", "generalized_cost", "seconds_per_episode")}
        for key, average in aggregated[slug].items():
            std = float(np.std([data["statistics"][key] for data in cohort], ddof=1)) if len(cohort) > 1 else None
            lines.append(f"| {LABELS[slug]} | {key} | {len(cohort)} | {_fmt(average)} | {_fmt(std)} |")
        diagnostics[slug] = {window: {key: _mean(data["windows"][window][key] for data in cohort)
                                     for key in ("critic_loss", "td_error_std", *ACTOR_METRICS)} for window in WINDOWS}
    lines += ["", "## 配对利润差：Dueling − 对照", "", "按 seed 和日期对齐。5000 次分层配对 bootstrap：先重采样 3 个 seed，再在每个选中 seed 内重采样测试日期；每次保留成对利润差。固定随机种子 20261003。仅 3 个训练 seed，区间用于本研究的探索性判断。",
              "", "| 对照 | 配对数 | 差值均值 | 95% CI | 胜率 |", "|---|---:|---:|---|---:|"]
    pairs = {}
    if len(runs) == 9:
        dueling = {seed: runs[(DUELING, seed)]["daily"] for seed in (42, 43, 44)}
        for slug in (H24, MATD3Q):
            control = {seed: runs[(slug, seed)]["daily"] for seed in (42, 43, 44)}
            delta = paired_deltas(dueling, control)
            if delta.shape != (3, 55):
                raise ValueError("Formal report requires 55 days x 3 seeds")
            pairs[slug] = paired_bootstrap(delta)
            p = pairs[slug]
            lines.append(f"| {LABELS[slug]} | {p['pairs']} | {_fmt(p['mean'])} | [{_fmt(p['ci_low'])}, {_fmt(p['ci_high'])}] | {_fmt(p['win_rate'])} |")
    else:
        lines.append("| H24 / MATD3Q | — | 待完成 | 待完整 9 次实验 | — |")
    lines += ["", "## 过程诊断", "", "窗口端点均包含；历史 episode 从 1 开始，0–500 实际为 1–500。先对每个 seed 的有效 episode/agent 诊断取均值，再对 seed 取均值；缺测不补零。",
              "", "| 算法 | episode 窗口 | " + " | ".join(("critic_loss", "td_error_std", *ACTOR_METRICS)) + " |",
              "|---|---|" + "---:|" * (2 + len(ACTOR_METRICS))]
    for slug, windows in diagnostics.items():
        for window, values in windows.items():
            lines.append(f"| {LABELS[slug]} | {window[0]}–{window[1]} | " + " | ".join(_fmt(v) for v in values.values()) + " |")
    lines += ["", "## 测试成本分解（每日均值，三 seed 等权）", "", "| 算法 | grid | peak | curtail | comfort | EV | 弃电 kWh |", "|---|---:|---:|---:|---:|---:|---:|"]
    for slug in LABELS:
        cohort = [data for (algorithm, _), data in runs.items() if algorithm == slug]
        values = [_mean(data["statistics"][key] for data in cohort) for key in
                  ("cost_grid", "cost_peak", "cost_curtail", "cost_comfort", "cost_ev", "curtailment_kwh")]
        lines.append(f"| {LABELS[slug]} | " + " | ".join(_fmt(v) for v in values) + " |")
    decision = dict(conclusion="待完成", mechanism=None, performance=None)
    if len(runs) == 9:
        decision = judge_study(diagnostics[DUELING][WINDOWS[-1]], diagnostics[MATD3Q][WINDOWS[-1]],
                               pairs[H24]["mean"], pairs[H24]["ci_low"])
    lines += ["", "## 研究判定", "", f"研究判定：{decision['conclusion']}。",
              f"机制成立：{decision['mechanism']}；性能成立：{decision['performance']}。",
              "机制使用本批 MATD3Q 的同期 dispatch 梯度，另要求 Dueling 饱和率 ≤0.35、adv share ∈[0.02,0.8]。性能要求对 H24 利润均值更高且配对 CI 下界 >0。",
              "", "## 参数量与全部来源", ""]
    for job in completed:
        data = runs[(job["algorithm"], job["seed"])]
        lines += [f"- {LABELS[job['algorithm']]} seed {job['seed']}: `{data['path']}`；参数量 `{json.dumps(data['parameters'], ensure_ascii=False)}`。"]
    if len(runs) == 9:
        ratio = aggregated[DUELING]["seconds_per_episode"] / aggregated[H24]["seconds_per_episode"]
        lines += ["", f"正式全过程秒/episode 比 Dueling/H24：{ratio:.6g}；1.3 上限验收：{'通过' if ratio <= 1.3 else '未通过'}。"]
    lines += ["", "表中时间来自 training_summary，包含训练流水线内的验证/学习曲线开销。成本来自保存的 test_summary.algorithm；Actor 参数量来自训练历史，Critic 参数量来自最佳 checkpoint 的浮点权重（固定索引为整数 buffer）。没有重新执行模型、环境或 MILP。所有运行仍为 working results，未 promote。", ""]
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    evidence = dict(completed_runs=len(runs), paired_comparisons=pairs, decision=decision,
                    aggregate=aggregated, diagnostic_windows={slug: {f"{low}-{high}": value for (low, high), value in windows.items()}
                    for slug, windows in diagnostics.items()})
    Path(output).with_suffix(".json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return evidence
