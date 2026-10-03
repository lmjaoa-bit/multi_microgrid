"""Read-only, realized-return audit of the two supported Q-guided algorithms.

This is a diagnostic adapter, not the shared training/evaluation pipeline. It
deliberately never calls update/save/replay.add and never invokes a MILP solver.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path, PureWindowsPath
import time
from typing import Any

import numpy as np
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv

SLUGS = ("matd3_dispatch_q_guided", "matd3_hierarchical_transformer_q_readout")
NAMES = {SLUGS[0]: "MATD3-Q", SLUGS[1]: "T-Q（设备预测读取）"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, set):
        return sorted(clean(v) for v in value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def fingerprint(value) -> str:
    encoded = json.dumps(clean(value), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass
class SourceRun:
    path: Path
    slug: str
    environment: dict
    config: dict
    metadata: dict
    episodes: int
    best_episode: int
    files: dict[str, str]

    @property
    def checkpoint(self):
        return self.path / "checkpoints/best_model.pt"

    def receipt(self):
        return dict(algorithm=self.slug, display_name=NAMES[self.slug], run_directory=str(self.path),
                    checkpoint=str(self.checkpoint), training_episodes=self.episodes,
                    best_episode=self.best_episode, end_time=self.metadata["end_time"], sha256=self.files)


def validate_source_run(path, slug, *, min_episodes=5000) -> SourceRun:
    """Reject incomplete/smoke/final-checkpoint sources before constructing a model."""
    path = Path(path).resolve()
    try:
        meta = read_json(path / "run_metadata.json")
        if not isinstance(meta, dict):
            raise ValueError("Run metadata must be a JSON object")
        if (meta.get("algorithm_slug") != slug or meta.get("run_purpose") != "formal_training"
                or meta.get("formal_test_completed") is not True or not meta.get("end_time")):
            raise ValueError("Not a completed formal training run of the requested algorithm")
        datetime.fromisoformat(meta["end_time"])
        snapshot = yaml.safe_load((path / "config_snapshot.yaml").read_text(encoding="utf-8"))
        config, environment = snapshot["algorithm"], snapshot["environment"]
        if config["algorithm"]["name"] != slug:
            raise ValueError("Snapshot algorithm does not match the requested plugin")
        planned = int(config["training"]["episodes"])
        summary = read_json(path / "training_summary.json")
        if not isinstance(summary, dict):
            raise ValueError("Training summary must be a JSON object")
        episodes = int(summary["episodes"])
        if episodes < min_episodes or episodes != planned or episodes != int(meta["train_episodes"]):
            raise ValueError("Training episodes are short, unfinished, or inconsistent")
        expected_steps = episodes * int(environment["time"]["episode_length"])
        if int(summary["total_steps"]) != expected_steps:
            raise ValueError("Training did not finish all expected environment steps")
        history = read_json(path / "training_history.json")
        if not isinstance(history, list) or len(history) != episodes:
            raise ValueError("Training history is missing or truncated")
        if (any(not isinstance(row, dict) for row in history)
                or [row.get("episode") for row in history] != list(range(1, episodes + 1))):
            raise ValueError("Training history episode numbering is incomplete")
        selection = summary["checkpoint_selection"]
        if (PureWindowsPath(selection["best_checkpoint"]).name != "best_model.pt"
                or selection.get("constraint_eligible") is not True):
            raise ValueError("Best-checkpoint selection is missing or ineligible")
        best_episode = int(selection["best_episode"])
        if best_episode != int(summary["best_episode"]) or not 1 <= best_episode <= episodes:
            raise ValueError("Best-checkpoint episode records disagree")
        test = read_json(path / "test_summary.json")
        if not isinstance(test, dict):
            raise ValueError("Formal test summary must be a JSON object")
        checkpoint = path / "checkpoints/best_model.pt"
        checkpoint_hash = sha256(checkpoint)
        if (test.get("status") != "ok" or test.get("algorithm_slug") != slug
                or PureWindowsPath(test["checkpoint_used"]).name != "best_model.pt"
                or test.get("checkpoint_sha256") != checkpoint_hash
                or int(test["checkpoint_episode"]) != best_episode
                or int(test["test_days_count"]) < int(config["training"]["test_episodes"])):
            raise ValueError("Formal test did not finish with this exact best_model.pt")
        names = ("config_snapshot.yaml", "run_metadata.json", "training_summary.json", "training_history.json", "test_summary.json")
        files = {name: sha256(path / name) for name in names}
        files["checkpoints/best_model.pt"] = checkpoint_hash
        return SourceRun(path, slug, environment, config, meta, episodes, best_episode, files)
    except (OSError, KeyError, TypeError, OverflowError, yaml.YAMLError) as exc:
        raise ValueError(f"Incomplete or invalid source run: {path}: {exc}") from exc


def find_source_run(root, slug, *, min_episodes=5000):
    """Pick newest eligible completed run; latest_run.json is not an authority."""
    eligible, excluded = [], []
    runs = Path(root) / "runs"
    for path in sorted(runs.iterdir()) if runs.is_dir() else []:
        if not path.is_dir():
            continue
        try:
            eligible.append(validate_source_run(path, slug, min_episodes=min_episodes))
        except ValueError as exc:
            excluded.append(dict(run_directory=str(path.resolve()), reason=str(exc)))
    if not eligible:
        raise ValueError(f"No eligible full training run for {slug}; excluded={excluded}")
    eligible.sort(key=lambda run: (datetime.fromisoformat(run.metadata["end_time"]).timestamp(), str(run.path)))
    return eligible[-1], excluded


class FrozenPolicy:
    """Explicit adapter for current Q-guided models; no private dependency in training."""
    def __init__(self, source: SourceRun, env, device):
        if source.slug not in SLUGS:
            raise ValueError("Unsupported critic diagnostic adapter")
        self.source = source
        self.model = get_algorithm_spec(source.slug).create(env, source.config, device=device,
                                                           seed=int(source.metadata["seed"]))
        self.model.load(source.checkpoint)
        for modules in (self.model.actors, self.model.critics, self.model.target_actors, self.model.target_critics):
            modules.eval()
            modules.requires_grad_(False)
        self.gamma = float(self.model.gamma)
        self.ids = list(env.agent_ids)

    def actions(self, observations):
        return self.model.select_actions(observations, noise_std=0.0)

    @torch.no_grad()
    def q_values(self, state, candidates):
        model = self.model
        raw = torch.as_tensor(np.stack([c.raw for c in candidates]), dtype=torch.float32, device=model.device)
        states = torch.as_tensor(state, dtype=torch.float32, device=model.device)[None].expand(len(candidates), -1)
        effective, _ = model.project_actions(states, raw)
        rows = [torch.stack(model.critics[mg](states, effective.flatten(1)), dim=-1).squeeze(1)
                for mg in self.ids]
        values = torch.stack(rows, dim=1).cpu().numpy().astype(np.float64)
        if values.shape != (len(candidates), len(self.ids), 2) or not np.isfinite(values).all():
            raise ValueError("Nonfinite or invalid critic values")
        return values


@dataclass
class Candidate:
    raw: np.ndarray
    agent_index: int | None
    control_index: int | None


def axis_candidates(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (3, 2) or not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1.000001:
        raise ValueError("Expected three bounded [ESS, DR] actions")
    result = [Candidate(raw.copy(), None, None)]
    for agent in range(3):
        for control in range(2):
            for target in (-.8, .8):
                changed = raw.copy()
                changed[agent, control] = target
                result.append(Candidate(changed, agent, control))
    return result


def score_choices(realized, predicted, *, reward_scale, tie_cost):
    realized, predicted = np.asarray(realized), np.asarray(predicted)
    if (realized.ndim != 1 or realized.shape != predicted.shape or not len(realized)
            or not np.isfinite(realized).all() or not np.isfinite(predicted).all()
            or reward_scale <= 0 or tie_cost < 0):
        raise ValueError("Invalid/nonfinite choices or reward/tie scale")
    selected, best = int(np.argmax(predicted)), int(np.argmax(realized))
    pairs = correct = 0
    for i in range(len(realized)):
        for j in range(i):
            difference = float(realized[i] - realized[j])
            if abs(difference) * reward_scale <= tie_cost:
                continue
            pairs += 1
            correct += int(difference * float(predicted[i] - predicted[j]) > 0)
    return dict(selected_index=selected, best_index=best,
                selection_regret_cost=float((realized[best] - realized[selected]) * reward_scale),
                actor_improvement_cost=float((realized[best] - realized[0]) * reward_scale),
                rank_pair_count=pairs, rank_correct_count=correct,
                mean_absolute_q_error=float(np.mean(np.abs(predicted - realized))))


@dataclass
class StateCase:
    case_id: str
    origin: str
    day: str
    seed: int
    hour: int
    category: str
    prefix: list
    state: np.ndarray
    observations: dict
    signature: str
    context: dict
    forecast_seed_base: int | None = None


def _object_state(obj):
    supported = (dict, list, tuple, set, np.ndarray, np.generic, float, int, str, bool, type(None))
    return {k: clean(v) for k, v in vars(obj).items() if isinstance(v, supported)}


def environment_signature(env, observations):
    # Include latent inventories/EV departures, caches, complete forecast arrays,
    # and coordination history, not just the flattened public observation.
    return fingerprint(dict(hour=env.t, global_state=env.global_state(), observations=observations,
                            peak=env.daily_load_peak_kw, history=env.history,
                            batteries={mg: _object_state(core.battery) for mg, core in env.cores.items()},
                            devices={mg: _object_state(dev) for mg, dev in env.dr_devices.items()},
                            forecasts=_object_state(env.forecasts),
                            exogenous=env._episode_exogenous, rng=env.data.rng.bit_generator.state))


def restore_case(env, case, *, forecast_noise):
    observations, _ = env.reset(day=case.day, seed=case.seed, deterministic=True, forecast_noise=forecast_noise)
    for actions in case.prefix:
        observations, _, done, _ = env.step(actions)
        if done:
            raise ValueError("Prefix unexpectedly ended the episode")
    if env.t != case.hour or environment_signature(env, observations) != case.signature:
        raise ValueError(f"State reconstruction failed for {case.case_id}")
    return observations


def _team_reward(env, rewards):
    team = np.asarray([float(rewards[mg]) for mg in env.agent_ids])
    if not np.isfinite(team).all() or not np.allclose(team, team[0], rtol=0, atol=1e-8):
        raise ValueError("Expected one finite shared team reward")
    return float(team[0])


def rollout_return(env, policy, first_actions, *, capture_first=None):
    total, discount, steps, actions = 0., 1., 0, first_actions
    while True:
        observations, rewards, done, _ = env.step(actions)
        reward = _team_reward(env, rewards)
        total += discount * reward
        steps += 1
        if steps == 1 and capture_first is not None:
            capture_first.update(signature=environment_signature(env, observations), reward=reward)
        if done:
            return total, steps
        discount *= policy.gamma
        actions = policy.actions(observations)


def evaluate_candidates(env, policy, case, candidates, *, forecast_noise):
    raw = torch.as_tensor(np.stack([c.raw for c in candidates]), dtype=torch.float64)
    states = torch.as_tensor(case.state[None], dtype=torch.float64).expand(len(candidates), -1)
    with torch.no_grad():
        projected, _ = env.action_projector(states, raw)
    effective = projected.numpy()
    returns, aliases, seen, outcomes = [], [], [], []
    rollouts = steps = 0
    for cid, candidate in enumerate(candidates):
        restore_case(env, case, forecast_noise=forecast_noise)
        steps += case.hour
        first = {mg: candidate.raw[i].copy() for i, mg in enumerate(env.agent_ids)}
        possible = [i for i in seen if np.allclose(effective[i], effective[cid], rtol=0, atol=1e-9)]
        duplicate = None
        if possible:
            observations, rewards, _, _ = env.step(first)
            steps += 1
            outcome = dict(signature=environment_signature(env, observations), reward=_team_reward(env, rewards))
            duplicate = next((i for i in possible if outcome == outcomes[i]), None)
            if duplicate is None:
                # Equal projector vectors are insufficient if environment side
                # effects differ. Reconstruct and evaluate the entire branch.
                restore_case(env, case, forecast_noise=forecast_noise)
                steps += case.hour
        if duplicate is None:
            outcome = {}
            realized, suffix_steps = rollout_return(env, policy, first, capture_first=outcome)
            rollouts += 1
            steps += suffix_steps
            seen.append(cid)
        else:
            realized = returns[duplicate]
        returns.append(realized)
        outcomes.append(outcome)
        aliases.append(duplicate)
    return dict(returns=np.asarray(returns), duplicate_of=aliases, effective=effective, rollouts=rollouts, steps=steps)


def seasonal_days(valid_days, limit=4, *, days_per_season=1):
    if days_per_season not in (1, 3, 6):
        raise ValueError("days_per_season must be 1, 3, or 6")
    dates = sorted({str(day.date()) for day in valid_days})
    selected = []
    for quarter in range(4):
        available = [d for d in dates if (datetime.fromisoformat(d).month - 1) // 3 == quarter]
        if not available:
            raise ValueError("Validation data does not cover all four seasons; pass explicit --days")
        months = sorted({d[:7] for d in available})
        if days_per_season == 1:
            month_days = [d for d in available if d[:7] == months[0]]
            selected.append(month_days[len(month_days) // 2])
            continue
        if len(months) != 3:
            raise ValueError("Expanded sampling requires all three months of every season")
        per_month = days_per_season // 3
        for month in months:
            month_days = [d for d in available if d[:7] == month]
            if len(month_days) < per_month:
                raise ValueError("Not enough distinct validation dates in a month")
            indices = [int((i + 1) * len(month_days) / (per_month + 1)) for i in range(per_month)]
            selected.extend(month_days[i] for i in indices)
    return selected[:limit]


def context_scores(env, state, raw):
    index = {name: i for i, name in enumerate(env.global_state_builder.schema())}
    get = lambda name: float(state[index[name]])
    scores = dict(ordinary=0., ev_deadline=None, hvac_boundary=None, ess_boundary=None, peak_pressure=None)
    boundary = []
    for mg in env.agent_ids:
        cfg = env.cfg_by_id[mg]
        battery = cfg["battery"]
        soc = get(f"{mg}_soc")
        boundary.append(min(soc - battery["soc_min"], battery["soc_max"] - soc))
        dr = cfg["dr"]
        if dr["type"] == "commercial_hvac" and get(f"{mg}_occupied") > .5:
            scores["hvac_boundary"] = -min(get(f"{mg}_comfort_lower_margin"), get(f"{mg}_comfort_upper_margin"))
        if dr["type"] == "ev_charging":
            slack = []
            for group in dr["cohorts"]:
                key = f"{mg}_ev_{group['group_id']}"
                remaining = get(key + "_remaining_energy") * group["count"] * group["capacity_per_vehicle_kwh"]
                if get(key + "_connected") > .5 and remaining > 1e-6:
                    hours = get(key + "_hours_to_departure") * 24
                    power = get(key + "_rated_power") * dr["max_charge_kw"]
                    slack.append(hours - remaining / (dr["eta_charge"] * max(power, 1e-9)))
            if slack:
                scores["ev_deadline"] = -min(slack)
    scores["ess_boundary"] = -min(boundary)
    with torch.no_grad():
        effective, _ = env.action_projector(torch.as_tensor(state[None], dtype=torch.float64),
                                            torch.as_tensor(raw[None], dtype=torch.float64))
    load = 0.
    for i, mg in enumerate(env.agent_ids):
        dr = env.cfg_by_id[mg]["dr"]
        scale = (dr["rated_power_kw"] if dr["type"] == "industrial_mill" else
                 dr["heating_max_kw"] if dr["type"] == "commercial_hvac" else dr["max_charge_kw"])
        dr_value = float(effective[0, i, 2])
        load += get(f"{mg}_load") * env.global_state_builder.norm[mg]["load"]
        load += (abs(dr_value) if dr["type"] == "commercial_hvac" else dr_value) * scale
    scores["peak_pressure"] = load - get("team_load_peak_so_far") * env.global_state_builder.team_power_scale_kw
    return scores


def collect_cases(env, policies, days, seed, state_limit, forecast_noise):
    if not 1 <= state_limit <= 6 * len(days):
        raise ValueError("--states must be between 1 and six times the number of selected days")
    pool = []
    for day_index, day in enumerate(days):
        for policy in policies:
            observations, _ = env.reset(day=day, seed=seed + day_index, deterministic=True, forecast_noise=forecast_noise)
            prefix = []
            for hour in range(env.episode_length):
                actions = policy.actions(observations)
                raw = np.stack([actions[mg] for mg in env.agent_ids])
                state = env.global_state()
                scores = context_scores(env, state, raw)
                pool.append(StateCase(f"{day}_seed{seed + day_index}_{policy.source.slug}_{hour:02d}", policy.source.slug, day,
                                      seed + day_index, hour, "unselected", list(prefix), state,
                                      {mg: value.copy() for mg, value in observations.items()},
                                      environment_signature(env, observations), scores, seed))
                prefix.append({mg: value.copy() for mg, value in actions.items()})
                observations, _, _, _ = env.step(actions)
    return select_cases(pool, [p.source.slug for p in policies], days, seed, state_limit)


def select_cases(pool, origins, days, seed, state_limit):
    """Keep source/category balance while replacing duplicate full-state fingerprints."""
    selected, used = [], set()
    categories = ("ordinary", "ev_deadline", "hvac_boundary", "ess_boundary", "peak_pressure", "ordinary")
    rng = np.random.default_rng(seed)
    for day_index, day in enumerate(days):
        for slot, category in enumerate(categories):
            origin = origins[(slot + day_index) % len(origins)]
            available = [c for c in pool if c.day == day and c.origin == origin and c.signature not in used]
            if not available:
                raise ValueError(f"Not enough distinct states for {day}, {origin}")
            critical = [c for c in available if c.context.get(category) is not None]
            if category == "ordinary" or not critical:
                case = available[int(rng.integers(len(available)))]
                case.category = "ordinary" if category == "ordinary" else category + "_unavailable_fallback"
            else:
                case = max(critical, key=lambda c: (c.context[category], -c.hour))
                case.category = category
            selected.append(case)
            used.add(case.signature)
            if len(selected) == state_limit:
                return selected
    return selected


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_scores(rows, *, tie_cost=1.):
    if not rows:
        raise ValueError("Cannot summarize an empty score group")
    opportunity = np.asarray([r["actor_improvement_cost"] for r in rows])
    result = dict(group_count=len(rows), mean_actor_candidate_improvement_cost=float(opportunity.mean()),
                  median_actor_candidate_improvement_cost=float(np.median(opportunity)),
                  actor_improvable_count=int((opportunity > tie_cost).sum()))
    for kind in ("q1", "q2", "qmin"):
        regret = np.asarray([r[f"{kind}_selection_regret_cost"] for r in rows])
        # selected_cost - actor_cost = regret - actor_opportunity
        change = regret - opportunity
        pairs = sum(r[f"{kind}_rank_pair_count"] for r in rows)
        correct = sum(r[f"{kind}_rank_correct_count"] for r in rows)
        values = dict(mean_selection_regret_cost=float(regret.mean()),
                      median_selection_regret_cost=float(np.median(regret)),
                      p90_selection_regret_cost=float(np.quantile(regret, .9)),
                      max_selection_regret_cost=float(regret.max()),
                      top_five_regret_share=float(np.sort(regret)[-5:].sum() / regret.sum()) if regret.sum() > 0 else 0.,
                      rank_correct_count=correct, rank_pair_count=pairs,
                      rank_accuracy=correct / pairs if pairs else None,
                      mean_selected_change_cost=float(change.mean()),
                      harmful_selection_count=int((change > tie_cost).sum()),
                      helpful_selection_count=int((change < -tie_cost).sum()))
        result.update({f"{kind}_{key}": value for key, value in values.items()})
    return result


def stratified_summary(group_rows, tie_cost):
    models, flat = {}, []
    for slug in SLUGS:
        rows = [r for r in group_rows if r["algorithm"] == slug]
        model = summarize_scores(rows, tie_cost=tie_cost)
        flat.append(dict(algorithm=slug, scope="overall", scope_value="all", **model))
        for level, key in (("by_forecast_seed", "forecast_seed_base"), ("by_season", "season"),
                           ("by_day", "day"), ("by_origin", "origin"), ("by_control", "coordinate")):
            buckets = {}
            for row in rows:
                value = f"{row['agent']}_{row['control']}" if key == "coordinate" else str(row[key])
                buckets.setdefault(value, []).append(row)
            model[level] = {}
            for value, bucket in sorted(buckets.items()):
                metrics = summarize_scores(bucket, tie_cost=tie_cost)
                model[level][value] = metrics
                flat.append(dict(algorithm=slug, scope=level, scope_value=value, **metrics))
        models[slug] = model
    return models, flat


def _report(output, summary, group_rows):
    lines = ["# Critic 第一阶段精简诊断", "", "这是固定策略下的历史场景回放诊断，不是 MILP 最优价值、Actor 教师数据或结构优劣的最终证明。", "",
             f"验证日期：{', '.join(summary['days'])}。预测噪声基础种子：{summary['forecast_seeds']}。共同状态数：{summary['state_count']}，不同完整状态指纹数：{summary['unique_state_count']}。", "",
             "每个模型使用自己的冻结联合策略继续执行；只在当前一步替换候选动作。不同模型的参考 Q 含义因此各自对应自身策略。", "",
             "| 模型 | 控制量 | Q1 选错的平均成本 | 当前动作的平均候选改进空间 | 有意义动作对的排序正确率 |", "|---|---|---:|---:|---:|"]
    for slug in SLUGS:
        for agent in ("mg1", "mg2", "mg3"):
            for control in ("ESS", "DR"):
                rows = [r for r in group_rows if r["algorithm"] == slug and r["agent"] == agent and r["control"] == control]
                if not rows:
                    continue
                pairs = sum(r["q1_rank_pair_count"] for r in rows)
                correct = sum(r["q1_rank_correct_count"] for r in rows)
                rate = f"{correct / pairs:.1%}" if pairs else "无可区分动作对"
                regret = np.mean([r["q1_selection_regret_cost"] for r in rows])
                opportunity = np.mean([r["actor_improvement_cost"] for r in rows])
                lines.append(f"| {NAMES[slug]} | {agent} {control} | {regret:.3f} | {opportunity:.3f} | {rate} |")
    for level, title in (("by_forecast_seed", "按预测噪声种子"), ("by_season", "按季度"),
                         ("by_origin", "按起点策略来源")):
        lines += ["", f"## {title}", "", "| 模型 | 分组 | Q1 平均误选代价 | Q1 排序正确率 | minQ 平均误选代价 | 候选改进空间 |", "|---|---|---:|---:|---:|---:|"]
        for slug, model in summary["models"].items():
            for value, metrics in model[level].items():
                rate = metrics["q1_rank_accuracy"]
                rate_label = f"{rate:.1%}" if rate is not None else "无可区分动作对"
                label = NAMES.get(value, value)
                lines.append(f"| {NAMES[slug]} | {label} | {metrics['q1_mean_selection_regret_cost']:.3f} | {rate_label} | {metrics['qmin_mean_selection_regret_cost']:.3f} | {metrics['mean_actor_candidate_improvement_cost']:.3f} |")
    lines += ["", "## 严重误选案例", "", "| 模型 | 日期 | 实际种子 | 时刻 | 起点来源 | 控制量 | 误选代价 | 原动作候选改进空间 |", "|---|---|---:|---:|---|---|---:|---:|"]
    for slug in SLUGS:
        rows = sorted((r for r in group_rows if r["algorithm"] == slug), key=lambda r: r["q1_selection_regret_cost"], reverse=True)
        for row in rows[:5]:
            lines.append(f"| {NAMES[slug]} | {row['day']} | {row['seed']} | {row['hour']} | {NAMES[row['origin']]} | {row['agent']} {row['control']} | {row['q1_selection_regret_cost']:.3f} | {row['actor_improvement_cost']:.3f} |")
    lines += ["", "## 解读准则", "", "- 优先看 Q1，因为 Actor 主损失使用 Q1；Q2 和 min(Q1,Q2) 的结果同时保存在 CSV 中。",
              "- 选错代价明显：优先检查价值评价。排序较好而当前动作差：进一步检查局部梯度、饱和和投影；不能直接判定 Actor 容量不足。",
              "- 候选是单个控制量的两个方向；这里只衡量候选集内的机会，尚未检验复杂联合协调。",
              "- 实际回报是一条外部条件下的实现值，不等于观测条件下的精确期望 Q。TD3 目标策略平滑也可能带来价值差异。",
              f"- 本次仅覆盖 {len(summary['days'])} 个验证日和 {len(summary['forecast_seeds'])} 个预测噪声基础种子，不把相关的候选分支视为独立日期样本。",
              "- 汇总代价按状态与控制量平均，不是日均政策成本，也不能相加得到可实现的每日节省。日期按日历预先选取，设备关键状态是定向诊断样本。",
              "- stratified_scores.csv 和 summary.json 同时记录分日期、种子、季度、起点来源、控制量的结果，以及中位数、P90、最大值和前五例误选代价占比。",
              "- 成本单位由原 reward_scale 换算，gamma=1 时是剩余综合成本；接近的动作按配置容差视为并列。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_probe(*, root, output, state_limit=None, day_limit=4, days=None, device="cpu", seed=424242,
              tie_cost=1., run_paths=None, check_only=False, days_per_season=1, forecast_seeds=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if ((state_limit is not None and state_limit < 1) or not 1 <= day_limit <= 4
            or days_per_season not in (1, 3, 6) or tie_cost < 0 or not np.isfinite(tie_cost)):
        raise ValueError("Invalid state/day limit or tie cost")
    forecast_seeds = list(forecast_seeds) if forecast_seeds is not None else [seed]
    if (not forecast_seeds or any(not isinstance(s, int) or s < 0 or s >= 2**32 for s in forecast_seeds)
            or len(set(forecast_seeds)) != len(forecast_seeds)):
        raise ValueError("Forecast seeds must be distinct nonnegative integer seeds")
    sources, excluded = [], []
    for slug in SLUGS:
        if run_paths and run_paths.get(slug):
            source = validate_source_run(run_paths[slug], slug)
        else:
            source, rejected = find_source_run(root / "outputs/working_runs" / slug, slug)
            excluded.extend(rejected)
        sources.append(source)
        print(f"Selected {NAMES[slug]}: {source.path}\n  full episodes={source.episodes}, best episode={source.best_episode}, checkpoint={source.checkpoint}", flush=True)
    environments = []
    for source in sources:
        config = dict(source.environment)
        config.pop("_project_root", None)
        environments.append(config)
    if fingerprint(environments[0]) != fingerprint(environments[1]):
        raise ValueError("Source environment snapshots differ; these runs cannot be compared")
    config = dict(environments[0], _project_root=str(root))
    if config.get("action_design") != "single_dispatch_dr" or config.get("reward", {}).get("semantics_version") != "generalized_system_cost_v2":
        raise ValueError("Diagnostic requires the current shared-reward dispatch environment")
    data_path = Path(config["data"]["val_path"])
    if not data_path.is_absolute():
        data_path = root / data_path
    protected = {root / "algorithms/matd3" / name: sha256(root / "algorithms/matd3" / name)
                 for name in ("actor.py", "critic.py", "matd3.py", "replay_buffer.py", "plugin.py")}
    for source in sources:
        protected.update({source.path / name: value for name, value in source.files.items()})
    protected[data_path] = sha256(data_path)
    torch.set_num_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available; use --device cpu")
    env = MultiMicrogridEnv(config, split="val")
    selected_days = list(days) if days else seasonal_days(
        env.data.valid_days, day_limit if days_per_season == 1 else None, days_per_season=days_per_season)
    valid = {str(day.date()) for day in env.data.valid_days}
    if (not selected_days or len(set(selected_days)) != len(selected_days) or not set(selected_days) <= valid):
        raise ValueError("All --days must be distinct valid validation dates")
    scenario_seeds = [base + i for base in forecast_seeds for i in range(len(selected_days))]
    if len(set(scenario_seeds)) != len(scenario_seeds) or max(scenario_seeds) >= 2**32:
        raise ValueError("Forecast base seeds produce overlapping or invalid actual scenario seeds")
    max_states = 6 * len(selected_days) * len(forecast_seeds)
    state_limit = max_states if state_limit is None else state_limit
    if not len(forecast_seeds) <= state_limit <= max_states:
        raise ValueError("Requested states must cover each forecast seed and not exceed six per day/seed")
    policies = [FrozenPolicy(source, env, device) for source in sources]
    forecast_noise = bool(config["observation"].get("evaluation_forecast_noise", False))
    manifest = dict(schema_version=2, purpose="offline_critic_diagnosis", status="checked" if check_only else "running",
                    created_at=datetime.now().astimezone().isoformat(), sources=[s.receipt() for s in sources],
                    excluded_runs=excluded, split="val", days=selected_days, seed=seed,
                    forecast_seeds=forecast_seeds, actual_scenario_seeds=scenario_seeds,
                    day_selection="explicit" if days else "calendar_only", days_per_season=days_per_season,
                    state_limit=state_limit, state_selection="six categories per day/seed; duplicate complete state fingerprints replaced",
                    device=device, forecast_noise=forecast_noise, exploration_noise=0., tie_cost=tie_cost,
                    validation_data_sha256=sha256(data_path), environment_hash=fingerprint(environments[0]),
                    diagnostic_code_sha256=sha256(Path(__file__)),
                    candidate_design="current + one coordinate at -0.8/+0.8; aliases require equal actual first reward and full next state",
                    continuation="own frozen joint policy, deterministic actions",
                    reference_semantics="realized return, not exact conditional expectation or MILP optimal Q")
    print(f"Validation days: {', '.join(selected_days)}; forecast base seeds={forecast_seeds}; states={state_limit}; maximum candidate rollouts={state_limit * 13 * 2}", flush=True)
    if check_only:
        return manifest
    forbidden = [root / "outputs/working_runs", root / "outputs/releases", root / "algorithms/matd3"]
    if any(output.is_relative_to(path) for path in forbidden + [source.path for source in sources]):
        raise ValueError("Diagnostic output must not modify source runs or retained releases")
    if output.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    candidate_rows, group_rows = [], []
    start, rollouts, steps, aliases = time.perf_counter(), 0, 0, 0
    reward_scale = float(config["cost"]["reward_scale"])
    try:
        cases = []
        per_seed, remainder = divmod(state_limit, len(forecast_seeds))
        for i, base_seed in enumerate(forecast_seeds):
            cases.extend(collect_cases(env, policies, selected_days, base_seed,
                                       per_seed + (i < remainder), forecast_noise))
        if len({c.signature for c in cases}) != len(cases) or len({c.case_id for c in cases}) != len(cases):
            raise ValueError("Expanded state pool has duplicate states or identifiers")
        write_json(output / "state_pool.json", [vars(c) for c in cases])
        for case_index, case in enumerate(cases):
            for policy in policies:
                current_actions = policy.actions(case.observations)
                original = np.stack([current_actions[mg] for mg in env.agent_ids])
                candidates = axis_candidates(original)
                values = policy.q_values(case.state, candidates)
                evaluated = evaluate_candidates(env, policy, case, candidates, forecast_noise=forecast_noise)
                effective, returns = evaluated["effective"], evaluated["returns"]
                rollouts += evaluated["rollouts"]
                steps += evaluated["steps"]
                aliases += sum(i is not None for i in evaluated["duplicate_of"])
                for cid, candidate in enumerate(candidates):
                    duplicate = evaluated["duplicate_of"][cid]
                    realized = float(returns[cid])
                    row = dict(case_id=case.case_id, origin=case.origin, day=case.day, seed=case.seed,
                               forecast_seed_base=case.forecast_seed_base, hour=case.hour,
                               category=case.category, algorithm=policy.source.slug, candidate_id=cid,
                               changed_agent="current" if candidate.agent_index is None else env.agent_ids[candidate.agent_index],
                               changed_control="current" if candidate.control_index is None else ("ESS", "DR")[candidate.control_index],
                               duplicate_of="" if duplicate is None else duplicate,
                               raw_actions=json.dumps(candidate.raw.tolist()), effective_actions=json.dumps(effective[cid].tolist()),
                               realized_return=realized, realized_remaining_cost=-reward_scale * realized)
                    for i, mg in enumerate(env.agent_ids):
                        row[f"{mg}_q1"], row[f"{mg}_q2"] = values[cid, i]
                    candidate_rows.append(row)
                for agent_index, mg in enumerate(env.agent_ids):
                    for control_index, control in enumerate(("ESS", "DR")):
                        ids = [0] + [i for i, c in enumerate(candidates) if c.agent_index == agent_index and c.control_index == control_index]
                        row = dict(case_id=case.case_id, origin=case.origin, day=case.day, seed=case.seed,
                                   forecast_seed_base=case.forecast_seed_base,
                                   season=(datetime.fromisoformat(case.day).month - 1) // 3 + 1, hour=case.hour,
                                   category=case.category, algorithm=policy.source.slug, agent=mg, control=control)
                        for kind, predictions in (("q1", values[ids, agent_index, 0]), ("q2", values[ids, agent_index, 1]),
                                                  ("qmin", values[ids, agent_index].min(axis=-1))):
                            metrics = score_choices(returns[ids], predictions, reward_scale=reward_scale, tie_cost=tie_cost)
                            row["actor_improvement_cost"] = metrics.pop("actor_improvement_cost")
                            metrics["selected_index"] = ids[metrics["selected_index"]]
                            metrics["best_index"] = ids[metrics["best_index"]]
                            row.update({f"{kind}_{k}": v for k, v in metrics.items()})
                        group_rows.append(row)
                _csv(output / "candidate_results.csv", candidate_rows)
                _csv(output / "state_scores.csv", group_rows)
                print(f"State {case_index + 1}/{len(cases)} [{case.category}, {case.day} h{case.hour:02d}] {NAMES[policy.source.slug]}: {evaluated['rollouts']}/{len(candidates)} unique rollouts; elapsed={time.perf_counter() - start:.1f}s", flush=True)
        changed = [str(path) for path, digest in protected.items() if sha256(path) != digest]
        if changed:
            raise ValueError(f"Source integrity check failed: {changed}")
        summary = dict(status="complete", state_count=len(cases), unique_state_count=len({c.signature for c in cases}),
                       model_count=len(policies), days=selected_days, forecast_seeds=forecast_seeds,
                       candidate_rows=len(candidate_rows), candidate_rollouts=rollouts, physical_aliases=aliases,
                       environment_steps_including_prefix=steps, reconstruction_failures=0,
                       source_integrity_verified=True, elapsed_seconds=time.perf_counter() - start,
                       reward_scale=reward_scale, gamma={p.source.slug: p.gamma for p in policies},
                       tie_cost=tie_cost, models={})
        summary["models"], stratified = stratified_summary(group_rows, tie_cost)
        _csv(output / "stratified_scores.csv", stratified)
        write_json(output / "summary.json", summary)
        _report(output, summary, group_rows)
        manifest.update(status="complete", completed_at=datetime.now().astimezone().isoformat(), source_integrity_verified=True)
        write_json(output / "manifest.json", manifest)
        print(f"Diagnosis complete: {output / 'report.md'}", flush=True)
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
