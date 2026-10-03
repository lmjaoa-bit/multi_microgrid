"""Read-only local test of value directions, projection and Actor sensitivity.

No optimizer steps, replay writes, checkpoint saves or MILP calls are made.
Parameter derivatives describe a single-state loss, not a minibatch Adam step.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch

from diagnostics.critic_probe import (
    Candidate, FrozenPolicy, NAMES, SLUGS, StateCase, _csv, evaluate_candidates,
    fingerprint, read_json, sha256, validate_source_run, write_json,
)
from envs.multi_microgrid_env import MultiMicrogridEnv


@dataclass
class LocalCandidate(Candidate):
    kind: str = 'axis'
    radius: float = 0.
    sign: int = 0


def local_candidates(raw, radii):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (3, 2) or not np.isfinite(raw).all() or np.abs(raw).max() > 1.000001:
        raise ValueError('Expected three bounded two-channel actions')
    if not radii or any(not np.isfinite(r) or not 0 < r <= .2 for r in radii):
        raise ValueError('Local radii must be positive and at most .2')
    result = [LocalCandidate(raw.copy(), None, None, 'current')]
    for agent in range(3):
        for control in range(2):
            for radius in radii:
                for sign in (-1, 1):
                    changed = raw.copy()
                    changed[agent, control] = np.clip(raw[agent, control] + sign * radius, -1., 1.)
                    result.append(LocalCandidate(changed, agent, control, 'axis', radius, sign))
    return result


def direction_candidates(raw, gradients, radius):
    result = []
    for agent, metric in enumerate(gradients):
        for kind, key in (('critic_direction', 'raw_gradient'), ('actor_direction', 'actor_response')):
            direction = np.asarray(metric[key], dtype=np.float64)
            if direction.shape != (2,) or not np.isfinite(direction).all():
                raise ValueError('Nonfinite local direction')
            norm = float(np.abs(direction).max())
            changed = raw.copy()
            if norm > 1e-12:
                changed[agent] = np.clip(raw[agent] + radius * direction / norm, -1., 1.)
            result.append(LocalCandidate(changed, agent, None, kind, radius, 1))
    return result


def select_local_cases(pool, state_limit=None):
    """One calendar-fixed case per day/seed; slots rotate by month, not outcomes."""
    days = sorted({case.day for case in pool})
    seeds = sorted({case.forecast_seed_base for case in pool})
    buckets = {}
    for case in pool:
        buckets.setdefault((case.day, case.forecast_seed_base), []).append(case)
    selected = []
    for seed_index, seed in enumerate(seeds):
        for day_index, day in enumerate(days):
            cases = buckets.get((day, seed), [])
            if len(cases) != 6:
                raise ValueError('Input probe must contain six cases per day and base seed')
            selected.append(cases[(day_index // 2 + seed_index * 3) % 6])
    if state_limit is not None:
        if not 1 <= state_limit <= len(selected):
            raise ValueError('State limit exceeds the fixed local case pool')
        selected = selected[:state_limit]
    return selected


def projection_trace(model, state, raw):
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=model.device)[None]
    raw_tensor = torch.tensor(raw, dtype=torch.float32, device=model.device, requires_grad=True)[None]
    effective, _ = model.project_actions(state_tensor, raw_tensor)
    flat = effective.flatten(1)
    jacobian = torch.stack([
        torch.autograd.grad(flat[0, i], raw_tensor, retain_graph=True)[0].flatten()
        for i in range(flat.shape[1])
    ])
    gradients, physical, values, residual = [], [], [], 0.
    for mg in model.agent_ids:
        q1 = model.critics[mg].q1(state_tensor, flat)
        direct = torch.autograd.grad(q1.sum(), raw_tensor, retain_graph=True)[0].flatten()
        partial = torch.autograd.grad(q1.sum(), effective, retain_graph=True)[0].flatten()
        residual = max(residual, float((direct - partial @ jacobian).abs().max().detach()))
        gradients.append(direct.detach().cpu().numpy())
        physical.append(partial.detach().cpu().numpy())
        values.append(float(q1.detach().sum()))
    arrays = [jacobian.detach().cpu().numpy(), np.asarray(gradients), np.asarray(physical)]
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError('Nonfinite projection/Critic derivative')
    return dict(jacobian=arrays[0].astype(np.float64), q1_raw_gradients=arrays[1].astype(np.float64),
                q1_effective_gradients=arrays[2].astype(np.float64), q1_values=values,
                chain_residual_max=residual)


def _norm(gradients):
    return sum(float(g.detach().double().square().sum()) for g in gradients if g is not None) ** .5


def _dot(left, right):
    return sum(float((a.detach().double() * b.detach().double()).sum())
               for a, b in zip(left, right) if a is not None and b is not None)


def actor_trace(model, active, state, observations, raw, raw_gradient):
    actor = model.actors[active]
    parameters = tuple(actor.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(True)
        obs = torch.as_tensor(observations, dtype=torch.float32, device=model.device)[None]
        if callable(getattr(actor, 'action_logits', None)):
            logits = actor.action_logits(obs)
        else:
            logits = actor.network[:-1](obs)
        actions = logits.tanh()
        pre = [torch.autograd.grad(logits[0, i], parameters, retain_graph=True, allow_unused=True) for i in range(2)]
        post = [torch.autograd.grad(actions[0, i], parameters, retain_graph=True, allow_unused=True) for i in range(2)]
        # Pull back the exact current-action Q derivative, avoiding tiny differences
        # between no-grad Transformer fast paths and the differentiable forward.
        upstream = torch.as_tensor(raw_gradient, dtype=torch.float32, device=model.device)
        main_loss = -(actions[0] * upstream).sum()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=model.device)[None]
        raw_tensor = torch.as_tensor(raw, dtype=torch.float32, device=model.device)[None]
        auxiliary, metrics = model.actor_auxiliary_loss(active, state_tensor, raw_tensor, obs)
        main_grad = torch.autograd.grad(main_loss, parameters, retain_graph=True, allow_unused=True)
        total_grad = torch.autograd.grad(main_loss + auxiliary, parameters, retain_graph=True, allow_unused=True)
        total_norm = _norm(total_grad)
        response = np.asarray([-_dot(g, total_grad) / total_norm if total_norm > 1e-20 else 0. for g in post])
        result = dict(raw_gradient=np.asarray(raw_gradient), actor_response=response,
            actor_forward_actions=actions.detach().cpu().numpy()[0],
            tanh_factors=(1 - actions.detach().square()).cpu().numpy()[0],
            pre_jacobian_norms=np.asarray([_norm(g) for g in pre]),
            post_jacobian_norms=np.asarray([_norm(g) for g in post]),
            main_gradient_norm=_norm(main_grad), total_gradient_norm=total_norm,
            auxiliary_loss=float(auxiliary.detach()),
            auxiliary_trigger=float(metrics.get('dispatch_q_guidance_trigger_rate', torch.tensor(0.)).detach()))
        if any(not np.isfinite(np.asarray(value)).all() for value in result.values()):
            raise ValueError('Nonfinite Actor derivative')
        return result
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def classify_coordinate(rows, metric, *, broad_opportunity, auxiliary_trigger, tie_cost):
    meaningful = [r for r in rows if abs(r['realized_gain_cost']) > tie_cost]
    value_wrong = sum(r['realized_gain_cost'] * r['predicted_gain_cost'] < 0
                      and abs(r['predicted_gain_cost']) > .01 for r in meaningful)
    smooth = [r for r in meaningful if r['physical_linear_residual'] <= max(1e-5, .25 * r['physical_delta_norm'])]
    linear_conflict = sum(r['realized_gain_cost'] * r['linear_gain_cost'] < 0
                         and abs(r['linear_gain_cost']) > .01 for r in smooth)
    q1_nonlinear = sum(abs(r['predicted_gain_cost'] - r['linear_gain_cost']) >
                      max(.01, .25 * abs(r['predicted_gain_cost'])) for r in smooth)
    local_opportunity = max([0.] + [r['realized_gain_cost'] for r in rows])
    return dict(local_opportunity_cost=local_opportunity, meaningful_local_pair_count=len(meaningful),
        local_value_wrong_count=value_wrong, local_gradient_testable_count=len(smooth),
        local_linear_direction_conflict_count=linear_conflict,
        local_q1_nonlinear_count=q1_nonlinear,
        local_projection_nonlinear_count=len(meaningful) - len(smooth),
        projection_barrier_opportunity=bool(metric['projection_jacobian_norm'] <= 1e-9
            and max([0.] + [r['physical_delta_norm'] for r in rows]) <= 1e-8 and broad_opportunity > 1.),
        actor_saturation_opportunity=bool(metric['tanh_factor'] <= .01
            and (broad_opportunity > 1. or local_opportunity > tie_cost)),
        auxiliary_available=bool(auxiliary_trigger > 0.))


def _tensor_fingerprint(model):
    import hashlib
    digest = hashlib.sha256()
    for label in ('actors', 'critics', 'target_actors', 'target_critics'):
        container = getattr(model, label)
        for key, tensor in list(container.named_parameters()) + list(container.named_buffers()):
            digest.update((label + key + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_phase_one(path):
    path = Path(path).resolve()
    manifest, summary = read_json(path/'manifest.json'), read_json(path/'summary.json')
    if (manifest.get('status') != 'complete' or not manifest.get('source_integrity_verified')
            or summary.get('status') != 'complete' or len(manifest.get('days', [])) < 24
            or len(manifest.get('forecast_seeds', [])) != 2 or summary.get('state_count', 0) < 288):
        raise ValueError('Require the completed expanded 24-day/two-seed phase-one probe')
    rows = read_json(path/'state_pool.json')
    cases = []
    for row in rows:
        row = dict(row)
        row['state'] = np.asarray(row['state'], dtype=np.float32)
        row['observations'] = {mg: np.asarray(value, dtype=np.float32) for mg, value in row['observations'].items()}
        row['prefix'] = [{mg: np.asarray(value, dtype=np.float32) for mg, value in actions.items()} for actions in row['prefix']]
        cases.append(StateCase(**row))
    if len(cases) != summary['state_count'] or len({c.signature for c in cases}) != len(cases):
        raise ValueError('Input state pool is inconsistent or duplicated')
    with (path/'state_scores.csv').open(encoding='utf-8-sig', newline='') as stream:
        scores = {(r['case_id'], r['algorithm'], r['agent'], r['control']): r for r in csv.DictReader(stream)}
    with (path/'candidate_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        baseline = {(r['case_id'], r['algorithm']): float(r['realized_return'])
                    for r in csv.DictReader(stream) if int(r['candidate_id']) == 0}
    return manifest, cases, scores, baseline


def find_expanded_probe(root):
    eligible = []
    directory = Path(root)/'outputs/diagnostics/critic_probe'
    for path in directory.iterdir() if directory.exists() else []:
        if not path.is_dir():
            continue
        try:
            manifest, _, _, _ = load_phase_one(path)
            eligible.append((datetime.fromisoformat(manifest['created_at']).timestamp(), path))
        except (ValueError, OSError, KeyError, TypeError):
            continue
    if not eligible:
        raise ValueError('No completed expanded phase-one probe; pass --probe')
    return max(eligible, key=lambda entry: entry[0])[1]


def _summarize(coordinates, directions):
    models = {}
    for slug in SLUGS:
        rows = [r for r in coordinates if r['algorithm'] == slug]
        meaningful = sum(r['meaningful_local_pair_count'] for r in rows)
        testable = sum(r['local_gradient_testable_count'] for r in rows)
        model = dict(coordinate_count=len(rows), meaningful_local_pair_count=meaningful,
            local_value_wrong_count=sum(r['local_value_wrong_count'] for r in rows),
            local_gradient_testable_count=testable,
            local_linear_direction_conflict_count=sum(r['local_linear_direction_conflict_count'] for r in rows),
            local_q1_nonlinear_count=sum(r['local_q1_nonlinear_count'] for r in rows),
            projection_barrier_opportunity_count=sum(r['projection_barrier_opportunity'] for r in rows),
            projection_barrier_with_auxiliary_count=sum(r['projection_barrier_opportunity'] and r['auxiliary_available'] for r in rows),
            actor_saturation_opportunity_count=sum(r['actor_saturation_opportunity'] for r in rows),
            mean_local_opportunity_cost=float(np.mean([r['local_opportunity_cost'] for r in rows])),
            mean_broad_opportunity_cost=float(np.mean([r['broad_opportunity_cost'] for r in rows])))
        for kind in ('critic_direction', 'actor_direction'):
            subset = [r for r in directions if r['algorithm'] == slug and r['kind'] == kind]
            gains = np.asarray([r['realized_gain_cost'] for r in subset])
            model[kind] = dict(count=len(subset), mean_gain_cost=float(gains.mean()),
                helpful_count=int((gains > .1).sum()), harmful_count=int((gains < -.1).sum()),
                physically_flat_count=sum(r['physical_delta_norm'] <= 1e-8 for r in subset))
        models[slug] = model
    return models


def _report(output, summary):
    lines = ['# Critic—投影—Actor 学习链条诊断', '',
        f"固定共同起点：{summary['state_count']}。半径：{summary['radii']}。实际分支回放：{summary['candidate_rollouts']}。", '',
        '只改变当前一步，后续执行自身冻结策略；没有训练、参数更新或 MILP。', '',
        '| 模型 | 有意义局部动作对 | Q1 局部价值与真实方向冲突 | 投影近似线性时梯度与有限收益冲突 | 投影近似线性时 Q1 一阶偏差明显 | 投影屏障且有较大动作机会 | 屏障中辅助引导启用 | 饱和且有改进机会 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for slug, model in summary['models'].items():
        lines.append(f"| {NAMES[slug]} | {model['meaningful_local_pair_count']} | {model['local_value_wrong_count']} | {model['local_linear_direction_conflict_count']}/{model['local_gradient_testable_count']} | {model['local_q1_nonlinear_count']} | {model['projection_barrier_opportunity_count']} | {model['projection_barrier_with_auxiliary_count']} | {model['actor_saturation_opportunity_count']} |")
    lines += ['', '## 同幅度方向回放', '',
        '对每个微网，分别沿 Critic 的原始动作梯度、Actor 单状态总损失的参数下降方向所诱导的动作方向，归一化为同一原始动作半径。',
        '两个动作通道一起变化，边界处会截断。归一化只比较方向，不表示实际一次参数更新能达到该幅度。', '',
        '| 模型 | 方向 | 平均实际成本改善 | 改善次数 | 恶化次数 | 物理动作未变次数 |', '|---|---|---:|---:|---:|---:|']
    for slug, model in summary['models'].items():
        for kind, label in [('critic_direction', 'Critic 原始动作梯度'), ('actor_direction', 'Actor 参数方向（含现有辅助损失）')]:
            metric = model[kind]
            lines.append(f"| {NAMES[slug]} | {label} | {metric['mean_gain_cost']:.4f} | {metric['helpful_count']} | {metric['harmful_count']} | {metric['physically_flat_count']} |")
    lines += ['', '## 判断边界', '',
        '- Q1 前向价值增益和当前梯度线性增益分开记录。梯度与有限收益冲突不等于导数错误：即使投影近似线性，Q1 曲率和策略回报非线性也可能使有限步长改变方向。Q1 一阶偏差单列，未过滤这些冲突。',
        '- 饱和机会使用 tanh 导数不超过 0.01；投影屏障要求当前局部投影导数和全部小扰动物理变化均为零，并且第一阶段远程候选可改善超过 1。',
        '- 现有 ESS 辅助引导单独记录；主梯度为零不等于全部训练信号为零。',
        '- 这些标志可以重叠，不能相加当作 Actor/Critic 的因果贡献比例。',
        '- Actor 导数对应当前单状态与当前其他 Actor 动作，不是实际历史 minibatch、Adam 动量或完整训练的重演。',
        '- 两个网络的绝对参数梯度范数受参数数目和特征尺度影响，不可用来直接判定表达能力优劣。',
        '- 真回报是固定历史未来的实现值；不等于观测条件下精确期望 Q，也不能据此证明 Transformer 结构失败。',
        '- 成本增益是当前一步扰动后的剩余综合成本差，不能叠加当作可实现的每日节省。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf8')


def run_chain_probe(*, root, probe, output, state_limit=None, radii=(.02, .1), device='cpu', check_only=False):
    root, probe, output = Path(root).resolve(), Path(probe).resolve(), Path(output).resolve()
    phase_one, pool, broad, baseline = load_phase_one(probe)
    selected = select_local_cases(pool, state_limit)
    sources = []
    for slug in SLUGS:
        entries = [entry for entry in phase_one['sources'] if entry['algorithm'] == slug]
        if len(entries) != 1:
            raise ValueError('Phase-one model source is inconsistent')
        source = validate_source_run(entries[0]['run_directory'], slug)
        if source.files != entries[0]['sha256']:
            raise ValueError('A phase-one training source changed; cannot mix versions')
        sources.append(source)
    environments = []
    for source in sources:
        config = dict(source.environment)
        config.pop('_project_root', None)
        environments.append(config)
    if fingerprint(environments[0]) != fingerprint(environments[1]) or fingerprint(environments[0]) != phase_one['environment_hash']:
        raise ValueError('Environment differs from phase one')
    config = dict(environments[0], _project_root=str(root))
    data_path = Path(config['data']['val_path'])
    data_path = data_path if data_path.is_absolute() else root/data_path
    if sha256(data_path) != phase_one['validation_data_sha256']:
        raise ValueError('Validation data changed since phase one')
    torch.set_num_threads(1)
    env = MultiMicrogridEnv(config, split='val')
    policies = [FrozenPolicy(source, env, device) for source in sources]
    local_candidates(np.zeros((3, 2)), radii)  # Validate before creating any output.
    if len(set(radii)) != len(radii):
        raise ValueError('Repeated local radii')
    protected = {data_path: sha256(data_path)}
    for filename in ('manifest.json', 'summary.json', 'state_pool.json', 'state_scores.csv', 'candidate_results.csv'):
        protected[probe/filename] = sha256(probe/filename)
    for source in sources:
        protected.update({source.path/name: digest for name, digest in source.files.items()})
    for name in ('actor.py', 'critic.py', 'matd3.py', 'replay_buffer.py', 'plugin.py'):
        path = root/'algorithms/matd3'/name
        protected[path] = sha256(path)
    snapshots = [_tensor_fingerprint(policy.model) for policy in policies]
    manifest = dict(status='checked' if check_only else 'running', created_at=datetime.now().astimezone().isoformat(),
        purpose='offline_learning_chain_diagnosis', phase_one_directory=str(probe),
        phase_one_manifest_sha256=protected[probe/'manifest.json'], sources=[s.receipt() for s in sources],
        state_count=len(selected), selected_case_ids=[case.case_id for case in selected], radii=list(radii),
        direction_radius=min(radii), local_tie_cost=.1, device=device, exploration_noise=0.,
        forecast_noise=phase_one['forecast_noise'],
        selection='one pre-existing case per day/seed; slot=(day_index//2+seed_index*3)%6; no outcome selection',
        diagnostic_code_sha256=sha256(Path(__file__)))
    print(f"Learning-chain probe: states={len(selected)}, models=2, radii={radii}, sources best episodes={[s.best_episode for s in sources]}", flush=True)
    if check_only:
        return manifest
    forbidden = [probe, root/'outputs/working_runs', root/'outputs/releases', root/'algorithms/matd3'] + [s.path for s in sources]
    if output.exists() or any(output.is_relative_to(path) for path in forbidden):
        raise ValueError('Choose a new output directory outside source evidence and training results')
    output.mkdir(parents=True)
    write_json(output/'manifest.json', manifest)
    write_json(output/'state_pool.json', [vars(case) for case in selected])
    coordinates, directions, candidate_rows = [], [], []
    rollouts = steps = aliases = 0
    started = time.perf_counter()
    scale = float(config['cost']['reward_scale'])
    try:
        for case_index, case in enumerate(selected):
            for policy in policies:
                model, slug = policy.model, policy.source.slug
                actions = policy.actions(case.observations)
                raw = np.stack([actions[mg] for mg in env.agent_ids])
                trace = projection_trace(model, case.state, raw)
                if trace['chain_residual_max'] > 1e-4:
                    raise ValueError('Projection chain-rule check failed')
                actor_metrics = []
                for i, mg in enumerate(env.agent_ids):
                    metric = actor_trace(model, mg, case.state, case.observations[mg], raw,
                                         trace['q1_raw_gradients'][i, 2*i:2*i+2])
                    if np.max(np.abs(metric['actor_forward_actions'] - raw[i])) > 1e-4:
                        raise ValueError('Differentiable Actor forward differs from deployed current action')
                    actor_metrics.append(metric)
                candidates = local_candidates(raw, radii) + direction_candidates(raw, actor_metrics, min(radii))
                predicted = policy.q_values(case.state, candidates)
                evaluated = evaluate_candidates(env, policy, case, candidates, forecast_noise=phase_one['forecast_noise'])
                returns, effective = evaluated['returns'], evaluated['effective'].reshape(len(candidates), -1)
                if abs(returns[0] - baseline[(case.case_id, slug)]) > 1e-7:
                    raise ValueError('Current-policy return differs from phase one')
                rollouts += evaluated['rollouts']
                steps += evaluated['steps']
                aliases += sum(alias is not None for alias in evaluated['duplicate_of'])
                local_rows = {}
                for cid, candidate in enumerate(candidates):
                    if cid == 0:
                        continue
                    i, k = candidate.agent_index, candidate.control_index
                    delta = (candidate.raw - raw).flatten().astype(np.float64)
                    physical_delta = effective[cid] - effective[0]
                    row = dict(case_id=case.case_id, algorithm=slug, origin=case.origin, day=case.day,
                        seed=case.seed, forecast_seed_base=case.forecast_seed_base, hour=case.hour,
                        agent=env.agent_ids[i], control=('ESS' if k == 0 else 'DR') if k is not None else 'joint',
                        candidate_id=cid, kind=candidate.kind, radius=candidate.radius, requested_sign=candidate.sign,
                        actual_raw_delta_norm=float(np.abs(delta).max()),
                        realized_gain_cost=float(scale*(returns[cid]-returns[0])),
                        predicted_gain_cost=float(scale*(predicted[cid,i,0]-predicted[0,i,0])),
                        q2_predicted_gain_cost=float(scale*(predicted[cid,i,1]-predicted[0,i,1])),
                        linear_gain_cost=float(scale*np.dot(trace['q1_raw_gradients'][i], delta)),
                        q1_linear_residual_cost=float(scale*(predicted[cid,i,0]-predicted[0,i,0]) - scale*np.dot(trace['q1_raw_gradients'][i], delta)),
                        physical_delta_norm=float(np.linalg.norm(physical_delta)),
                        physical_linear_residual=float(np.linalg.norm(physical_delta-trace['jacobian']@delta)),
                        duplicate_of='' if evaluated['duplicate_of'][cid] is None else evaluated['duplicate_of'][cid])
                    candidate_rows.append(row)
                    if candidate.kind == 'axis':
                        local_rows.setdefault((i,k), []).append(row)
                    else:
                        directions.append(row.copy())
                for i, mg in enumerate(env.agent_ids):
                    actor = actor_metrics[i]
                    for k, control in enumerate(('ESS','DR')):
                        metric = dict(projection_jacobian_norm=float(np.linalg.norm(trace['jacobian'][:,2*i+k])),
                            tanh_factor=float(actor['tanh_factors'][k]),
                            main_signal=float(abs(actor['raw_gradient'][k])*actor['post_jacobian_norms'][k]),
                            actor_response=float(actor['actor_response'][k]))
                        previous = broad[(case.case_id,slug,mg,control)]
                        opportunity = float(previous['actor_improvement_cost'])
                        classification = classify_coordinate(local_rows[(i,k)], metric,
                            broad_opportunity=opportunity, auxiliary_trigger=actor['auxiliary_trigger'] if k == 0 else 0., tie_cost=.1)
                        coordinates.append(dict(case_id=case.case_id, algorithm=slug, origin=case.origin, day=case.day,
                            seed=case.seed, forecast_seed_base=case.forecast_seed_base, hour=case.hour,
                            agent=mg, control=control, current_raw=float(raw[i,k]),
                            broad_opportunity_cost=opportunity, broad_q1_regret_cost=float(previous['q1_selection_regret_cost']),
                            raw_q1_gradient=float(actor['raw_gradient'][k]),
                            pre_jacobian_norm=float(actor['pre_jacobian_norms'][k]),
                            post_jacobian_norm=float(actor['post_jacobian_norms'][k]),
                            main_actor_gradient_norm=actor['main_gradient_norm'], total_actor_gradient_norm=actor['total_gradient_norm'],
                            auxiliary_trigger_agent=actor['auxiliary_trigger'], chain_residual_max=trace['chain_residual_max'],
                            **metric, **classification))
                _csv(output/'local_candidate_results.csv', candidate_rows)
                _csv(output/'coordinate_results.csv', coordinates)
                _csv(output/'direction_results.csv', directions)
                print(f"Case {case_index+1}/{len(selected)} {case.day} h{case.hour:02d} {NAMES[slug]}: {evaluated['rollouts']}/{len(candidates)} rollouts, elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if any(sha256(path) != digest for path, digest in protected.items()):
            raise ValueError('Source files changed during diagnosis')
        if snapshots != [_tensor_fingerprint(policy.model) for policy in policies]:
            raise ValueError('Model tensors changed during diagnosis')
        summary = dict(status='complete', state_count=len(selected), source_integrity_verified=True,
            model_tensors_unchanged=True, candidate_rollouts=rollouts, physical_aliases=aliases,
            environment_steps_including_prefix=steps, elapsed_seconds=time.perf_counter()-started,
            radii=list(radii), candidate_rows=len(candidate_rows), coordinate_rows=len(coordinates),
            direction_rows=len(directions), models=_summarize(coordinates,directions))
        write_json(output/'summary.json', summary)
        _report(output, summary)
        manifest.update(status='complete', completed_at=datetime.now().astimezone().isoformat(),
                        source_integrity_verified=True, model_tensors_unchanged=True)
        write_json(output/'manifest.json', manifest)
        print(f"Learning-chain diagnosis complete: {output/'report.md'}", flush=True)
        return summary
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output/'manifest.json', manifest)
        raise
