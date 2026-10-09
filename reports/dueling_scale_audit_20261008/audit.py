"""Read-only V/A scale diagnosis of the fixed episode-1900 checkpoint.

No optimizer step, replay mutation, training, or environment/config edits.
Run with MKL_THREADING_LAYER=SEQUENTIAL and OMP_NUM_THREADS=1.
"""
from pathlib import Path
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'reports/dr_critic_audit_20261007'
spec = importlib.util.spec_from_file_location('other_dr', PARENT / 'audit_other_dr.py')
other = importlib.util.module_from_spec(spec)
spec.loader.exec_module(other)
base = other.base
np, pd, torch = base.np, base.pd, base.torch


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    source = other.verify_source()
    manifest = json.loads((PARENT / 'source_manifest.json').read_text())
    for name, expected in manifest['additional_source_hashes'].items():
        assert base.digest(ROOT / name) == expected
    env, model, config = base.load()
    assert model.gamma == 1., 'Return-to-go calculation assumes gamma=1'
    model_hash = {name: base.digest(base.OUT / 'source_snapshot' / name) for name in source['snapshot_hashes']}
    originals = pd.concat([
        pd.read_csv(base.OUT / 'counterfactual_probes.csv').assign(agent='mg2'),
        pd.read_csv(PARENT / 'counterfactual_probes.csv')], ignore_index=True)
    wanted = {(mg, date, int(hour)): g for (mg, date, hour), g in originals.groupby(['agent','date','hour'])}
    states, effective_actions, meta, candidate_rows = [], [], [], []
    checks = dict(candidate_v_spread_max=0., candidate_q_match_max=0.,
                  candidate_dq_minus_dadv_max=0., action_gradient_q_minus_adv_max=0.)
    days = [str(x.date()) for x in env.data.valid_days]
    for day_idx, day in enumerate(days):
        obs, _ = env.reset(day=day, seed=base.SEED + day_idx, deterministic=True, forecast_noise=True)
        while True:
            hour = env.t
            actions = model.select_actions(obs, noise_std=0.)
            state = env.global_state().copy()
            raw = torch.tensor(np.stack([actions[mg] for mg in model.agent_ids])[None])
            with torch.no_grad():
                eff, _ = model.project_actions(torch.tensor(state[None]), raw)
            states.append(state); effective_actions.append(eff.flatten(1)[0].numpy())
            for mg in model.agent_ids:
                key = (mg, day, hour)
                if key not in wanted:
                    continue
                group = wanted[key].copy().reset_index(drop=True)
                idx = model.agent_ids.index(mg)
                raws = raw.repeat(len(group)+1, 1, 1)
                raws[1:, idx, 1] = torch.tensor(group.candidate_raw.to_numpy(), dtype=torch.float32)
                ss = torch.tensor(np.repeat(state[None], len(raws), axis=0))
                with torch.no_grad():
                    aa, _ = model.project_actions(ss, raws)
                    parts = model.critics[mg].decompose(ss, aa.flatten(1))
                values = {k: v[:,0].numpy() for k,v in parts.items()}
                checks['candidate_v_spread_max'] = max(checks['candidate_v_spread_max'],
                    *(float(np.ptp(values[k])) for k in ['v1','v2']))
                for branch in [1,2]:
                    dq = values[f'q{branch}'][1:] - values[f'q{branch}'][0]
                    da = values[f'adv{branch}'][1:] - values[f'adv{branch}'][0]
                    checks['candidate_dq_minus_dadv_max'] = max(checks['candidate_dq_minus_dadv_max'], float(abs(dq-da).max()))
                    checks['candidate_q_match_max'] = max(checks['candidate_q_match_max'], float(abs(dq-group[f'dq{branch}'].to_numpy()).max()))
                    group[f'dadv{branch}'] = da
                    group[f'v{branch}'] = values[f'v{branch}'][1:]
                    group[f'adv{branch}'] = values[f'adv{branch}'][1:]
                    group[f'new_dq{branch}'] = dq
                candidate_rows.extend(group.to_dict('records'))
            obs, rewards, done, _ = env.step(actions)
            meta.append(dict(date=day, hour=hour, reward=float(rewards['mg1'])))
            if done:
                break
        if (day_idx+1) % 10 == 0:
            print(f'replayed {day_idx+1}/{len(days)} days', flush=True)
    state_tensor = torch.tensor(np.stack(states))
    action_tensor = torch.tensor(np.stack(effective_actions))
    timeline = pd.DataFrame(meta)
    timeline['return_to_go'] = timeline.groupby('date').reward.transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    baselines = []
    for mg_idx, mg in enumerate(model.agent_ids):
        arrays = {k: [] for k in ['v1','v2','adv1','adv2','q1','q2','grad_dr']}
        for start in range(0,len(states),128):
            ss = state_tensor[start:start+128]
            aa = action_tensor[start:start+128].clone().requires_grad_(True)
            parts = model.critics[mg].decompose(ss, aa)
            grad_q = torch.autograd.grad(parts['q1'].sum(), aa, retain_graph=True)[0]
            grad_a = torch.autograd.grad(parts['adv1'].sum(), aa)[0]
            checks['action_gradient_q_minus_adv_max'] = max(checks['action_gradient_q_minus_adv_max'], float(abs(grad_q-grad_a).max()))
            for k in arrays:
                value = grad_q[:,mg_idx*3+2] if k == 'grad_dr' else parts[k][:,0]
                arrays[k].extend(value.detach().numpy().tolist())
        baselines.append(timeline.assign(agent=mg, **arrays))
    baseline = pd.concat(baselines, ignore_index=True)
    candidates = pd.DataFrame(candidate_rows)
    candidates['power_delta_kw'] = candidates.candidate_kw-candidates.reference_kw
    assert len(baseline) == 3960 and len(candidates) == len(originals) == 1232
    assert checks['candidate_q_match_max'] < 2e-5
    assert checks['candidate_dq_minus_dadv_max'] < 2e-5
    assert checks['action_gradient_q_minus_adv_max'] == 0
    assert checks['candidate_v_spread_max'] < 2e-5
    scale_stats, delta_stats = [], []
    for mg, g in baseline.groupby('agent'):
        for branch in [1,2]:
            v, a, q = (g[f'{k}{branch}'] for k in ['v','adv','q'])
            scale_stats.append(dict(agent=mg, branch=branch, hours=len(g),
                mean_abs_v=float(v.abs().mean()), mean_abs_adv=float(a.abs().mean()),
                mean_abs_adv_over_mean_abs_v=float(a.abs().mean()/v.abs().mean()),
                median_abs_v=float(v.abs().median()), median_abs_adv=float(a.abs().median()),
                adv_std=float(a.std(ddof=0)), q_std=float(q.std(ddof=0)),
                adv_std_over_q_std=float(a.std(ddof=0)/q.std(ddof=0)),
                mc_q_rmse=float(np.sqrt(((q-g.return_to_go)**2).mean())),
                dr_effective_action_gradient_rms=float(np.sqrt((g.grad_dr**2).mean())) if branch == 1 else None))
        local = candidates[(candidates.agent == mg)&np.isclose(abs(candidates.power_delta_kw),1.,atol=1e-5,rtol=0)]
        assert len(local) in [50,95,48]
        meaningful = local[local.gain.abs() > .001]
        for branch in [1,2]:
            da = local[f'dadv{branch}']; error = da-local.gain
            delta_stats.append(dict(agent=mg, branch=branch, local_pairs=len(local), meaningful_pairs=len(meaningful),
                median_abs_actual_delta=float(local.gain.abs().median()),
                median_abs_predicted_delta=float(da.abs().median()),
                median_abs_prediction_error=float(error.abs().median()),
                predicted_under_actual_fraction=float((da.abs()<local.gain.abs()).mean()),
                meaningful_sign_wrong=int(sum(meaningful[f'dadv{branch}']*meaningful.gain < 0)),
                meaningful_predicted_near_zero=int(sum(meaningful[f'dadv{branch}'].abs() < 1e-5)),
                prediction_worse_than_zero_in_local_mse=bool((error**2).mean() > (local.gain**2).mean()),
                local_prediction_mse=float((error**2).mean()), zero_prediction_mse=float((local.gain**2).mean())))
    history_path = Path(source['source_run']) / 'artifacts/training_history.json'
    history_content = history_path.read_bytes()
    history_sha = base.digest(history_path)
    assert base.hashlib.sha256(history_content).hexdigest() == history_sha
    history = json.loads(history_content)
    training = []
    for h in history:
        for mg, record in h.get('actor_diagnostics', {}).items():
            if 'critic_adv_abs_mean' in record:
                training.append(dict(episode=h['episode'], agent=mg, **{k:v for k,v in record.items() if k.startswith('critic_') or k in ['dr_saturation_rate','dr_action_grad_rms']}))
    training_frame = pd.DataFrame(training)
    training_stats = []
    for low, high in [(1,100),(1600,1900),(4700,5000)]:
        for mg, g in training_frame[training_frame.episode.between(low,high)].groupby('agent'):
            training_stats.append(dict(episode_range=[low,high], agent=mg, diagnostic_episodes=len(g),
                median_adv_abs_mean=float(g.critic_adv_abs_mean.median()),
                median_abs_v_mean=float(g.critic_v_mean.abs().median()),
                median_adv_share=float(g.critic_adv_share.median()),
                median_dr_action_gradient_rms=float(g.critic_dqda_dr_rms.median()),
                median_actor_dr_saturation_rate=float(g.dr_saturation_rate.median())))
    baseline.to_csv(OUT/'baseline_decomposition.csv',index=False)
    candidates.to_csv(OUT/'candidate_decomposition.csv',index=False)
    training_frame.to_csv(OUT/'training_diagnostics.csv',index=False)
    other.verify_source()
    assert all(base.digest(base.OUT/'source_snapshot'/name) == expected for name,expected in model_hash.items())
    summary = dict(checkpoint_episode=1900, checkpoint_sha256=source['checkpoint_sha256'],
        days=55, scale_stats=scale_stats, local_delta_stats=delta_stats,
        training_trend=training_stats, last_training_episode=history[-1]['episode'], checks=checks,
        caveats=['Frozen episode-1900 model; late training rows are historical diagnostics of evolving models, not this snapshot.',
                 'V is Q at the zero EFFECTIVE joint action, not necessarily a feasible or policy-average baseline.',
                 'Advantage is centred relative to the joint zero reference, includes all agents/devices, and depends on reference choice.',
                 'Candidate state sample is stratified, not random. Local true gain is deterministic online-policy return difference; training target is smoothed.',
                 'Training diagnostic episode averages are recorded sample statistics, not all updates. No saved replay/optimizer/gradient trajectory is available.',
                 'No retraining or ablation performed; scale observations do not establish that V dominance caused advantage errors.'])
    write_json('audit_summary.json',summary)
    write_json('source_manifest.json',dict(parent_manifest=str(base.OUT/'source_manifest.json'),
        checkpoint_sha256=source['checkpoint_sha256'], training_history_sha256=history_sha,
        training_history_last_episode=history[-1]['episode'], diagnostic_script_sha256=base.digest(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):base.digest(p) for p in [base.OUT/'counterfactual_probes.csv', PARENT/'counterfactual_probes.csv']},
        learning_updates=0, source_hashes_unchanged=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
