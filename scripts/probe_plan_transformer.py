from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from algorithms.registry import get_algorithm_spec
from diagnostics.plan_transformer_probe import probe_forecast_sensitivity
from envs.multi_microgrid_env import MultiMicrogridEnv
from utils.torch_runtime import configure_torch_runtime


def fingerprints(directory):
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


def run_probe(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(run) or output.is_relative_to(ROOT/'outputs/releases'):
        raise ValueError('Use a new independent study output directory')
    status = json.loads((run/'run_status.json').read_text(encoding='utf-8'))
    summary = json.loads((run/'test_summary.json').read_text(encoding='utf-8'))
    if status['status'] != 'COMPLETED' or summary['algorithm_slug'] != 'matd3_plan_transformer':
        raise ValueError('Probe requires a completed Plan-Transformer run')
    checkpoint = run/'checkpoints/best_model.pt'
    if Path(summary['checkpoint_used']).name != checkpoint.name or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != summary['checkpoint_sha256']:
        raise ValueError('Probe checkpoint differs from the formal-test checkpoint')
    before = fingerprints(run)
    snapshot = yaml.safe_load((run/'config_snapshot.yaml').read_text(encoding='utf-8'))
    settings = snapshot['algorithm']
    configure_torch_runtime(settings,'cpu')
    env = MultiMicrogridEnv(snapshot['environment'], split='test')
    spec = get_algorithm_spec('matd3_plan_transformer')
    model = spec.create(env, settings, device='cpu', seed=summary['test_seed'])
    model.load(checkpoint)
    rows = [json.loads(x) for x in (run/'auxiliary_plan/formal_test_plan_attention.jsonl').read_text(encoding='utf-8').splitlines()]
    first = rows[0]
    selected = [r for r in rows if r['day']==first['day'] and r['t']==0]
    observations = {r['agent']:np.asarray(r['observations'],dtype=np.float32) for r in selected}
    if set(observations) != set(env.agent_ids) or len(selected) != len(env.agent_ids):
        raise ValueError('Saved initial joint observations are incomplete')
    if any(r['checkpoint_sha256'] != summary['checkpoint_sha256'] for r in selected):
        raise ValueError('Saved observations belong to another checkpoint')
    # The loaded environment supplies schema/config only; no reset or step occurs.
    schema = model.observation_schemas['mg1']
    values = observations['mg1']
    peak = max(range(1,24), key=lambda k:values[schema.index(f'load_t+{k}')])
    neighbour = peak+1 if peak<23 else peak-1
    cases = [dict(name='move_forecast_peak',kind='swap_energy_hours',hours=[peak,neighbour]),
             dict(name='move_forecast_price',kind='swap_price_hours',hours=[peak,neighbour]),
             dict(name='increase_midday_surplus',kind='surplus_pv',hours=[12],agent='mg1')]
    reports = probe_forecast_sensitivity(model,observations,cases)
    if fingerprints(run) != before:
        raise RuntimeError('Source run changed during read-only probe')
    output.mkdir(parents=True)
    report = dict(diagnostic_only=True, source_run=str(run), checkpoint_sha256=summary['checkpoint_sha256'],
                  day=first['day'], t=0, cases=reports, source_unchanged=True)
    (output/'forecast_sensitivity.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (output/'README.md').write_text('Saved-observation sensitivity study.\n\nThis measures action, auxiliary plan, and attention responses to controlled forecast changes. It does not measure realized dispatch benefits or establish causal performance. The source run is unchanged.\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='Inspect saved Plan-Transformer forecast sensitivity')
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run_probe(args.run,args.output)
    print(f'Saved independent sensitivity study: {args.output.resolve()}')


if __name__ == '__main__':
    main()
