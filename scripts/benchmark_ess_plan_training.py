"""Matched local train-only timing; not a learned-policy performance study."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from algorithms.registry import get_algorithm_spec
from algorithms.matd3_ess_plan_transformer.diagnostics import write_json
from core.reward_v2 import stable_contract_hash
from envs.multi_microgrid_env import MultiMicrogridEnv


def overhead_fraction(on_seconds,off_seconds):
    if not np.isfinite(on_seconds) or not np.isfinite(off_seconds) or off_seconds <= 0:
        raise ValueError('Timing must be finite with a positive baseline')
    return on_seconds/off_seconds-1.


def benchmark_case(config,seed,episodes,output_directory,*,enabled,device='cpu',slug='matd3_ess_plan_transformer'):
    cfg = copy.deepcopy(config)
    cfg['algorithm']['name'] = slug
    cfg['algorithm'].update(learning_starts=128,replay_capacity=4096,warmup_steps=128)
    if slug == 'matd3_ess_plan_transformer':
        cfg['algorithm']['ess_plan']['enabled'] = enabled
    spec = get_algorithm_spec(slug)
    env_cfg = yaml.safe_load((ROOT/'configs/cooperative.yaml').read_text(encoding='utf-8'))
    env = MultiMicrogridEnv(spec.build_environment_config(env_cfg),split='train')
    model = spec.create(env,cfg,device=device,seed=seed)
    batch_size = cfg['algorithm']['batch_size']
    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize()
    def day(index,capture):
        obs,reset = env.reset(seed=seed+index)
        state,done = reset['global_state'],False
        while not done:
            if capture and hasattr(model,'capture_training_context'):
                model.capture_training_context(env,obs,state,dict(episode_index=50+index,day=str(reset['day'])))
            actions = model.select_actions(obs,noise_std=.15)
            following,rewards,done,info = env.step(actions)
            model.replay.add(obs,state,actions,rewards[env.agent_ids[0]],following,info['global_state_after'],done)
            model.update(batch_size)
            obs,state = following,info['global_state_after']
    # Identical warm-up incl. actual update kernels. No expert work is hidden here.
    sync(); warm_start=time.perf_counter()
    for index in range(-6,0):
        day(index,False)
    sync(); warm_seconds=time.perf_counter()-warm_start
    if hasattr(model,'configure_diagnostics'):
        model.configure_diagnostics(Path(output_directory)/'auxiliary_plan',dict(benchmark=True,seed=seed))
    sync(); started=time.perf_counter()
    for index in range(episodes):
        day(index,True)
    sync(); elapsed=time.perf_counter()-started
    summary = model.plan_summary() if hasattr(model,'plan_summary') else {}
    result = dict(slug=slug,enabled=enabled,seed=seed,episodes=episodes,seconds=elapsed,
                  cold_warmup_seconds=warm_seconds,updates=model.update_steps,
                  config_hash=stable_contract_hash(cfg),device=device)
    result.update(summary)
    write_json(Path(output_directory)/'timing.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm-config',type=Path,default=ROOT/'configs/algorithms/matd3_ess_plan_transformer.yaml')
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--pairs',type=int,default=3)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.episodes <= 0 or args.pairs < 3:
        parser.error('Require positive episodes and at least three pairs')
    if args.output.exists():
        parser.error('Output must be a new study directory')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg=yaml.safe_load(args.algorithm_config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True)
    pairs=[]
    for i in range(args.pairs):
        records={}
        for enabled in ([False,True] if i%2 == 0 else [True,False]):
            name='on' if enabled else 'off'
            records[name]=benchmark_case(cfg,42+i,args.episodes,args.output/f'pair_{i}_{name}',enabled=enabled,device=args.device)
            print(f'pair {i} {name}: {records[name]["seconds"]:.3f}s',flush=True)
        pairs.append(dict(on=records['on'],off=records['off'],extra_fraction=overhead_fraction(
            records['on']['seconds'],records['off']['seconds'])))
        write_json(args.output/'pairs.json',pairs)
    baseline=benchmark_case(cfg,42,args.episodes,args.output/'mlp_reference',enabled=False,device=args.device,
                             slug='matd3_mlp_dueling_critic_h24_q_guided')
    adequate=all(p['on']['valid_labels'] > 0 and p['on']['supervision_updates'] > 0 for p in pairs)
    median=float(np.median([p['extra_fraction'] for p in pairs]))
    result=dict(pairs=pairs,median_extra_fraction=median,target_fraction=.2,
                supervision_exercised=adequate,sample_budget_met=adequate and median <= .2,
                mlp_reference=baseline,scope='local matched train-only sample; not 5000-episode guarantee',
                source_config_hash=stable_contract_hash(cfg),torch_version=torch.__version__,
                numpy_version=np.__version__,threads=1)
    write_json(args.output/'summary.json',result)
    print(f'median extra: {median:.2%}; labels+updates: {adequate}',flush=True)
    if not result['sample_budget_met']:
        raise SystemExit('Runtime or supervision acceptance failed; inspect saved records')


if __name__ == '__main__':
    main()
