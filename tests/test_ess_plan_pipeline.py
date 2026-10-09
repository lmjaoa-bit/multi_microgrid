import pytest
from pathlib import Path
import subprocess
import sys
import yaml

from scripts.benchmark_ess_plan_training import overhead_fraction, benchmark_case
from algorithms.registry import get_algorithm_spec


def test_runtime_overhead_includes_generated_label_cost():
    assert overhead_fraction(120.,100.) == pytest.approx(.2)
    with pytest.raises(ValueError):
        overhead_fraction(1.,0.)


def test_real_benchmark_emits_disabled_supervision_record(tmp_path):
    import torch
    root=Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    cfg=yaml.safe_load((root/'configs/smoke/matd3_ess_plan_transformer.yaml').read_text(encoding='utf-8'))
    cfg['algorithm']['network'].update(embed_dim=8,num_heads=2,num_layers=1,hidden_dim=16)
    cfg['algorithm']['critic_network']['value'].update(embed_dim=8,num_heads=2,num_layers=1,feedforward_dim=16)
    cfg['algorithm']['critic_network']['advantage']['hidden_dim']=16
    record=benchmark_case(cfg,42,1,tmp_path,enabled=False)
    assert record['seconds'] > 0 and record['enabled'] is False
    assert record['teacher_calls'] == record['supervision_updates'] == 0
    assert (tmp_path/'timing.json').is_file()


def test_plugin_config_and_smoke_use_isolated_slug():
    root=Path(__file__).resolve().parents[1]
    spec=get_algorithm_spec('matd3_ess_plan_transformer')
    formal=yaml.safe_load(spec.default_config_path.read_text(encoding='utf-8'))
    smoke=yaml.safe_load((root/'configs/smoke/matd3_ess_plan_transformer.yaml').read_text(encoding='utf-8'))
    assert formal['algorithm']['name'] == smoke['algorithm']['name'] == spec.slug
    assert formal['algorithm']['network']['forecast_horizon'] == 24
    assert formal['algorithm']['ess_plan']['teacher_fraction'] == .15
    assert smoke['training']['episodes'] == 3 and smoke['training']['test_episodes'] == 2
    assert smoke['algorithm']['ess_plan']['enabled']


def test_new_plugin_is_independent_of_removable_predecessors():
    root=Path(__file__).resolve().parents[1]
    script='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname.startswith(('algorithms.matd3_plan_transformer','algorithms.matd3_mlp_dueling_critic_h24_q_guided')):
   raise ImportError('predecessor unavailable')
sys.meta_path.insert(0,Block())
from algorithms.registry import get_algorithm_spec
assert get_algorithm_spec('matd3_ess_plan_transformer').algorithm_class.SLUG == 'matd3_ess_plan_transformer'
assert get_algorithm_spec('matd3').slug == 'matd3_dispatch'
assert get_algorithm_spec('matd3_hierarchical_transformer').slug == 'matd3_hierarchical_transformer'
'''
    subprocess.run([sys.executable,'-c',script],cwd=root,check=True,capture_output=True,text=True)
