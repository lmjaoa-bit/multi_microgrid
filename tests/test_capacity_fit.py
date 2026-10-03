from pathlib import Path
import importlib
import importlib.util
import numpy as np
import pytest
import torch


def fitting():
    assert importlib.util.find_spec('diagnostics.capacity_fit') is not None
    return importlib.import_module('diagnostics.capacity_fit')


def test_date_membership_and_all_four_partitions_are_checked():
    manifest=dict(fit_days=['a'],dev_days=['b'],confirmation_days=['c'],audit_days=['d'])
    blocks={k:{'day':np.array([v])} for k,v in [('fit','a'),('dev','b'),('confirmation','c'),('audit','d')]}
    fitting().validate_partition_membership(manifest,blocks)
    blocks['fit']['day']=np.array(['c'])
    with pytest.raises(ValueError,match='date'):
        fitting().validate_partition_membership(manifest,blocks)
    manifest['audit_days']=['a']
    with pytest.raises(ValueError,match='overlap'):
        fitting().validate_partition_membership(manifest,{})


def test_gain_error_uses_each_case_own_baseline_and_original_agent_column():
    block=dict(y=np.array([-10.,-9.,-11.,-20.,-19.]),case_id=np.array(['a','a','a','b','b']),
               control=np.array(['current','mg1_ESS','mg2_DR','current','mg2_DR']))
    pred=np.array([[-8.,-7.,-6.],[-7.,-7.,-6.],[-8.,-8.,-6.],[-10.,-12.,-14.],[-10.,-11.,-14.]])
    assert fitting().gain_mae(block,pred,reward_scale=100.)==pytest.approx(0.)


def test_value_supervision_learns_action_gain_and_keeps_labels_train_only():
    from diagnostics.capacity_probe import DenseValue
    rng=np.random.default_rng(1)
    state=rng.uniform(-1,1,(120,1)).astype(np.float32)
    action=rng.uniform(-1,1,(120,1)).astype(np.float32)
    x=np.c_[state,action];base_x=np.c_[state,np.zeros_like(action)]
    y=-(state[:,0]**2)+(action[:,0]*.6)
    train=dict(x=x,y=y,base_x=base_x,base_y=-(state[:,0]**2))
    model,history=fitting().fit_value_supervised(DenseValue(2,16),train,train,seed=17,learning_rate=.01,
        updates=350,batch_size=64,device='cpu',evaluate_every=50)
    with torch.no_grad():
        pred=model(torch.tensor(x));base=model(torch.tensor(base_x))
    assert np.mean(np.abs((pred-base).numpy()-.6*action[:,0]))<.025
    assert history[-1]['training_loss']<history[0]['training_loss']
    assert torch.isfinite(model(torch.tensor([[100.,1.]]))).all()


def test_day_pairing_does_not_count_seeds_as_independent_dates():
    rows=[dict(date=d,seed=s,cost=x) for d,x in [('a',10.),('b',20.)] for s in (1,2)]
    other=[dict(date=d,seed=s,cost=x) for d,x in [('a',9.),('b',19.)] for s in (1,2)]
    result=fitting().paired_day_difference(other,rows,field='cost')
    assert result['date_count']==2
    assert result['mean_difference']==pytest.approx(-1.)
    assert result['descriptive_95_interval']==pytest.approx([-1.,-1.])


def test_diagnostic_writer_retries_transient_windows_file_lock(tmp_path,monkeypatch):
    module=fitting();attempts=[]
    original=module._write_json
    def temporary_lock(path,value):
        attempts.append(1)
        if len(attempts)<3:raise PermissionError('temporary lock')
        original(path,value)
    monkeypatch.setattr(module,'_write_json',temporary_lock)
    module.write_json(tmp_path/'check.json',{'complete':True})
    assert len(attempts)==3
    assert 'true' in (tmp_path/'check.json').read_text()


def test_actor_supervision_can_recover_saturated_intents_without_changing_teacher():
    from torch import nn
    class ToyActor(nn.Module):
        def __init__(self):
            super().__init__();self.net=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,2))
        def action_logits(self,x):return self.net(x)
        def forward(self,x):return self.action_logits(x).tanh()
    rng=np.random.default_rng(2)
    obs=rng.normal(size=(100,3)).astype(np.float32)
    target=np.stack([obs[:,0]*3.,obs[:,1]*2.],1)[:,None,:]
    before=target.copy()
    actors=nn.ModuleDict({'mg1':ToyActor()})
    actors,history=fitting().fit_actor_supervised(actors,{'mg1':obs},{'mg1':obs},target,target,
        seed=17,learning_rate=.01,updates=450,batch_size=64,device='cpu',evaluate_every=50)
    prediction=fitting().predict_actors(actors,{'mg1':obs})
    assert np.sqrt(np.mean((prediction-np.tanh(target))**2))<.04
    np.testing.assert_array_equal(target,before)
    assert history[-1]['training_loss']<history[0]['training_loss']


def test_state_pool_matches_protected_date_state_and_observations_and_records_hash(tmp_path):
    module=fitting()
    manifest=dict(fit_days=['a'],dev_days=['b'],confirmation_days=['c'],audit_days=['d'],state_schema=['s'])
    record=dict(case_id='one',split='fit',day='a',seed=1,origin='p',state=[.3],
                observations={'mg1':[.8],'mg2':[.7],'mg3':[.6]})
    path=tmp_path/'state_pool.json'
    module.write_json(path,[record])
    values={'fit':dict(case_id=np.array(['one']),day=np.array(['a']),x=np.array([[.3,.4]]))}
    actors={'fit':dict(day=np.array(['a']),seed=np.array([1]),origin=np.array(['p']),state=np.array([[.3]]),
        mg1=np.array([[.8]]),mg2=np.array([[.7]]),mg3=np.array([[.6]]))}
    module.validate_state_pool(tmp_path,manifest,values,actors)
    assert (tmp_path/'state_pool_integrity.json').exists()
    record['observations']['mg1']=[.9]
    module.write_json(path,[record])
    with pytest.raises(ValueError,match='state pool changed'):
        module.validate_state_pool(tmp_path,manifest,values,actors)


def test_state_pool_cannot_pair_another_day_to_value_returns(tmp_path):
    module=fitting()
    manifest=dict(fit_days=['a'],dev_days=['b'],confirmation_days=['c'],audit_days=['d'],state_schema=['s'])
    record=dict(case_id='one',split='fit',day='b',seed=1,origin='p',state=[.3],
                observations={'mg1':[.8],'mg2':[.7],'mg3':[.6]})
    module.write_json(tmp_path/'state_pool.json',[record])
    values={'fit':dict(case_id=np.array(['one']),day=np.array(['a']),x=np.array([[.3,.4]]))}
    with pytest.raises(ValueError,match='date'):
        module.validate_state_pool(tmp_path,manifest,values,{'fit':{}})
