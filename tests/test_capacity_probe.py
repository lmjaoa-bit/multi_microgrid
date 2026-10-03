from pathlib import Path
import importlib
import importlib.util
import numpy as np
import pytest
import torch


def probe():
    assert importlib.util.find_spec('diagnostics.capacity_probe') is not None, 'Capacity diagnostic is missing'
    return importlib.import_module('diagnostics.capacity_probe')


def test_h8_mask_copies_and_removes_local_peer_and_validity_future():
    schema=['soc','load_t+1','load_t+9','mg2_energy_balance_t+12','forecast_valid_t+24']
    obs=np.array([.5,.3,.8,.9,1.],dtype=np.float32)
    before=obs.copy()
    masked=probe().h8_observation(obs,schema)
    np.testing.assert_array_equal(masked,np.array([.5,.3,0.,0.,0.],dtype=np.float32))
    np.testing.assert_array_equal(obs,before)


def test_calendar_partition_keeps_whole_days_disjoint_and_ignores_outcomes():
    dates=[f'2018-{m:02d}-{d:02d}' for m in range(1,13) for d in range(1,22)]
    fit,dev=probe().training_dates(dates)
    assert len(fit)==24 and len(dev)==12 and not set(fit)&set(dev)
    confirm=probe().confirmation_dates(dates,excluded=fit+dev)
    assert len(confirm)==12 and not set(confirm)&set(fit+dev)


def test_critic_forecast_tokens_preserve_all_fields_and_action_derivatives():
    schema=['soc','mg1_net_load_t+1','forecast_valid_t+1','mg1_net_load_t+2','forecast_valid_t+2']
    net=probe().SequenceValue(schema,action_dim=2,embed=8,heads=2,layers=1,feedforward=16)
    x=torch.tensor([[.3,.2,1.,.5,1.,.1,.2]],requires_grad=True)
    value=net(x)
    assert value.shape==(1,)
    grad=torch.autograd.grad(value.sum(),x)[0]
    assert torch.isfinite(value).all() and torch.isfinite(grad).all()
    assert grad[0,-2:].abs().sum()>0
    covered=net.current_indices.tolist()+net.forecast_indices.flatten().tolist()
    assert sorted(covered)==list(range(5))


def test_grouped_value_metrics_keep_baseline_and_do_not_sum_candidate_opportunities():
    truth=np.array([-10.,-9.,-11.,-20.,-19.,-21.])
    pred=np.array([-10.,-11.,-9.,-20.,-21.,-19.])
    groups=['day1_a']*3+['day2_b']*3
    controls=['current','mg1_ESS','mg1_ESS']*2
    result=probe().value_metrics(truth,pred,groups,controls,reward_scale=100.,tie_cost=1.)
    assert result['mean_selection_regret_cost']==pytest.approx(200.)
    assert result['mean_oracle_gain_cost']==pytest.approx(100.)
    assert result['rank_accuracy']==0.


def test_scaler_is_fitted_on_training_only_and_constant_columns_are_finite():
    train=np.array([[1.,2.],[3.,2.]],dtype=np.float32)
    scaler=probe().TrainScaler.fit(train)
    assert scaler.mean.tolist()==[2.,2.]
    assert np.isfinite(scaler.apply(np.array([[100.,2.]],dtype=np.float32))).all()
    assert scaler.mean.tolist()==[2.,2.]


@pytest.fixture(scope='module')
def real_t_actor():
    from diagnostics.capacity_data import sources_and_config
    root=Path(__file__).resolve().parents[1]
    path=root/'outputs/diagnostics/critic_probe/expanded_24d_2seeds_2026-10-02'
    if not (path/'manifest.json').exists():pytest.skip('Original best-model evidence absent')
    _,_,_,_,policies=sources_and_config(root,path)
    return policies[1].model.actors['mg2']


def test_semantic_mlp_and_current_t_have_same_lossy_renewable_forecast_information(real_t_actor):
    actor=real_t_actor
    obs=torch.zeros(1,len(actor.schema))
    names={n:i for i,n in enumerate(actor.schema)}
    obs[:,names['forecast_valid_t+1']]=1.
    obs[:,names['pv_t+1']]=.4
    obs[:,names['wind_t+1']]=.3
    changed=obs.clone()
    scales=actor.power_normalizer_tensor
    changed[:,names['pv_t+1']]+=.01
    changed[:,names['wind_t+1']]-=.01*scales[1]/scales[2]
    features=probe().SemanticFeatures(actor)
    with torch.no_grad():
        torch.testing.assert_close(features(obs),features(changed),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(actor._build_forecast_features(obs),actor._build_forecast_features(changed),atol=1e-6,rtol=1e-6)
    assert not torch.equal(obs[:,probe().h8_indices(actor.schema)],changed[:,probe().h8_indices(actor.schema)])


def test_both_flat_students_match_current_t_parameter_budget(real_t_actor):
    target=sum(p.numel() for p in real_t_actor.parameters())
    for semantic in (False,True):
        net=probe().FlatStudent(real_t_actor.schema,real_t_actor,semantic)
        count=sum(p.numel() for p in net.parameters())
        assert abs(count-target)/target<.015
        assert net(torch.zeros(2,len(real_t_actor.schema))).shape==(2,2)


def test_semantic_features_match_every_t_encoder_input_and_do_not_add_raw_clock(real_t_actor):
    actor=real_t_actor
    obs=torch.randn(2,len(actor.schema))*.1
    obs[:,actor.hour_idx_tensor]=torch.tensor([[.3,.7],[.6,.4]])
    for k in range(1,9):obs[:,actor.schema.index(f'forecast_valid_t+{k}')]=0.
    captured={}
    hooks=[]
    for name in ('ess_encoder','dr_encoder','supply_encoder','trade_encoder','peer_encoder','forecast_encoder'):
        hooks.append(getattr(actor,name).register_forward_pre_hook(lambda mod,args,n=name:captured.update({n:args[0].detach().clone()})))
    with torch.no_grad():
        actor._build_tokens(obs)
        expected=torch.cat([captured[n].flatten(1) for n in ('ess_encoder','dr_encoder','supply_encoder','trade_encoder','peer_encoder','forecast_encoder')],-1)
        feature=probe().SemanticFeatures(actor)
        torch.testing.assert_close(feature(obs),expected)
        changed=obs.clone();changed[:,actor.hour_idx_tensor]=0.
        torch.testing.assert_close(feature(obs),feature(changed))
    for hook in hooks:hook.remove()


def test_capacity_output_cannot_be_a_new_child_of_original_audit(tmp_path):
    from diagnostics.capacity_data import prepare_capacity_data
    audit=tmp_path/'completed_audit';audit.mkdir()
    with pytest.raises(ValueError,match='separate diagnostic'):
        prepare_capacity_data(tmp_path,audit,audit/'new_child')


def test_semantic_forecast_has_identical_nonfinite_handling(real_t_actor):
    actor=real_t_actor
    obs=torch.zeros(2,len(actor.schema))
    for k in range(1,9):obs[:,actor.schema.index(f'forecast_valid_t+{k}')]=1.
    obs[:,actor.schema.index('load_t+1')]=float('inf')
    obs[:,actor.all_peer_forecast_indices[0]]=float('nan')
    with torch.no_grad():
        features=probe().SemanticFeatures(actor)(obs)
        expected=actor._build_forecast_features(obs).flatten(1)
    torch.testing.assert_close(features[:,-expected.shape[1]:],expected)
