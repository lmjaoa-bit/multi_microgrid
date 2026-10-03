"""Fixed-data representation probes. These are diagnostic networks, not RL plugins."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
from diagnostics.critic_probe import score_choices

LEAD = re.compile(r't\+(\d+)$')


def h8_observation(observation, schema):
    result = np.asarray(observation, dtype=np.float32).copy()
    for i, name in enumerate(schema):
        match = LEAD.search(name)
        if match and int(match.group(1)) > 8:
            result[..., i] = 0.
    return result


def h8_indices(schema):
    return [i for i, name in enumerate(schema) if not LEAD.search(name) or int(LEAD.search(name).group(1)) <= 8]


def training_dates(dates):
    months = sorted({str(d)[:7] for d in dates})
    if len(months) != 12:
        raise ValueError('Require all twelve months')
    fit, dev = [], []
    for month in months:
        available = sorted({str(d) for d in dates if str(d).startswith(month)})
        if len(available) < 4:
            raise ValueError('Insufficient calendar dates')
        picks = [available[int(k * len(available)/4)] for k in (1,2,3)]
        fit.extend([picks[0],picks[2]])
        dev.append(picks[1])
    return fit, dev


def confirmation_dates(dates, excluded):
    available = sorted({str(d) for d in dates} - set(excluded))
    months = sorted({str(d)[:7] for d in dates})
    result = []
    for month in months:
        choices = [d for d in available if d.startswith(month)]
        if not choices:
            raise ValueError('No unreviewed confirmation date in month')
        result.append(choices[len(choices)//2])
    return result


@dataclass
class TrainScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, training):
        values = np.asarray(training,dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError('Expected finite training matrix')
        return cls(values.mean(0),np.maximum(values.std(0),1e-3))

    def apply(self, values):
        return (values-self.mean)/self.std


class DenseValue(nn.Module):
    def __init__(self, input_dim, width=128):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(input_dim,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))

    def forward(self,x):
        return self.network(x).squeeze(-1)


class SequenceValue(nn.Module):
    """Full state is partitioned losslessly into current/action and time tokens."""
    def __init__(self,schema,action_dim=9,embed=48,heads=4,layers=2,feedforward=128):
        super().__init__()
        future={}
        current=[]
        for i,name in enumerate(schema):
            match=LEAD.search(name)
            if match:future.setdefault(int(match.group(1)),[]).append(i)
            else:current.append(i)
        keys=sorted(future)
        if not keys or len({len(future[k]) for k in keys})!=1:
            raise ValueError('Forecast tokens must have equal width')
        self.register_buffer('current_indices',torch.tensor(current,dtype=torch.long))
        self.register_buffer('forecast_indices',torch.tensor([future[k] for k in keys],dtype=torch.long))
        self.state_dim=len(schema)
        self.current=nn.Sequential(nn.Linear(len(current)+action_dim,embed),nn.LayerNorm(embed),nn.GELU())
        self.forecast=nn.Linear(len(future[keys[0]]),embed)
        self.position=nn.Parameter(torch.zeros(len(keys),embed))
        nn.init.normal_(self.position,std=.02)
        block=nn.TransformerEncoderLayer(embed,heads,feedforward,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.Linear(embed*2,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,x):
        state=x[:,:self.state_dim]
        current=self.current(torch.cat([state.index_select(-1,self.current_indices),x[:,self.state_dim:]],-1))
        f=state[:,self.forecast_indices]
        tokens=torch.cat([current[:,None],self.forecast(f)+self.position[None]],1)
        out=self.transformer(tokens)
        return self.head(torch.cat([out[:,0],out[:,1:].mean(1)],-1)).squeeze(-1)


def value_metrics(truth,predicted,groups,controls,*,reward_scale,tie_cost=.1):
    truth=np.asarray(truth,dtype=np.float64)
    predicted=np.asarray(predicted,dtype=np.float64)
    groups=np.asarray(groups);controls=np.asarray(controls)
    metrics=[]
    for group in sorted(set(groups)):
        group_rows=np.flatnonzero(groups==group)
        base=group_rows[controls[group_rows]=='current']
        if len(base)!=1:raise ValueError('Exactly one baseline required per case')
        for control in sorted(set(controls[group_rows])-{'current'}):
            rows=np.r_[base,group_rows[controls[group_rows]==control]]
            selected_prediction=predicted[rows]
            if predicted.ndim==2:
                column=int(str(control).split('_')[0].replace('mg',''))-1
                selected_prediction=selected_prediction[:,column]
            metrics.append(score_choices(truth[rows],selected_prediction,reward_scale=reward_scale,tie_cost=tie_cost))
    if not metrics:raise ValueError('No coordinate groups')
    correct=sum(r['rank_correct_count'] for r in metrics)
    count=sum(r['rank_pair_count'] for r in metrics)
    return dict(group_count=len(metrics),mean_selection_regret_cost=float(np.mean([r['selection_regret_cost'] for r in metrics])),
        mean_oracle_gain_cost=float(np.mean([r['actor_improvement_cost'] for r in metrics])),
        rank_accuracy=correct/count if count else None,rank_correct_count=correct,rank_pair_count=count,
        value_mae_cost=float(np.mean(np.abs((truth[:,None] if predicted.ndim==2 else truth)-predicted))*reward_scale))


class SemanticFeatures(nn.Module):
    """Exactly the non-learning inputs consumed by current T encoders and skips."""
    def __init__(self,actor):
        super().__init__()
        self.schema=actor.schema
        self.forecast_horizon=actor.forecast_horizon
        self.dr_type=actor.dr_type
        self.ev_cohort_mode=actor.ev_cohort_mode
        self.num_peers=actor.num_peers
        for name in ('ess_idx_tensor','dr_idx_tensor','supply_idx_tensor','trade_idx_tensor',
                     'all_peer_indices','all_forecast_indices','all_peer_forecast_indices',
                     'forecast_load_indices','hour_idx_tensor','power_normalizer_tensor','battery_static',
                     'temp_mask','step_sin','step_cos'):
            self.register_buffer(name,getattr(actor,name).detach().clone())
        self.peer_dim=actor.peer_dim
        self.sys_index=self.schema.index('previous_system_p2p')

    def forward(self,obs):
        b=len(obs)
        ess=obs.index_select(-1,self.ess_idx_tensor)
        dr=obs.index_select(-1,self.dr_idx_tensor)
        if self.dr_type=='ev_charging' and not self.ev_cohort_mode:
            dr=torch.cat([dr,(dr[:,2:3]/dr[:,3:4].clamp_min(1.))/100.],-1)
        supply=obs.index_select(-1,self.supply_idx_tensor)
        scales=self.power_normalizer_tensor
        balance=-(supply[:,0:1]*scales[0]-supply[:,1:2]*scales[1]-supply[:,2:3]*scales[2])/scales[3]
        supply=torch.cat([supply,balance,torch.sign(balance)],-1)
        trade=obs.index_select(-1,self.trade_idx_tensor)
        trade=torch.cat([trade,1-obs[:,self.sys_index:self.sys_index+1].clamp(0,1)],-1)
        peer=obs.index_select(-1,self.all_peer_indices)
        raw=obs.index_select(-1,self.all_forecast_indices).reshape(b,self.forecast_horizon,6)
        raw=torch.nan_to_num(raw,nan=0.,posinf=0.,neginf=0.)
        valid=raw[:,:,5:6].clamp(0,1)
        net=(raw[:,:,0:1]*scales[0]-raw[:,:,1:2]*scales[1]-raw[:,:,2:3]*scales[2])/scales[3]
        hour=obs.index_select(-1,self.hour_idx_tensor)
        sin0=hour[:,0:1,None];cos0=hour[:,1:2,None]
        sin_k=sin0*self.step_cos+cos0*self.step_sin
        cos_k=cos0*self.step_cos-sin0*self.step_sin
        peer_future=obs.index_select(-1,self.all_peer_forecast_indices).reshape(b,self.forecast_horizon,self.num_peers)
        peer_future=torch.nan_to_num(peer_future,nan=0.,posinf=0.,neginf=0.)
        base_load=obs.index_select(-1,self.forecast_load_indices).unsqueeze(-1)
        base_load=torch.nan_to_num(base_load,nan=0.,posinf=0.,neginf=0.)
        forecast=torch.cat([net,raw[:,:,3:4],raw[:,:,4:5]*self.temp_mask,valid,sin_k,cos_k,peer_future,base_load],-1)
        forecast=torch.where((valid>=.5).expand_as(forecast),forecast,torch.zeros_like(forecast))
        return torch.cat([ess,self.battery_static[None].expand(b,-1),dr,supply,trade,peer,forecast.flatten(1)],-1)


def matched_width(input_dim,target_params,outputs=2):
    return min(range(16,513),key=lambda w:abs(w*w+(input_dim+outputs+2)*w+outputs-target_params))


class FlatStudent(nn.Module):
    def __init__(self,schema,reference_actor,semantic=False):
        super().__init__()
        self.semantic=semantic
        if semantic:
            self.features=SemanticFeatures(reference_actor)
            with torch.no_grad():width=self.features(torch.zeros(1,len(schema))).shape[-1]
        else:
            self.register_buffer('indices',torch.tensor(h8_indices(schema),dtype=torch.long))
            width=len(self.indices)
        target=sum(p.numel() for p in reference_actor.parameters())
        hidden=matched_width(width,target)
        self.network=nn.Sequential(nn.Linear(width,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,2))

    def action_logits(self,obs):
        return self.network(self.features(obs) if self.semantic else obs.index_select(-1,self.indices))

    def forward(self,obs,collect_diagnostics=False):
        return self.action_logits(obs).tanh()


class MaskedTeacher:
    def __init__(self,policy):
        self.policy=policy
        self.source=policy.source
        self.gamma=policy.gamma
        self.ids=policy.ids

    def actions(self,observations):
        obs={mg:h8_observation(observations[mg],self.policy.model.observation_schemas[mg]) for mg in self.ids}
        return self.policy.actions(obs)

    @torch.no_grad()
    def logits(self,mg,observations):
        actor=self.policy.model.actors[mg]
        obs=h8_observation(observations,actor_schema(self.policy,mg))
        values=torch.as_tensor(obs,device=self.policy.model.device,dtype=torch.float32)
        return actor.network[:-1](values).cpu().numpy()


def actor_schema(policy,mg):
    return policy.model.observation_schemas[mg]
