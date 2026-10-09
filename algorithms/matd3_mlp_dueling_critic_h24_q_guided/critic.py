from __future__ import annotations

import re

import torch
from torch import nn

LEAD = re.compile(r"_t\+(\d+)$")
VALUE_DEFAULTS = dict(forecast_horizon=24, embed_dim=48, num_heads=4,
                      num_layers=2, feedforward_dim=128)
ADV_DEFAULTS = dict(hidden_dim=128, use_value_context=True, detach_value_context=True,
                    action_reinjection=True, reference_action="zero")


def resolve_critic_config(settings):
    """Validate and complete the checkpoint-bound two-level network contract."""
    if not isinstance(settings, dict) or set(settings) - {"value", "advantage"}:
        raise ValueError("Unknown or invalid Critic network settings")
    resolved = {}
    for branch, defaults in (("value", VALUE_DEFAULTS), ("advantage", ADV_DEFAULTS)):
        supplied = settings.get(branch, {})
        if not isinstance(supplied, dict) or set(supplied) - set(defaults):
            raise ValueError(f"Unknown or invalid Critic {branch} settings")
        resolved[branch] = {key: supplied.get(key, value) for key, value in defaults.items()}
    value, advantage = resolved["value"], resolved["advantage"]
    for name, dimension in (*value.items(), ("hidden_dim", advantage["hidden_dim"])):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Critic {name} must be a positive integer")
    if value["forecast_horizon"] != 24:
        raise ValueError("Dueling Critic requires forecast_horizon=24 (H24)")
    if value["embed_dim"] % value["num_heads"]:
        raise ValueError("Critic embed_dim must be divisible by num_heads")
    for name in ("use_value_context", "detach_value_context", "action_reinjection"):
        if not isinstance(advantage[name], bool):
            raise ValueError(f"Critic {name} must be bool")
    if advantage["reference_action"] != "zero":
        raise ValueError("Critic reference_action must be 'zero'")
    return resolved


class DuelingTemporalValue(nn.Module):
    """State-only H24 Transformer V plus zero-centred action-direct MLP A."""

    def __init__(self, schema, action_dim=9, *, value=None, advantage=None):
        super().__init__()
        settings = resolve_critic_config({
            "value": {} if value is None else value,
            "advantage": {} if advantage is None else advantage,
        })
        value, advantage = settings["value"], settings["advantage"]
        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError("Critic action_dim must be a positive integer")
        # Schema parsing/masking are copied from H24 TemporalValue; no runtime
        # dependency on that removable experiment is introduced.
        if len(set(schema)) != len(schema):
            raise ValueError("Critic state schema has duplicate features")
        current, future = [], {}
        for index, name in enumerate(schema):
            match = LEAD.search(name)
            if match:
                future.setdefault(int(match.group(1)), []).append(index)
            else:
                current.append(index)
        if sorted(future) != list(range(1, 25)):
            raise ValueError("Critic state schema must provide the complete H24 horizon")
        channels = [[LEAD.sub("", schema[i]) for i in future[k]] for k in range(1, 25)]
        if any(names != channels[0] for names in channels[1:]):
            raise ValueError("Forecast tokens must have identical ordered channels")
        if "forecast_valid" not in channels[0]:
            raise ValueError("Critic requires forecast validity masks")
        self.validity_column = channels[0].index("forecast_valid")
        self.state_dim = len(schema)
        self.action_dim = action_dim
        self.use_value_context = advantage["use_value_context"]
        self.detach_value_context = advantage["detach_value_context"]
        self.action_reinjection = advantage["action_reinjection"]
        self.register_buffer("current_indices", torch.tensor(current, dtype=torch.long))
        self.register_buffer("forecast_indices", torch.tensor([future[k] for k in range(1, 25)], dtype=torch.long))
        embed_dim = value["embed_dim"]
        self.current = nn.Sequential(nn.Linear(len(current), embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.forecast = nn.Linear(len(channels[0]), embed_dim)
        self.position = nn.Parameter(torch.empty(24, embed_dim))
        nn.init.normal_(self.position, std=.02)
        block = nn.TransformerEncoderLayer(
            embed_dim, value["num_heads"], value["feedforward_dim"], dropout=0., activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, value["num_layers"], enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(embed_dim * 2, 64), nn.GELU(), nn.Linear(64, 1))
        self.advantage_state_norm = nn.LayerNorm(len(current))
        hidden_dim = advantage["hidden_dim"]
        context_dim = 2 * embed_dim if self.use_value_context else 0
        self.advantage_hidden1 = nn.Linear(len(current) + context_dim + action_dim, hidden_dim)
        self.advantage_hidden2 = nn.Linear(hidden_dim + (action_dim if self.action_reinjection else 0), hidden_dim)
        self.advantage_head = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.advantage_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.advantage_head.bias)

    def _value_context(self, state):
        current_state = state.index_select(-1, self.current_indices)
        current = self.current(current_state)
        forecast = state[:, self.forecast_indices]
        valid = torch.nan_to_num(forecast[:, :, self.validity_column], nan=0., posinf=0., neginf=0.) >= .5
        forecast = torch.where(valid[:, :, None], forecast, torch.zeros_like(forecast))
        forecast = torch.nan_to_num(forecast, nan=0., posinf=0., neginf=0.)
        tokens = torch.cat([current[:, None], self.forecast(forecast) + self.position[None]], 1)
        padding = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], 1)
        output = self.transformer(tokens, src_key_padding_mask=padding)
        pooled = torch.where(valid[:, :, None], output[:, 1:], torch.zeros_like(output[:, 1:])).sum(1)
        pooled = pooled / valid.sum(1, keepdim=True).clamp_min(1)
        context = torch.cat([output[:, 0], pooled], -1)
        return current_state, context

    def decompose(self, state, joint_action):
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state shape [batch, {self.state_dim}]")
        if joint_action.shape != (state.shape[0], self.action_dim):
            raise ValueError(f"Expected effective joint action shape [batch, {self.action_dim}]")
        current_state, context = self._value_context(state)
        return self._decompose_context(current_state, context, joint_action)

    def _decompose_context(self, current_state, context, joint_action):
        value = self.head(context)
        features = [self.advantage_state_norm(current_state)]
        if self.use_value_context:
            features.append(context.detach() if self.detach_value_context else context)
        state_features = torch.cat(features, -1)
        # One 2B MLP pass evaluates the proposed action and its zero reference.
        actions = torch.cat([joint_action, torch.zeros_like(joint_action)], 0)
        combined = torch.cat([state_features.repeat(2, 1), actions], -1)
        hidden = self.advantage_hidden1(combined).relu()
        if self.action_reinjection:
            hidden = torch.cat([hidden, actions], -1)
        advantages = self.advantage_head(self.advantage_hidden2(hidden).relu())
        actual, reference = advantages.chunk(2, dim=0)
        centred = actual - reference
        return value, centred, value + centred

    @torch.no_grad()
    def q_candidates(self, state, joint_actions):
        """Score several actions per state, encoding its H24 context once."""
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state shape [batch, {self.state_dim}]")
        if (joint_actions.ndim != 3 or joint_actions.shape[0] != len(state)
                or joint_actions.shape[-1] != self.action_dim or joint_actions.shape[1] < 1):
            raise ValueError(f"Expected candidate action shape [batch, candidates, {self.action_dim}]")
        current, context = self._value_context(state)
        count = joint_actions.shape[1]
        current = current[:, None].expand(-1, count, -1).reshape(len(state) * count, -1)
        context = context[:, None].expand(-1, count, -1).reshape(len(state) * count, -1)
        return self._decompose_context(current, context, joint_actions.flatten(0, 1))[2].reshape(len(state), count)

    def forward(self, state, joint_action):
        return self.decompose(state, joint_action)[2]


class DuelingTwinCritic(nn.Module):
    """Independent Q1/Q2, preserving the projected TD3 critic interface."""

    def __init__(self, schema, action_dim=9, *, value=None, advantage=None):
        super().__init__()
        self.q1_network = DuelingTemporalValue(schema, action_dim, value=value, advantage=advantage)
        self.q2_network = DuelingTemporalValue(schema, action_dim, value=value, advantage=advantage)

    def forward(self, state, joint_action):
        return self.q1_network(state, joint_action), self.q2_network(state, joint_action)

    def q1(self, state, joint_action):
        return self.q1_network(state, joint_action)

    def q_candidates(self, state, joint_actions):
        return (self.q1_network.q_candidates(state, joint_actions),
                self.q2_network.q_candidates(state, joint_actions))

    def decompose(self, state, joint_action):
        v1, adv1, q1 = self.q1_network.decompose(state, joint_action)
        v2, adv2, q2 = self.q2_network.decompose(state, joint_action)
        return dict(v1=v1, adv1=adv1, q1=q1, v2=v2, adv2=adv2, q2=q2)
