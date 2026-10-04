from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn

from algorithms.matd3_dispatch.algorithm import Matd3Dispatch
from core.reward_v2 import build_reward_v2_contract
from .actor import PlanTransformerActor
from .forecast_scenario import PlanningContextRejected, capture_planning_context
from .plan_losses import plan_loss_terms
from .plan_teacher import PlanTeacher
from .replay_buffer import PlanReplayBuffer
from .diagnostics import PlanDiagnosticsRecorder, write_json


class Matd3PlanTransformer(Matd3Dispatch):
    """Projected MATD3 with a causally supervised, auxiliary H24 plan."""

    def __init__(self, env, config, device='cpu', seed=42):
        super().__init__(env, config, device=device, seed=seed)
        if not env.reward_v2_enabled or int(env.config['observation']['future_horizon']) != 24:
            raise ValueError('Plan-Transformer requires Reward V2 and H24 observations')
        settings = config.get('algorithm', config)
        network = deepcopy(settings.get('network', {}))
        auxiliary = deepcopy(settings.get('auxiliary_plan', {}))
        self.loss_weights = tuple(float(auxiliary.get(k, v)) for k, v in
                                 (('lambda_plan', .1), ('lambda_dyn', .01), ('lambda_first', .05)))
        if any(not np.isfinite(x) or x < 0 for x in self.loss_weights):
            raise ValueError('Auxiliary loss weights must be finite and nonnegative')
        self.auxiliary_enabled = bool(auxiliary.get('enabled', True)) and any(self.loss_weights)
        if not self.auxiliary_enabled:
            self.loss_weights = (0., 0., 0.)
        self.capture_hours = tuple(auxiliary.get('capture_hours', [0, 6, 12, 18]))
        if any(isinstance(h, bool) or not isinstance(h, int) or not 0 <= h < 24 for h in self.capture_hours):
            raise ValueError('Capture hours must be integers in [0, 23]')
        self.auxiliary_batch_size = int(auxiliary.get('batch_size', 32))
        if self.auxiliary_batch_size <= 0:
            raise ValueError('Auxiliary batch size must be positive')
        self.teacher = PlanTeacher(auxiliary.get('teacher', {}), seed + 100003)
        self.replay = PlanReplayBuffer(int(settings.get('replay_capacity', 200000)),
            auxiliary.get('capacity', 2048), seed, seed + 200003)
        self.battery_configs = {mg: deepcopy(env.cfg_by_id[mg]['battery']) for mg in self.agent_ids}
        self.delta_t = env.delta_t
        self.actors = nn.ModuleDict({mg: PlanTransformerActor(self.observation_schemas[mg],
            self.agent_ids, mg, env.cfg_by_id[mg], env.obs_builder.norm[mg], network)
            for mg in self.agent_ids}).to(self.device)
        self.target_actors = deepcopy(self.actors).to(self.device)
        self.actor_optimizers = {mg: torch.optim.Adam(actor.parameters(), lr=float(settings.get('actor_lr', 2e-4)))
                                 for mg, actor in self.actors.items()}
        self._actor_parameters = {mg: tuple(actor.parameters()) for mg, actor in self.actors.items()}
        self.actor_auxiliary_update_count = 0
        self.context_rejection_count = 0
        self.rejection_reasons = {}
        self.recorder = None
        masks = 'forecast:t+k<24;plan:t+j<24;soc_after:t+j+1'
        peers = {mg: [p for p in self.agent_ids if p != mg] for mg in self.agent_ids}
        self.actor_contract = dict(version='now-cross-attention-plan-v1', network=network,
            auxiliary_plan=auxiliary, observation_schemas=self.observation_schemas,
            normalizers=deepcopy(env.obs_builder.norm), batteries=self.battery_configs,
            peer_orders=peers, masks=masks, power_sign='positive_discharge_negative_charge', gamma=self.gamma)
        self.reward_v2_contract = build_reward_v2_contract(env.config, self.observation_schemas,
            self.state_schema, dict(by_microgrid=deepcopy(env.obs_builder.norm),
                team_power_scale_kw=float(env.obs_builder.team_power_scale_kw),
                carbon_factor_scale=float(env.obs_builder.carbon_factor_scale)),
            algorithm_slug='matd3_plan_transformer', actor_forecast_context_version='causal-h24-plan-v1',
            peer_orders=peers, mask_semantics=masks, gamma=self.gamma, action_semantics=self.projector.semantics)

    def configure_diagnostics(self, directory, metadata):
        self.recorder = PlanDiagnosticsRecorder(directory, metadata, self.actor_contract)

    def set_diagnostic_phase(self, phase, metadata):
        if self.recorder is None:
            raise ValueError('Configure diagnostics before setting a phase')
        self.recorder.set_phase(phase, metadata)
        if metadata.get('training_complete'):
            summary = dict(enabled=self.auxiliary_enabled, teacher_solve_calls=self.teacher.solve_calls,
                valid_labels=len(self.replay.labels), actor_auxiliary_update_count=self.actor_auxiliary_update_count,
                context_rejections=self.context_rejection_count, rejection_reasons=self.rejection_reasons)
            write_json(self.recorder.directory/'training_supervision_summary.json', summary)
            if self.auxiliary_enabled and (not self.replay.labels or not self.actor_auxiliary_update_count):
                raise RuntimeError('Enabled plan supervision did not produce labels and Actor updates')
        if metadata.get('formal_test_complete'):
            write_json(self.recorder.directory/'formal_test_teacher_summary.json',
                       dict(teacher_solve_calls=self.teacher.solve_calls))

    def capture_training_context(self, env, observations, global_state, metadata):
        if (not self.auxiliary_enabled or env.split != 'train' or env.t not in self.capture_hours or
                self.teacher.solve_calls >= self.teacher.settings['max_labels']):
            return
        try:
            context = capture_planning_context(env, observations, global_state, metadata)
        except PlanningContextRejected as error:
            self.context_rejection_count += 1
            reason = str(error)
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
            if self.recorder is not None:
                self.recorder.record_rejection(reason, metadata)
            return
        label = self.teacher.solve(context)
        self.replay.bind_context(context, label)
        if label is not None and not label.valid:
            reason = label.rejection_reason
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        if self.recorder is not None and label is not None:
            self.recorder.record_label(label)

    @torch.no_grad()
    def inspect_policy(self, observations):
        return {mg: self.actors[mg].forward_with_plan(torch.as_tensor(observations[mg],
                    dtype=torch.float32, device=self.device).unsqueeze(0), collect_attention=True)
                for mg in self.agent_ids}

    def select_actions(self, observations, noise_std=0.0):
        # Use one identical forward path with diagnostics enabled or disabled.
        outputs = self.inspect_policy(observations)
        if self.recorder is not None:
            self.recorder.record_policy(observations, outputs)
        actions = {}
        for mg, output in outputs.items():
            action = output.actions[0].cpu().numpy().copy()
            if noise_std > 0:
                action += self.rng.normal(0, noise_std, size=action.shape)
            actions[mg] = np.clip(action, -1, 1).astype(np.float32)
        return actions

    @staticmethod
    def _norm(parameters, gradients=None):
        values = gradients if gradients is not None else [p.grad for p in parameters]
        return sum(float(g.detach().square().sum()) for g in values if g is not None) ** .5

    def update(self, batch_size):
        if len(self.replay) < batch_size:
            return {}
        batch = self.replay.sample(batch_size, self.agent_ids, self.device)
        state, next_state = batch['global_state'], batch['next_global_state']
        effective, _ = self.project_actions(state, torch.stack([batch['actions'][mg] for mg in self.agent_ids], 1))
        with torch.no_grad():
            target_actions = []
            for mg in self.agent_ids:
                action = self.target_actors[mg](batch['next_observations'][mg])
                noise = torch.randn_like(action).mul(self.target_policy_noise).clamp(-self.target_noise_clip, self.target_noise_clip)
                target_actions.append((action + noise).clamp(-1, 1))
            target_effective, _ = self.project_actions(next_state, torch.stack(target_actions, 1))
            targets = {}
            for mg in self.agent_ids:
                q1, q2 = self.target_critics[mg](next_state, target_effective.flatten(1))
                targets[mg] = batch['team_reward'] + (1 - batch['done']) * self.gamma * torch.minimum(q1, q2)
        records, q_stats = {}, {}
        for mg in self.agent_ids:
            q1, q2 = self.critics[mg](state, effective.flatten(1))
            c1, c2 = nn.functional.mse_loss(q1, targets[mg]), nn.functional.mse_loss(q2, targets[mg])
            loss = c1 + c2
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{mg}: nonfinite Critic loss')
            self.critic_optimizers[mg].zero_grad(set_to_none=True)
            loss.backward()
            grad = self._norm(self.critics[mg].parameters())
            if not np.isfinite(grad):
                raise FloatingPointError(f'{mg}: nonfinite Critic gradient')
            self.critic_optimizers[mg].step()
            td = targets[mg] - torch.minimum(q1, q2)
            q = torch.minimum(q1.detach(), q2.detach())
            q_stats[mg] = dict(mean=float(q.mean()), min=float(q.min()), max=float(q.max()), std=float(q.std(unbiased=False)))
            records[mg] = dict(critic_loss=float(loss.detach()), critic1_loss=float(c1.detach()),
                critic2_loss=float(c2.detach()), critic_grad_norm=grad, q1_mean=float(q1.detach().mean()),
                q2_mean=float(q2.detach().mean()), q1_q2_gap=float((q1.detach().mean()-q2.detach().mean()).abs()),
                target_q_mean=float(targets[mg].mean()), td_error_mean=float(td.detach().mean()),
                td_error_std=float(td.detach().std(unbiased=False)))
        metrics = {key: float(np.mean([v[key] for v in records.values()])) for key in next(iter(records.values()))}
        metrics.update(critic_losses={mg: v['critic_loss'] for mg,v in records.items()},
            critic_grad_norms={mg:v['critic_grad_norm'] for mg,v in records.items()}, q_stats=q_stats)
        self.update_steps += 1
        if self.update_steps % self.policy_delay == 0:
            auxiliary_batch = self.replay.sample_auxiliary(self.auxiliary_batch_size, self.agent_ids, self.device) if self.auxiliary_enabled else None
            losses, diagnostics, layers = {}, {}, {}
            for active_index, active in enumerate(self.agent_ids):
                actor, critic = self.actors[active], self.critics[active]
                for p in critic.parameters():
                    p.requires_grad_(False)
                try:
                    parts = []
                    for mg in self.agent_ids:
                        with torch.set_grad_enabled(mg == active):
                            parts.append(self.actors[mg](batch['observations'][mg]))
                    policy_effective, _ = self.project_actions(state, torch.stack(parts, 1))
                    rl = -critic.q1(state, policy_effective.flatten(1)).mean()
                    aux_loss = rl.new_zeros(())
                    details = {}
                    if auxiliary_batch is not None:
                        outputs = {}
                        for mg in self.agent_ids:
                            with torch.set_grad_enabled(mg == active):
                                outputs[mg] = self.actors[mg].forward_with_plan(auxiliary_batch.observations[mg])
                        _, plans = self.project_actions(auxiliary_batch.global_state,
                            torch.stack([outputs[mg].actions for mg in self.agent_ids], 1))
                        terms = plan_loss_terms(outputs[active], auxiliary_batch, active,
                            plans[active_index], self.battery_configs[active], self.delta_t)
                        aux_loss = sum(w * term for w,term in zip(self.loss_weights, (terms.plan, terms.dynamics, terms.first)))
                        details = dict(plan_loss=float(terms.plan.detach()), dynamics_loss=float(terms.dynamics.detach()),
                            first_loss=float(terms.first.detach()), **terms.metrics)
                    attention_parameters = tuple(actor.layers.parameters())
                    rl_grad = torch.autograd.grad(rl, attention_parameters, retain_graph=True, allow_unused=True)
                    aux_grad = torch.autograd.grad(aux_loss, attention_parameters, retain_graph=True, allow_unused=True) if auxiliary_batch is not None else []
                    loss = rl + aux_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'{active}: nonfinite Actor loss')
                    self.actor_optimizers[active].zero_grad(set_to_none=True)
                    loss.backward()
                    parameters = self._actor_parameters[active]
                    before = self._norm(parameters)
                    torch.nn.utils.clip_grad_norm_(parameters, self.actor_max_grad_norm, error_if_nonfinite=True)
                    diagnostics[active] = dict(total_grad_norm=before, post_clip_grad_norm=self._norm(parameters),
                        clip_applied=float(before > self.actor_max_grad_norm), rl_loss=float(rl.detach()),
                        auxiliary_loss=float(aux_loss.detach()), rl_attention_grad_norm=self._norm((), rl_grad),
                        auxiliary_attention_grad_norm=self._norm((), aux_grad), auxiliary_label_count=len(self.replay.labels),
                        auxiliary_valid_slots=float(auxiliary_batch.mask.sum()) if auxiliary_batch is not None else 0., **details)
                    layers[active] = {f'attention_layer_{i}': self._norm(layer.parameters()) for i,layer in enumerate(actor.layers)}
                    self.actor_optimizers[active].step()
                    losses[active] = float(loss.detach())
                    self._soft_update(actor, self.target_actors[active])
                    self._soft_update(critic, self.target_critics[active])
                finally:
                    for p in critic.parameters():
                        p.requires_grad_(True)
            if auxiliary_batch is not None:
                self.actor_auxiliary_update_count += 1
            metrics.update(actor_loss=float(np.mean(list(losses.values()))), actor_losses=losses,
                actor_diagnostics=diagnostics, actor_layer_grad_norms=layers,
                actor_grad_norm=float(np.mean([v['total_grad_norm'] for v in diagnostics.values()])),
                actor_grad_norms={mg:v['total_grad_norm'] for mg,v in diagnostics.items()})
        metrics.update(auxiliary_label_count=len(self.replay.labels), actor_auxiliary_update_count=self.actor_auxiliary_update_count,
            actor_parameter_counts=self.actor_parameter_counts(), actor_parameter_count=float(np.mean(list(self.actor_parameter_counts().values()))))
        return metrics

    def _checkpoint_contract(self):
        return dict(algorithm_slug='matd3_plan_transformer', action_semantics=self.projector.semantics,
            projection_signature=self.projection_signature, actor_contract=deepcopy(self.actor_contract),
            reward_v2_contract=deepcopy(self.reward_v2_contract), observation_schemas=self.observation_schemas,
            state_schema=self.state_schema)

    def save(self, path):
        payload = self._checkpoint_contract()
        payload.update({name: getattr(self,name).state_dict() for name in ('actors','target_actors','critics','target_critics')})
        payload['update_steps'] = self.update_steps
        torch.save(payload, Path(path))

    def load(self, path):
        payload = torch.load(Path(path), map_location=self.device, weights_only=True)
        for key, value in self._checkpoint_contract().items():
            if payload.get(key) != value:
                raise ValueError(f'Checkpoint {key} mismatch')
        for name in ('actors','target_actors','critics','target_critics'):
            expected, saved = getattr(self,name).state_dict(), payload.get(name, {})
            if set(saved) != set(expected) or any(not isinstance(saved[k], torch.Tensor) or
                    saved[k].shape != expected[k].shape or not torch.isfinite(saved[k]).all() for k in expected):
                raise ValueError(f'Checkpoint {name} tensors mismatch')
        steps = payload.get('update_steps')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('Invalid checkpoint update_steps')
        for name in ('actors','target_actors','critics','target_critics'):
            getattr(self,name).load_state_dict(payload[name])
        self.update_steps = steps
