from __future__ import annotations

import copy
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

# The formal plugin installs the shared projected TD3 update and diagnostic hook.
from algorithms.matd3_hierarchical_transformer.plugin import Matd3HierarchicalTransformer
from algorithms.matd3_hierarchical_transformer.actor import HierarchicalTransformerActor
from algorithms.matd3_ess_plan_transformer.actor import EssPlanActor
from algorithms.matd3_ess_plan_transformer.critic import DuelingTwinCritic, resolve_critic_config
from core.reward_v2 import stable_contract_hash
from .forecast_scenario import capture_forecast_context
from .plan_teacher import PlanTeacher
from .plan_buffer import PlanBuffer
from .plan_supervision import TeacherBudget, plan_supervision_loss
from .diagnostics import write_json


class Matd3EssPlanTransformer(Matd3HierarchicalTransformer):
    """H24 ESS sequence policy, existing DR MLP and dueling twin critics."""

    SLUG = "matd3_ess_plan_transformer"

    def __init__(self, env, config, device="cpu", seed=42):
        config = copy.deepcopy(config)
        settings = config.get("algorithm", config)
        actor_settings = settings.setdefault("network", {})
        actor_settings.setdefault("include_peer_forecast", True)
        actor_settings.setdefault("forecast_horizon", 24)
        horizon = actor_settings["forecast_horizon"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon != 24:
            raise ValueError("MLP Actor requires integer forecast_horizon=24 (H24)")
        if not actor_settings.get("include_peer_forecast", True):
            raise ValueError("MLP Actor requires the current peer forecast features")
        # The formal parent admits peer forecasts only with H8. Use that path
        # solely to initialize shared projection/replay/update machinery, then
        # construct H24 feature references locally. No parent policy is trained.
        parent_config = copy.deepcopy(config)
        parent_config.get("algorithm", parent_config)["network"]["forecast_horizon"] = 8
        super().__init__(env, parent_config, device=device, seed=seed)
        reference_settings = {
            key: int(actor_settings.get(key, default))
            for key, default in (("embed_dim", 32), ("num_heads", 4),
                                 ("num_layers", 2), ("hidden_dim", 64))
        }
        self.actors = nn.ModuleDict({
            mg: EssPlanActor(HierarchicalTransformerActor(
                self.observation_schemas[mg], self.agent_ids, mg,
                mg_config=env.cfg_by_id[mg],
                power_normalizers=self.actor_feature_normalizers[mg],
                forecast_horizon=24, include_peer_forecast=True,
                peer_forecast_features=self.peer_forecast_features,
                **reference_settings,
            ), env.cfg_by_id[mg]["battery"], env.delta_t)
            for mg in self.agent_ids
        }).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.actor_optimizers = {mg: torch.optim.Adam(actor.parameters(), lr=float(settings.get("actor_lr", 2e-4)))
                                 for mg, actor in self.actors.items()}
        self._actor_parameters = {mg: tuple(actor.parameters()) for mg, actor in self.actors.items()}
        self._actor_branch_parameters = {mg: {"joint": params} for mg, params in self._actor_parameters.items()}
        self._actor_layer_parameters = {
            mg: {f"linear_{i}": tuple(layer.parameters()) for i, layer in enumerate(actor.network) if isinstance(layer, nn.Linear)}
            for mg, actor in self.actors.items()
        }
        self._diagnostic_plain_actor = True
        self._diagnostic_variant = self.SLUG
        self.critic_network_config = resolve_critic_config(settings.get("critic_network", {}))
        self.critics = nn.ModuleDict({
            mg: DuelingTwinCritic(self.state_schema, len(self.agent_ids) * 3, **self.critic_network_config)
            for mg in self.agent_ids
        }).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device)
        self.critic_optimizers = {mg: torch.optim.Adam(critic.parameters(), lr=float(settings.get("critic_lr", 3e-4)))
                                  for mg, critic in self.critics.items()}
        self.actor_network_config = {
            "architecture": "ess_plan_transformer_h24_v1", "forecast_horizon": 24,
            "ess_network": dict(reference_settings, dropout=0.0),
            "feature_semantics": self.actor_feature_semantics,
            "hidden_dims_by_agent": {mg: [actor.hidden_dim, actor.hidden_dim] for mg, actor in self.actors.items()},
            "include_peer_forecast": True, "include_base_load_forecast": True,
            "peer_forecast_features": list(self.peer_forecast_features),
            "parameter_budget_reference": "t_q_device_forecast_readout_h24_v1",
        }
        self.actor_forecast_context_version = "semantic_mlp_peer_balance_base_load_h24_v1"
        self.learning_starts = int(settings.get("learning_starts", 1000))
        if not 0 <= self.learning_starts <= self.replay.capacity:
            raise ValueError("learning_starts must be between 0 and replay_capacity")
        self.dispatch_random_action_prob = float(settings.get("dispatch_random_action_prob", .10))
        if not math.isfinite(self.dispatch_random_action_prob) or not 0 <= self.dispatch_random_action_prob <= 1:
            raise ValueError("dispatch_random_action_prob must be in [0, 1]")
        self.dispatch_q_guidance_weight = float(settings.get("dispatch_q_guidance_weight", .01))
        self.dispatch_q_guidance_margin = float(settings.get("dispatch_q_guidance_margin", .01))
        for name in ("dispatch_q_guidance_weight", "dispatch_q_guidance_margin"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.dispatch_candidates = torch.tensor([-.9, -.5, 0., .5, .9], device=self.device)
        if self.reward_v2_contract is not None:
            contract = copy.deepcopy(self.reward_v2_contract)
            contract.pop("signature", None)
            contract.update(algorithm_slug=self.SLUG, actor_forecast_context_version=self.actor_forecast_context_version,
                            actor_network_config=self.actor_network_config, critic_network_config=self.critic_network_config,
                            critic_feature_semantics="full_state_h24_dueling_v_transformer_a_mlp_zero_ref_v1")
            contract["signature"] = stable_contract_hash(contract)
            self.reward_v2_contract = contract

        self.plan_settings = copy.deepcopy(settings.get('ess_plan', {}))
        self.plan_weight = float(self.plan_settings.get('lambda_plan', .1))
        if not math.isfinite(self.plan_weight) or self.plan_weight < 0:
            raise ValueError('Invalid lambda_plan')
        self.plan_enabled = bool(self.plan_settings.get('enabled', True)) and self.plan_weight > 0
        self.teacher = PlanTeacher(self.plan_settings.get('solve_seconds', 1.))
        self.plan_buffer = PlanBuffer(self.plan_settings.get('capacity', 512), seed+200003)
        self.teacher_budget = TeacherBudget(self.plan_settings.get('teacher_fraction', .15), self.teacher.time_limit_seconds)
        self.plan_update_interval = int(self.plan_settings.get('update_interval', 10))
        self.plan_batch_size = int(self.plan_settings.get('batch_size', 16))
        self.capture_every = int(self.plan_settings.get('capture_every_episodes', 10))
        self.capture_start = int(self.plan_settings.get('capture_start_episode', 50))
        self.capture_hours = tuple(self.plan_settings.get('capture_hours', [0,6,12,18]))
        self.max_labels = int(self.plan_settings.get('max_labels', 512))
        if min(self.plan_update_interval,self.plan_batch_size,self.capture_every,self.max_labels) <= 0:
            raise ValueError('Plan counts must be positive')
        if self.capture_start < 0 or not self.capture_hours or any(isinstance(h,bool) or not isinstance(h,int) or not 0 <= h < 24 for h in self.capture_hours):
            raise ValueError('Invalid plan capture schedule')
        self.plan_supervision_updates = 0
        self.plan_skip_counts = Counter()
        self.plan_reject_counts = Counter()
        self._plan_batch_step = None
        self._plan_batch = None
        self._diagnostics_directory = None
        self._diagnostic_phase = 'training'
        self._policy_record_index = 0
        self._phase_days = ()
        self._plan_started = time.perf_counter()

    def capture_training_context(self, env, observations, global_state, metadata):
        if not self.plan_enabled or env.split != 'train' or self._diagnostic_phase != 'training':
            return
        episode = int(metadata['episode_index'])
        if episode < self.capture_start or episode % self.capture_every:
            return
        hour = self.capture_hours[(episode//self.capture_every)%len(self.capture_hours)]
        if env.t != hour:
            return
        if self.teacher.solve_calls >= self.max_labels:
            self.plan_skip_counts['label_limit'] += 1
            return
        nonteacher = time.perf_counter()-self._plan_started-self.teacher_budget.seconds
        if not self.teacher_budget.admit(nonteacher):
            self.plan_skip_counts['time_budget'] += 1
            return
        started = time.perf_counter()
        try:
            context = capture_forecast_context(env,observations,global_state,metadata)
            self.teacher.time_limit_seconds = min(float(self.plan_settings.get('solve_seconds',1.)),
                                                  self.teacher_budget.remaining(nonteacher))
            label = self.teacher.solve(context,self.actors)
            if label.valid:
                self.plan_buffer.add(label)
            else:
                self.plan_reject_counts[label.rejection_reason] += 1
            if self._diagnostics_directory:
                write_json(self._diagnostics_directory/'labels'/f'label_{self.teacher.solve_calls:05d}.json',label)
        except (ValueError, RuntimeError) as exc:
            self.plan_reject_counts[type(exc).__name__] += 1
            if self._diagnostics_directory:
                write_json(self._diagnostics_directory/'last_rejection.json',dict(error=str(exc),metadata=dict(metadata)))
        finally:
            self.teacher_budget.record(time.perf_counter()-started)

    def configure_diagnostics(self,directory,metadata):
        self._diagnostics_directory = Path(directory)
        self._diagnostics_directory.mkdir(parents=True,exist_ok=True)
        self._plan_started = time.perf_counter()
        write_json(self._diagnostics_directory/'contract.json',dict(settings=self.plan_settings,
            architecture=self.actor_network_config,metadata=dict(metadata),inference_uses_teacher=False))

    def plan_summary(self):
        return dict(enabled=self.plan_enabled,teacher_calls=self.teacher.solve_calls,
            teacher_total_seconds=self.teacher_budget.seconds,valid_labels=len(self.plan_buffer),
            supervision_updates=self.plan_supervision_updates,rejections=dict(self.plan_reject_counts),
            skipped=dict(self.plan_skip_counts),inference_uses_teacher=False,
            label_source_split='train',overhead_target_fraction=.2,overhead_target_measured=False)

    def set_diagnostic_phase(self,phase,metadata):
        if phase is not None:
            self._diagnostic_phase = phase
            if phase == 'formal_test':
                self._formal_teacher_calls = self.teacher.solve_calls
                self._policy_record_index = 0
                self._phase_days = tuple(metadata.get('days', ()))
        if not self._diagnostics_directory:
            return
        if metadata.get('training_complete'):
            write_json(self._diagnostics_directory/'training_supervision_summary.json',self.plan_summary())
            if self.plan_enabled and (not len(self.plan_buffer) or not self.plan_supervision_updates):
                raise RuntimeError('ESS plan supervision produced no valid labels or updates')
        if metadata.get('formal_test_complete'):
            calls = self.teacher.solve_calls-getattr(self,'_formal_teacher_calls',self.teacher.solve_calls)
            write_json(self._diagnostics_directory/'formal_test_teacher_summary.json',dict(
                teacher_calls=calls, policy_plan_records=self._policy_record_index))
            if calls:
                raise RuntimeError('Formal test unexpectedly invoked teacher')

    def load_training_state(self,directory):
        raise RuntimeError('ESS planning supports inference checkpoints only; training resume is unsupported')

    def select_actions(self, observations, noise_std=0.):
        actions = super().select_actions(observations, noise_std=noise_std)
        if noise_std > 0 and self.dispatch_random_action_prob > 0:
            for mg in self.agent_ids:
                if self.rng.random() < self.dispatch_random_action_prob:
                    actions[mg][0] = np.float32(self.rng.uniform(-1., 1.))
        if noise_std == 0 and self._diagnostic_phase == 'formal_test' and self._diagnostics_directory:
            self._record_policy_plan(observations, actions)
        return actions

    @torch.inference_mode()
    def _record_policy_plan(self, observations, actions):
        # Formal evaluation is a known sequence of 24-hour days. Bound evidence
        # to that sequence; no expert or simulation is called by this hook.
        index = self._policy_record_index
        if not self._phase_days or index >= 24*len(self._phase_days):
            return
        agents = {}
        for mg, actor in self.actors.items():
            obs = torch.as_tensor(observations[mg], dtype=torch.float32, device=self.device)[None]
            plan = actor.forward_with_plan(obs)
            agents[mg] = dict(action=np.asarray(actions[mg]),
                current_soc=float(obs[0, actor.soc_index]),
                intents=plan.intents[0].cpu().numpy(), request_kw=plan.request_kw[0].cpu().numpy(),
                nominal_soc_after=plan.nominal_soc_after[0].cpu().numpy(), mask=plan.mask[0].cpu().numpy())
        actor = self.actors[self.agent_ids[0]]
        clock = np.asarray(observations[self.agent_ids[0]])[actor.hour_indices]
        hour = int(round(math.atan2(float(clock[0]), float(clock[1]))*24/(2*math.pi))) % 24
        write_json(self._diagnostics_directory/'policy_plans'/f'formal_test_{index:05d}.json',
            dict(date=self._phase_days[index//24], hour=hour, record_index=index, agents=agents,
                 semantics='nominal requests; actual clearing and hourly replanning may differ'))
        self._policy_record_index += 1

    def update(self, batch_size):
        if len(self.replay) < self.learning_starts:
            return {}
        return super().update(batch_size)

    def actor_auxiliary_loss(self, active, state, joint_raw, observations):
        loss, metrics = self._ess_q_guidance_loss(active, state, joint_raw, observations)
        if self.plan_enabled and len(self.plan_buffer) and self.update_steps % (self.policy_delay*self.plan_update_interval) == 0:
            if self._plan_batch_step != self.update_steps:
                self._plan_batch = self.plan_buffer.sample(self.plan_batch_size,self.device)
                self._plan_batch_step = self.update_steps
                self.plan_supervision_updates += 1
            actor = self.actors[active]
            batch = self._plan_batch[active]
            extra, detail = plan_supervision_loss(actor.forward_with_plan(batch['observations']),batch,actor.battery)
            loss = loss+self.plan_weight*extra
            metrics.update(detail)
        # Shared due() has already advanced this stateful cursor before entering
        # the Actor phase. Observe that advance once, then reuse it for all agents.
        boundary = self.diagnostic_schedule.next_boundary
        previous = getattr(self, "_dueling_diagnostic_boundary", self.diagnostic_schedule.interval)
        if boundary != previous:
            self._dueling_diagnostic_boundary = boundary
            self._dueling_diagnostic_step = self.update_steps
        if getattr(self, "_dueling_diagnostic_step", -1) == self.update_steps:
            metrics.update(self._dueling_critic_diagnostics(active, state, joint_raw))
        return loss, metrics

    @torch.no_grad()
    def _dueling_critic_diagnostics(self, active, state, joint_raw):
        effective, _ = self.project_actions(state.detach(), joint_raw.detach())
        action = effective.flatten(1).detach()
        critic = self.critics[active]
        parts = critic.decompose(state.detach(), action)
        value, advantage, q = parts["v1"], parts["adv1"], parts["q1"]
        # Sample std matches the study definition; singleton diagnostics use
        # population std so their zero variance remains finite.
        correction = len(state) > 1
        value_centered = value - value.mean()
        advantage_centered = advantage - advantage.mean()
        correlation = (value_centered * advantage_centered).mean() / (
            value.std(unbiased=False) * advantage.std(unbiased=False) + 1e-8
        )
        with torch.enable_grad():
            differentiable_action = action.clone().requires_grad_(True)
            gradient = torch.autograd.grad(
                critic.q1(state.detach(), differentiable_action).sum(), differentiable_action,
            )[0].detach()
        index = self.agent_ids.index(active) * 3
        return {
            "critic_v_mean": value.mean(),
            "critic_adv_std": advantage.std(unbiased=False),
            "critic_adv_abs_mean": advantage.abs().mean(),
            "critic_adv_share": advantage.std(unbiased=correction) / (q.std(unbiased=correction) + 1e-8),
            "critic_dqda_dispatch_rms": gradient[:, index:index + 2].square().mean().sqrt(),
            "critic_dqda_dr_rms": gradient[:, index + 2].square().mean().sqrt(),
            "critic_v_adv_corr": correlation,
        }

    def _ess_q_guidance_loss(self, active, state, joint_raw, observations):
        """Existing ESS-only conservative Q recovery, applied to MLP logits."""
        index = self.agent_ids.index(active)
        zero = joint_raw[:, index, 0].sum() * 0.
        metrics = {name: zero.detach() for name in
                   ("dispatch_q_guidance_trigger_rate", "dispatch_q_guidance_loss", "dispatch_q_guidance_q_gain")}
        if self.dispatch_q_guidance_weight == 0:
            return zero, metrics
        with torch.no_grad():
            raw = joint_raw.detach()
            intent = raw[:, index, 0]
            charge = state[:, self.projector.index[f"{active}_feasible_charge_max"]]
            discharge = state[:, self.projector.index[f"{active}_feasible_discharge_max"]]
            blocked = ((intent > 0) & (discharge <= 1e-8)) | ((intent < 0) & (charge <= 1e-8))
            if not blocked.any():
                return zero, metrics
            selected_state, selected_raw = state[blocked], raw[blocked]
            effective, _ = self.project_actions(selected_state, selected_raw)
            count, candidates = len(selected_state), self.dispatch_candidates.numel()
            probes = selected_raw[:, None].expand(-1, candidates, -1, -1).clone()
            probes[:, :, index, 0] = self.dispatch_candidates
            probe_state = selected_state[:, None].expand(-1, candidates, -1).reshape(count * candidates, -1)
            probe_effective, _ = self.project_actions(probe_state, probes.reshape(count * candidates, len(self.agent_ids), 2))
            # Baseline and probes share the same state-only value context. Reuse
            # it within this call; never cache across updates or different states.
            actions = torch.cat([effective.flatten(1)[:, None],
                                 probe_effective.reshape(count, candidates, -1)], dim=1)
            scores1, scores2 = self.critics[active].q_candidates(selected_state, actions)
            gains = torch.minimum(scores1[:, 1:] - scores1[:, :1],
                                  scores2[:, 1:] - scores2[:, :1])
            gain, choice = gains.max(1)
            approved = gain > self.dispatch_q_guidance_margin
            if not approved.any():
                return zero, metrics
            targets = torch.atanh(self.dispatch_candidates[choice])
        logits = self.actors[active].action_logits(observations[blocked])[:, 0]
        loss = self.dispatch_q_guidance_weight * (logits[approved] - targets[approved]).square().sum() / len(state)
        metrics.update(dispatch_q_guidance_trigger_rate=approved.float().sum() / len(state),
                       dispatch_q_guidance_loss=loss.detach(), dispatch_q_guidance_q_gain=gain[approved].mean())
        return loss, metrics

    def _checkpoint_metadata(self):
        return {
            "algorithm_slug": self.SLUG,
            "actor_feature_semantics": self.actor_feature_semantics,
            "actor_feature_normalizers": copy.deepcopy(self.actor_feature_normalizers),
            "actor_network_config": copy.deepcopy(self.actor_network_config),
            "critic_network_config": copy.deepcopy(self.critic_network_config),
            "action_semantics": self.projector.semantics,
            "projection_signature": self.projection_signature,
            "observation_schemas": self.observation_schemas,
            "state_schema": self.state_schema,
            "actor_forecast_context_version": self.actor_forecast_context_version,
            "peer_forecast_features": list(self.peer_forecast_features),
            "reward_v2_contract": copy.deepcopy(self.reward_v2_contract),
            "ess_plan": copy.deepcopy(self.plan_settings),
            "ess_plan_version": "forecast_storage_proposal_verified_v1",
        }

    def save(self, path):
        checkpoint = self._checkpoint_metadata()
        checkpoint.update({name: getattr(self, name).state_dict() for name in
                           ("actors", "target_actors", "critics", "target_critics")})
        checkpoint["update_steps"] = self.update_steps
        torch.save(checkpoint, Path(path))

    def load(self, path):
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        for name, expected in self._checkpoint_metadata().items():
            if checkpoint.get(name) != expected:
                raise ValueError(f"Checkpoint structure/contract mismatch: {name}")
        self._validate_actor_feature_metadata(checkpoint)
        # Preflight all networks, including fixed-feature buffers, before any write.
        for name in ("actors", "target_actors", "critics", "target_critics"):
            module = getattr(self, name)
            expected = module.state_dict()
            saved = checkpoint.get(name, {})
            parameters = dict(module.named_parameters())
            if saved.keys() != expected.keys():
                raise ValueError(f"Checkpoint state keys mismatch: {name}")
            for key, value in expected.items():
                tensor = saved[key]
                if not isinstance(tensor, torch.Tensor) or tensor.shape != value.shape or tensor.dtype != value.dtype:
                    raise ValueError(f"Checkpoint state shape/dtype mismatch: {name}.{key}")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Checkpoint nonfinite state: {name}.{key}")
                if key not in parameters and not torch.equal(tensor, value):
                    raise ValueError(f"Checkpoint fixed state mismatch: {name}.{key}")
        update_steps = checkpoint.get("update_steps", 0)
        if isinstance(update_steps, bool) or not isinstance(update_steps, int) or update_steps < 0:
            raise ValueError("Checkpoint update_steps must be a non-negative integer")
        for name in ("actors", "target_actors", "critics", "target_critics"):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.update_steps = update_steps
